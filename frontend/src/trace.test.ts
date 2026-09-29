import { describe, expect, it } from "vitest";
import { continuation, formatArgs, initialAssistantState, reduce, summaryLine, turnsUsed } from "./trace";
import type { ChatEvent, Stage } from "./types";

const MAX = 7;
const status = (stage: Stage, turn: number | null): ChatEvent => ({
  type: "status",
  data: { stage, turn, max_turns: MAX, message: "" },
});
const call = (id: string, turn: number, name = "semantic_search"): ChatEvent => ({
  type: "tool_call",
  data: { id, name, arguments: { queries: ["q"] }, turn },
});
const result = (id: string, summary: string): ChatEvent => ({
  type: "tool_result",
  data: { id, name: "semantic_search", summary },
});
const answered = (turn: number, citations = [1]): ChatEvent => ({
  type: "research_answer",
  data: { turn, status: "answered", answer: "A", citations, reason: "" },
});
const evaluation = (turn: number, verdict: "supported" | "unsupported"): ChatEvent => ({
  type: "evaluation",
  data: { turn, verdict, independent_answer: "", feedback: `fb${turn}` },
});

const run = (events: ChatEvent[]) => events.reduce(reduce, initialAssistantState());

describe("reduce", () => {
  it("builds one flat timeline with several answer/evaluation cycles", () => {
    const s = run([
      status("research", 1),
      call("a", 1),
      result("a", "1 query, 5 hits"),
      status("research", 2),
      answered(2, [10]),
      status("evaluate", 2),
      evaluation(2, "unsupported"),
      status("research", 3),
      call("b", 3),
      result("b", "2 chunks"),
      status("research", 4),
      answered(4, [10, 11]),
      status("evaluate", 4),
      evaluation(4, "supported"),
      status("respond", null),
      { type: "outcome", data: { result: "supported", turns_used: 4 } },
      { type: "citations", data: [] },
      { type: "token", data: { delta: "Hel" } },
      { type: "token", data: { delta: "lo" } },
      { type: "done", data: {} },
    ]);
    expect(s.turns.map((t) => t.turn)).toEqual([1, 2, 3, 4]);
    expect(s.maxTurns).toBe(MAX);
    expect(s.turns[1].answer?.citations).toEqual([10]);
    expect(s.turns[1].evaluation?.verdict).toBe("unsupported");
    expect(s.turns[3].answer?.citations).toEqual([10, 11]);
    expect(s.turns[3].evaluation?.verdict).toBe("supported");
    expect(s.turns[0].answer).toBeUndefined();
    expect(continuation(s.turns[1], s.maxTurns)).toEqual({ kind: "continue", turnsLeft: 5 });
    expect(continuation(s.turns[3], s.maxTurns)).toBeNull();
    expect(s.outcome).toEqual({ result: "supported", turns_used: 4 });
    expect(summaryLine(s)).toBe("4 turns · 2 tool calls · ✓ Supported");
    expect(s.text).toBe("Hello");
    expect(s.phase).toBe("done");
  });

  it("attaches tool results to the turn that made the call", () => {
    const s = run([
      status("research", 1),
      call("a", 1),
      call("b", 1, "keyword_search"),
      status("research", 2),
      call("c", 2, "fetch"),
      // Results may arrive late and out of order.
      result("c", "2 chunks"),
      result("a", "1 query, 5 hits"),
    ]);
    const [t1, t2] = s.turns;
    expect(t1.calls.map((c) => [c.call.id, c.result?.summary])).toEqual([
      ["a", "1 query, 5 hits"],
      ["b", undefined],
    ]);
    expect(t2.calls.map((c) => [c.call.id, c.result?.summary])).toEqual([["c", "2 chunks"]]);
  });

  it("creates a turn from a tool_call even without a preceding status", () => {
    const s = run([call("x", 3)]);
    expect(s.turns.map((t) => t.turn)).toEqual([3]);
    expect(s.maxTurns).toBeNull();
    expect(continuation({ turn: 3, calls: [], answer: { turn: 3, status: "invalid", answer: "", citations: [], reason: "e" } }, null))
      .toEqual({ kind: "continue", turnsLeft: null });
  });

  it("ends on a not_found answer without evaluation", () => {
    const s = run([
      status("research", 1),
      call("a", 1),
      result("a", "1 query, 0 hits"),
      status("research", 2),
      {
        type: "research_answer",
        data: { turn: 2, status: "not_found", answer: "", citations: [], reason: "searched X, no Y" },
      },
      { type: "outcome", data: { result: "not_found", turns_used: 2 } },
      { type: "citations", data: [] },
      { type: "token", data: { delta: "Couldn't find it." } },
      { type: "done", data: {} },
    ]);
    const t2 = s.turns[1];
    expect(t2.answer?.status).toBe("not_found");
    expect(t2.answer?.reason).toBe("searched X, no Y");
    expect(t2.evaluation).toBeUndefined();
    expect(continuation(t2, s.maxTurns)).toBeNull();
    expect(summaryLine(s)).toBe("2 turns · 1 tool call · ✗ Not found");
  });

  it("marks invalid answers as continuing research", () => {
    const s = run([
      status("research", 1),
      {
        type: "research_answer",
        data: { turn: 1, status: "invalid", answer: "", citations: [], reason: "citations must be chunk ids" },
      },
      status("research", 2),
      call("a", 2),
    ]);
    expect(s.turns[0].answer?.status).toBe("invalid");
    expect(s.turns[0].answer?.reason).toBe("citations must be chunk ids");
    expect(continuation(s.turns[0], s.maxTurns)).toEqual({ kind: "continue", turnsLeft: 6 });
    expect(s.turns[1].calls).toHaveLength(1);
    expect(s.outcome).toBeNull();
    expect(turnsUsed(s)).toBe(2);
  });

  it("reports out_of_turns when the budget runs out", () => {
    const events: ChatEvent[] = [];
    for (let t = 1; t <= 6; t++) events.push(status("research", t), call(`c${t}`, t), result(`c${t}`, "hits"));
    events.push(status("research", 7), answered(7), status("evaluate", 7), evaluation(7, "unsupported"));
    events.push({ type: "outcome", data: { result: "out_of_turns", turns_used: 7 } }, { type: "done", data: {} });
    const s = run(events);
    expect(s.turns).toHaveLength(7);
    expect(continuation(s.turns[6], s.maxTurns)).toEqual({ kind: "exhausted" });
    expect(summaryLine(s)).toBe("7 turns · 6 tool calls · Out of turns");
  });
});

describe("formatArgs", () => {
  it("renders queries, ids and top_k compactly", () => {
    expect(formatArgs({ queries: ["a", "b"], top_k: 5 })).toBe('"a", "b" · top_k: 5');
    expect(formatArgs({ chunk_ids: [1, 2] })).toBe("chunk_ids: [1, 2]");
  });
});
