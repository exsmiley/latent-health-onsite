import { describe, expect, it } from "vitest";
import { initialAssistantState, reduce } from "./trace";
import type { ChatEvent } from "./types";

describe("reduce", () => {
  it("attaches tool results by id and groups by round/turn", () => {
    const events: ChatEvent[] = [
      { type: "status", data: { stage: "research", round: 1, turn: 1, message: "" } },
      { type: "tool_call", data: { id: "a", name: "semantic_search", arguments: { queries: ["x"] }, round: 1, turn: 1 } },
      { type: "tool_call", data: { id: "b", name: "fetch", arguments: { chunk_ids: [1] }, round: 1, turn: 1 } },
      { type: "tool_result", data: { id: "b", name: "fetch", summary: "1 chunk" } },
      { type: "evaluation", data: { round: 1, verdict: "unsupported", independent_answer: "", feedback: "f" } },
      { type: "status", data: { stage: "research", round: 2, turn: 1, message: "" } },
      { type: "token", data: { delta: "Hel" } },
      { type: "token", data: { delta: "lo" } },
      { type: "done", data: {} },
    ];
    const s = events.reduce(reduce, initialAssistantState());
    expect(s.rounds.map((r) => r.round)).toEqual([1, 2]);
    const calls = s.rounds[0].turns[0].calls;
    expect(calls[0].result).toBeUndefined();
    expect(calls[1].result?.summary).toBe("1 chunk");
    expect(s.rounds[0].evaluation?.verdict).toBe("unsupported");
    expect(s.text).toBe("Hello");
    expect(s.phase).toBe("done");
  });
});
