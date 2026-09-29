// Pure reducer that folds the SSE event stream into the view state of one assistant message.
// The research agent has one flat budget of turns; the view model is a single timeline of turns.
// A turn holds its tool calls and, when the agent gave a final answer on that turn, the
// research_answer and the evaluator's verdict on it.
// Timings are client-side: each event is stamped with its arrival time (`now`, ms). Events
// stream live as the pipeline runs, so arrival times track the server's step boundaries.

import type {
  ChatEvent,
  Citation,
  EvaluationData,
  OutcomeData,
  OutcomeResult,
  ResearchAnswerData,
  StatusData,
  ToolCallData,
  ToolResultData,
} from "./types";

/** A timed step: `end` stays unset while the step is still running. */
export interface Span {
  start: number;
  end?: number;
}

export interface ToolCallState {
  call: ToolCallData;
  result?: ToolResultData;
  span?: Span;
}

export interface TurnState {
  turn: number;
  calls: ToolCallState[];
  answer?: ResearchAnswerData;
  evaluation?: EvaluationData;
  /** The research part of the turn: from its `status` to the next `status` (or the end). */
  span?: Span;
  /** The evaluator's check of this turn's answer. */
  evalSpan?: Span;
}

export type Phase = "running" | "done" | "stopped";

export interface AssistantState {
  status: StatusData | null;
  /** Turn budget, from the latest `status` event. */
  maxTurns: number | null;
  turns: TurnState[];
  outcome: OutcomeData | null;
  citations: Citation[] | null;
  text: string;
  errors: string[];
  phase: Phase;
  /** The whole query: from send to `done` (or stop / connection loss). */
  span: Span;
  /** Writing the final answer: from the `respond` status to the end. */
  respondSpan: Span | null;
}

export const initialAssistantState = (now: number = Date.now()): AssistantState => ({
  status: null,
  maxTurns: null,
  turns: [],
  outcome: null,
  citations: null,
  text: "",
  errors: [],
  phase: "running",
  span: { start: now },
  respondSpan: null,
});

const close = (span: Span | undefined, now: number): Span | undefined =>
  span && span.end == null ? { ...span, end: now } : span;

/** Close every research span still open; a new `status` or the end of the run ends it. */
const closeTurns = (turns: TurnState[], now: number): TurnState[] =>
  turns.map((t) => (t.span && t.span.end == null ? { ...t, span: close(t.span, now) } : t));

/** End the run: close every open step and set the final phase. */
export function finish(state: AssistantState, phase: Phase, now: number = Date.now()): AssistantState {
  return {
    ...state,
    phase,
    span: close(state.span, now)!,
    respondSpan: state.respondSpan && close(state.respondSpan, now)!,
    turns: closeTurns(state.turns, now).map((t) => ({
      ...t,
      evalSpan: close(t.evalSpan, now),
      calls: t.calls.map((c) => ({ ...c, span: close(c.span, now) })),
    })),
  };
}

function withTurn(turns: TurnState[], turn: number, update: (t: TurnState) => TurnState): TurnState[] {
  const idx = turns.findIndex((t) => t.turn === turn);
  if (idx === -1) return [...turns, update({ turn, calls: [] })].sort((a, b) => a.turn - b.turn);
  return turns.map((t, i) => (i === idx ? update(t) : t));
}

export function reduce(state: AssistantState, ev: ChatEvent, now: number = Date.now()): AssistantState {
  switch (ev.type) {
    case "status": {
      const d = ev.data;
      let turns = closeTurns(state.turns, now);
      if (d.stage === "research" && d.turn != null)
        turns = withTurn(turns, d.turn, (t) => ({ ...t, span: t.span ?? { start: now } }));
      if (d.stage === "evaluate" && d.turn != null)
        turns = withTurn(turns, d.turn, (t) => ({ ...t, evalSpan: { start: now } }));
      const respondSpan = d.stage === "respond" ? (state.respondSpan ?? { start: now }) : state.respondSpan;
      return { ...state, status: d, maxTurns: d.max_turns ?? state.maxTurns, turns, respondSpan };
    }
    case "tool_call": {
      const d = ev.data;
      const c: ToolCallState = { call: d, span: { start: now } };
      return { ...state, turns: withTurn(state.turns, d.turn, (t) => ({ ...t, calls: [...t.calls, c] })) };
    }
    case "tool_result": {
      const d = ev.data;
      return {
        ...state,
        turns: state.turns.map((t) =>
          t.calls.some((c) => c.call.id === d.id)
            ? {
                ...t,
                calls: t.calls.map((c) => (c.call.id === d.id ? { ...c, result: d, span: close(c.span, now) } : c)),
              }
            : t,
        ),
      };
    }
    case "research_answer":
      return { ...state, turns: withTurn(state.turns, ev.data.turn, (t) => ({ ...t, answer: ev.data })) };
    case "evaluation":
      return {
        ...state,
        turns: withTurn(state.turns, ev.data.turn, (t) => ({
          ...t,
          evaluation: ev.data,
          evalSpan: close(t.evalSpan, now),
        })),
      };
    case "outcome":
      return { ...state, outcome: ev.data };
    case "citations":
      return { ...state, citations: ev.data };
    case "token":
      return { ...state, text: state.text + ev.data.delta };
    case "error":
      return { ...state, errors: [...state.errors, ev.data.message] };
    case "done":
      return finish(state, "done", now);
    default:
      return state;
  }
}

export function toolCallCount(state: AssistantState): number {
  return state.turns.reduce((n, t) => n + t.calls.length, 0);
}

/** Turns used so far: the outcome's count when known, else the highest turn seen. */
export function turnsUsed(state: AssistantState): number {
  if (state.outcome) return state.outcome.turns_used;
  return state.turns.reduce((m, t) => Math.max(m, t.turn), 0);
}

/**
 * What follows a turn's final answer in the timeline, or null when nothing does.
 * An unsupported verdict or an invalid answer sends research on within the same budget.
 */
export function continuation(
  t: TurnState,
  maxTurns: number | null,
): { kind: "continue"; turnsLeft: number | null } | { kind: "exhausted" } | null {
  const rejected = t.answer?.status === "invalid" || t.evaluation?.verdict === "unsupported";
  if (!rejected) return null;
  if (maxTurns == null) return { kind: "continue", turnsLeft: null };
  const left = maxTurns - t.turn;
  return left > 0 ? { kind: "continue", turnsLeft: left } : { kind: "exhausted" };
}

export const OUTCOME_LABEL: Record<OutcomeResult, string> = {
  supported: "✓ Supported",
  not_found: "✗ Not found",
  out_of_turns: "Out of turns",
};

/** Length of a span; an open span runs until `now`. Null when the step never started. */
export function spanMs(span: Span | null | undefined, now: number): number | null {
  if (!span) return null;
  return Math.max(0, (span.end ?? now) - span.start);
}

/** Time per stage; research and evaluation sum over turns. Null for a stage that never ran. */
export function stageTimes(
  state: AssistantState,
  now: number,
): { research: number | null; evaluate: number | null; respond: number | null; total: number } {
  const sum = (spans: (Span | undefined)[]) => {
    const ms = spans.map((s) => spanMs(s, now)).filter((x): x is number => x != null);
    return ms.length ? ms.reduce((a, b) => a + b, 0) : null;
  };
  return {
    research: sum(state.turns.map((t) => t.span)),
    evaluate: sum(state.turns.map((t) => t.evalSpan)),
    respond: spanMs(state.respondSpan, now),
    total: spanMs(state.span, now)!,
  };
}

/** "850ms", "3.2s", "1m 05s". */
export function formatDuration(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  const s = Math.round(ms / 1000);
  return `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`;
}

const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`;

/** Collapsed summary line, e.g. "4 turns · 7 tool calls · ✓ Supported". */
export function summaryLine(state: AssistantState): string {
  const parts = [plural(turnsUsed(state), "turn")];
  const calls = toolCallCount(state);
  if (calls > 0) parts.push(plural(calls, "tool call"));
  if (state.outcome) parts.push(OUTCOME_LABEL[state.outcome.result]);
  return parts.join(" · ");
}

/** Compact one-line rendering of tool-call arguments. */
export function formatArgs(args: Record<string, unknown>): string {
  if (!args || typeof args !== "object") return "";
  const parts: string[] = [];
  const queries = args.queries;
  if (Array.isArray(queries)) parts.push(queries.map((q) => `"${String(q)}"`).join(", "));
  for (const key of ["chunk_ids", "article_ids"]) {
    const v = args[key];
    if (Array.isArray(v) && v.length) parts.push(`${key}: [${v.join(", ")}]`);
  }
  const handled = new Set(["queries", "chunk_ids", "article_ids", "top_k"]);
  for (const [k, v] of Object.entries(args)) {
    if (handled.has(k)) continue;
    parts.push(`${k}: ${typeof v === "string" ? v : JSON.stringify(v)}`);
  }
  if (typeof args.top_k === "number") parts.push(`top_k: ${args.top_k}`);
  return parts.join(" · ");
}
