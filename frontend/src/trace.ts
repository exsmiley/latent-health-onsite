// Pure reducer that folds the SSE event stream into the view state of one assistant message.
// The research agent has one flat budget of turns; the view model is a single timeline of turns.
// A turn holds its tool calls and, when the agent gave a final answer on that turn, the
// research_answer and the evaluator's verdict on it.

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

export interface ToolCallState {
  call: ToolCallData;
  result?: ToolResultData;
}

export interface TurnState {
  turn: number;
  calls: ToolCallState[];
  answer?: ResearchAnswerData;
  evaluation?: EvaluationData;
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
}

export const initialAssistantState = (): AssistantState => ({
  status: null,
  maxTurns: null,
  turns: [],
  outcome: null,
  citations: null,
  text: "",
  errors: [],
  phase: "running",
});

function withTurn(turns: TurnState[], turn: number, update: (t: TurnState) => TurnState): TurnState[] {
  const idx = turns.findIndex((t) => t.turn === turn);
  if (idx === -1) return [...turns, update({ turn, calls: [] })].sort((a, b) => a.turn - b.turn);
  return turns.map((t, i) => (i === idx ? update(t) : t));
}

export function reduce(state: AssistantState, ev: ChatEvent): AssistantState {
  switch (ev.type) {
    case "status": {
      const d = ev.data;
      const turns =
        d.stage === "research" && d.turn != null ? withTurn(state.turns, d.turn, (t) => t) : state.turns;
      return { ...state, status: d, maxTurns: d.max_turns ?? state.maxTurns, turns };
    }
    case "tool_call": {
      const d = ev.data;
      return { ...state, turns: withTurn(state.turns, d.turn, (t) => ({ ...t, calls: [...t.calls, { call: d }] })) };
    }
    case "tool_result": {
      const d = ev.data;
      return {
        ...state,
        turns: state.turns.map((t) =>
          t.calls.some((c) => c.call.id === d.id)
            ? { ...t, calls: t.calls.map((c) => (c.call.id === d.id ? { ...c, result: d } : c)) }
            : t,
        ),
      };
    }
    case "research_answer":
      return { ...state, turns: withTurn(state.turns, ev.data.turn, (t) => ({ ...t, answer: ev.data })) };
    case "evaluation":
      return { ...state, turns: withTurn(state.turns, ev.data.turn, (t) => ({ ...t, evaluation: ev.data })) };
    case "outcome":
      return { ...state, outcome: ev.data };
    case "citations":
      return { ...state, citations: ev.data };
    case "token":
      return { ...state, text: state.text + ev.data.delta };
    case "error":
      return { ...state, errors: [...state.errors, ev.data.message] };
    case "done":
      return { ...state, phase: "done" };
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
