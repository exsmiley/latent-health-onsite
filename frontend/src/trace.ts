// Pure reducer that folds the SSE event stream into the view state of one assistant message.

import type {
  ChatEvent,
  Citation,
  EvaluationData,
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
}

export interface RoundState {
  round: number;
  turns: TurnState[];
  answer?: ResearchAnswerData;
  evaluation?: EvaluationData;
}

export type Phase = "running" | "done" | "stopped";

export interface AssistantState {
  status: StatusData | null;
  rounds: RoundState[];
  citations: Citation[] | null;
  text: string;
  errors: string[];
  phase: Phase;
}

export const initialAssistantState = (): AssistantState => ({
  status: null,
  rounds: [],
  citations: null,
  text: "",
  errors: [],
  phase: "running",
});

function withRound(
  rounds: RoundState[],
  round: number,
  update: (r: RoundState) => RoundState,
): RoundState[] {
  const idx = rounds.findIndex((r) => r.round === round);
  if (idx === -1) {
    const next = [...rounds, update({ round, turns: [] })];
    return next.sort((a, b) => a.round - b.round);
  }
  return rounds.map((r, i) => (i === idx ? update(r) : r));
}

function withTurn(r: RoundState, turn: number, update: (t: TurnState) => TurnState): RoundState {
  const idx = r.turns.findIndex((t) => t.turn === turn);
  const turns =
    idx === -1
      ? [...r.turns, update({ turn, calls: [] })].sort((a, b) => a.turn - b.turn)
      : r.turns.map((t, i) => (i === idx ? update(t) : t));
  return { ...r, turns };
}

export function reduce(state: AssistantState, ev: ChatEvent): AssistantState {
  switch (ev.type) {
    case "status": {
      const d = ev.data;
      const rounds = withRound(state.rounds, d.round, (r) =>
        d.stage === "research" && d.turn != null ? withTurn(r, d.turn, (t) => t) : r,
      );
      return { ...state, status: d, rounds };
    }
    case "tool_call": {
      const d = ev.data;
      return {
        ...state,
        rounds: withRound(state.rounds, d.round, (r) =>
          withTurn(r, d.turn, (t) => ({ ...t, calls: [...t.calls, { call: d }] })),
        ),
      };
    }
    case "tool_result": {
      const d = ev.data;
      return {
        ...state,
        rounds: state.rounds.map((r) => ({
          ...r,
          turns: r.turns.map((t) => ({
            ...t,
            calls: t.calls.map((c) => (c.call.id === d.id ? { ...c, result: d } : c)),
          })),
        })),
      };
    }
    case "research_answer":
      return { ...state, rounds: withRound(state.rounds, ev.data.round, (r) => ({ ...r, answer: ev.data })) };
    case "evaluation":
      return { ...state, rounds: withRound(state.rounds, ev.data.round, (r) => ({ ...r, evaluation: ev.data })) };
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

/** Compact one-line rendering of tool-call arguments. */
export function formatArgs(name: string, args: Record<string, unknown>): string {
  if (!args || typeof args !== "object") return "";
  const parts: string[] = [];
  const queries = args.queries;
  if (Array.isArray(queries)) parts.push(queries.map((q) => `"${String(q)}"`).join(", "));
  for (const key of ["chunk_ids", "article_ids", "citations"]) {
    const v = args[key];
    if (Array.isArray(v) && v.length) parts.push(`${key}: [${v.join(", ")}]`);
  }
  if (name === "submit_answer" && typeof args.answer === "string") {
    const a = args.answer;
    parts.unshift(`"${a.length > 80 ? a.slice(0, 77) + "..." : a}"`);
  }
  const handled = new Set(["queries", "chunk_ids", "article_ids", "citations", "answer", "top_k"]);
  for (const [k, v] of Object.entries(args)) {
    if (handled.has(k)) continue;
    parts.push(`${k}: ${typeof v === "string" ? v : JSON.stringify(v)}`);
  }
  if (typeof args.top_k === "number") parts.push(`top_k: ${args.top_k}`);
  return parts.join(" · ");
}
