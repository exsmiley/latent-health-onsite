// Event and data types for the HTTP / SSE contract in docs/ARCHITECTURE.md.

export type Role = "user" | "assistant";

export interface ChatMessage {
  role: Role;
  content: string;
}

export type Stage = "research" | "evaluate" | "respond";

export interface StatusData {
  stage: Stage;
  /** research: the turn starting (0: pre-retrieval); evaluate: the turn whose answer is checked; respond: null. */
  turn: number | null;
  max_turns: number;
  message: string;
}

export interface ToolCallData {
  id: string;
  name: string;
  arguments: Record<string, unknown>;
  /** 0: pre-retrieval, run by the harness before turn 1. */
  turn: number;
}

export interface ToolResultData {
  id: string;
  name: string;
  summary: string;
}

export type ResearchAnswerStatus = "answered" | "not_found" | "invalid";

/** Every final (tool-call-free) message from the research agent. "invalid" carries the error in `reason`. */
export interface ResearchAnswerData {
  turn: number;
  status: ResearchAnswerStatus;
  answer: string;
  citations: number[];
  reason: string;
}

export type Verdict = "supported" | "unsupported";

export interface EvaluationData {
  turn: number;
  verdict: Verdict;
  independent_answer: string;
  feedback: string;
}

export type OutcomeResult = "supported" | "not_found" | "out_of_turns";

/** Sent once, right before `citations`. */
export interface OutcomeData {
  result: OutcomeResult;
  turns_used: number;
}

export interface Citation {
  n: number;
  chunk_id: number;
  article_id: number;
  title: string;
  section: string | null;
  url: string;
}

export interface TokenData {
  delta: string;
}

export interface ErrorData {
  message: string;
}

export type ChatEvent =
  | { type: "status"; data: StatusData }
  | { type: "tool_call"; data: ToolCallData }
  | { type: "tool_result"; data: ToolResultData }
  | { type: "research_answer"; data: ResearchAnswerData }
  | { type: "evaluation"; data: EvaluationData }
  | { type: "outcome"; data: OutcomeData }
  | { type: "citations"; data: Citation[] }
  | { type: "token"; data: TokenData }
  | { type: "done"; data: Record<string, never> }
  | { type: "error"; data: ErrorData };

export type ChatEventType = ChatEvent["type"];

export const EVENT_TYPES: readonly ChatEventType[] = [
  "status",
  "tool_call",
  "tool_result",
  "research_answer",
  "evaluation",
  "outcome",
  "citations",
  "token",
  "done",
  "error",
];

/** `GET /api/chunks/{id}` response (rag.tools.models.Chunk). */
export interface Chunk {
  chunk_id: number;
  article_id: number;
  title: string;
  url: string;
  section: string | null;
  chunk_index: number;
  text: string;
}
