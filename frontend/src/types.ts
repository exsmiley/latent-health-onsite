// Event and data types for the HTTP / SSE contract in docs/ARCHITECTURE.md.

export type Role = "user" | "assistant";

export interface ChatMessage {
  role: Role;
  content: string;
}

export type Stage = "research" | "evaluate" | "respond";

export interface StatusData {
  stage: Stage;
  round: number;
  turn: number | null;
  message: string;
}

export interface ToolCallData {
  id: string;
  name: string;
  arguments: Record<string, unknown>;
  round: number;
  turn: number;
}

export interface ToolResultData {
  id: string;
  name: string;
  summary: string;
}

export interface ResearchAnswerData {
  round: number;
  answer: string;
  citations: number[];
}

export interface EvaluationData {
  round: number;
  verdict: "supported" | "unsupported";
  independent_answer: string;
  feedback: string;
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
