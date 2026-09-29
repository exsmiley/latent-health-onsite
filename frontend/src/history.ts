import type { AssistantState } from "./trace";
import type { ChatMessage } from "./types";

export interface HistoryExchange {
  question: string;
  assistant: AssistantState;
}

/** An answer counts as part of the conversation only if it finished cleanly. */
export function isCompleteAnswer(s: AssistantState): boolean {
  return s.phase === "done" && s.errors.length === 0 && s.text.trim() !== "";
}

/**
 * Conversation history for the API: questions plus final answer text only (never the trace).
 * Stopped, failed or empty answers are left out, so a truncated reply is never passed off as
 * a finished one.
 */
export function toHistory(exchanges: HistoryExchange[]): ChatMessage[] {
  const out: ChatMessage[] = [];
  for (const ex of exchanges) {
    out.push({ role: "user", content: ex.question });
    if (isCompleteAnswer(ex.assistant)) out.push({ role: "assistant", content: ex.assistant.text });
  }
  return out;
}
