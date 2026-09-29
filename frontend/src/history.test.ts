import { describe, expect, it } from "vitest";
import { toHistory } from "./history";
import { initialAssistantState, type AssistantState } from "./trace";

const answer = (over: Partial<AssistantState>): AssistantState => ({
  ...initialAssistantState(),
  ...over,
});

describe("toHistory", () => {
  it("includes finished answers", () => {
    const h = toHistory([{ question: "Q1", assistant: answer({ phase: "done", text: "A1 [1]" }) }]);
    expect(h).toEqual([
      { role: "user", content: "Q1" },
      { role: "assistant", content: "A1 [1]" },
    ]);
  });

  it("drops stopped, failed, still-running and empty answers but keeps their questions", () => {
    const h = toHistory([
      { question: "stopped", assistant: answer({ phase: "stopped", text: "Half an ans" }) },
      { question: "failed", assistant: answer({ phase: "done", text: "Partial", errors: ["boom"] }) },
      { question: "running", assistant: answer({ phase: "running", text: "Stream" }) },
      { question: "empty", assistant: answer({ phase: "done", text: "  " }) },
      { question: "ok", assistant: answer({ phase: "done", text: "Fine." }) },
    ]);
    expect(h).toEqual([
      { role: "user", content: "stopped" },
      { role: "user", content: "failed" },
      { role: "user", content: "running" },
      { role: "user", content: "empty" },
      { role: "user", content: "ok" },
      { role: "assistant", content: "Fine." },
    ]);
  });
});
