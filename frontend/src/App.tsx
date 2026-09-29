import { useCallback, useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import { isMockMode, streamChat } from "./api";
import { AssistantMessage } from "./components/AssistantMessage";
import { initialAssistantState, reduce, type AssistantState } from "./trace";
import type { ChatEvent, ChatMessage } from "./types";

interface Exchange {
  id: number;
  question: string;
  assistant: AssistantState;
}

const EXAMPLES = [
  "When was the Eiffel Tower built, and how tall is it?",
  "Why is the sky blue?",
  "Who was Ada Lovelace?",
];

/** Conversation history for the API: questions plus final answer text only (never the trace). */
function toHistory(exchanges: Exchange[]): ChatMessage[] {
  const out: ChatMessage[] = [];
  for (const ex of exchanges) {
    out.push({ role: "user", content: ex.question });
    if (ex.assistant.text.trim()) out.push({ role: "assistant", content: ex.assistant.text });
  }
  return out;
}

export default function App() {
  const [exchanges, setExchanges] = useState<Exchange[]>([]);
  const [input, setInput] = useState("");
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const stickToBottom = useRef(true);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [busy, setBusy] = useState(false);
  const mock = isMockMode();

  // Keep the view pinned to the bottom while streaming, unless the user scrolled up.
  useEffect(() => {
    const onScroll = () => {
      const el = document.scrollingElement ?? document.documentElement;
      stickToBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 120;
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);
  useEffect(() => {
    if (stickToBottom.current) bottomRef.current?.scrollIntoView({ block: "end" });
  }, [exchanges]);

  const update = useCallback((id: number, fn: (s: AssistantState) => AssistantState) => {
    setExchanges((xs) => xs.map((x) => (x.id === id ? { ...x, assistant: fn(x.assistant) } : x)));
  }, []);

  const send = useCallback(
    async (question: string) => {
      const q = question.trim();
      if (!q || abortRef.current) return;
      const id = Date.now();
      const messages = [...toHistory(exchanges), { role: "user" as const, content: q }];
      setExchanges((xs) => [...xs, { id, question: q, assistant: initialAssistantState() }]);
      setInput("");
      stickToBottom.current = true;

      const ctrl = new AbortController();
      abortRef.current = ctrl;
      setBusy(true);
      let sawDone = false;
      let sawError = false;
      const onEvent = (ev: ChatEvent) => {
        if (ev.type === "done") sawDone = true;
        if (ev.type === "error") sawError = true;
        update(id, (s) => reduce(s, ev));
      };
      try {
        await streamChat(messages, onEvent, ctrl.signal);
        if (!sawDone) {
          update(id, (s) => ({
            ...s,
            phase: "done",
            errors: sawError ? s.errors : [...s.errors, "The connection closed before the answer finished."],
          }));
        }
      } catch (e) {
        if (ctrl.signal.aborted) {
          update(id, (s) => ({ ...s, phase: "stopped" }));
        } else {
          const msg = e instanceof Error ? e.message : String(e);
          update(id, (s) => ({ ...s, phase: "done", errors: [...s.errors, msg] }));
        }
      } finally {
        abortRef.current = null;
        setBusy(false);
        textareaRef.current?.focus();
      }
    },
    [exchanges, update],
  );

  const stop = () => abortRef.current?.abort();

  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    void send(input);
  };
  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      void send(input);
    }
  };

  // Auto-grow the textarea.
  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = `${Math.min(ta.scrollHeight, 200)}px`;
  }, [input]);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="logo" aria-hidden>
            W
          </span>
          Wiki RAG
        </div>
        <div className="topbar-right">
          {mock && <span className="tag mock">mock mode</span>}
          {exchanges.length > 0 && (
            <button className="ghost-btn" onClick={() => !busy && setExchanges([])} disabled={busy}>
              New chat
            </button>
          )}
        </div>
      </header>

      <main className="log">
        {exchanges.length === 0 ? (
          <div className="empty">
            <h1>Ask Simple English Wikipedia</h1>
            <p className="muted">
              Answers are researched, checked by an evaluator against the cited passages, and then written
              with citations.
            </p>
            <div className="examples">
              {EXAMPLES.map((ex) => (
                <button key={ex} className="example" onClick={() => void send(ex)}>
                  {ex}
                </button>
              ))}
            </div>
          </div>
        ) : (
          exchanges.map((x) => (
            <section key={x.id} className="exchange">
              <div className="msg user">{x.question}</div>
              <AssistantMessage state={x.assistant} />
            </section>
          ))
        )}
        <div ref={bottomRef} />
      </main>

      <form className="composer" onSubmit={onSubmit}>
        <div className="composer-inner">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Ask a question…"
            rows={1}
            autoFocus
            aria-label="Question"
          />
          {busy ? (
            <button type="button" className="send stop" onClick={stop}>
              Stop
            </button>
          ) : (
            <button type="submit" className="send" disabled={!input.trim()}>
              Send
            </button>
          )}
        </div>
      </form>
    </div>
  );
}
