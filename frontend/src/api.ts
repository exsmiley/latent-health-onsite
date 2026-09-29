import { SseParser } from "./sse";
import { EVENT_TYPES, type ChatEvent, type ChatEventType, type ChatMessage, type Chunk } from "./types";
import { mockChunk, mockChatStream } from "./mock";
import type { EvalRun, EvalRunListItem } from "./evalTypes";

export type EventHandler = (event: ChatEvent) => void;

/** Mock mode: `?mock=1` in the URL, or VITE_MOCK=1 at build/dev time. */
export function isMockMode(): boolean {
  if (import.meta.env.VITE_MOCK === "1") return true;
  try {
    return new URLSearchParams(window.location.search).get("mock") === "1";
  } catch {
    return false;
  }
}

/**
 * POST /api/chat and dispatch each parsed SSE event to `onEvent`.
 * Resolves when the stream ends (or rejects with AbortError on abort).
 */
export async function streamChat(
  messages: ChatMessage[],
  onEvent: EventHandler,
  signal: AbortSignal,
): Promise<void> {
  if (isMockMode()) return mockChatStream(messages, onEvent, signal);

  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({ messages }),
    signal,
  });
  if (!res.ok || !res.body) {
    const detail = await res.text().catch(() => "");
    throw new Error(
      `Request failed: ${res.status} ${res.statusText}${detail ? ` - ${detail.slice(0, 300)}` : ""}`,
    );
  }

  const parser = new SseParser((msg) => {
    if (!(EVENT_TYPES as readonly string[]).includes(msg.event)) return; // unknown event: ignore
    let data: unknown;
    try {
      data = msg.data === "" ? {} : JSON.parse(msg.data);
    } catch {
      onEvent({ type: "error", data: { message: `Malformed ${msg.event} event data` } });
      return;
    }
    onEvent({ type: msg.event as ChatEventType, data } as ChatEvent);
  });

  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      parser.feed(value);
    }
    parser.end();
  } finally {
    reader.releaseLock();
  }
}

const chunkCache = new Map<number, Promise<Chunk>>();

/** GET /api/chunks/{id}, memoized. Failed requests are evicted so they can be retried. */
export function fetchChunk(id: number): Promise<Chunk> {
  let p = chunkCache.get(id);
  if (!p) {
    p = isMockMode()
      ? mockChunk(id)
      : fetch(`/api/chunks/${id}`).then(async (r) => {
          if (r.status === 404) throw new Error(`Chunk ${id} not found`);
          if (!r.ok) throw new Error(`Failed to load chunk ${id} (${r.status})`);
          return (await r.json()) as Chunk;
        });
    p.catch(() => chunkCache.delete(id));
    chunkCache.set(id, p);
  }
  return p;
}

async function getJson<T>(url: string, what: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`Failed to load ${what} (${r.status})`);
  return (await r.json()) as T;
}

/** GET /api/evals: every eval run, newest first. */
export async function fetchEvalRuns(): Promise<EvalRunListItem[]> {
  return (await getJson<{ runs: EvalRunListItem[] }>("/api/evals", "eval runs")).runs;
}

/** GET /api/evals/{id}: one run with its summary and per-question records. */
export function fetchEvalRun(id: string): Promise<EvalRun> {
  return getJson<EvalRun>(`/api/evals/${encodeURIComponent(id)}`, `eval run ${id}`);
}
