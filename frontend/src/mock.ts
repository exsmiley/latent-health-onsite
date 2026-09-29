// In-browser fake backend for `?mock=1` / VITE_MOCK=1. Emits a scripted, realistic event
// sequence with delays: round 1 (2 research turns) -> unsupported -> round 2 -> supported ->
// citations -> streamed tokens -> done.

import type { ChatEvent, ChatMessage, Chunk } from "./types";

const WIKI = "https://simple.wikipedia.org/wiki/";

const CHUNKS: Record<number, Chunk> = {
  4101: {
    chunk_id: 4101,
    article_id: 812,
    title: "Eiffel Tower",
    url: `${WIKI}Eiffel%20Tower`,
    section: null,
    chunk_index: 0,
    text:
      "The Eiffel Tower is a tower made of iron in Paris, France. It was designed by the engineer " +
      "Gustave Eiffel and his company. It was built for the 1889 World's Fair (Exposition " +
      "Universelle), which celebrated 100 years since the French Revolution. Building started in " +
      "January 1887 and the tower was finished in March 1889.\n\nAt first, many artists and writers " +
      "in Paris did not like the tower. Today it is one of the most famous buildings in the world.",
  },
  4102: {
    chunk_id: 4102,
    article_id: 812,
    title: "Eiffel Tower",
    url: `${WIKI}Eiffel%20Tower`,
    section: "Visitors",
    chunk_index: 2,
    text:
      "The tower is the most visited paid monument in the world. About 7 million people visit it " +
      "every year. Visitors can walk up the stairs to the first and second levels, or take a lift. " +
      "The top level can only be reached by lift.",
  },
  4103: {
    chunk_id: 4103,
    article_id: 812,
    title: "Eiffel Tower",
    url: `${WIKI}Eiffel%20Tower`,
    section: "Structure",
    chunk_index: 1,
    text:
      "The tower is 330 metres (1,083 ft) tall, including the antennas at the top. Without the " +
      "antennas it is 300 metres (984 ft). It was the tallest man-made structure in the world " +
      "until the Chrysler Building in New York City was finished in 1930.\n\nThe tower has three " +
      "levels for visitors. It weighs about 10,100 tonnes.",
  },
  9377: {
    chunk_id: 9377,
    article_id: 2290,
    title: "Gustave Eiffel",
    url: `${WIKI}Gustave%20Eiffel`,
    section: "Career",
    chunk_index: 3,
    text:
      "Alexandre Gustave Eiffel (1832 - 1923) was a French civil engineer. He is best known for " +
      "the Eiffel Tower in Paris. His company also built the metal frame inside the Statue of " +
      "Liberty in New York.",
  },
};

const FINAL_ANSWER =
  "The **Eiffel Tower** was built between **January 1887 and March 1889** for the 1889 World's " +
  "Fair in Paris, which marked 100 years since the French Revolution [1]. It was designed by " +
  "Gustave Eiffel's company [1][3].\n\n" +
  "It is **330 metres (1,083 ft)** tall including its antennas, or 300 metres without them [2]. " +
  "When it opened it was the tallest man-made structure in the world, a record it held until " +
  "the Chrysler Building was finished in 1930 [2].";

type Step = [delayMs: number, event: ChatEvent];

function script(question: string): Step[] {
  const q = question.length > 60 ? question.slice(0, 57) + "..." : question;
  return [
    [250, { type: "status", data: { stage: "research", round: 1, turn: 1, message: `Researching "${q}"` } }],
    [500, { type: "tool_call", data: { id: "call_a1", name: "semantic_search", round: 1, turn: 1, arguments: { queries: ["When was the Eiffel Tower built?", "Eiffel Tower construction history"], top_k: 5 } } }],
    [80, { type: "tool_call", data: { id: "call_a2", name: "keyword_search", round: 1, turn: 1, arguments: { queries: ["Eiffel Tower 1889"], top_k: 5 } } }],
    [650, { type: "tool_result", data: { id: "call_a2", name: "keyword_search", summary: "1 query, 5 hits" } }],
    [200, { type: "tool_result", data: { id: "call_a1", name: "semantic_search", summary: "2 queries, 10 hits" } }],
    [300, { type: "status", data: { stage: "research", round: 1, turn: 2, message: "Reading chunks" } }],
    [500, { type: "tool_call", data: { id: "call_a3", name: "fetch", round: 1, turn: 2, arguments: { chunk_ids: [4101, 4102] } } }],
    [450, { type: "tool_result", data: { id: "call_a3", name: "fetch", summary: "2 chunks" } }],
    [700, { type: "research_answer", data: { round: 1, answer: "The Eiffel Tower was built from 1887 to 1889 for the World's Fair. It is about 330 metres tall.", citations: [4101, 4102] } }],
    [250, { type: "status", data: { stage: "evaluate", round: 1, turn: null, message: "Checking the answer against the cited chunks" } }],
    [1100, { type: "evaluation", data: { round: 1, verdict: "unsupported", independent_answer: "The Eiffel Tower was built between 1887 and 1889 for the 1889 World's Fair. The cited chunks do not give its height.", feedback: "The construction dates are supported by chunk 4101, but neither cited chunk states the height of 330 metres. Cite a chunk that gives the height." } }],
    [300, { type: "status", data: { stage: "research", round: 2, turn: 1, message: "Retrying with evaluator feedback" } }],
    [550, { type: "tool_call", data: { id: "call_b1", name: "keyword_search", round: 2, turn: 1, arguments: { queries: ["Eiffel Tower height metres", "Eiffel Tower tall antennas"], top_k: 5 } } }],
    [600, { type: "tool_result", data: { id: "call_b1", name: "keyword_search", summary: "2 queries, 9 hits" } }],
    [300, { type: "status", data: { stage: "research", round: 2, turn: 2, message: "Reading chunks" } }],
    [450, { type: "tool_call", data: { id: "call_b2", name: "fetch", round: 2, turn: 2, arguments: { chunk_ids: [4103, 9377] } } }],
    [400, { type: "tool_result", data: { id: "call_b2", name: "fetch", summary: "2 chunks" } }],
    [650, { type: "research_answer", data: { round: 2, answer: "The Eiffel Tower was built from January 1887 to March 1889 for the 1889 World's Fair, designed by Gustave Eiffel's company. It is 330 m tall with antennas (300 m without).", citations: [4101, 4103, 9377] } }],
    [250, { type: "status", data: { stage: "evaluate", round: 2, turn: null, message: "Checking the answer against the cited chunks" } }],
    [1000, { type: "evaluation", data: { round: 2, verdict: "supported", independent_answer: "Built 1887-1889 for the World's Fair by Gustave Eiffel's company; 330 m tall including antennas.", feedback: "All claims are supported by the cited chunks." } }],
    [250, { type: "status", data: { stage: "respond", round: 2, turn: null, message: "Writing the final answer" } }],
    [300, {
      type: "citations",
      data: [
        { n: 1, chunk_id: 4101, article_id: 812, title: "Eiffel Tower", section: null, url: `${WIKI}Eiffel%20Tower` },
        { n: 2, chunk_id: 4103, article_id: 812, title: "Eiffel Tower", section: "Structure", url: `${WIKI}Eiffel%20Tower` },
        { n: 3, chunk_id: 9377, article_id: 2290, title: "Gustave Eiffel", section: "Career", url: `${WIKI}Gustave%20Eiffel` },
      ],
    }],
  ];
}

function sleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) return reject(new DOMException("Aborted", "AbortError"));
    const t = setTimeout(() => {
      signal.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    const onAbort = () => {
      clearTimeout(t);
      reject(new DOMException("Aborted", "AbortError"));
    };
    signal.addEventListener("abort", onAbort, { once: true });
  });
}

export async function mockChatStream(
  messages: ChatMessage[],
  onEvent: (e: ChatEvent) => void,
  signal: AbortSignal,
): Promise<void> {
  const question = messages[messages.length - 1]?.content ?? "";
  if (/\berror\b/i.test(question)) {
    // Handy for demoing the inline error state: ask anything containing "error".
    await sleep(400, signal);
    onEvent({ type: "status", data: { stage: "research", round: 1, turn: 1, message: "Researching" } });
    await sleep(600, signal);
    onEvent({ type: "error", data: { message: "Mock error: the research agent failed (simulated)." } });
    onEvent({ type: "done", data: {} });
    return;
  }
  for (const [delay, ev] of script(question)) {
    await sleep(delay, signal);
    onEvent(ev);
  }
  // Stream the final answer in small, irregular pieces, like a model would.
  const pieces = FINAL_ANSWER.match(/\S+\s*|\s+/g) ?? [];
  for (const p of pieces) {
    await sleep(25 + Math.random() * 45, signal);
    onEvent({ type: "token", data: { delta: p } });
  }
  await sleep(100, signal);
  onEvent({ type: "done", data: {} });
}

export async function mockChunk(id: number): Promise<Chunk> {
  await new Promise((r) => setTimeout(r, 300));
  const c = CHUNKS[id];
  if (!c) throw new Error(`Chunk ${id} not found`);
  return c;
}
