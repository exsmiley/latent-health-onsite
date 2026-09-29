// In-browser fake backend for `?mock=1` / VITE_MOCK=1. Emits scripted, realistic event
// sequences with delays, following the contract in docs/ARCHITECTURE.md:
// - default: turn 1 searches, turn 2 fetches, turn 3 answers -> unsupported, turn 4 searches +
//   fetches, turn 5 answers -> supported -> outcome -> citations -> streamed tokens -> done.
// - question contains "notfound": two turns of searching, turn 3 answers "not_found" -> outcome
//   -> empty citations -> fixed "couldn't find" token -> done.
// - question contains "error": an error event.

import type { ChatEvent, ChatMessage, Chunk, Stage } from "./types";

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

const MAX_TURNS = 7;

const NOT_FOUND_TEXT = "Sorry, I couldn't find the answer to that in Simple English Wikipedia.";

const status = (stage: Stage, turn: number | null, message: string): ChatEvent => ({
  type: "status",
  data: { stage, turn, max_turns: MAX_TURNS, message },
});

const call = (id: string, name: string, turn: number, args: Record<string, unknown>): ChatEvent => ({
  type: "tool_call",
  data: { id, name, turn, arguments: args },
});

const result = (id: string, name: string, summary: string): ChatEvent => ({
  type: "tool_result",
  data: { id, name, summary },
});

function short(question: string): string {
  return question.length > 60 ? question.slice(0, 57) + "..." : question;
}

function supportedScript(question: string): Step[] {
  return [
    [250, status("research", 1, `Researching "${short(question)}"`)],
    [500, call("call_1a", "semantic_search", 1, { queries: ["When was the Eiffel Tower built?", "Eiffel Tower construction history"], top_k: 5 })],
    [80, call("call_1b", "keyword_search", 1, { queries: ["Eiffel Tower 1889"], top_k: 5 })],
    [650, result("call_1b", "keyword_search", "1 query, 5 hits")],
    [200, result("call_1a", "semantic_search", "2 queries, 10 hits")],
    [300, status("research", 2, "Reading chunks")],
    [500, call("call_2a", "fetch", 2, { chunk_ids: [4101, 4102] })],
    [450, result("call_2a", "fetch", "2 chunks")],
    [300, status("research", 3, "Deciding whether the evidence is enough")],
    [700, {
      type: "research_answer",
      data: {
        turn: 3,
        status: "answered",
        answer: "The Eiffel Tower was built from 1887 to 1889 for the World's Fair. It is about 330 metres tall.",
        citations: [4101, 4102],
        reason: "",
      },
    }],
    [250, status("evaluate", 3, "Checking the answer against the cited chunks")],
    [1100, {
      type: "evaluation",
      data: {
        turn: 3,
        verdict: "unsupported",
        independent_answer: "The Eiffel Tower was built between 1887 and 1889 for the 1889 World's Fair. The cited chunks do not give its height.",
        feedback: "The construction dates are supported by chunk 4101, but neither cited chunk states the height of 330 metres. Cite a chunk that gives the height.",
      },
    }],
    [300, status("research", 4, "Continuing research with the evaluator's feedback")],
    [550, call("call_4a", "keyword_search", 4, { queries: ["Eiffel Tower height metres", "Eiffel Tower tall antennas"], top_k: 5 })],
    [60, call("call_4b", "fetch", 4, { chunk_ids: [4103, 9377] })],
    [450, result("call_4b", "fetch", "2 chunks")],
    [200, result("call_4a", "keyword_search", "2 queries, 9 hits")],
    [300, status("research", 5, "Deciding whether the evidence is enough")],
    [650, {
      type: "research_answer",
      data: {
        turn: 5,
        status: "answered",
        answer: "The Eiffel Tower was built from January 1887 to March 1889 for the 1889 World's Fair, designed by Gustave Eiffel's company. It is 330 m tall with antennas (300 m without).",
        citations: [4101, 4103, 9377],
        reason: "",
      },
    }],
    [250, status("evaluate", 5, "Checking the answer against the cited chunks")],
    [1000, {
      type: "evaluation",
      data: {
        turn: 5,
        verdict: "supported",
        independent_answer: "Built 1887-1889 for the World's Fair by Gustave Eiffel's company; 330 m tall including antennas.",
        feedback: "All claims are supported by the cited chunks.",
      },
    }],
    [250, status("respond", null, "Writing the final answer")],
    [100, { type: "outcome", data: { result: "supported", turns_used: 5 } }],
    [200, {
      type: "citations",
      data: [
        { n: 1, chunk_id: 4101, article_id: 812, title: "Eiffel Tower", section: null, url: `${WIKI}Eiffel%20Tower` },
        { n: 2, chunk_id: 4103, article_id: 812, title: "Eiffel Tower", section: "Structure", url: `${WIKI}Eiffel%20Tower` },
        { n: 3, chunk_id: 9377, article_id: 2290, title: "Gustave Eiffel", section: "Career", url: `${WIKI}Gustave%20Eiffel` },
      ],
    }],
  ];
}

function notFoundScript(question: string): Step[] {
  return [
    [250, status("research", 1, `Researching "${short(question)}"`)],
    [500, call("call_1a", "semantic_search", 1, { queries: ["Eiffel Tower paint colour 1889", "original colour of the Eiffel Tower"], top_k: 5 })],
    [80, call("call_1b", "keyword_search", 1, { queries: ["Eiffel Tower paint"], top_k: 5 })],
    [600, result("call_1a", "semantic_search", "2 queries, 10 hits")],
    [150, result("call_1b", "keyword_search", "1 query, 0 hits")],
    [300, status("research", 2, "Deciding whether the evidence is enough")],
    [500, call("call_2a", "keyword_search", 2, { queries: ["Venetian red", "Eiffel Tower repainted"], top_k: 5 })],
    [550, result("call_2a", "keyword_search", "2 queries, 0 hits")],
    [300, status("research", 3, "Deciding whether the evidence is enough")],
    [700, {
      type: "research_answer",
      data: {
        turn: 3,
        status: "not_found",
        answer: "",
        citations: [],
        reason: "Searched semantically and by keyword for the tower's original paint colour and repainting history. The Eiffel Tower chunks cover its construction, height and visitors, but none mention its colour.",
      },
    }],
    [200, { type: "outcome", data: { result: "not_found", turns_used: 3 } }],
    [100, { type: "citations", data: [] }],
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
    onEvent(status("research", 1, "Researching"));
    await sleep(600, signal);
    onEvent({ type: "error", data: { message: "Mock error: the research agent failed (simulated)." } });
    onEvent({ type: "done", data: {} });
    return;
  }
  const notFound = /notfound/i.test(question);
  for (const [delay, ev] of notFound ? notFoundScript(question) : supportedScript(question)) {
    await sleep(delay, signal);
    onEvent(ev);
  }
  if (notFound) {
    // The orchestrator sends the fixed "couldn't find" reply as a single token.
    await sleep(150, signal);
    onEvent({ type: "token", data: { delta: NOT_FOUND_TEXT } });
    await sleep(100, signal);
    onEvent({ type: "done", data: {} });
    return;
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
