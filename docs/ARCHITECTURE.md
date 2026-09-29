# Architecture & interface contract

RAG question answering over Simple English Wikipedia (`20220301.simple`, 205,328 articles,
~215M chars). Python 3.12 + uv, Postgres 17 + pgvector, OpenAI (Responses API for chat,
`text-embedding-3-small` for vectors), FastAPI SSE backend, Vite + React + TS frontend.

Shared, already-written files (do not change without coordinating):
`pyproject.toml`, `docker-compose.yml`, `db/schema.sql`, `src/rag/config.py`, `src/rag/db.py`,
`src/rag/llm.py`, `src/rag/tools/models.py`, `src/rag/cli.py`, this document.

## Data

Parquet at `data/simplewiki-20220301.parquet`, columns `id, url, title, text` (all strings).
Text is plain prose, no wiki markup, **no links** (links are out of scope). Paragraphs are
separated by `\n\n`. Section headings are a single short line with no terminal punctuation,
usually with a trailing space, e.g. `"The Month "`. About half of them are their own paragraph;
the rest are the first line of a paragraph with the body following after a single `\n`. Lists
appear as paragraphs whose lines start with a space. Most articles end with a block of category
lines (`1930 births\nLiving people`). See `rag.ingest.chunker` for the heuristic in full.

Full-dataset chunking gives 384k chunks: p50 111 / p90 276 / max 6.9k tokens, 54M
`embed_text` tokens in total, costing about $1.08 to embed.

## Chunking rules

- Token counts use tiktoken `cl100k_base`.
- Split into paragraphs on blank lines. **Never split inside a paragraph.**
- Greedily pack consecutive paragraphs up to `chunk_target_tokens` (350).
- A section heading starts a new chunk, unless the current chunk is below
  `chunk_min_tokens` (100), in which case the heading and its content keep packing into it.
  `chunks.section` is the heading in force at the chunk's first paragraph (NULL for the lead).
- A single paragraph larger than the target becomes its own chunk as-is. The only exception is
  a paragraph over 8,000 tokens (the embedding input limit): split it on sentence boundaries.
- Heading lines are not included in `text`; they go into `section`.
- `embed_text = f"{title} > {section}\n\n{text}"` (or `f"{title}\n\n{text}"` when section is NULL).
- No overlap between chunks: boundaries are paragraph/section aligned and the title/section
  prefix carries the context.

## Modules and ownership

| Module | Owner | Public API |
|---|---|---|
| `rag.ingest.chunker` | ingest | `chunk_article(title: str, text: str) -> list[ChunkDraft]` (pure; `ChunkDraft` has `chunk_index, section, text, embed_text, token_count`) |
| `rag.ingest.pipeline` | ingest | `async run_ingest(limit: int \| None, batch_size: int) -> None`. Downloads the parquet if missing, then streams articles in dataset order. Idempotent: skips article ids already present. |
| `rag.embed.pool` | embed | `async run_embed(workers: int \| None, batch_size: int \| None, limit: int \| None) -> None`. One producer pages through `chunks WHERE embedding IS NULL` by id (keyset) into an `asyncio.Queue`; N workers call `rag.llm.embed_texts` on `embed_text` batches and `UPDATE` the rows. Retries 429/5xx/timeouts with exponential backoff and jitter. Logs progress (chunks/s, ETA). Resumable. |
| `rag.tools.search` | tools | `async semantic_search(queries: list[str], top_k: int = 5) -> list[QueryResults]` embeds all queries in one request, then runs the per-query vector searches concurrently. |
| `rag.tools.keyword` | tools | `async keyword_search(queries: list[str], top_k: int = 5) -> list[QueryResults]` uses Postgres FTS (`websearch_to_tsquery('english', q)` against `chunks.tsv`, ranked by `ts_rank_cd`), with the queries run concurrently. |
| `rag.tools.fetch` | tools | `async fetch(chunk_ids: list[int] = [], article_ids: list[int] = []) -> FetchResult`. Articles return full text, uncapped, plus their `chunk_ids`. |
| `rag.tools.registry` | tools | `TOOL_SPECS: list[dict]` (OpenAI Responses API function-tool definitions for `semantic_search`, `keyword_search`, `fetch`), `async dispatch(name: str, args: dict) -> str` (runs a tool and returns a compact JSON string; search hits carry full `text` and `token_count`) and `present(outputs: list[tuple[str, str]], shown: set[int], full_text: int \| None = None) -> list[str]` (shapes one turn's outputs for the model, see "Search hits"). |
| `rag.agents.*` | agents | See below. |
| `rag.api.app` | agents | FastAPI `app`. See the HTTP/SSE contract. |
| `frontend/` | frontend | Chat UI. |

Result models live in `rag.tools.models` (`SearchHit`, `QueryResults`, `Chunk`, `Article`,
`FetchResult`). Search only returns chunks with a non-NULL embedding (semantic) or any chunk
(keyword). The keyword `score` is `ts_rank_cd`, so it is not comparable across queries or with
semantic scores. The blurb is the first `blurb_words` (40) words of `chunks.text`, followed by "…" if
truncated.

### Search hits

Hits carry full chunk text so the agent can cite straight from a search, without a fetch turn.
After all of a turn's tool calls finish, `registry.present` shapes the search outputs (in call
order) against the agent's `shown` set (chunk ids whose full text the model has already seen):

- The first `hit_full_text` (3) hits of each query not already in `shown` get `text`, picked
  rank by rank across every query of both search tools until `hit_text_budget_tokens` (5000,
  by `chunks.token_count`) is spent. Chunks that don't fit are skipped.
- Every other hit gets a `blurb`, and must be fetched before it is cited.
- A chunk already listed earlier in the turn, or in `shown`, gets only `"seen": true` (no text).
- Chunks returned by `fetch` (including an article's `chunk_ids`) are added to `shown`.

Each hit keeps `chunk_id, article_id, title, section, score`. The UI summary is unchanged.

## Agent pipeline

There is a single budget of `settings.research_max_turns` (7) research turns per question; there
are no rounds. A *turn* is one research-agent model call. Evaluator and responder calls do not
count as turns.

```
turn 0 (pre-retrieval, no model call, not a turn):
    semantic_search(question) + keyword_search(its names/numbers) ──► fetch the top hits
    └─ added to the conversation as function_call/function_call_output pairs
for turn in 1..7:
    research agent model call (tools: semantic_search, keyword_search, fetch; tool_choice auto)
    ├─ made tool calls ──► run them concurrently, feed results back, next turn
    └─ no tool calls   ──► its final message is a structured answer (JSON schema below)
         ├─ invalid (bad JSON, empty/unknown/non-chunk citations) ──► error fed back, next turn
         ├─ status "not_found" ──► fixed "couldn't find" reply, done (no evaluation)
         └─ status "answered" or "premise_false"
                            ──► Evaluator (sees ONLY the user's question + cited chunk texts)
              ├─ supported   ──► Responder ──stream──► user, done
              └─ unsupported ──► feedback added to the conversation, next turn
out of turns ──► fixed "couldn't find" reply
```

- **Pre-retrieval** (`rag.agents.pre_retrieval`, `ResearchAgent.pre_retrieve`; on unless
  `settings.research_pre_retrieve` is false). Before turn 1 the harness itself, with no model
  call, runs `semantic_search` (top 8) on the question and `keyword_search` (top 5) on a query
  made of the question's capitalized names (consecutive ones as a quoted phrase) and numbers,
  at most 4 terms (skipped if there are none; a whole question ANDs every word and rarely
  matches). It then fetches the top 5 semantic and top 2 keyword hits in full, dropping chunks
  over 4,000 characters (big lists), so the model can cite without a turn of reading. Its
  searches go through `registry.present` with no full-text hits (that fetch supplies the text,
  so nothing is shown twice), and the fetched chunks are added to `shown`. For a
  question with history, the semantic query is the previous user and assistant messages
  (400 chars each, `[n]` markers removed) followed by the question, and the keyword query
  falls back to that text when the question itself has no names. The calls go into the
  conversation after the question as ordinary `function_call`/`function_call_output` pairs,
  exactly as if the model had made them: the model already knows how to read and cite tool
  outputs, and the fetched chunk ids sit next to the hits they came from. They emit
  `status`/`tool_call`/`tool_result` events with turn 0, don't count toward `turns_used`, and
  count as a search for the "not_found" rule if either search succeeded. The system prompt
  tells the model to read them first and answer on turn 1 when they suffice, but only after
  checking that every step of the question's chain and every item or side is stated in a chunk
  it has read, and that facts about a subject come from the subject's own article (pre-retrieved
  chunks resemble the question; they don't necessarily answer it).
- **Research agent** (`rag.agents.research`). Model `settings.chat_model` via the Responses API.
  Tools: `semantic_search`, `keyword_search`, `fetch`, always with `tool_choice: "auto"`: it is
  never forced to answer. It ends research by replying WITHOUT a tool call. That final message
  must match the strict JSON schema `{question: str, status: "answered" | "premise_false" |
  "not_found", answer: str, citations: int[], reason: str}`, set via the Responses API `text.format`. `question` is
  the user's latest question with only pronouns and follow-up references resolved: it must keep
  every constraint and step, and is never shortened to a sub-question or the last hop of a
  chain. The evaluator judges against the user's ORIGINAL latest question; the rewrite is given
  to it only as a second, labelled "references resolved" version when there is chat history
  (with no history it is ignored). The evaluator never sees the chat history. For "answered"
  and "premise_false", `answer` and chunk-id `citations` are required and `reason` is ""; for
  "not_found", `reason` says what was searched and what was missing, and `answer`/`citations`
  are empty. "premise_false" means the cited chunks contradict an assumption of the question
  (e.g. "Who is the prince of Azerbaijan?" when it is a republic with a president): `answer`
  corrects the premise and gives the closest true answer the corpus has, and the citations back
  the contradicting facts and that answer. It is used only when read chunks contradict the
  premise; a fact that is merely absent is "not_found". It may answer after any turn. A
  "not_found" or "premise_false" is rejected (fed back as an error) until at least one search
  has run. The system prompt must explain that:
  - full-text hits can be cited directly, blurb-only hits must be fetched first, and `seen`
    hits refer to text shown earlier;
  - it should prefer fetching specific chunks over whole articles;
  - **citations must be chunk ids**, and the evaluator sees only the cited chunk text and will
    not accept whole articles;
  - it should cite the minimal sufficient set of chunks, except that a chain needs a chunk for
    every hop, and comparisons, superlatives, counts and "which of these" questions need
    evidence for every entity compared, not just the winner;
  - every link of a chain must be found in the index, even if it knows the entity (each chunk
    is shown to the evaluator under its title/section heading);
  - it should never guess, and should answer "not_found" after about two turns of fruitless
    focused searching, but first broaden (related titles, synonyms, historical senses such as
    "crown prince", "governor of", the dynasty) rather than rephrase the same phrase;
  - when a name has several referents in the corpus, it answers the main readings briefly, or
    the likeliest one and names the alternative;
  - when the chunks contradict the question's premise, it answers "premise_false";
  - it should answer as early as the evidence allows.
  Before each turn a developer note gives the remaining turns. The last one says that tool calls
  made on that turn can't be followed up, and that running out means the user is told the
  answer couldn't be found. That is information, not forcing.
  A response containing any function call is a tool turn, and any commentary message text in it
  is ignored. Tool calls made on the last turn are not run: each gets a `tool_result` with
  summary "not run: no turns left".
- **Evaluator** (`rag.agents.evaluator`). A single model call, with structured JSON output
  `{requirements: str, independent_answer: str, verdict: "supported" | "unsupported", feedback:
  str}`. Its input is the user's original latest question (plus, for follow-ups, the agent's
  references-resolved version, which may only replace pronouns and references), the cited chunk
  texts and the candidate answer. It first lists the question's requirements (every link of the
  chain it describes, every constraint and compared entity, and exactly what is asked: which
  entity, at which hop), then answers from the cited chunk texts alone, checking that the
  passages establish EACH link. A final fact backed by the passages is not enough unless they
  also show it belongs to the thing the question describes. A link backed only by the
  candidate's claims or by outside knowledge, a compared entity without evidence, or an answer a
  hop short or too far means `unsupported`. Each cited chunk is shown as `[n] {title} >
  {section}` (or `[n] {title}`) followed by its text, the same prefix as `embed_text`, so chunks
  are self-identifying: "He was born in 1879" under "Albert Einstein" names its subject. The
  prompt says the heading resolves identity only and is not evidence for any other fact. It
  unpacks nested "the X of the Y of Z" phrases from the inside out, one hop per "of", and checks
  that the answer is the outermost entity. A direct possessive or description ("Gustave
  Eiffel's company") is enough for a naming or ownership link. An answer to an ambiguous
  question that covers the main readings, or the likeliest one plus the alternative, is
  acceptable. The candidate answer is labelled with its type; a premise correction is
  `supported` when the passages establish positive facts contradicting the premise (a negative
  such as "no prince" may be concluded from "is a republic with a president"), any closest true
  answer is backed, and the correction addresses the question. The
  feedback names the missing link or what's contradicted. An empty or malformed
  evaluator reply counts as `unsupported`. A failure while checking citations (e.g. a DB blip)
  is an `invalid` answer the agent can resubmit, not a request error. "Not found"-style
  text in an answer is always `unsupported`.
- **Responder** (`rag.agents.responder`). A streaming model call that takes the question, the
  research answer, the evaluator's independent answer and the cited chunks. It writes a concise
  final answer with inline `[n]` markers, where n indexes the citations list. For a verified
  "premise_false" answer it opens with a short correction and then gives the closest true
  answer. It runs only for supported answers (a verified premise correction has outcome
  `supported`). For not found or out of turns, the orchestrator emits an empty `citations`
  event and a single fixed "couldn't find the answer" `token` instead.
- **Orchestrator** (`rag.agents.orchestrator`). `async run(question: str, history: list[dict]) ->
  AsyncIterator[Event]` yields the events below. `async run_cli(question)` prints them.

## HTTP / SSE contract (backend ⇄ frontend)

- `GET /api/health` returns `{"ok": true}`
- `GET /api/chunks/{id}` returns a `Chunk` JSON, or 404
- `POST /api/chat` takes `{"messages": [{"role": "user"|"assistant", "content": str}, ...]}`, where
  the last message is the new question. It responds with `text/event-stream`. Each event is
  `event: <type>\ndata: <json>\n\n`:

| type | data |
|---|---|
| `status` | `{stage: "research"\|"evaluate"\|"respond", turn: int\|null, max_turns: int, message: str}` (research: the turn starting, 0 for pre-retrieval; evaluate: the turn whose answer is checked; respond: null) |
| `tool_call` | `{id: str, name: str, arguments: object, turn: int}` (turn 0: pre-retrieval, run by the harness before turn 1) |
| `tool_result` | `{id: str, name: str, summary: str}` (e.g. "3 queries, 15 hits") |
| `research_answer` | `{turn: int, status: "answered"\|"premise_false"\|"not_found"\|"invalid", answer: str, citations: int[], reason: str}` (every final message from the agent; "premise_false" is an answer correcting a false premise, evaluated like "answered" and shown as "premise corrected"; "invalid" carries the error in `reason` and research continues) |
| `evaluation` | `{turn: int, verdict: "supported"\|"unsupported", independent_answer: str, feedback: str}` |
| `outcome` | `{result: "supported"\|"not_found"\|"out_of_turns", turns_used: int}` (sent once, right before `citations`) |
| `citations` | `[{n: int, chunk_id: int, article_id: int, title: str, section: str\|null, url: str}]` (sent before tokens) |
| `token` | `{delta: str}` (final answer text) |
| `done` | `{}` |
| `error` | `{message: str}` |

The backend runs as a long-lived process (`uv run rag serve`) with CORS allowing
`http://localhost:5180`. The frontend dev server proxies `/api` to `http://localhost:8100`.

## Dev commands

```
docker compose up -d db
uv sync
uv run rag ingest --limit 10 && uv run rag embed
uv run rag serve
cd frontend && npm install && npm run dev
uv run pytest          # tests marked `db` need the database
```
