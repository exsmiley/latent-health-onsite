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
| `rag.tools.registry` | tools | `TOOL_SPECS: list[dict]` (OpenAI Responses API function-tool definitions for `semantic_search`, `keyword_search`, `fetch`) and `async dispatch(name: str, args: dict) -> str` (runs a tool and returns the compact JSON string given back to the model). |
| `rag.agents.*` | agents | See below. |
| `rag.api.app` | agents | FastAPI `app`. See the HTTP/SSE contract. |
| `frontend/` | frontend | Chat UI. |

Result models live in `rag.tools.models` (`SearchHit`, `QueryResults`, `Chunk`, `Article`,
`FetchResult`). Search only returns chunks with a non-NULL embedding (semantic) or any chunk
(keyword). The keyword `score` is `ts_rank_cd`, so it is not comparable across queries or with
semantic scores. The blurb is the first `blurb_words` (40) words of `chunks.text`, followed by "…" if
truncated.

## Agent pipeline

```
question ──► Research agent (≤7 turns) ──submit_answer(answer, chunk_ids)──►
             Evaluator (sees ONLY question + cited chunk texts loaded from the DB)
               ├─ supported ──► Responder ──stream──► user
               └─ not supported ──► back to Research agent with feedback (new 7-turn round)
                    after max_research_rounds (3): a fixed "couldn't find the answer" reply, no citations
```

- **Research agent** (`rag.agents.research`). Model `settings.chat_model` via the Responses API.
  Tools: `semantic_search`, `keyword_search`, `fetch`, `submit_answer`. A *turn* is one model
  call, which may emit several tool calls; these execute concurrently. On the last turn,
  `tool_choice` is forced to `submit_answer`. The system prompt must explain that:
  prefer fetching specific chunks over whole articles; **citations must be chunk ids**; the
  evaluator will only see cited chunk text and will not accept whole articles; cite the minimal
  sufficient set of chunks. `submit_answer` args: `{answer: str, citations: list[int]}`. On a
  retry round, the agent keeps its prior conversation and receives the evaluator's feedback.
  Invalid or nonexistent citation ids are returned to the agent as a tool error, which costs a
  turn.
- **Evaluator** (`rag.agents.evaluator`). A single model call, with structured JSON output
  `{independent_answer: str, verdict: "supported" | "unsupported", feedback: str}`. It answers
  the question from the cited chunk texts alone, then judges whether that answer agrees with the
  research agent's answer. The feedback says what's missing or contradicted. A "not found"
  answer is always `unsupported`, so the research agent keeps looking.
- **Responder** (`rag.agents.responder`). A streaming model call that takes the question, the
  research answer, the evaluator's independent answer and the cited chunks. It writes a concise
  final answer with inline `[n]` markers, where n indexes the citations list. It runs only for
  supported answers. If no round is supported, the orchestrator skips it and emits an empty
  `citations` event and a single fixed "couldn't find the answer" `token`.
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
| `status` | `{stage: "research"\|"evaluate"\|"respond", round: int, turn: int\|null, message: str}` |
| `tool_call` | `{id: str, name: str, arguments: object, round: int, turn: int}` |
| `tool_result` | `{id: str, name: str, summary: str}` (e.g. "3 queries, 15 hits") |
| `research_answer` | `{round: int, answer: str, citations: int[]}` |
| `evaluation` | `{round: int, verdict: "supported"\|"unsupported", independent_answer: str, feedback: str}` |
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
