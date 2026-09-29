# Stress test: how many questions at once?

Goal: find how many concurrent questions the system can realistically serve, where latency
degrades, what fails first, and why. The single-request baseline is from the earlier speedup
study (branch `speedup-brainstorm`, `docs/perf/SPEEDUP.md`): a median of about 8–10 s per
question, nearly all of it OpenAI model calls, each with a fixed floor of about 1.0–1.3 s.

## Tools

| file | what it does |
|---|---|
| `scripts/stress_test.py` | Closed-loop HTTP load generator for `POST /api/chat` (httpx, streaming SSE parsing). Also samples server-side signals. |
| `scripts/stress_db.py` | DB layer only: concurrent `semantic_search` / `keyword_search` / `fetch` straight at `rag.tools` through the production pool, with no ASGI and no chat-model calls. |
| `scripts/stress_logging.json` | uvicorn `--log-config` that puts the `httpx2` (the OpenAI SDK's transport), `openai` and `psycopg.pool` loggers at INFO, with timestamps. |
| `tests/stress/` | Unit tests for SSE parsing, stats, early stopping and log counting, run against a tiny fake SSE ASGI app. |

### Load generator (`stress_test.py`)

- At each level N (`--levels 1,2,4,8,16,32,64`), N workers each send a request as soon as their
  previous one ends, until `--requests-per-level` requests have gone out (default `max(8, 2N)`).
- Questions come from `evals/questions.jsonl`, shuffled once with `--seed`, then handed out
  round-robin. The cursor carries across levels, so levels get fresh questions until the set of
  81 wraps.
- Per request it records the time to the first SSE event, the time to the first `token` (TTFT),
  and the total time to `done`. It also records the outcome (`result`, `turns_used`) and the kind
  of failure: an `error` event, a non-200 status (`http_error`), a `timeout` (`--timeout`,
  default 300 s), a `transport` error, or `incomplete` (the stream ended without `done`).
- Per level it reports:
  - throughput, as completed answers per minute of wall time;
  - p50/p90/p99 of the total time and of TTFT, over completed requests only;
  - the error rate, split by kind;
  - supported / not_found / out_of_turns counts and the mean turns.
- Early stop: after a level whose error rate is over 20% (`--max-error-rate`), or whose p50 is
  over 4x the first level's p50 (`--p50-factor`).
- `--max-requests` refuses a plan larger than a budget, and `--dry-run` prints the plan.
- It writes the full JSON (per request, plus 1 Hz server samples) to
  `evals/results/stress/*.json` (gitignored) and a small `*.summary.json` next to it.

Server-side signals, collected without changing any production code path:

| signal | source | column |
|---|---|---|
| OpenAI calls, by endpoint and status; 429s; 5xx | `httpx2` INFO lines (`HTTP Request: POST https://api.openai.com/... "HTTP/1.1 429 ..."`) in the server log, sliced per level by byte offset | `oai`, `429` |
| SDK retries | `openai._base_client` INFO `Retrying request ...` | `retry` |
| Pool in-use peak and queueing | replaying the `psycopg.pool` INFO `connection requested` / `given` / `returning` lines. `plQ` counts connection requests that arrived while all 20 were in use. | `use`, `plQ` |
| Server's TCP connections to Postgres | `lsof -p <pid> -iTCP:5433`, every second | `dbc` |
| Active backends on the `rag` DB, from all clients | `pg_stat_activity`, every second | `pgA` |
| Server CPU (average and peak 1 s step) and RSS | `ps -o time=,rss=` deltas, every second | `cpu%`, `cpuPk` |

The pg sample at 1 Hz misses most queries, which take only milliseconds. The pool replay from the
log is the reliable pool-saturation signal.

### Running it

```bash
# 1. Server from this worktree on :8300, with the OpenAI/pool loggers on (same app as `rag serve`)
uv run uvicorn rag.api.app:app --host 127.0.0.1 --port 8300 \
    --log-config scripts/stress_logging.json > evals/results/stress/server.log 2>&1 &

# 2. Load
uv run python scripts/stress_test.py --levels 1,2 --requests-per-level 4 \
    --server-log evals/results/stress/server.log --label smoke

# DB layer only (one embeddings request, then no OpenAI calls)
uv run python scripts/stress_db.py --levels 1,2,4,8,16,32,64,128 --duration 5
uv run python scripts/stress_db.py --levels 16,32,64 --pool-size 60   # pool-size what-if
```

`rag serve` is just `uvicorn.run("rag.api.app:app", ...)`. Running uvicorn directly only adds the
log config.

## Smoke results (2026-09-29)

Levels 1 and 2, 4 requests each (8 live questions). Summary: `docs/perf/results/smoke.summary.json`.

```
  N  req   ok ans/min |    p50    p90    p99 | ttft50 ttft90 ttft99 |  err% | sup  nf oot turns |  oai  429 retry | dbc use  plQ  cpu% cpuPk
  1    4    4    5.45 |   10.3   13.9   14.7 |   10.0   13.4   14.2 |   0.0 |   4   0   0  3.00 |   24    0     0 |   3   3    0   1.3  12.7
  2    4    4    9.15 |   10.5   14.0   15.1 |   10.0   13.5   14.6 |   0.0 |   4   0   0  3.50 |   27    0     0 |   3   3    0   1.5   3.6
```

- It matches the baseline: p50 is about 10 s at N=1, and N=2 doubles throughput with no latency
  cost.
- TTFT is about 0.4 s below the total. Nearly all the wait is research and evaluation before the
  first token, and the streamed answer itself is short.
- Each question makes about 6 OpenAI calls (5 Responses and 1 embeddings), about 8 pool checkouts,
  and at most 3 pooled connections at once.
- The server uses about 0.14 CPU-seconds per question, so the average CPU at N=1–2 is under 2%.

## DB-layer ceiling (no model calls)

`stress_db.py` ran 5 s per level through the production pool (`max_size=20`). Each worker cycles
through this mix: `semantic_search` of 3 queries (3 pooled connections at once), `keyword_search`
of 3 queries, `fetch` of 5 chunks, and `fetch` of 1 article. `sql/s` counts each per-query SQL.
Pool columns come from `psycopg_pool` stats: `waitPk` is the peak number of waiters, `queued` is
the number of requests that had to wait, and `wait_ms` is the mean wait per connection request.
Data: `docs/perf/results/db-pool20.json`.

```
   N   ops/s   sql/s |   p50ms   p90ms   p99ms |  sem50   kw50  fch50  art50 | poolPk inUse waitPk queued wait_ms
   1   341.2   682.7 |     2.4     6.2     7.6 |      6      3      1      1 |      6     3      0      0     0.0
   2   663.3  1327.0 |     2.2     6.3     7.7 |      6      3      1      1 |      6     6      0      0     0.0
   4  1114.4  2229.0 |     2.5     7.4    10.6 |      7      4      2      1 |     12    12      0     46     0.0
   8  1384.3  2768.8 |     4.7    11.0    16.5 |     10      6      3      3 |     20    20      4    544    0.03
  16  1508.8  3017.9 |    10.0    16.7    23.4 |     15     12      7      7 |     20    20     20  14580    3.38
  32  1498.1  2996.8 |    20.6    28.3    36.0 |     26     22     18     18 |     20    20     54  15025   13.75
  64  1469.9  2942.1 |    43.3    51.4    58.6 |     49     45     40     40 |     20    20    124  14820   35.79
 128  1460.7  2920.8 |    87.0    95.6   107.4 |     92     88     85     84 |     20    20    249  14835   79.16
```

The same test with a script-only 60-connection pool (`db-pool60.json`):

```
  16  1353.3  2707.6 |    10.3    21.3    33.8 |     19     12      6      6 |     45    43     23   1735    0.63
  32  1607.2  3217.6 |    15.8    39.6    68.3 |     35     19     10     10 |     60    60     24   6556    1.19
  64  1652.1  3306.0 |    35.3    60.8    91.4 |     54     39     28     27 |     60    60     96  16579   18.75
```

- The ceiling is about 1,500 tool ops/s, or about 3,000 SQL/s. The pool of 20 fills at about 8
  concurrent tool calls. From 16 up, throughput is flat, and latency grows linearly as pool
  queueing: a mean wait of 3 ms at N=16 and 79 ms at N=128.
- Tripling the pool to 60 adds only about 10% throughput and makes the tail worse. Past about 20
  connections, the limit is the single Python event loop and psycopg per-query overhead (plus
  Postgres), not the pool size.
- Compared with the real workload, a question needs about 8 pool checkouts over about 10 s, so
  under 1 checkout/s per in-flight question. Even 100 concurrent questions would need well under
  100 checkouts/s, less than 10% of the ceiling. The pool would saturate only with hundreds of
  tool calls truly simultaneous. **The DB and pool are not the first bottleneck.**

## Early hypotheses (to be checked by the full ramp)

1. **The OpenAI rate limit fails first.** Each in-flight question makes about 0.6 OpenAI calls/s
   and uses about 16k input tokens (about 2/3 cached) per about 10 s. At N=64 that is about
   40 req/s (about 2,300 RPM) and about 6M input TPM. That is likely above the key's TPM limit.
   The expected failure mode is 429s absorbed by SDK retries (default `max_retries=2`, exponential
   backoff). Those show up first as rising p50/p90 and `retry` counts. Once retries run out, they
   surface as `error` events (`RateLimitError`), which the orchestrator turns into
   `error` + `done`.
2. **The pool of 20 is not the limit** (see above). It would matter only if tool calls got much
   slower (for example, a cold cache or a larger `ef_search`).
3. **A single uvicorn process** uses about 0.14 CPU-s per question, so one core would cap out
   around 400 questions/min, far above any plausible OpenAI-limited throughput. It could matter
   later through event-loop latency (JSON parsing of large Responses payloads, SSE formatting)
   rather than raw CPU.
4. The capacity guess, before the ramp: throughput scales roughly linearly
   (N x about 6 answers/min) until the TPM limit, then flattens while latency climbs. There are no
   server-side guards: no concurrency cap, no queue, and no per-request deadline besides the
   client's.

## Full ramp: results

_Pending. The full ramp was not run in the first session: the live-load run was blocked pending
explicit user approval (it shares the OpenAI key with other benchmark work)._

Planned command (at most 264 questions, early stop at 20% errors or 4x p50):

```bash
uv run uvicorn rag.api.app:app --host 127.0.0.1 --port 8300 \
    --log-config scripts/stress_logging.json > evals/results/stress/server.log 2>&1 &
uv run python scripts/stress_test.py --levels 1,2,4,8,16,32,64 --timeout 300 --max-requests 300 \
    --server-log evals/results/stress/server.log --label ramp
```

Table: _TBD_

Capacity estimate, bottleneck and evidence: _TBD_

## Recommendations

_TBD after the full ramp. The candidates to size from it are:_

- a server-side concurrency cap (a semaphore around `orchestrator.run`, returning 503 or queueing
  above it);
- the pool size;
- the OpenAI SDK `max_retries` and timeout;
- a per-request server deadline.
