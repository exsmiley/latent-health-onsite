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

## Early hypotheses (written before the ramp; see the results below)

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

## Full ramp: results (2026-09-29)

The full ramp covered levels 1 to 64 with the default `max(8, 2N)` requests per level: 264
questions in total. No early-stop rule fired. The lead ran it in this worktree after the user
approved it, and nothing else was using the OpenAI key during the run. The summary is in
`docs/perf/results/ramp.summary.json`. The raw per-request JSON and the server log are in the
gitignored `evals/results/stress/`.

```bash
uv run uvicorn rag.api.app:app --host 127.0.0.1 --port 8300 \
    --log-config scripts/stress_logging.json > evals/results/stress/server.log 2>&1 &
uv run python scripts/stress_test.py --levels 1,2,4,8,16,32,64 --timeout 300 --max-requests 300 \
    --server-log evals/results/stress/server.log --label ramp 2>&1 | tee evals/results/stress/ramp.log
```

```
  N  req   ok ans/min |    p50    p90    p99 | ttft50 ttft90 ttft99 | err% | sup  nf oot turns |  oai  429 retry | dbc use  plQ pgA  cpu% cpuPk
  1    8    8    5.68 |   10.5   12.7   14.2 |   10.1   12.2   13.6 |  0.0 |   8   0   0  3.12 |   51    0     0 |   4   3    0   3   1.0   9.9
  2    8    8   10.06 |   10.7   13.9   18.8 |   10.3   13.4   18.4 |  0.0 |   8   0   0  3.62 |   57    0     0 |   5   4    0   4   1.1   3.7
  4    8    8   16.70 |   11.9   15.8   19.9 |   11.2   15.5   19.6 |  0.0 |   8   0   0  3.88 |   60    0     0 |   5   3    0   2   1.9   5.5
  8   16   16   33.87 |   10.3   17.5   19.5 |    9.7   17.0   19.0 |  0.0 |  16   0   0  3.38 |  109    0     0 |   5   5    0   5   3.6   7.3
 16   32   32   62.62 |    9.6   14.6   17.5 |    9.1   13.3   17.2 |  0.0 |  32   0   0  3.22 |  204    0     1 |   7   6    0   3   6.0  12.8
 32   64   64   72.15 |   10.5   16.2   32.2 |   10.0   15.8   31.7 |  0.0 |  61   2   1  3.47 |  434    0     0 |  10   9    0   1   6.3  20.9
 64  128  128  152.19 |   10.9   16.4   27.2 |   10.5   16.1   26.8 |  0.0 | 126   1   1  3.33 |  852    0     0 |  14  13    0   6  11.3  39.1
```

Derived per level, from the full JSON:

| N | wall s | mean latency s | slowest s | N / mean latency (answers/min) | worker busy % | OpenAI calls/q | pool checkouts/q | server CPU-s/q | RSS peak MB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 84.6 | 10.6 | 14.3 | 5.7 | 100 | 6.4 | 9.0 | 0.11 | 104 |
| 2 | 47.7 | 11.2 | 19.3 | 10.7 | 94 | 7.1 | 10.5 | 0.07 | 106 |
| 4 | 28.7 | 12.3 | 20.4 | 19.4 | 86 | 7.5 | 10.6 | 0.07 | 107 |
| 8 | 28.3 | 11.4 | 19.6 | 42.0 | 81 | 6.8 | 10.9 | 0.06 | 108 |
| 16 | 30.7 | 10.5 | 18.3 | 91.3 | 69 | 6.4 | 8.8 | 0.06 | 110 |
| 32 | 53.2 | 11.8 | 44.1 | 162.4 | 44 | 6.8 | 10.2 | 0.05 | 116 |
| 64 | 50.5 | 12.1 | 37.6 | 317.7 | 48 | 6.7 | 10.1 | 0.05 | 127 |

"Worker busy %" is the sum of request times divided by (N x wall time). It is the share of the
level during which all N slots were actually occupied.

### Caveat: measured throughput is limited by the tail, not by the server

This is a closed loop with only 2N requests per level, so each worker sends about 2 requests and
then goes idle while the level's slowest request finishes. At N=32, one question (q075: 7 turns,
44.1 s, about 4x the median) set most of the 53 s wall time, and workers were busy only 44% of it.
That is why N=32 shows 72 answers/min rather than about 125.

The measured `ans/min` for N ≥ 16 therefore **understates** capacity. The better estimate of
steady-state throughput is Little's law: N / mean latency, about 318 answers/min at N=64. Latency
did not grow with N, so that holds for as long as the arrival rate stays below it. q075 is simply a
slow question: the same question took 24.8 s at N=64 and ran out of turns. The p99 values at
N=32 and N=64 reflect the question mix, not load.

### Capacity: what we can support, with evidence

**Tested and supported: 64 questions in flight on one uvicorn process, with no degradation.**
That is about 150 answers/min measured, and about 320/min steady state by Little's law.

- **Latency is flat.** p50 stays between 9.6 and 11.9 s and p90 at or below 17.5 s from N=1 to
  N=64. The mean goes from 10.6 to 12.1 s. The 4x-p50 stop rule was never close: the worst level
  reached 1.1x.
- **Errors: 0 of 264.** There were 0 HTTP 429s and 0 5xx from OpenAI. Of 1,767 OpenAI calls there
  was one SDK retry (N=16), and no HTTP status was logged for it, so it was a connection or
  timeout blip, not a rate limit.
- **Outcomes don't move with load.** Answers were 95–100% supported per level, with 3.1–3.9 mean
  turns.
- **The server stays responsive.** The time to the first SSE event, a proxy for event-loop lag,
  had p99 ≤ 50 ms at N=64. Server CPU averaged 11% (39% for the worst 1 s) at N=64. RSS grew from
  104 to 127 MB, about 0.35 MB per in-flight question.

**Measured per-question cost:**
- about 6.7 OpenAI calls (about 5.4 Responses and 1.3 embeddings);
- about 10 pool checkouts, each held for a few ms;
- about 0.05 server CPU-seconds;
- about 16k input tokens (about 2/3 cached) and about 600 output tokens. The token figures come
  from the earlier eval runs; this ramp did not measure tokens.

**The ceiling was not found.** Here is where the next limits probably are, extrapolated from the
costs above. At N=64 the load was about 320 answers/min, or roughly 36 OpenAI calls/s, about
2,100 RPM, about 5M input TPM and about 0.2M output TPM.

| next limit | used at N=64 | extrapolated ceiling | confidence |
|---|---|---|---|
| **OpenAI rate limits on this key** (TPM, then RPM) | about 5M input TPM (about 1.7M uncached) and about 2,100 RPM, with no 429s | Unknown: the key's limits are not recorded anywhere, and all we know is that they sit above this. If the key is at a tier with about 10M TPM, the wall is around N≈120–130. This is **the most likely first limit**, and the failure would be 429s, then SDK retries (2, backing off up to 8 s), then `RateLimitError` error events. | low: check the `x-ratelimit-limit-tokens` / `-requests` response headers |
| Single uvicorn event loop | 11% average, 39% peak | About 0.05 CPU-s per question means one core could do roughly 1,000+ answers/min, or N≈200 at 100% average. The 1 s peaks are about 3.5x the average, though, so loop stalls (large Responses JSON, SSE formatting) would start adding latency around N≈150–250. | medium |
| Postgres pool (20) | peak 13 in use, 0 queued | In-use grows by about N/5, so the pool reaches 20 around N≈100. The DB-only test shows queueing past that costs milliseconds, not seconds, so it is harmless next to 10 s of model time. | high |
| DB layer (about 1,500 tool ops/s, about 3,000 SQL/s) | about 53 checkouts/s | That is about 2–3% of the ceiling, which would be reached only around N≈1,000+. | high |
| Postgres `max_connections=100` (shared) | peak 27 total on the DB | It matters only with several workers, because each has its own pool of up to 20. | high |

Bottom line: **the system is OpenAI-bound, not infrastructure-bound, at every level tested.**
Latency is set by the chain of model calls (about 5 sequential Responses calls at about
1–2 s each). Our own server, pool and DB used less than an eighth of their capacity at N=64. The
first real wall is most likely the key's token-per-minute limit, and after that a single event
loop somewhere past N≈150.

## Recommendations

1. **Add a server-side concurrency cap with fast rejection.** Today nothing bounds concurrency: a
   burst above the OpenAI limit would turn into slow retries and then error events for everyone
   in flight. Put an `asyncio.Semaphore` around `orchestrator.run` in `/api/chat`, and let a few
   more requests wait for a short time (about 5 s). Beyond that, answer **429 Too Many Requests
   with `Retry-After`** (use 503 if the cause is a dependency, such as sustained OpenAI 429s).
   Start the cap at **64 per process**, the tested level, and raise it only after a ramp above 64.
   Expose in-flight, queued and rejected counts in the health endpoint or logs.
2. **Set per-call model timeouts and a per-request deadline.** The SDK default is `timeout=600 s`
   (connect 5 s) with `max_retries=2`, so one hung Responses call can hold a request, and its slot
   under the cap, for up to about 30 minutes. Hung calls were seen in earlier runs. Set the client
   to about 60 s for non-streaming research and evaluator calls, and a read timeout of about 30 s
   between stream events for the responder. Keep `max_retries=2`. Add a whole-request deadline of
   about 120 s (p99 today is 27–32 s; the slowest was 44 s) that emits an `error` event and stops
   the pipeline.
3. **Keep the pool at 20.** The peak was 13 at N=64, and the DB-only check shows a larger pool
   adds little (+10% throughput at 60) and worsens the tail. Revisit only if the cap goes above
   about 100 per process; 30 would be enough then. Lower the pool's checkout `timeout` from its
   30 s default to about 5 s, so a stuck DB fails fast as a tool error instead of stalling turns.
   With several workers, keep workers x `max_size` under the shared `max_connections=100`.
4. **Don't add uvicorn workers yet.** One process is at 11% CPU at N=64. Add workers, each with
   its own cap and pool, only when a ramp shows event-loop lag, such as first-event p99 over
   about 200 ms or CPU peaks near 100%.
5. **Find the real ceiling next. Each step needs the user's approval, because it spends real
   OpenAI budget.**
   - First, at no load cost, read the key's `x-ratelimit-limit-requests` / `-tokens` headers from
     one Responses call. They tell us where the TPM/RPM wall is.
   - Prefer an **open-loop** run, which avoids the tail artifact above: Poisson arrivals at fixed
     rates (for example 150, 300, 450 and 600 answers/min for 2–3 minutes each). This needs a
     small `--rate` mode in `stress_test.py`. It measures the latency and 429s at a given arrival
     rate, which is what a concurrency cap has to be sized against.
   - Alternatively, run the closed loop at N=128 and N=256 with at least 4N requests per level, to
     keep workers busy for most of the level. That is about 1,500 questions. At about 16k input
     and 600 output tokens each, this is about 25M input tokens, so budget it explicitly.
   - Either way, keep the early-stop rules on and watch the `429`/`retry` columns first.
