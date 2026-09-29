# Evaluation question set

There are two files:

| file | questions | purpose |
|---|---:|---|
| `questions.jsonl` | 81 | Main set: hard, mostly 2–3 hop questions that should be solvable within the production 7-turn budget |
| `questions_super_hard.jsonl` | 29 | [Super-hard tier](#super-hard-tier): 6–8 step chains and 10–14 article fan-outs, built to find the breaking point |

The rest of this section and the next few describe the main set.

`questions.jsonl` holds 81 hard questions, each answerable from this corpus: the
`20220301.simple` Simple English Wikipedia snapshot as chunked into the `chunks` table. Most of
them need several retrieval steps. The set is weighted toward questions whose evidence spans
several chunks, either in different articles (bridge / multi-hop) or in different parts of one
article.

**The expected answer is what this corpus says**, and that is not always what is true today (see
[Caveats](#caveats)). Every question was written from chunk text read in the DB, never from world
knowledge.

## Schema

One JSON object per line:

| field | type | meaning |
|---|---|---|
| `id` | str | `q001` … `q081` |
| `question` | str | The question as a user would ask it. Multi-hop questions describe intermediate entities without naming them. |
| `answer` | str | Short canonical answer. It may carry a parenthetical with the numbers used. |
| `answer_aliases` | [str] | Acceptable variants: spellings, shorter forms, alternative phrasings. |
| `type` | str | `multi_article`, `multi_chunk`, `comparison`, `aggregation`, `temporal` or `single_hop` (see below) |
| `hops` | int | Number of retrieval/reasoning steps, usually the number of distinct facts needed |
| `difficulty` | str | `easy`, `medium` or `hard` |
| `supporting_chunks` | [obj] | `{chunk_id, article_id, title, section, chunk_index, evidence}`. `evidence` is an exact sentence (or two) from that chunk's `text` that supports one step. A chunk may appear twice when two non-contiguous sentences in it are needed. |
| `reasoning` | str | How the hops connect |
| `notes` | str | Alternative supporting chunks, pinned answers, and `CAVEAT:` notes where the snapshot differs from reality |

`chunk_id` is a BIGSERIAL and changes on re-ingest. `(article_id, chunk_index)` stays stable as
long as the chunker is unchanged, so a future eval runner should match on that pair (or on
`article_id` alone for a looser recall metric). The supporting chunks are a sufficient set, not
the only one. Where another article states the same fact, `notes` usually says so.

### Types

- **multi_article**: facts from 2+ articles chained together, e.g. *Esperanto → its creator
  Zamenhof → his birthplace Białystok → the Biała River*.
- **multi_chunk**: 2+ chunks of the same article (e.g. a lead-section birth date plus a later
  "Family" section), with every chunk needed.
- **comparison**: comparing numbers or dates across entities (older, higher, bigger capacity).
- **aggregation**: listing or collecting items spread across chunks or articles.
- **temporal**: date arithmetic or ordering across facts.
- **single_hop**: easy controls answered by one chunk.

Most comparison, aggregation and temporal questions also span several articles. The type records
the dominant reasoning operation.

## Breakdown

| type | count | share | easy | medium | hard |
|---|---:|---:|---:|---:|---:|
| multi_article | 36 | 44.4% | 0 | 23 | 13 |
| multi_chunk | 15 | 18.5% | 0 | 13 | 2 |
| comparison | 10 | 12.3% | 1 | 7 | 2 |
| temporal | 9 | 11.1% | 1 | 6 | 2 |
| aggregation | 6 | 7.4% | 0 | 3 | 3 |
| single_hop | 5 | 6.2% | 5 | 0 | 0 |
| **total** | **81** | | **7** | **52** | **22** |

- **Hops:** 1-hop 5, 2-hop 35, 3-hop 38, 4-hop 2, 5-hop 1.
- **Chunks:** 58 of the 81 questions cite chunks from 2 or more articles. There are 220
  supporting-chunk entries covering 166 distinct chunks.
- **Domains**, in roughly equal thirds:
  - **science and technology:** penicillin, the Curies, Apollo 11, programming languages, elements,
    volcanoes, vaccines, animals
  - **history, politics, geography and religion:** Napoleon, Luther, US presidents, Gandhi,
    Mongols, Greek philosophers, mountains, rivers, lakes
  - **arts, sports and business:** composers, Queen, the Beatles, Picasso, Van Gogh, Tolkien,
    stadiums, Pelé, Michael Jordan, IKEA, LEGO, Nintendo, McDonald's, Esperanto

## How it was built

1. Candidate topics were picked by domain. For each one, the relevant articles were read chunk by
   chunk from the DB (`chunks.text`), looking for facts that chain together: X's article names Y,
   Y's article gives a place, and the place's article gives a river, a capital or a height.
2. Each question was written from the chunk text. Intermediate entities are described, not named
   ("the Australian pathologist who shared the 1945 Nobel Prize…"). The exact supporting sentences
   were copied into `evidence`.
3. **Findability:** for each key chunk we checked that a reasonable `semantic_search` or
   `keyword_search` query (`rag.tools`) returns it near the top, usually in the top 3 and always in
   the top 10. Where the obvious query misses a chunk but another chunk states the same fact, the
   notes say so (e.g. q035).
4. We dropped or rewrote candidates that:
   - were ambiguous;
   - were contradicted across chunks (e.g. the Mendel article says "35 years later" for
     1866→1900; conflicting Beatles dates);
   - collapsed into a single chunk (e.g. Henry VIII's six wives, all listed in "Wives of King
     Henry VIII");
   - leaned on date pages such as "1928" or "September 3", which are never used as supporting
     chunks.
5. Metadata (`article_id`, `title`, `section`, `chunk_index`) was filled in from the DB, not typed
   by hand.

## Validation

```
uv run python evals/validate.py [path ...]   # defaults to both questions.jsonl and questions_super_hard.jsonl
```

The script checks that:
- every line parses and has the required fields, types and enum values for its tier;
- ids and question texts are unique across all the files checked together;
- super-hard questions have `tier`, `sequential_depth`, `breadth` and `min_turns_estimate`, with
  `1 <= sequential_depth <= hops`, `min_turns_estimate >= sequential_depth`, and `breadth` equal
  to the number of distinct `article_id`s in `supporting_chunks`;
- every supporting chunk exists with the recorded `article_id`, `chunk_index` and `title` (a
  `section` mismatch is only a warning);
- each `evidence` string appears in that chunk's text after whitespace normalization.

For each file it prints counts by type, difficulty and hops. For the super-hard tier it prints
counts by type and the distributions of hops, depth, breadth and `min_turns_estimate`. It exits
non-zero on any error and only runs SELECTs. Both files currently pass with 0 errors and 0
warnings.

## Super-hard tier

`questions_super_hard.jsonl` (ids `sh001`–`sh029`) is a stress test. **Its purpose is to find
the breaking point, not to measure day-to-day quality.** With the production budget of 7 research
turns, many deep chains (depth above about 6) are *expected* to run out of turns even when every
step is found. Low scores on this tier at budget 7 are therefore not a regression. That is why it
should also be run with an extended budget (see [Benchmarking
requirements](#benchmarking-requirements)).

It was built the same way as the main set: questions were written from chunk text read in the DB,
with evidence recorded for every hop, metadata filled in from the DB, and each hop's key chunk
checked for findability. Because a single fuzzy link ruins a long chain, the rules were stricter:
- every link must be stated explicitly in chunk text;
- no link may come only from a category line or rest on an inferred fact;
- a candidate is dropped when articles contradict each other on the fact it needs.

Several candidates were dropped for these reasons, e.g. a Norway "independent in the 20th
century" question (its lead says "independent since 1814"), noble-gas discoverers (Ramsay's
article contradicts Helium/Radon), and stadium capacities. No chain from the main set is reused.

### Extra fields

| field | meaning |
|---|---|
| `tier` | Always `"super_hard"` |
| `difficulty` | Always `"super_hard"` |
| `type` | `deep_chain`, `wide` or `deep_wide` (see below) |
| `hops` | Total number of lookups |
| `sequential_depth` | Longest chain of dependent lookups: the minimum number of research turns even with unlimited parallel tool calls |
| `breadth` | Number of distinct articles in `supporting_chunks` (checked by the validator) |
| `min_turns_estimate` | Estimated turns for an efficient agent, including fetches and the final-answer turn (formula below) |

`min_turns_estimate = sequential_depth + min(sequential_depth, F) + 1`. Here F is the number of
supporting articles whose evidence is *not* inside the 40-word search blurb, which is roughly the
number of fetch rounds needed. The formula assumes parallel searches and fetches within a turn,
plus one fetch turn per dependent step where the blurb does not already show the fact. It is a
consistent, mechanical estimate, not a bound. A lucky agent that recognizes intermediate entities
from its own knowledge can beat it; an agent that searches badly will exceed it.

### Types

- **deep_chain** (10): sequential chains of 6–8 dependent steps, where each answer is the next
  search key. Example (sh007): *Tristan und Isolde premiere conductor → his father-in-law Liszt →
  Liszt's teacher Czerny → Czerny's teacher Beethoven → Beethoven's teacher Haydn → Haydn's
  employer Porpora → Porpora's first opera → Nero.*
- **wide** (11): 10–14 independent lookups (one per article), then combined by ranking, counting,
  finding the earliest or latest, or matching equal years. Depth is 1–2, so an agent that runs
  searches in parallel should manage in 2–5 turns. Serial agents will not.
- **deep_wide** (8): a 3–5 step chain whose end is a set of 5 or more entities, then an
  aggregation over them. Example: *Bath museum → Herschel → Uranus → its five biggest moons → who
  discovered each, and when.*

### Breakdown

| type | n | hops | sequential_depth | breadth | min_turns_estimate |
|---|---:|---|---|---|---|
| deep_chain | 10 | 6–8 | 6–8 | 5–8 | 9–17 |
| wide | 11 | 10–14 | 1–2 | 10–14 | 2–5 |
| deep_wide | 8 | 7–9 | 3–5 | 6–9 | 5–9 |

- **Distributions over all 29 questions:**
  - depth: 1: 8, 2: 3, 3: 5, 4: 2, 5: 1, 6: 7, 7: 2, 8: 1
  - breadth: 5: 1, 6: 6, 7: 6, 8: 4, 9: 1, 10: 4, 12: 5, 13: 1, 14: 1
- **Over the production budget:** 11 of 29 questions have `min_turns_estimate` above 7. All 10
  deep chains are among them.
- **Domain split:** science/technology 10, history/politics/geography 9, arts/music/business 10.

### Super-hard caveats

- **Mergeable hops:** some chains have a chunk that states two links at once, so an agent can
  merge hops. The recorded `sequential_depth` accounts for this: sh002 and sh003 are 7 hops but
  depth 6. sh008's Perugia chunk says "near the Tiber", which allows a depth-5 shortcut; the notes
  flag this.
- **Keyword search only:** a few key chunks are reliably found by `keyword_search` but not by
  `semantic_search`, e.g. the Korolev/Kerimov clue (sh001), "Le Réveillon" (sh009), Czerny →
  Beethoven (sh007), the 1913 Ballets Russes description (sh027) and "Le Cateau-Cambrésis"
  (sh028).
- **Pinned to corpus wording:**
  - sh023: the six noble gases as listed in the Noble gas article; oganesson is excluded.
  - sh029: Tailleferre's birthplace is Saint-Maur-des-Fossés, so she does not count as Paris-born.
  - sh004: the Franz Joseph article wrongly calls Franz Ferdinand his "brother"; the question only
    uses "successor".
  - sh025: Nur-Sultan/Astana, with a move date of 1997 vs 1998; the question asks only for "late
    1990s".
- **Snapshot age:** sh014 describes Ken Mattingly as living (he died in 2023). The answer is
  unaffected.
- **Wide questions list their entities:** they name their entity set explicitly or define it
  closed, e.g. "presidents numbered 29th–38th". Answers depend on the exact dates in each
  article, and several include deliberate near-ties (Watt and Coulomb both born in 1736;
  Lawrence and Fermi both born in 1901; Nixon and Ford both born in 1913).
- **Guessable intermediates:** a strong model may guess intermediate entities (Marx, Lenin,
  Stalin) without searching. Every hop still needs a retrieved chunk as evidence for a supported
  answer.

## Benchmarking requirements

These are the requirements the eval runner was built against (see [Running the
evals](#running-the-evals) for what it records). It should drive
`rag.agents.orchestrator.run()`, or `POST /api/chat`, once per question with empty history, and
record the following for each question and run.

**Per question:**
- **Total wall-clock time.** This is the headline metric: time from submitting the question to
  the `done` (or `error`) event, including research, evaluation and responding.
- **Turns used:** `outcome.turns_used`, and the budget it ran under.
- **Outcome:** `supported`, `not_found`, `out_of_turns` or `error` (an exception or SSE `error`
  event).
- **Correctness.** Normalize the final answer and the research `answer` (case, punctuation,
  articles, number formatting), then match them against `answer` and `answer_aliases`. If there is
  no match, fall back to an LLM judge that sees the question, the expected answer and aliases,
  `notes`, and the system's answer, and returns correct, partially correct or incorrect. Record
  which method decided. Multi-part answers ("first X, last Y, count Z") need the judge to score
  each part.
- **Citation overlap.** Match the cited chunks to `supporting_chunks` on `(article_id,
  chunk_index)`, never on `chunk_id`, which changes on re-ingest. Report chunk-level recall and
  precision, plus article-level recall. `supporting_chunks` is *one sufficient set*, not the only
  one: the notes often name alternatives. So low overlap on a correct, supported answer is
  informational, not a failure. Correctness is the primary metric.
- **Time per turn:** the model-call latency and the tool-execution latency for each research turn.
- **Evaluator and responder time:** each evaluator call and the responder stream, including
  time to first token.
- **Token usage:** input, output and cached tokens per call (research, evaluator, responder),
  plus embedding tokens for `semantic_search`.
- **Trace data:** tool calls per turn, the number of evaluator rejections, and the verdicts from
  the `evaluation` events, kept for failure analysis.

**Reporting:**
- Break results down by tier (main vs super-hard), by type, and by difficulty (main set) or
  depth/breadth bucket (super-hard).
- For each group report accuracy, the outcome mix, and median and p90 wall-clock time. Also report
  median/p90 turns and tokens, and cost per question.
- For the super-hard tier, show accuracy and out-of-turns rate against `sequential_depth` and
  `min_turns_estimate`. That is the breaking-point curve.

**Budgets:** run the super-hard tier twice, at the production budget
(`research_max_turns = 7`) and at an extended budget (e.g. 15). This separates two failures:
- *Ran out of budget:* out-of-turns at 7 but correct at 15.
- *Reasoning or retrieval failed:* wrong, not found or out of turns even at 15.

The main set runs at the production budget. Run each configuration more than once, or at least
keep the seeds and model versions, because agent runs are not deterministic. Run questions with
bounded concurrency so wall-clock times aren't inflated by rate limits, and record the
concurrency level.

## Running the evals

`rag eval` runs the real pipeline (`rag.agents.orchestrator.run`) on the questions, times every
stage, counts tokens and grades each answer. It needs the DB and `OPENAI_API_KEY`.

```
uv run rag eval --set main --limit 5 --label smoke        # quick check
uv run rag eval --set main --label baseline-7turns         # the full main set
uv run rag eval --set super_hard --max-turns 15 --label sh-15
uv run rag eval --ids q001,q017 --no-judge                 # a few questions, exact match only
uv run rag eval-compare evals/results/A.jsonl evals/results/B.jsonl
```

(`uv run python -m rag.evals [options]` is the same as `rag eval [options]`.)

| option | default | meaning |
|---|---|---|
| `--set` | `main` | `main` (`questions.jsonl`), `super_hard` (`questions_super_hard.jsonl`) or `all` |
| `--file` | | any question JSONL (overrides `--set`) |
| `--ids`, `--limit` | | select questions by id / keep the first N |
| `--max-turns` | `research_max_turns` (7) | research-turn budget for this run |
| `--concurrency` | 4 | questions in flight at once (lower it on rate limits) |
| `--timeout` | 600 | per-question timeout in seconds; a timeout is recorded, the run continues |
| `--no-judge` | judge on | skip the LLM judge and grade by exact match only |
| `--out`, `--label` | `evals/results/` | where results go and the label in the file name |

### Output

Each run writes two files to `evals/results/` (git-ignored):

- `<UTC timestamp>_<label>.jsonl`: one record per question, appended as each question finishes.
  - `outcome`: `supported`, `not_found`, `out_of_turns` (from the pipeline's `outcome` event),
    `error` or `timeout`. Also `turns_used`, the `final_answer`, every `research_answers` and
    `evaluations` event, the `tool_calls` (with their turn), `evaluator_rejections`,
    `invalid_answers`, and the cited chunks with their `article_id` and `chunk_index`.
  - `timing` (seconds from the start of the question, `time.perf_counter()` as events arrive):
    `total_s` (start to `done`), `first_token_s` (first answer token), `research_turns` (each
    turn, from its `status` event to the next `status` event, split into `model_s`, up to the
    turn's first `tool_call`/`research_answer` event, and `tools_s`, from there to its last
    `tool_result`), `evaluator_calls`, and
    `responder_first_token_s` / `responder_total_s` (from the respond `status`; supported only).
  - `usage`: tokens summed over all pipeline model calls (`input_tokens` includes
    `cached_input_tokens`, `output_tokens` includes `reasoning_tokens`) plus embedding calls.
    `judge_usage` is counted separately.
  - `grading`: `exact`, `judge` (`{correct, partially_correct, reason}`), `correct` (the judge's
    verdict, or `exact` with `--no-judge` or if the judge call fails), `method` (`judge`,
    `exact`, or `no_answer` when there was no answer to judge), `partially_correct`,
    `citation_recall`, `citation_precision`, `citation_any_overlap` and
    `citation_article_recall`.
- `<same stem>.summary.json`: the run config (set, max_turns, concurrency, timeout, model names,
  git commit, wall-clock time) and the summary statistics printed at the end.

### Reading the summary

The printed table has one row per question `type` within each tier (`main`, or the `tier` field
of super-hard questions), plus an `ALL` row. A tier with several difficulties also gets a table
by `difficulty`, and the super-hard tier gets tables by `sequential_depth` and by
`min_turns_estimate` (the breaking-point curve).

- `acc`: final correctness (the judge). `exact`: normalized exact match. `part`: the judge's
  partially-correct rate (some parts of a multi-part answer right).
- `turns` / `trn90`: mean and p90 research turns used. `t_med` / `t_p90` / `t_max`: total
  wall-clock time per question.
- `tok_in` / `cached` / `tok_out`: mean tokens per question. `cost` is shown only when prices
  are configured (see below).
- `outcomes`: `sup` supported, `nf` not found, `oot` out of turns, `err` error, `to` timeout.

Then come a "where the time goes" line (mean research-turn time split into model call and
tool execution, mean evaluator-call time, responder
time to first token and total), the citation diagnostics, the 5 slowest questions and every
wrong answer with what was expected and what the judge said.

Time depends on `--concurrency` (API latency grows under load), so compare runs made at the
same concurrency.

**Grading.**
- *Exact match*: the answer, the answer without its parenthetical note, or any alias must occur
  in the final answer after normalization (casefold, accents, punctuation and a/an/the removed,
  whole words only). For an expected answer with `;`-separated parts, every part must occur.
- *Judge*: `settings.chat_model` with a strict JSON schema. It sees the question, the expected
  answer, the aliases, the question's `notes` and the final answer, and judges factual
  agreement with the expected answer only, not style. Not-found replies, errors and timeouts
  are wrong without a judge call. Every supported answer is judged (not only exact-match
  misses), so `exact` and the judge can be compared; exact match undercounts list answers
  written in prose, e.g. "Corsica, Elba and Saint Helena".
- *Citation recall* matches cited chunks to `supporting_chunks` on `(article_id, chunk_index)`.
  The supporting chunks are one sufficient route, not the only one, so treat it as a
  diagnostic, not as correctness.

**Cost.** Nothing in the repo states chat-model prices, so cost is off by default and only
tokens are shown. To estimate it, set USD per 1M tokens in the environment or `.env`:
`EVAL_PRICE_INPUT_PER_1M`, `EVAL_PRICE_CACHED_INPUT_PER_1M` (defaults to the input price),
`EVAL_PRICE_OUTPUT_PER_1M` and optionally `EVAL_PRICE_EMBEDDING_PER_1M` (`docs/ARCHITECTURE.md`
implies 0.02 for `text-embedding-3-small`).

**Comparing runs.** `rag eval-compare RUN_A RUN_B` takes two result `.jsonl` files (or their
`.summary.json`). It prints the accuracy, exact-match, turns, time (median/p90/max), token and
outcome changes overall, per tier and per type, then the questions that were fixed (wrong in A,
correct in B) or broken, and the largest per-question time changes.

## Caveats (main set)

**Snapshot versus reality.** The expected answer follows the corpus in each of these cases:
- **q003:** the Linus Pauling article says his 1954 Chemistry Nobel was for DNA structure (it was
  really for the chemical bond). The question uses only his 1962 Peace Prize.
- **q018:** the corpus dates Pompeii's destruction to 24 October 79 AD. Pompeii's founding is
  given only as "around 600 BC", so accept an answer of about 680 years.
- **q046:** the corpus uses the traditional date for the Buddha (about 563 BC). Later scholarly
  dates would change the order relative to Confucius.
- **q053:** the Kennedy article calls Oswald only "the main suspect".
- **q026:** "the last 100 years" is relative to when the article was written.

**Pinned or inferred answers:**
- **q071:** the corpus never says outright that Olga Koklova is Paul's mother. It follows from the
  1918 marriage, Paul's birth in 1921 and the 1935 divorce. The corpus spelling is "Koklova".
- **q059:** the Statue of Liberty article credits Maurice Koechlin, chief engineer of Eiffel's
  company, with the internal structure, while the Gustave Eiffel article credits Eiffel himself.
  The question is worded so both readings give 1832.
- **q008:** the C article credits both Thompson and Ritchie. Both received the Turing Award in
  1983.
- **q004:** Guido van Rossum's link to Haarlem appears only in the article's category line ("People
  from Haarlem").
- **q064:** the expected answer is the island Unguja. "Zanzibar" alone is too coarse, because it
  names the archipelago.

**Other caveats:**
- **Beethoven (q056, q077):** the corpus gives only his baptism date, not his birth date. Neither
  answer depends on it.
- **Numeric answers:** these come from the corpus's own figures, e.g. river lengths, mountain
  heights and stadium capacities, so they may differ slightly from other sources.
- **Changing chunk ids:** a re-chunk with different parameters invalidates `chunk_index`, and a
  re-ingest changes `chunk_id`. Run `validate.py` after either.
