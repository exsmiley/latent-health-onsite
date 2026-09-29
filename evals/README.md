# Evaluation question set

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
uv run python evals/validate.py [path]     # defaults to evals/questions.jsonl
```

The script checks that:
- every line parses and has the required fields, types and enum values;
- ids and question texts are unique;
- every supporting chunk exists with the recorded `article_id`, `chunk_index` and `title` (a
  `section` mismatch is only a warning);
- each `evidence` string appears in that chunk's text after whitespace normalization.

It prints counts by type, difficulty and hops, and exits non-zero on any error. It only runs
SELECTs. It currently passes with 0 errors and 0 warnings.

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
    `evaluations` event, the `tool_calls`, and the cited chunks with their `article_id` and
    `chunk_index`.
  - `timing` (seconds from the start of the question, `time.perf_counter()` as events arrive):
    `total_s` (start to `done`), `first_token_s` (first answer token), `research_turns` (each
    turn, from its `status` event to the next `status` event), `evaluator_calls`, and
    `responder_first_token_s` / `responder_total_s` (from the respond `status`; supported only).
  - `usage`: tokens summed over all pipeline model calls (`input_tokens` includes
    `cached_input_tokens`, `output_tokens` includes `reasoning_tokens`) plus embedding calls.
    `judge_usage` is counted separately.
  - `grading`: `exact`, `judge` (`{correct, partially_correct, reason}`), `correct` (the judge's
    verdict, or `exact` with `--no-judge` or if the judge call fails), `partially_correct`,
    `citation_recall`, `citation_any_overlap` and `citation_article_recall`.
- `<same stem>.summary.json`: the run config (set, max_turns, concurrency, timeout, model names,
  git commit, wall-clock time) and the summary statistics printed at the end.

### Reading the summary

The printed table has one row per question `type` within each tier (`main`, or the `tier` field
of super-hard questions), plus an `ALL` row. Super-hard tiers also get a table by
`sequential_depth`.

- `acc`: final correctness (the judge). `exact`: normalized exact match. `part`: the judge's
  partially-correct rate (some parts of a multi-part answer right).
- `turns`: mean research turns used. `t_med` / `t_p90` / `t_max`: total time per question.
- `tok_in` / `cached` / `tok_out`: mean tokens per question. `cost` is shown only when prices
  are configured (see below).
- `outcomes`: `sup` supported, `nf` not found, `oot` out of turns, `err` error, `to` timeout.

Then come a "where the time goes" line (mean research-turn and evaluator-call time, responder
time to first token and total), the citation diagnostics, the 5 slowest questions and every
wrong answer with what was expected and what the judge said.

Time depends on `--concurrency` (API latency grows under load), so compare runs made at the
same concurrency.

**Grading.**
- *Exact match*: the answer, the answer without its parenthetical note, or any alias must occur
  in the final answer after normalization (casefold, accents, punctuation and a/an/the removed,
  whole words only). For an expected answer with `;`-separated parts, every part must occur.
- *Judge*: `settings.chat_model` with a strict JSON schema. It judges factual agreement with
  the expected answer only, not style. Not-found replies, errors and timeouts are wrong without
  a judge call.
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

## Caveats

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
