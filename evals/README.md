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
