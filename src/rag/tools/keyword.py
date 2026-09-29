"""Keyword search: Postgres full-text search over chunks.tsv (GIN-indexed)."""

import asyncio

from rag import db
from rag.tools.blurb import make_blurb, normalize_queries
from rag.tools.models import QueryResults, SearchHit

# numnode(q) = 0 means the query had only stop words / no lexemes: return no hits (and skip
# Postgres' "doesn't contain lexemes" NOTICE path). `tsv @@ q` uses the GIN index chunks_tsv_idx.
_SQL = """
WITH q AS (SELECT websearch_to_tsquery('english', %(q)s) AS q)
SELECT c.id AS chunk_id, c.article_id, a.title, c.section, c.text, c.token_count,
       ts_rank_cd(c.tsv, q.q) AS score
FROM q
JOIN chunks c ON c.tsv @@ q.q
JOIN articles a ON a.id = c.article_id
WHERE numnode(q.q) > 0
ORDER BY score DESC, c.id
LIMIT %(k)s
"""


async def _search_one(query: str, top_k: int) -> QueryResults:
    async with db.connection() as conn:
        cur = await conn.execute(_SQL, {"q": query, "k": top_k})
        rows = await cur.fetchall()
    hits = [
        SearchHit(
            chunk_id=r["chunk_id"],
            article_id=r["article_id"],
            title=r["title"],
            section=r["section"],
            score=round(float(r["score"]), 4),
            blurb=make_blurb(r["text"]),
            text=r["text"],
            token_count=r["token_count"],
        )
        for r in rows
    ]
    return QueryResults(query=query, hits=hits)


async def keyword_search(queries: list[str], top_k: int = 5) -> list[QueryResults]:
    """Full-text search (websearch syntax), one concurrent query per input query.

    Queries are stripped and deduplicated (first occurrence wins); results keep that order.
    """
    qs = normalize_queries(queries)
    if not qs:
        return []
    top_k = max(1, int(top_k))
    return list(await asyncio.gather(*(_search_one(q, top_k) for q in qs)))
