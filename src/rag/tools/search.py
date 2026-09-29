"""Semantic (pgvector cosine) search over chunk embeddings."""

import asyncio

from pgvector import Vector

from rag import db, llm
from rag.tools.blurb import make_blurb, normalize_queries
from rag.tools.models import QueryResults, SearchHit

# The ORDER BY/LIMIT lives on `chunks` alone so the planner can use the HNSW index
# (chunks_embedding_hnsw_idx, vector_cosine_ops); the join to articles happens on the <= k rows.
# HNSW never indexes NULL vectors; the IS NOT NULL filter is kept for the seq-scan fallback.
_SQL = """
SELECT n.id AS chunk_id, n.article_id, a.title, n.section, n.text, n.token_count,
       1 - n.distance AS score
FROM (
    SELECT c.id, c.article_id, c.section, c.text, c.token_count,
           c.embedding <=> %(v)s AS distance
    FROM chunks c
    WHERE c.embedding IS NOT NULL
    ORDER BY c.embedding <=> %(v)s
    LIMIT %(k)s
) n
JOIN articles a ON a.id = n.article_id
ORDER BY n.distance, n.id
"""


def _ef_search(top_k: int) -> int:
    return max(40, 2 * top_k)


async def _search_one(query: str, vector: list[float], top_k: int) -> QueryResults:
    async with db.connection() as conn, conn.transaction():
        # SET cannot take bind parameters; the value is an int we computed.
        await conn.execute(f"SET LOCAL hnsw.ef_search = {_ef_search(top_k)}")
        cur = await conn.execute(_SQL, {"v": Vector(vector), "k": top_k})
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


async def semantic_search(queries: list[str], top_k: int = 5) -> list[QueryResults]:
    """Embed all queries in one request, then run one vector search per query concurrently.

    Queries are stripped and deduplicated (first occurrence wins); results keep that order.
    """
    qs = normalize_queries(queries)
    if not qs:
        return []
    top_k = max(1, int(top_k))
    vectors = await llm.embed_texts(qs)
    return list(await asyncio.gather(*(_search_one(q, v, top_k) for q, v in zip(qs, vectors))))
