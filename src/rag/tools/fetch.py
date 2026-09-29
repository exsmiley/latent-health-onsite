"""Fetch full chunk texts and/or whole articles by id."""

import asyncio

from rag import db
from rag.tools.models import Article, Chunk, FetchResult

_CHUNKS_SQL = """
SELECT c.id AS chunk_id, c.article_id, a.title, a.url, c.section, c.chunk_index, c.text
FROM chunks c
JOIN articles a ON a.id = c.article_id
WHERE c.id = ANY(%s)
"""

_ARTICLES_SQL = """
SELECT a.id AS article_id, a.title, a.url, a.text,
       COALESCE(
           (SELECT array_agg(c.id ORDER BY c.chunk_index) FROM chunks c WHERE c.article_id = a.id),
           '{}'
       ) AS chunk_ids
FROM articles a
WHERE a.id = ANY(%s)
"""


def _dedupe(ids: list[int]) -> list[int]:
    return list(dict.fromkeys(int(i) for i in ids))


async def _fetch_chunks(ids: list[int]) -> list[Chunk]:
    if not ids:
        return []
    async with db.connection() as conn:
        rows = await (await conn.execute(_CHUNKS_SQL, (ids,))).fetchall()
    by_id = {r["chunk_id"]: Chunk(**r) for r in rows}
    return [by_id[i] for i in ids if i in by_id]


async def _fetch_articles(ids: list[int]) -> list[Article]:
    if not ids:
        return []
    async with db.connection() as conn:
        rows = await (await conn.execute(_ARTICLES_SQL, (ids,))).fetchall()
    by_id = {r["article_id"]: Article(**r) for r in rows}
    return [by_id[i] for i in ids if i in by_id]


async def fetch(chunk_ids: list[int] = [], article_ids: list[int] = []) -> FetchResult:  # noqa: B006
    """Fetch chunks and/or whole articles (full text, uncapped) in the requested order.

    Duplicate ids are collapsed; ids that do not exist are reported in missing_*_ids.
    """
    cids, aids = _dedupe(chunk_ids), _dedupe(article_ids)
    chunks, articles = await asyncio.gather(_fetch_chunks(cids), _fetch_articles(aids))
    found_c = {c.chunk_id for c in chunks}
    found_a = {a.article_id for a in articles}
    return FetchResult(
        chunks=chunks,
        articles=articles,
        missing_chunk_ids=[i for i in cids if i not in found_c],
        missing_article_ids=[i for i in aids if i not in found_a],
    )
