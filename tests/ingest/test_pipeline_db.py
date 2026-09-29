"""DB tests for rag.ingest.pipeline, against a throwaway `rag_test_ingest` database."""

import os

import psycopg
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import pytest_asyncio

from rag import db
from rag.config import get_settings
from rag.ingest.chunker import chunk_article
from rag.ingest.pipeline import iter_article_batches, run_ingest

pytestmark = pytest.mark.db

HOST = "postgresql://rag:rag@localhost:5433"
TEST_DB = "rag_test_ingest"
TEST_URL = f"{HOST}/{TEST_DB}"
ADMIN_URL = f"{HOST}/postgres"

BIG = " ".join(f"word{i}" for i in range(150)) + "."
ARTICLES = [
    {"id": "10", "url": "https://x/A", "title": "Alpha", "text": f"Alpha lead.\n\nHistory \n{BIG}"},
    {"id": "11", "url": "https://x/B", "title": "Beta", "text": f"{BIG}\n\nUses \n\n{BIG}"},
    {"id": "12", "url": "https://x/E", "title": "Empty", "text": "   \n\n  "},
    {"id": "13", "url": "https://x/G", "title": "Gamma", "text": "Gamma is a letter."},
    {"id": "14", "url": "https://x/D", "title": "Delta", "text": "Delta is a river mouth."},
]


async def _admin(sql: str) -> None:
    async with await psycopg.AsyncConnection.connect(ADMIN_URL, autocommit=True) as conn:
        await conn.execute(sql)


def _set_env(key: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value


@pytest_asyncio.fixture(loop_scope="session")
async def ingest_db(tmp_path):
    try:
        await _admin(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)")
        await _admin(f"CREATE DATABASE {TEST_DB}")
    except psycopg.OperationalError as e:
        pytest.skip(f"Postgres not available: {e}")

    dataset = tmp_path / "mini.parquet"
    pq.write_table(pa.Table.from_pylist(ARTICLES), dataset, row_group_size=2)

    old = {k: os.environ.get(k) for k in ("DATABASE_URL", "DATASET_PATH")}
    await db.close_pool()
    os.environ["DATABASE_URL"] = TEST_URL
    os.environ["DATASET_PATH"] = str(dataset)
    get_settings.cache_clear()
    db._pool = None
    assert get_settings().database_url == TEST_URL
    try:
        await db.apply_schema()
        yield TEST_URL
    finally:
        await db.close_pool()
        for k, v in old.items():
            _set_env(k, v)
        get_settings.cache_clear()
        await _admin(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)")


async def _query(url: str, sql: str) -> list[tuple]:
    async with await psycopg.AsyncConnection.connect(url) as conn:
        cur = await conn.execute(sql)
        return await cur.fetchall()


def test_iter_article_batches_respects_limit(tmp_path):
    path = tmp_path / "m.parquet"
    pq.write_table(pa.Table.from_pylist(ARTICLES), path, row_group_size=2)
    batches = list(iter_article_batches(path, batch_size=2, limit=3))
    assert [len(b) for b in batches] == [2, 1]
    assert batches[0][0] == (10, "https://x/A", "Alpha", ARTICLES[0]["text"])
    assert sum(len(b) for b in iter_article_batches(path, batch_size=2, limit=None)) == 5


async def test_ingest_writes_articles_and_chunks(ingest_db):
    await run_ingest(limit=3, batch_size=2)  # Alpha, Beta, Empty

    arts = await _query(ingest_db, "SELECT id, title, url FROM articles ORDER BY id")
    assert arts == [(10, "Alpha", "https://x/A"), (11, "Beta", "https://x/B")]  # empty skipped

    rows = await _query(
        ingest_db,
        "SELECT article_id, chunk_index, section, text, embed_text, token_count,"
        " embedding IS NULL, tsv IS NOT NULL FROM chunks ORDER BY id",
    )
    expected = []
    for a in ARTICLES[:2]:
        for c in chunk_article(a["title"], a["text"]):
            expected.append(
                (int(a["id"]), c.chunk_index, c.section, c.text, c.embed_text, c.token_count,
                 True, True)
            )  # fmt: skip
    assert rows == expected
    assert [r[2] for r in rows] == [None, None, "Uses"]  # Alpha lead < min: History packs in


async def test_ingest_is_idempotent_and_resumes(ingest_db):
    await run_ingest(limit=2, batch_size=2)
    before = await _query(ingest_db, "SELECT id, article_id, chunk_index FROM chunks ORDER BY id")

    await run_ingest(limit=2, batch_size=2)  # nothing new
    assert (
        await _query(ingest_db, "SELECT id, article_id, chunk_index FROM chunks ORDER BY id")
        == before
    )

    await run_ingest(limit=None, batch_size=2)  # picks up the rest only
    arts = await _query(ingest_db, "SELECT id FROM articles ORDER BY id")
    assert [a[0] for a in arts] == [10, 11, 13, 14]
    counts = await _query(
        ingest_db, "SELECT article_id, count(*) FROM chunks GROUP BY article_id ORDER BY 1"
    )
    assert counts == [(10, 1), (11, 2), (13, 1), (14, 1)]
    dupes = await _query(
        ingest_db, "SELECT article_id, chunk_index FROM chunks GROUP BY 1, 2 HAVING count(*) > 1"
    )
    assert dupes == []
