"""Fixtures for tool tests: a throwaway Postgres database (never the main `rag` DB)."""

import os

import psycopg
import pytest
import pytest_asyncio
from pgvector import Vector
from pgvector.psycopg import register_vector_async

from rag import db
from rag.config import get_settings

HOST = "postgresql://rag:rag@localhost:5433"
TEST_DB = "rag_test_tools"
TEST_URL = f"{HOST}/{TEST_DB}"
ADMIN_URL = f"{HOST}/postgres"
DIM = 1536


def vec(**axes: float) -> list[float]:
    """Sparse hand-made embedding: vec(e0=1.0, e2=0.5) -> [1, 0, 0.5, 0, ...]."""
    v = [0.0] * DIM
    for k, val in axes.items():
        v[int(k[1:])] = val
    return v


LONG_TEXT = " ".join(f"word{i}" for i in range(60)) + " quantum   mechanics\n\nfinal paragraph."

ARTICLES = [
    (
        1,
        "Albert Einstein",
        "https://simple.wikipedia.org/wiki/Albert%20Einstein",
        "Albert Einstein was a German-born physicist.\n\nNobel Prize \n\nHe won it in 1921.",
    ),
    (
        2,
        "Marie Curie",
        "https://simple.wikipedia.org/wiki/Marie%20Curie",
        "Marie Curie was a Polish and French physicist and chemist.",
    ),
    (3, "Paris", "https://simple.wikipedia.org/wiki/Paris", "Paris is the capital of France."),
]

# (id, article_id, chunk_index, section, text, embedding)
CHUNKS = [
    (
        101,
        1,
        0,
        None,
        "Albert Einstein was a German-born physicist who developed the theory of relativity.",
        vec(e0=1.0),
    ),
    (
        102,
        1,
        1,
        "Nobel Prize",
        "He won the Nobel Prize in Physics in 1921 for his work on the photoelectric effect.",
        vec(e1=1.0),
    ),
    (103, 1, 2, "Later life", LONG_TEXT, None),  # not embedded yet
    (
        104,
        2,
        0,
        None,
        (
            "Marie Curie was a Polish and French physicist and chemist who studied radioactivity. "
            "She won a Nobel in two sciences."
        ),
        vec(e1=0.8, e2=0.6),
    ),
    (105, 2, 1, "Death", "She died in 1934 from aplastic anaemia.", vec(e3=1.0)),
]


async def _admin(sql: str) -> None:
    async with await psycopg.AsyncConnection.connect(ADMIN_URL, autocommit=True) as conn:
        await conn.execute(sql)


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def tools_db():
    try:
        await _admin(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)")
        await _admin(f"CREATE DATABASE {TEST_DB}")
    except psycopg.OperationalError as e:
        pytest.skip(f"Postgres not available: {e}")

    old_url = os.environ.get("DATABASE_URL")
    await db.close_pool()
    os.environ["DATABASE_URL"] = TEST_URL
    get_settings.cache_clear()
    db._pool = None
    assert get_settings().database_url == TEST_URL
    try:
        await db.apply_schema()
        async with await psycopg.AsyncConnection.connect(TEST_URL, autocommit=True) as conn:
            await register_vector_async(conn)
            async with conn.cursor() as cur:
                await cur.executemany(
                    "INSERT INTO articles (id, title, url, text) VALUES (%s, %s, %s, %s)", ARTICLES
                )
                titles = {a[0]: a[1] for a in ARTICLES}
                rows = []
                for cid, aid, idx, section, text, emb in CHUNKS:
                    head = f"{titles[aid]} > {section}" if section else titles[aid]
                    rows.append(
                        (
                            cid,
                            aid,
                            idx,
                            section,
                            text,
                            f"{head}\n\n{text}",
                            len(text.split()),
                            Vector(emb) if emb is not None else None,
                        )
                    )
                await cur.executemany(
                    "INSERT INTO chunks (id, article_id, chunk_index, section, text, embed_text,"
                    " token_count, embedding) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                    rows,
                )
        yield TEST_URL
    finally:
        await db.close_pool()
        if old_url is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = old_url
        get_settings.cache_clear()
        await _admin(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)")


class FakeEmbedder:
    """Stands in for rag.llm.embed_texts: returns preset vectors and records every call."""

    def __init__(self, table: dict[str, list[float]]):
        self.table = table
        self.calls: list[list[str]] = []

    async def __call__(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self.table[t] for t in texts]


@pytest.fixture
def fake_embed(monkeypatch):
    fake = FakeEmbedder(
        {
            "relativity": vec(e0=1.0),
            "nobel prize winners": vec(e1=1.0),
            "radioactivity chemist": vec(e2=1.0),
            "how did curie die": vec(e3=1.0),
        }
    )
    monkeypatch.setattr("rag.llm.embed_texts", fake)
    return fake
