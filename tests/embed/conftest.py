"""Fixtures for the embed pool tests.

DB tests use a dedicated, throwaway database (`rag_test_embed`), never the main `rag` database.
"""

from __future__ import annotations

import psycopg
import pytest
from embed_fakes import ADMIN_URL, TEST_DB, TEST_URL
from psycopg import sql

import rag.db
from rag.config import get_settings


@pytest.fixture(scope="module")
def test_database():
    """Create `rag_test_embed` (dropping any leftover), apply the schema, drop it afterwards."""
    try:
        admin = psycopg.connect(ADMIN_URL, autocommit=True, connect_timeout=3)
    except psycopg.OperationalError as e:
        pytest.skip(f"Postgres not reachable at {ADMIN_URL}: {e}")
    ident = sql.Identifier(TEST_DB)
    with admin:
        admin.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(ident))
        admin.execute(sql.SQL("CREATE DATABASE {}").format(ident))
    try:
        with psycopg.connect(TEST_URL, autocommit=True) as conn:
            conn.execute(rag.db.SCHEMA_PATH.read_text())
        yield TEST_URL
    finally:
        with psycopg.connect(ADMIN_URL, autocommit=True) as admin:
            admin.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(ident))


@pytest.fixture
async def test_db(test_database, monkeypatch):
    """Point settings and the shared pool at the test database for one test."""
    assert test_database.endswith("/" + TEST_DB)
    prev_pool = rag.db._pool
    monkeypatch.setenv("DATABASE_URL", test_database)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    get_settings.cache_clear()
    rag.db._pool = None
    assert get_settings().database_url == test_database
    try:
        yield test_database
    finally:
        await rag.db.close_pool()
        rag.db._pool = prev_pool
        monkeypatch.undo()
        get_settings.cache_clear()
