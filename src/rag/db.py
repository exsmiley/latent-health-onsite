from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from pgvector.psycopg import register_vector_async
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from rag.config import ROOT, get_settings

SCHEMA_PATH = ROOT / "db" / "schema.sql"

_pool: AsyncConnectionPool | None = None


async def _configure(conn: AsyncConnection) -> None:
    await register_vector_async(conn)
    conn.row_factory = dict_row


async def get_pool() -> AsyncConnectionPool:
    """Process-wide async pool. Connections return dict rows and accept numpy/list vectors."""
    global _pool
    if _pool is None:
        _pool = AsyncConnectionPool(
            get_settings().database_url,
            min_size=1,
            max_size=20,
            configure=_configure,
            open=False,
        )
        await _pool.open()
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


@asynccontextmanager
async def connection() -> AsyncIterator[AsyncConnection]:
    pool = await get_pool()
    async with pool.connection() as conn:
        yield conn


async def apply_schema() -> None:
    async with await AsyncConnection.connect(get_settings().database_url, autocommit=True) as conn:
        await conn.execute(SCHEMA_PATH.read_text())
