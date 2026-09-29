"""Streaming ingest: parquet -> chunker (process pool) -> Postgres via COPY.

Articles are read in dataset order in pyarrow record batches of ``batch_size`` rows, so only a
few batches are ever held as Python objects. For each batch the ids already in ``articles`` are
skipped, the rest are chunked in a worker process, and the articles plus their chunks are
written with ``COPY`` in one transaction per batch. Chunking of later batches overlaps with the
DB writes of earlier ones. Re-running is safe: completed batches are skipped by id.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
import urllib.request
from collections import deque
from collections.abc import Iterator
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
from psycopg import AsyncConnection
from psycopg.rows import dict_row

from rag.config import get_settings
from rag.db import apply_schema
from rag.ingest.chunker import chunk_article

log = logging.getLogger(__name__)

COLUMNS = ["id", "url", "title", "text"]
# Below this many articles a process pool costs more (spawn + imports) than it saves.
POOL_THRESHOLD = 5_000

Article = tuple[int, str, str, str]  # (id, url, title, text)
ChunkRow = tuple[int, int, str | None, str, str, int]  # COPY row for `chunks`


# ---------------------------------------------------------------------------------------------
# Dataset download


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=dest.name + ".", suffix=".part", dir=dest.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            done, last_log = 0, time.monotonic()
            while block := resp.read(1 << 20):
                out.write(block)
                done += len(block)
                if time.monotonic() - last_log > 5:
                    last_log = time.monotonic()
                    pct = f" ({done / total:.0%})" if total else ""
                    log.info("downloaded %.1f MB%s", done / 1e6, pct)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)


async def ensure_dataset() -> Path:
    s = get_settings()
    path = Path(s.dataset_path)
    if not path.exists():
        log.info("dataset missing, downloading %s -> %s", s.dataset_url, path)
        await asyncio.to_thread(_download, s.dataset_url, path)
        log.info("download complete (%.1f MB)", path.stat().st_size / 1e6)
    return path


# ---------------------------------------------------------------------------------------------
# Reading and chunking


def iter_article_batches(path: Path, batch_size: int, limit: int | None) -> Iterator[list[Article]]:
    """Yield lists of (id, url, title, text) in dataset order, at most `limit` articles total."""
    remaining = limit
    pf = pq.ParquetFile(path)
    for rb in pf.iter_batches(batch_size=batch_size, columns=COLUMNS):
        if remaining is not None:
            if remaining <= 0:
                return
            if rb.num_rows > remaining:
                rb = rb.slice(0, remaining)
            remaining -= rb.num_rows
        cols = rb.to_pydict()
        yield [
            (int(i), u, t, x)
            for i, u, t, x in zip(cols["id"], cols["url"], cols["title"], cols["text"])
        ]


def chunk_batch(articles: list[Article]) -> list[tuple[Article, list[ChunkRow]]]:
    """Chunk a batch of articles (runs in a worker process). Drops articles with no chunks."""
    out = []
    for art in articles:
        art_id, _url, title, text = art
        chunks = chunk_article(title, text)
        if not chunks:
            continue
        rows = [
            (art_id, c.chunk_index, c.section, c.text, c.embed_text, c.token_count) for c in chunks
        ]
        out.append((art, rows))
    return out


# ---------------------------------------------------------------------------------------------
# DB


async def _existing_ids(conn: AsyncConnection, ids: list[int]) -> set[int]:
    cur = await conn.execute("SELECT id FROM articles WHERE id = ANY(%s)", (ids,))
    return {r["id"] for r in await cur.fetchall()}


async def _write(conn: AsyncConnection, chunked: list[tuple[Article, list[ChunkRow]]]) -> int:
    """Insert articles and chunks in one transaction. Returns the number of chunks written."""
    n_chunks = 0
    async with conn.transaction(), conn.cursor() as cur:
        async with cur.copy("COPY articles (id, url, title, text) FROM STDIN") as cp:
            for art, _ in chunked:
                await cp.write_row(art)
        async with cur.copy(
            "COPY chunks (article_id, chunk_index, section, text, embed_text, token_count) "
            "FROM STDIN"
        ) as cp:
            for _, rows in chunked:
                for row in rows:
                    await cp.write_row(row)
                n_chunks += len(rows)
    return n_chunks


# ---------------------------------------------------------------------------------------------
# Entry point


async def run_ingest(limit: int | None, batch_size: int) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    path = await ensure_dataset()
    await apply_schema()  # idempotent; makes `rag ingest` work on a fresh database

    total_rows = pq.ParquetFile(path).metadata.num_rows
    planned = min(limit, total_rows) if limit is not None else total_rows
    workers = max(1, (os.cpu_count() or 2) - 1)
    use_pool = planned >= POOL_THRESHOLD
    executor: Executor = (
        ProcessPoolExecutor(max_workers=workers) if use_pool else ThreadPoolExecutor(max_workers=1)
    )
    max_inflight = workers * 2 if use_pool else 1

    loop = asyncio.get_running_loop()
    t0 = time.monotonic()
    seen = skipped = articles_written = chunks_written = 0
    pending: deque[asyncio.Future] = deque()

    async def drain_one(conn: AsyncConnection) -> None:
        nonlocal articles_written, chunks_written
        chunked = await pending.popleft()
        if chunked:
            chunks_written += await _write(conn, chunked)
            articles_written += len(chunked)
        elapsed = time.monotonic() - t0
        log.info(
            "%d/%d articles scanned, %d written, %d skipped, %d chunks (%.0f articles/s)",
            seen, planned, articles_written, skipped, chunks_written, seen / max(elapsed, 1e-9),
        )  # fmt: skip

    try:
        # A dedicated autocommit connection (not the shared pool), so each batch's
        # `conn.transaction()` is a real BEGIN/COMMIT.
        async with await AsyncConnection.connect(
            get_settings().database_url, autocommit=True, row_factory=dict_row
        ) as conn:
            for batch in iter_article_batches(path, batch_size, limit):
                seen += len(batch)
                existing = await _existing_ids(conn, [a[0] for a in batch])
                todo = [a for a in batch if a[0] not in existing and a[3].strip()]
                skipped += len(batch) - len(todo)
                pending.append(loop.run_in_executor(executor, chunk_batch, todo))
                while len(pending) >= max_inflight:
                    await drain_one(conn)
            while pending:
                await drain_one(conn)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    log.info(
        "ingest done in %.1fs: %d articles scanned, %d written, %d skipped, %d chunks",
        time.monotonic() - t0, seen, articles_written, skipped, chunks_written,
    )  # fmt: skip
