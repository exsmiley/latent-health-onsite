"""Worker pool that fills `chunks.embedding` for every chunk where it is NULL.

Layout:

    producer ──(bounded asyncio.Queue of batches)──► N workers ──► UPDATE ... FROM unnest(...)
                                                        │
                                                   embed_texts (OpenAI)

* The producer keyset-paginates `chunks WHERE embedding IS NULL AND id > $last ORDER BY id`,
  packs rows into batches capped by count (`batch_size`) and by estimated tokens
  (`MAX_TOKENS_PER_REQUEST`), and puts them on a bounded queue, so memory stays flat.
* Each worker embeds a batch and writes it back in one statement / one transaction, so the DB is
  consistent whenever the process stops. Rows that fail stay NULL and a rerun picks them up.
* Transient errors (429, 5xx, connection errors, timeouts) are retried with exponential backoff
  plus full jitter, honouring `retry-after(-ms)`. A 429 also sets a shared cool-down that pauses
  every worker. After `max_attempts` the batch is logged and skipped.
* Non-retryable errors (400 etc.) split the batch in half recursively to isolate the bad input.
* Auth / permission / unknown-model / quota errors abort the whole run immediately.
* First Ctrl-C: stop taking new batches and let in-flight requests finish. Second Ctrl-C: hard
  stop (the in-flight transactions roll back).
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import random
import signal
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime

import openai
from pgvector import Vector

from rag import db
from rag.config import get_settings
from rag.llm import embed_texts

log = logging.getLogger(__name__)

# The embeddings endpoint caps a request at 300k tokens (and 2048 inputs). Stay well under it,
# since the per-row estimate below is approximate.
MAX_TOKENS_PER_REQUEST = 250_000
MAX_INPUTS_PER_REQUEST = 2048

EmbedFn = Callable[[list[str]], Awaitable[list[list[float]]]]

_SELECT_PAGE = """
    SELECT id,
           embed_text,
           -- token_count covers `text`; add a generous allowance for the "title > section" prefix
           token_count + (length(embed_text) - length(text)) / 2 + 8 AS est_tokens
      FROM chunks
     WHERE embedding IS NULL AND id > %s
     ORDER BY id
     LIMIT %s
"""

_UPDATE_BATCH = """
    UPDATE chunks AS c
       SET embedding = v.e
      FROM unnest(%s::bigint[], %s::vector[]) AS v(id, e)
     WHERE c.id = v.id AND c.embedding IS NULL
"""


class EmbedFatalError(SystemExit):
    """Unrecoverable error (no/invalid API key, no quota, unknown model). Exits with a message."""


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 8
    base_delay: float = 1.0  # seconds; attempt n waits up to base * 2**(n-1)
    max_delay: float = 60.0
    min_rate_limit_pause: float = 1.0  # shared cool-down after a 429 without retry-after


DEFAULT_POLICY = RetryPolicy()


@dataclass(slots=True)
class Row:
    id: int
    text: str
    tokens: int


@dataclass
class Stats:
    target: int = 0  # rows this run intends to embed (initial NULL count, capped by limit)
    embedded: int = 0
    tokens: int = 0
    failed: int = 0  # rows skipped after exhausting retries or isolated as bad input
    requests: int = 0
    retries: int = 0
    rate_limited: int = 0
    splits: int = 0
    bad_ids: list[int] = field(default_factory=list)
    started: float = field(default_factory=time.monotonic)

    def line(self) -> str:
        elapsed = max(time.monotonic() - self.started, 1e-9)
        rate = self.embedded / elapsed
        remaining = max(self.target - self.embedded - self.failed, 0)
        eta = _fmt_duration(remaining / rate) if rate > 0 else "?"
        return (
            f"embedded {self.embedded:,}/{self.target:,} | {rate:,.1f} chunks/s | "
            f"{self.tokens / elapsed:,.0f} tok/s | remaining {remaining:,} | ETA {eta} | "
            f"failed {self.failed:,} | retries {self.retries:,} (429s {self.rate_limited:,})"
        )


def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


class Batcher:
    """Packs rows (in order) into batches capped by count and by estimated tokens."""

    def __init__(self, batch_size: int, max_tokens: int = MAX_TOKENS_PER_REQUEST):
        self.batch_size = max(1, min(batch_size, MAX_INPUTS_PER_REQUEST))
        self.max_tokens = max_tokens
        self._rows: list[Row] = []
        self._tokens = 0

    def add(self, row: Row) -> list[Row] | None:
        """Add a row; returns a full batch if adding it would overflow the current one."""
        out = None
        if self._rows and self._tokens + row.tokens > self.max_tokens:
            out = self.flush()
        self._rows.append(row)
        self._tokens += row.tokens
        if len(self._rows) >= self.batch_size:
            assert out is None  # batch_size >= 1, and a flush above leaves exactly one row
            out = self.flush()
        return out

    def flush(self) -> list[Row] | None:
        rows, self._rows, self._tokens = self._rows, [], 0
        return rows or None


def retry_after_seconds(exc: BaseException) -> float | None:
    """Parse `retry-after-ms` / `retry-after` (seconds or HTTP date) from an API error."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    if (ms := headers.get("retry-after-ms")) is not None:
        try:
            return max(float(ms) / 1000, 0.0)
        except ValueError:
            pass
    if (value := headers.get("retry-after")) is not None:
        try:
            return max(float(value), 0.0)
        except ValueError:
            try:
                return max(parsedate_to_datetime(value).timestamp() - time.time(), 0.0)
            except (TypeError, ValueError):
                return None
    return None


def backoff_delay(attempt: int, policy: RetryPolicy) -> float:
    """Exponential backoff with full jitter (attempt is 1-based)."""
    cap = min(policy.max_delay, policy.base_delay * 2 ** (attempt - 1))
    return random.uniform(cap / 2, cap)


class _NonRetryable(Exception):
    pass


class _GiveUp(Exception):
    pass


class _Stopped(Exception):
    pass


def _is_fatal(exc: BaseException) -> bool:
    if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
        return True
    if isinstance(exc, openai.NotFoundError):  # e.g. unknown embedding model
        return True
    if (
        isinstance(exc, openai.RateLimitError)
        and getattr(exc, "code", None) == "insufficient_quota"
    ):
        return True
    # Anything that isn't an API error is a bug or misconfiguration (e.g. a TypeError, a missing
    # API key); bisecting the batch would just skip every row one by one.
    return not isinstance(exc, openai.APIError)


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (openai.RateLimitError, openai.APIConnectionError)):
        return True  # APITimeoutError subclasses APIConnectionError
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code >= 500 or exc.status_code in (408, 409)
    return False


class EmbedPool:
    def __init__(
        self,
        workers: int,
        batch_size: int,
        limit: int | None,
        *,
        embed_fn: EmbedFn,
        policy: RetryPolicy,
        progress_interval: float,
        max_tokens: int | None = None,
    ):
        self.workers = max(1, workers)
        self.batch_size = batch_size
        self.limit = limit
        self.embed_fn = embed_fn
        self.policy = policy
        self.progress_interval = progress_interval
        self.max_tokens = max_tokens or MAX_TOKENS_PER_REQUEST
        self.stats = Stats()
        self.queue: asyncio.Queue[list[Row] | None] = asyncio.Queue(maxsize=self.workers * 2)
        self.stop = asyncio.Event()
        self.fatal: BaseException | None = None
        self._cool_until = 0.0

    # ----------------------------------------------------------------------------- run

    async def run(self) -> Stats:
        async with db.connection() as conn:
            cur = await conn.execute("SELECT count(*) AS n FROM chunks WHERE embedding IS NULL")
            total = (await cur.fetchone())["n"]
        self.stats.target = total if self.limit is None else min(total, self.limit)
        log.info(
            "embedding %s chunks (%s NULL in total) with %d workers, batch size %d",
            f"{self.stats.target:,}",
            f"{total:,}",
            self.workers,
            self.batch_size,
        )
        if self.stats.target == 0:
            return self.stats

        self.stats.started = time.monotonic()
        reporter = asyncio.create_task(self._report())
        try:
            await asyncio.gather(self._produce(), *(self._work(i) for i in range(self.workers)))
        finally:
            reporter.cancel()
        return self.stats

    def request_stop(self) -> None:
        if not self.stop.is_set():
            log.warning("stopping: finishing in-flight batches (press Ctrl-C again to abort)")
            self.stop.set()

    # ------------------------------------------------------------------------ producer

    async def _produce(self) -> None:
        try:
            await self._produce_batches()
        except Exception as e:  # e.g. the DB went away: stop cleanly, report at the end
            log.exception("producer failed")
            self._abort(e)
        # Workers drain the queue until they see a sentinel, so these puts cannot block forever.
        # (Skipped on cancellation, when the workers are being cancelled too.)
        for _ in range(self.workers):
            await self.queue.put(None)

    async def _produce_batches(self) -> None:
        batcher = Batcher(self.batch_size, self.max_tokens)
        page_size = max(1000, self.batch_size * self.workers)
        last_id, sent = 0, 0
        while not self.stop.is_set():
            n = page_size if self.limit is None else min(page_size, self.limit - sent)
            if n <= 0:
                break
            async with db.connection() as conn:
                cur = await conn.execute(_SELECT_PAGE, (last_id, n))
                page = await cur.fetchall()
            if not page:
                break
            last_id = page[-1]["id"]
            sent += len(page)
            for r in page:
                batch = batcher.add(Row(r["id"], r["embed_text"], r["est_tokens"]))
                if batch and not await self._put(batch):
                    return
            if len(page) < n:
                break
        if batch := batcher.flush():
            await self._put(batch)

    async def _put(self, batch: list[Row]) -> bool:
        if self.stop.is_set():
            return False
        await self.queue.put(batch)
        return True

    # ------------------------------------------------------------------------- workers

    async def _work(self, idx: int) -> None:
        while True:
            batch = await self.queue.get()
            if batch is None:
                return
            if self.stop.is_set():
                continue  # discard queued work; keep draining until our sentinel
            try:
                await self._handle(batch)
            except Exception:  # never let one batch kill the worker
                log.exception(
                    "worker %d: unexpected error on ids %d..%d; skipping",
                    idx,
                    batch[0].id,
                    batch[-1].id,
                )
                self.stats.failed += len(batch)

    async def _handle(self, rows: list[Row]) -> None:
        try:
            vectors = await self._embed_with_retry(rows)
        except _Stopped:
            return
        except _GiveUp:
            self.stats.failed += len(rows)
            return
        except _NonRetryable as e:
            cause = e.__cause__
            if len(rows) == 1:
                log.error(
                    "bad input: chunk id %d cannot be embedded (%s); skipping",
                    rows[0].id,
                    _describe(cause),
                )
                self.stats.failed += 1
                self.stats.bad_ids.append(rows[0].id)
                return
            log.warning(
                "non-retryable error on %d chunks (ids %d..%d): %s; splitting",
                len(rows),
                rows[0].id,
                rows[-1].id,
                _describe(cause),
            )
            self.stats.splits += 1
            mid = len(rows) // 2
            await self._handle(rows[:mid])
            await self._handle(rows[mid:])
            return
        await self._write(rows, vectors)

    async def _embed_with_retry(self, rows: list[Row]) -> list[list[float]]:
        texts = [r.text for r in rows]
        for attempt in range(1, self.policy.max_attempts + 1):
            await self._wait_cooldown()
            if self.stop.is_set():
                raise _Stopped
            try:
                self.stats.requests += 1
                vectors = await self.embed_fn(texts)
                if len(vectors) != len(rows):
                    raise ValueError(f"got {len(vectors)} vectors for {len(rows)} inputs")
                return vectors
            except Exception as e:
                if _is_fatal(e):
                    self._abort(e)
                    raise _Stopped from e
                if not _is_retryable(e):
                    raise _NonRetryable from e
                delay = backoff_delay(attempt, self.policy)
                hinted = retry_after_seconds(e)
                if hinted is not None:
                    delay = hinted + random.uniform(0, self.policy.base_delay / 4)
                if isinstance(e, openai.RateLimitError):
                    self.stats.rate_limited += 1
                    pause = max(delay, self.policy.min_rate_limit_pause)
                    self._cool_until = max(self._cool_until, time.monotonic() + pause)
                if attempt == self.policy.max_attempts:
                    log.error(
                        "giving up on %d chunks (ids %d..%d) after %d attempts: %s",
                        len(rows),
                        rows[0].id,
                        rows[-1].id,
                        attempt,
                        _describe(e),
                    )
                    raise _GiveUp from e
                self.stats.retries += 1
                log.warning(
                    "attempt %d/%d failed for %d chunks (%s); retrying in %.1fs",
                    attempt,
                    self.policy.max_attempts,
                    len(rows),
                    _describe(e),
                    delay,
                )
                if await self._sleep_or_stop(delay):
                    raise _Stopped from e
        raise AssertionError("unreachable")

    async def _write(self, rows: list[Row], vectors: list[list[float]]) -> None:
        async with db.connection() as conn:
            cur = await conn.execute(
                _UPDATE_BATCH, ([r.id for r in rows], [Vector(v) for v in vectors])
            )
            updated = cur.rowcount
        self.stats.embedded += updated
        self.stats.tokens += sum(r.tokens for r in rows)

    # ------------------------------------------------------------------------- helpers

    def _abort(self, exc: BaseException) -> None:
        if self.fatal is None:
            self.fatal = exc
            log.error("fatal error, aborting run: %s", _describe(exc))
        self.stop.set()

    async def _wait_cooldown(self) -> None:
        while (remaining := self._cool_until - time.monotonic()) > 0:
            if await self._sleep_or_stop(remaining):
                return

    async def _sleep_or_stop(self, seconds: float) -> bool:
        """Sleep, waking early if a stop is requested. Returns True if stopped."""
        try:
            await asyncio.wait_for(self.stop.wait(), timeout=max(seconds, 0))
            return True
        except TimeoutError:
            return False

    async def _report(self) -> None:
        while True:
            await asyncio.sleep(self.progress_interval)
            log.info(self.stats.line())


def _describe(exc: BaseException | None) -> str:
    if exc is None:
        return "unknown error"
    status = getattr(exc, "status_code", None)
    prefix = f"{type(exc).__name__}" + (f" {status}" if status else "")
    return f"{prefix}: {exc}"[:500]


def _ensure_logging() -> None:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if log.getEffectiveLevel() > logging.INFO:
        log.setLevel(logging.INFO)


async def run_embed(
    workers: int | None = None,
    batch_size: int | None = None,
    limit: int | None = None,
    *,
    embed_fn: EmbedFn | None = None,
    policy: RetryPolicy | None = None,
    progress_interval: float = 5.0,
) -> Stats:
    """Embed every chunk whose embedding is NULL (at most `limit`). Resumable and idempotent.

    Raises EmbedFatalError (a SystemExit with a message) on unrecoverable API errors.
    """
    _ensure_logging()
    settings = get_settings()
    if embed_fn is None:
        embed_fn = functools.partial(embed_texts, max_retries=0)  # the pool owns retries
        if not (settings.openai_api_key or os.environ.get("OPENAI_API_KEY")):
            raise EmbedFatalError(
                "OPENAI_API_KEY is not set. Put it in .env or the environment, then rerun "
                "`rag embed` (it resumes where it left off)."
            )

    pool = EmbedPool(
        workers or settings.embed_workers,
        batch_size or settings.embed_batch_size,
        limit,
        embed_fn=embed_fn,
        policy=policy or DEFAULT_POLICY,
        progress_interval=progress_interval,
    )

    loop = asyncio.get_running_loop()
    handler_installed = False
    try:
        loop.add_signal_handler(signal.SIGINT, _on_sigint, loop, pool)
        handler_installed = True
    except (NotImplementedError, RuntimeError, ValueError):
        pass  # not on the main thread / not supported; Ctrl-C then cancels hard

    owns_pool = db._pool is None
    try:
        stats = await pool.run()
    finally:
        if handler_installed:
            loop.remove_signal_handler(signal.SIGINT)
        if owns_pool:
            await db.close_pool()

    elapsed = time.monotonic() - stats.started
    log.info(
        "done in %s: embedded %s chunks (%s est. tokens) in %s requests; failed %s; "
        "retries %s (429s %s); splits %s%s",
        _fmt_duration(elapsed),
        f"{stats.embedded:,}",
        f"{stats.tokens:,}",
        f"{stats.requests:,}",
        f"{stats.failed:,}",
        f"{stats.retries:,}",
        f"{stats.rate_limited:,}",
        f"{stats.splits:,}",
        f"; bad chunk ids {stats.bad_ids[:20]}" if stats.bad_ids else "",
    )
    if stats.failed:
        log.info("failed rows are still NULL; rerun `rag embed` to retry them")
    if pool.fatal is not None:
        raise EmbedFatalError(f"embedding aborted: {_describe(pool.fatal)}")
    return stats


def _on_sigint(loop: asyncio.AbstractEventLoop, pool: EmbedPool) -> None:
    pool.request_stop()
    # A second Ctrl-C falls back to the default handler (KeyboardInterrupt -> hard stop).
    loop.remove_signal_handler(signal.SIGINT)
