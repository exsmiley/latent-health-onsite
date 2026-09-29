"""End-to-end tests for rag.embed.pool against a throwaway Postgres database (rag_test_embed)."""

from __future__ import annotations

import os
import signal
import time

import psycopg
import pytest
from embed_fakes import FakeEmbed, auth_error, embedded_state, fake_vector, seed, server_error

from rag.embed import pool
from rag.embed.pool import RetryPolicy

pytestmark = pytest.mark.db

FAST = RetryPolicy(max_attempts=4, base_delay=0.01, max_delay=0.05, min_rate_limit_pause=0.05)


def _parse(s: str) -> list[float]:
    return [float(x) for x in s.strip("[]").split(",")]


async def test_embeds_everything_except_bad_input(test_db, monkeypatch):
    ids = seed(test_db, n_articles=5, chunks_per_article=12, bad=[(3, 7)])
    bad_id = ids[3 * 12 + 7]
    fake = FakeEmbed(rate_limit_calls={2, 5}, retry_after="0.1")
    monkeypatch.setattr(pool, "embed_texts", fake)

    stats = await pool.run_embed(workers=4, batch_size=8, policy=FAST, progress_interval=0.05)

    state = embedded_state(test_db)
    assert [i for i, v in state.items() if v is None] == [bad_id]
    assert stats.embedded == len(ids) - 1
    assert stats.failed == 1 and stats.bad_ids == [bad_id]
    assert stats.rate_limited == 2 and stats.retries >= 2  # 429s were retried
    assert stats.splits == 3  # 8 -> 4 -> 2 -> 1
    assert fake.bad_request_calls == 4
    assert max(fake.batch_sizes) == 8

    # Vectors are written to the right rows.
    with psycopg.connect(test_db) as conn:
        rows = conn.execute("SELECT id, embed_text FROM chunks WHERE id = ANY(%s)", (ids[:3],))
        texts = dict(rows.fetchall())
    for i in ids[:3]:
        assert _parse(state[i]) == pytest.approx(fake_vector(texts[i]), abs=1e-6)

    # Shared cool-down: after a 429 (retry-after 0.1s) no worker starts a call before it expires.
    for t429 in fake.rate_limit_times:
        later = [t for t in fake.call_starts if t > t429]
        assert all(t >= t429 + 0.1 for t in later), "a worker ignored the shared cool-down"

    # A rerun changes nothing: only the bad row is still NULL and it fails again.
    rerun = FakeEmbed()
    monkeypatch.setattr(pool, "embed_texts", rerun)
    stats2 = await pool.run_embed(workers=4, batch_size=8, policy=FAST)
    assert stats2.target == 1 and stats2.embedded == 0 and stats2.bad_ids == [bad_id]
    assert embedded_state(test_db) == state


async def test_rerun_after_complete_run_is_noop(test_db, monkeypatch):
    seed(test_db, n_articles=2, chunks_per_article=5)
    monkeypatch.setattr(pool, "embed_texts", FakeEmbed())
    await pool.run_embed(workers=2, batch_size=4, policy=FAST)
    before = embedded_state(test_db)
    assert all(v is not None for v in before.values())

    fake = FakeEmbed()
    monkeypatch.setattr(pool, "embed_texts", fake)
    stats = await pool.run_embed(workers=2, batch_size=4, policy=FAST)
    assert stats.target == 0 and stats.embedded == 0 and fake.calls == 0
    assert embedded_state(test_db) == before


async def test_limit_is_respected(test_db, monkeypatch):
    ids = seed(test_db, n_articles=4, chunks_per_article=10)
    monkeypatch.setattr(pool, "embed_texts", FakeEmbed())

    stats = await pool.run_embed(workers=3, batch_size=4, limit=11, policy=FAST)
    state = embedded_state(test_db)
    done = [i for i, v in state.items() if v is not None]
    assert stats.embedded == 11 and done == ids[:11]  # lowest ids first

    # Resuming picks up where it left off.
    stats = await pool.run_embed(workers=3, batch_size=4, limit=100, policy=FAST)
    assert stats.embedded == len(ids) - 11
    assert all(v is not None for v in embedded_state(test_db).values())


async def test_persistent_server_errors_skip_batch_without_crashing(test_db, monkeypatch):
    seed(test_db, n_articles=2, chunks_per_article=6)
    fake = FakeEmbed(always=server_error())
    monkeypatch.setattr(pool, "embed_texts", fake)

    stats = await pool.run_embed(workers=2, batch_size=4, policy=FAST)
    assert stats.embedded == 0 and stats.failed == 12
    assert fake.calls == 3 * FAST.max_attempts  # 3 batches, each tried max_attempts times
    assert all(v is None for v in embedded_state(test_db).values())


async def test_auth_error_aborts_quickly(test_db, monkeypatch):
    seed(test_db, n_articles=3, chunks_per_article=10)
    fake = FakeEmbed(always=auth_error())
    monkeypatch.setattr(pool, "embed_texts", fake)

    start = time.monotonic()
    with pytest.raises(pool.EmbedFatalError, match="AuthenticationError"):
        await pool.run_embed(workers=4, batch_size=2, policy=RetryPolicy(base_delay=5))
    assert time.monotonic() - start < 2
    assert fake.calls <= 4  # each worker tried at most once, no retries
    assert all(v is None for v in embedded_state(test_db).values())


async def test_token_cap_splits_batches(test_db, monkeypatch):
    seed(test_db, n_articles=1, chunks_per_article=10)
    fake = FakeEmbed()
    monkeypatch.setattr(pool, "embed_texts", fake)
    # Each seeded chunk is ~40 + prefix allowance tokens, so a 150-token cap allows 2 per request.
    monkeypatch.setattr(pool, "MAX_TOKENS_PER_REQUEST", 150)
    stats = await pool.run_embed(workers=2, batch_size=100, policy=FAST)
    assert stats.embedded == 10
    assert max(fake.batch_sizes) == 2


async def test_ctrl_c_finishes_in_flight_and_leaves_db_consistent(test_db, monkeypatch):
    seed(test_db, n_articles=10, chunks_per_article=20)
    fake = FakeEmbed()

    async def interrupting(texts, **kwargs):
        if fake.calls == 3:
            os.kill(os.getpid(), signal.SIGINT)  # handled by run_embed's graceful handler
        return await fake(texts, **kwargs)

    monkeypatch.setattr(pool, "embed_texts", interrupting)
    stats = await pool.run_embed(workers=2, batch_size=5, policy=FAST)

    state = embedded_state(test_db)
    done = sum(v is not None for v in state.values())
    assert 0 < done < len(state)
    assert done == stats.embedded  # every successful request was written, nothing half-done
    assert done % 5 == 0  # whole batches only
