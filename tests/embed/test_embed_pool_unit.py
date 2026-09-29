"""Pure unit tests for rag.embed.pool (no DB, no network)."""

from __future__ import annotations

import openai
import pytest
from embed_fakes import (
    _response,
    auth_error,
    bad_request_error,
    rate_limit_error,
    server_error,
)

from rag.config import get_settings
from rag.embed import pool
from rag.embed.pool import Batcher, RetryPolicy, Row, backoff_delay, retry_after_seconds


def rows(tokens: list[int]) -> list[Row]:
    return [Row(i + 1, f"t{i}", t) for i, t in enumerate(tokens)]


def pack(batcher: Batcher, items: list[Row]) -> list[list[int]]:
    out = [b for r in items if (b := batcher.add(r))]
    if last := batcher.flush():
        out.append(last)
    return [[r.id for r in b] for b in out]


def test_batcher_caps_by_count():
    assert pack(Batcher(3), rows([10] * 7)) == [[1, 2, 3], [4, 5, 6], [7]]


def test_batcher_caps_by_tokens():
    # 100 + 100 fits in 250, the third would overflow
    assert pack(Batcher(10, max_tokens=250), rows([100, 100, 100, 100, 60])) == [
        [1, 2],
        [3, 4],
        [5],
    ]


def test_batcher_oversized_row_is_its_own_batch():
    assert pack(Batcher(10, max_tokens=100), rows([10, 500, 10])) == [[1], [2], [3]]


def test_batcher_batch_size_one():
    assert pack(Batcher(1, max_tokens=5), rows([10, 10])) == [[1], [2]]


def test_retry_after_parsing():
    assert retry_after_seconds(rate_limit_error("2")) == 2.0
    assert retry_after_seconds(rate_limit_error(None)) is None
    err = openai.RateLimitError("x", response=_response(429, {"retry-after-ms": "250"}), body=None)
    assert retry_after_seconds(err) == 0.25
    assert retry_after_seconds(ValueError()) is None


def test_backoff_is_exponential_and_capped():
    p = RetryPolicy(base_delay=1.0, max_delay=10.0)
    for attempt, cap in [(1, 1), (2, 2), (3, 4), (4, 8), (5, 10), (9, 10)]:
        for _ in range(20):
            assert cap / 2 <= backoff_delay(attempt, p) <= cap


def test_error_classification():
    assert pool._is_retryable(rate_limit_error())
    assert pool._is_retryable(server_error())
    assert pool._is_retryable(openai.APITimeoutError(request=_response(500).request))
    assert pool._is_retryable(openai.APIConnectionError(request=_response(500).request))
    assert not pool._is_retryable(bad_request_error())
    assert pool._is_fatal(auth_error())
    assert pool._is_fatal(openai.OpenAIError("The api_key client option must be set"))
    quota = rate_limit_error()
    quota.code = "insufficient_quota"
    assert pool._is_fatal(quota)
    assert not pool._is_fatal(rate_limit_error())
    assert not pool._is_fatal(bad_request_error())


async def test_missing_api_key_fails_fast(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    # Make sure we would notice if it tried to reach the DB.
    monkeypatch.setenv("DATABASE_URL", "postgresql://nobody@127.0.0.1:1/none")
    get_settings.cache_clear()
    try:
        with pytest.raises(pool.EmbedFatalError, match="OPENAI_API_KEY is not set"):
            await pool.run_embed()
    finally:
        monkeypatch.undo()
        get_settings.cache_clear()
