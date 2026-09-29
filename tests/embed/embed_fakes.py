"""Fakes and helpers for the embed pool tests (imported by conftest and tests)."""

from __future__ import annotations

import asyncio
import hashlib
import os
import random
import time

import httpx2
import openai
import psycopg

ADMIN_URL = os.environ.get("RAG_TEST_ADMIN_URL", "postgresql://rag:rag@localhost:5433/postgres")
TEST_DB = "rag_test_embed"
TEST_URL = ADMIN_URL.rsplit("/", 1)[0] + "/" + TEST_DB
DIM = 1536


def fake_vector(text: str) -> list[float]:
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
    rng = random.Random(seed)
    return [rng.uniform(-1, 1) for _ in range(DIM)]


def _response(status: int, headers: dict[str, str] | None = None) -> httpx2.Response:
    request = httpx2.Request("POST", "https://api.openai.com/v1/embeddings")
    return httpx2.Response(status, request=request, headers=headers or {})


def rate_limit_error(retry_after: str | None = "0.05") -> openai.RateLimitError:
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    return openai.RateLimitError("rate limited", response=_response(429, headers), body=None)


def bad_request_error() -> openai.BadRequestError:
    return openai.BadRequestError("invalid input", response=_response(400), body=None)


def server_error() -> openai.InternalServerError:
    return openai.InternalServerError("boom", response=_response(503), body=None)


def auth_error() -> openai.AuthenticationError:
    return openai.AuthenticationError("bad key", response=_response(401), body=None)


class FakeEmbed:
    """Deterministic stand-in for `rag.llm.embed_texts`.

    * raises a 429 on the call numbers in `rate_limit_calls` (1-based),
    * raises a 400 whenever the batch contains a text including `bad_marker`,
    * or raises `always` on every call if set.
    """

    def __init__(
        self,
        rate_limit_calls: set[int] = frozenset(),
        bad_marker: str = "POISON",
        always: Exception | None = None,
        retry_after: str | None = "0.05",
    ):
        self.rate_limit_calls = set(rate_limit_calls)
        self.bad_marker = bad_marker
        self.always = always
        self.retry_after = retry_after
        self.calls = 0
        self.batch_sizes: list[int] = []
        self.call_starts: list[float] = []
        self.rate_limit_times: list[float] = []
        self.bad_request_calls = 0

    async def __call__(self, texts: list[str], **_kwargs) -> list[list[float]]:
        self.calls += 1
        now = time.monotonic()
        self.call_starts.append(now)
        self.batch_sizes.append(len(texts))
        if self.always is not None:
            raise self.always
        if self.calls in self.rate_limit_calls:
            self.rate_limit_times.append(now)
            raise rate_limit_error(self.retry_after)
        if any(self.bad_marker in t for t in texts):
            self.bad_request_calls += 1
            raise bad_request_error()
        await asyncio.sleep(0.002)  # yield so workers interleave
        return [fake_vector(t) for t in texts]


def seed(url: str, n_articles: int = 5, chunks_per_article: int = 12, bad: tuple = ()) -> list[int]:
    """Replace all data with synthetic articles/chunks. `bad` = (article_idx, chunk_idx) pairs
    whose text contains the poison marker. Returns the chunk ids in order."""
    with psycopg.connect(url) as conn:
        conn.execute("TRUNCATE articles, chunks RESTART IDENTITY CASCADE")
        for a in range(n_articles):
            title = f"Article {a}"
            conn.execute(
                "INSERT INTO articles (id, title, url, text) VALUES (%s, %s, %s, %s)",
                (1000 + a, title, f"https://example.org/{a}", "..."),
            )
            for c in range(chunks_per_article):
                section = None if c == 0 else f"Section {c // 4}"
                text = f"Paragraph {c} of {title}. " * 5
                if (a, c) in bad:
                    text += "POISON"
                prefix = f"{title} > {section}" if section else title
                conn.execute(
                    "INSERT INTO chunks (article_id, chunk_index, section, text, embed_text, "
                    "token_count) VALUES (%s, %s, %s, %s, %s, %s)",
                    (1000 + a, c, section, text, f"{prefix}\n\n{text}", 40),
                )
        return [r[0] for r in conn.execute("SELECT id FROM chunks ORDER BY id")]


def embedded_state(url: str) -> dict[int, str | None]:
    with psycopg.connect(url) as conn:
        rows = conn.execute("SELECT id, embedding::text FROM chunks ORDER BY id").fetchall()
    return dict(rows)
