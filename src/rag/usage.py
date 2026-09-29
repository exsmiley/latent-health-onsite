"""Optional, concurrency-safe token-usage accounting.

`track_usage()` installs a `Usage` accumulator in a ContextVar for the enclosing block; model
calls (`rag.agents.model.create` / `stream_text`) and embedding calls (`rag.llm.embed_texts`)
add to it via `record_response` / `record_embedding`. When no tracker is active these are no-ops.

Each asyncio task runs in a copy of its parent's context, so a tracker set inside a per-question
task only sees that task's calls (and those of tasks it spawns, e.g. concurrent tool calls).
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Usage:
    calls: int = 0  # chat/Responses API calls that reported usage
    input_tokens: int = 0  # includes cached input tokens
    cached_input_tokens: int = 0
    output_tokens: int = 0  # includes reasoning tokens
    reasoning_tokens: int = 0
    embedding_calls: int = 0
    embedding_tokens: int = 0

    def add(self, other: "Usage") -> None:
        for k, v in asdict(other).items():
            setattr(self, k, getattr(self, k) + v)

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


_current: ContextVar[Usage | None] = ContextVar("rag_usage", default=None)


@contextmanager
def track_usage() -> Iterator[Usage]:
    """`with track_usage() as usage:` sums the usage of every model call made in the block."""
    usage = Usage()
    token = _current.set(usage)
    try:
        yield usage
    finally:
        _current.reset(token)


def _int(obj: Any, name: str) -> int:
    value = getattr(obj, name, None) if obj is not None else None
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def record_response(usage_obj: Any) -> None:
    """Add a Responses API `ResponseUsage` (or None) to the active tracker, if any."""
    acc = _current.get()
    if acc is None or usage_obj is None:
        return
    acc.calls += 1
    acc.input_tokens += _int(usage_obj, "input_tokens")
    acc.output_tokens += _int(usage_obj, "output_tokens")
    acc.cached_input_tokens += _int(
        getattr(usage_obj, "input_tokens_details", None), "cached_tokens"
    )
    acc.reasoning_tokens += _int(
        getattr(usage_obj, "output_tokens_details", None), "reasoning_tokens"
    )


def record_embedding(usage_obj: Any) -> None:
    """Add an embeddings API usage object (or None) to the active tracker, if any."""
    acc = _current.get()
    if acc is None:
        return
    acc.embedding_calls += 1
    acc.embedding_tokens += _int(usage_obj, "prompt_tokens") or _int(usage_obj, "total_tokens")
