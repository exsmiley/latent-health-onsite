"""Short previews of chunk text shown in search hits."""

from rag.config import get_settings

ELLIPSIS = "…"


def make_blurb(text: str, words: int | None = None) -> str:
    """First `words` (default settings.blurb_words) words of `text`, whitespace-normalized.

    Appends "…" when the text was truncated.
    """
    n = get_settings().blurb_words if words is None else words
    parts = text.split()
    if len(parts) <= n:
        return " ".join(parts)
    return " ".join(parts[:n]) + ELLIPSIS


def normalize_queries(queries: list[str]) -> list[str]:
    """Strip whitespace, drop empty strings and exact duplicates, keep first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for q in queries:
        q = " ".join(q.split())
        if q and q not in seen:
            seen.add(q)
            out.append(q)
    return out
