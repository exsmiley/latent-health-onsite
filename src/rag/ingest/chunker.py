"""Paragraph-aligned chunking of Simple English Wikipedia articles.

See docs/ARCHITECTURE.md ("Chunking rules"). Summary of the choices made here:

* Paragraphs are separated by one or more blank (or whitespace-only) lines. A paragraph is
  never split, except one over ``MAX_PARAGRAPH_TOKENS`` (the embedding input limit), which is split on sentence
  boundaries into pieces of at most ``chunk_target_tokens``.
* Heading detection is a heuristic, because the dataset is plain text. A heading is a line that
  (see :func:`is_heading`) does not start with whitespace (that marks a list item), is short
  (at most 8 words / 80 chars, or 12 words / 100 chars when it has the trailing space the
  dump leaves after headings), does not start with a lowercase letter, and does not end with
  sentence or clause punctuation (``. ! ? : ; ,``, ignoring closing quotes/brackets).
* In the real data a heading is either its own paragraph (``"The Month \\n\\n..."``) or the
  first line of a paragraph, followed by its body after a single newline
  (``"April in poetry \\nPoets use April..."``). Both are handled: in the second case the
  heading line is peeled off and the rest of the paragraph is the body (see :func:`_peelable`
  for the guard against bare lists such as the trailing category block). Only the first line
  of a paragraph is ever considered, so short lines further down are never taken for headings.
* Consecutive headings with no content between them (section then subsection) are joined as a
  path of at most two levels, e.g. ``"Events in April > Fixed Events"``. Known-empty sections
  (References, Notes, ...) are replaced instead of joined, so ``References`` followed by
  ``Other websites`` gives ``"Other websites"``. A heading that follows content replaces the
  section entirely (heading levels are not recoverable from the plain text).
* An article made only of heading-like lines (a stub without a final period) is kept as a
  single lead chunk of those lines instead of being dropped.
* Whitespace-only paragraphs are dropped; paragraph text is right-stripped but keeps leading
  spaces (they mark list items). Chunk ``text`` is its paragraphs joined with ``\\n\\n``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import tiktoken

from rag.config import get_settings

MAX_PARAGRAPH_TOKENS = 8000  # text-embedding-3-small input limit is 8191
PARAGRAPH_SEP = "\n\n"

_PARA_SPLIT = re.compile(r"\n(?:[^\S\n]*\n)+")  # one or more blank lines; keeps leading spaces
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_TERMINAL_PUNCT = ".!?:;,"
# Headings of sections whose content the dump dropped (reference lists), so they are almost
# always immediately followed by a sibling heading. They are replaced rather than joined.
_EMPTY_SECTIONS = {"references", "notes", "footnotes", "sources", "citations"}
_CLOSERS = "\"')]}”’»"


@dataclass(frozen=True, slots=True)
class ChunkDraft:
    chunk_index: int
    section: str | None
    text: str
    embed_text: str
    token_count: int


@lru_cache(maxsize=1)
def _encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoder().encode_ordinary(text))


def is_heading(line: str) -> bool:
    """Heuristic: is this single line a section heading?"""
    if not line or line[0].isspace():
        return False  # empty, or a list item
    s = line.strip()
    if not s or "\n" in s:
        return False
    trailing_space = line != line.rstrip()
    max_words, max_chars = (12, 100) if trailing_space else (8, 80)
    if len(s) > max_chars or len(s.split()) > max_words:
        return False
    if s[0].islower() or s[0] in "|!{}=" or not any(ch.isalnum() for ch in s):
        return False  # sentence fragment, wiki-table debris ("|-"), or pure punctuation
    core = s.rstrip(_CLOSERS)
    return bool(core) and core[-1] not in _TERMINAL_PUNCT


def _peelable(first: str, rest: str) -> bool:
    """Is the heading-like first line of a multi-line paragraph really a heading?

    A trailing space (how the dump renders headings) is decisive. Without one, the line only
    counts as a heading if the body is not itself a run of heading-like short lines: that shape
    is a bare list, most often the category block at the end of an article
    ("1930 births\nLiving people\nAmerican movie actors").
    """
    if first != first.rstrip():
        return True
    return not all(is_heading(line) for line in rest.split("\n") if line.strip())


def split_blocks(text: str) -> list[tuple[str, str]]:
    """Split article text into ("heading" | "para", text) blocks, in order."""
    blocks: list[tuple[str, str]] = []
    for raw in _PARA_SPLIT.split(text):
        raw = raw.strip("\n")
        if not raw.strip():
            continue
        first, nl, rest = raw.partition("\n")
        if is_heading(first) and (not nl or _peelable(first, rest)):
            blocks.append(("heading", first.strip()))
            if rest.strip():
                blocks.append(("para", rest.strip("\n").rstrip()))
        else:
            blocks.append(("para", raw.rstrip()))
    return blocks


def _split_oversize(para: str, limit: int) -> list[str]:
    """Split a huge paragraph on sentence boundaries into pieces of <= limit tokens."""
    enc = _encoder()
    pieces: list[str] = []
    cur: list[str] = []
    cur_tokens = 0
    for sent in _SENTENCE_SPLIT.split(para):
        n = len(enc.encode_ordinary(sent))
        if n > limit:  # a "sentence" with no boundaries at all: hard split on tokens
            if cur:
                pieces.append(" ".join(cur))
                cur, cur_tokens = [], 0
            toks = enc.encode_ordinary(sent)
            pieces.extend(enc.decode(toks[i : i + limit]) for i in range(0, len(toks), limit))
            continue
        if cur and cur_tokens + 1 + n > limit:
            pieces.append(" ".join(cur))
            cur, cur_tokens = [], 0
        cur.append(sent)
        cur_tokens += n + (1 if cur_tokens else 0)
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def _nest(parent: str | None, heading: str) -> str:
    """Join consecutive headings as a path, keeping at most the last two levels."""
    if not parent:
        return heading
    last = parent.rsplit(" > ", 1)[-1]
    if last.lower() in _EMPTY_SECTIONS:
        return heading
    return f"{last} > {heading}"


def make_embed_text(title: str, section: str | None, text: str) -> str:
    return f"{title} > {section}\n\n{text}" if section else f"{title}\n\n{text}"


def chunk_article(
    title: str,
    text: str,
    *,
    target_tokens: int | None = None,
    min_tokens: int | None = None,
) -> list[ChunkDraft]:
    """Chunk one article into whole-paragraph chunks. Pure apart from reading settings."""
    if not text or not text.strip():
        return []
    if target_tokens is None or min_tokens is None:
        s = get_settings()
        target_tokens = s.chunk_target_tokens if target_tokens is None else target_tokens
        min_tokens = s.chunk_min_tokens if min_tokens is None else min_tokens

    enc = _encoder()
    groups: list[tuple[str | None, list[str]]] = []  # (section, paragraphs)
    cur: list[str] = []
    cur_tokens = 0
    cur_section: str | None = None  # section of the chunk being built
    section: str | None = None  # heading currently in force
    prev_was_heading = False

    def flush() -> None:
        nonlocal cur, cur_tokens
        if cur:
            groups.append((cur_section, cur))
        cur, cur_tokens = [], 0

    blocks = split_blocks(text)
    if not any(kind == "para" for kind, _ in blocks):
        # Nothing but heading-like lines, e.g. a stub with no final period ("Flagstaff is a
        # railway station in Melbourne, Australia"). Keep the lines as body text rather than
        # dropping the article.
        blocks = [("para", b) for _, b in blocks]

    for kind, block in blocks:
        if kind == "heading":
            section = _nest(section, block) if prev_was_heading else block
            prev_was_heading = True
            if cur_tokens >= min_tokens:
                flush()
            continue
        prev_was_heading = False

        n = len(enc.encode_ordinary(block))
        if n > target_tokens:
            flush()
            parts = _split_oversize(block, target_tokens) if n > MAX_PARAGRAPH_TOKENS else [block]
            for part in parts:
                groups.append((section, [part]))
            continue
        if cur and cur_tokens + 1 + n > target_tokens:  # +1 for the "\n\n" separator
            flush()
        if not cur:
            cur_section = section
            cur_tokens = n
        else:
            cur_tokens += 1 + n
        cur.append(block)
    flush()

    chunks: list[ChunkDraft] = []
    for i, (sec, paras) in enumerate(groups):
        body = PARAGRAPH_SEP.join(paras)
        chunks.append(
            ChunkDraft(
                chunk_index=i,
                section=sec,
                text=body,
                embed_text=make_embed_text(title, sec, body),
                token_count=len(enc.encode_ordinary(body)),
            )
        )
    return chunks
