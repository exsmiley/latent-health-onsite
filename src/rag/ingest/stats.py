"""Dry run: chunk the whole dataset without touching the DB and report chunk/token stats.

uv run python -m rag.ingest.stats [--row-groups N]
"""

from __future__ import annotations

import argparse
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import pyarrow.parquet as pq

from rag.config import get_settings
from rag.ingest.chunker import chunk_article, count_tokens

EMBED_USD_PER_M_TOKENS = 0.02


def _chunk_row_group(args: tuple[str, int]) -> tuple[list[int], list[int], int, int, Counter]:
    path, rg = args
    table = pq.ParquetFile(path).read_row_group(rg, columns=["title", "text"])
    body_tokens: list[int] = []
    embed_tokens: list[int] = []
    articles = empty = 0
    sections: Counter = Counter()

    for title, text in zip(table.column("title").to_pylist(), table.column("text").to_pylist()):
        articles += 1
        chunks = chunk_article(title, text)
        if not chunks:
            empty += 1
        for c in chunks:
            body_tokens.append(c.token_count)
            embed_tokens.append(count_tokens(c.embed_text))
            sections[c.section is not None] += 1
    return body_tokens, embed_tokens, articles, empty, sections


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--row-groups", type=int, default=None, help="only the first N row groups")
    args = ap.parse_args()

    path = str(get_settings().dataset_path)
    n_rg = pq.ParquetFile(path).num_row_groups
    if args.row_groups:
        n_rg = min(n_rg, args.row_groups)
    t0 = time.monotonic()
    body: list[int] = []
    embed: list[int] = []
    articles = empty = 0
    sections: Counter = Counter()
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        for b, e, a, z, s in ex.map(_chunk_row_group, [(path, i) for i in range(n_rg)]):
            body += b
            embed += e
            articles += a
            empty += z
            sections += s
    elapsed = time.monotonic() - t0

    body.sort()
    embed.sort()

    def pct(xs: list[int], q: float) -> int:
        return xs[min(len(xs) - 1, int(q / 100 * len(xs)))]

    print(f"articles: {articles:,} ({empty:,} produced no chunks)  chunked in {elapsed:.1f}s")
    print(f"chunks: {len(body):,}  (with section: {sections[True]:,}, lead: {sections[False]:,})")
    print(f"chunks per article: {len(body) / max(articles, 1):.2f}")
    for name, xs in (("text", body), ("embed_text", embed)):
        print(
            f"{name} tokens/chunk: p10={pct(xs, 10)} p50={pct(xs, 50)} p90={pct(xs, 90)} "
            f"p99={pct(xs, 99)} max={xs[-1]} mean={sum(xs) / len(xs):.0f}  total={sum(xs):,}"
        )
    for lim in (100, 350, 1000, 8000):
        n = sum(1 for x in body if x > lim) if lim > 100 else sum(1 for x in body if x < lim)
        print(f"chunks {'<' if lim == 100 else '>'} {lim} tokens: {n:,}")
    cost = sum(embed) / 1e6 * EMBED_USD_PER_M_TOKENS
    print(f"estimated embedding cost (embed_text @ ${EMBED_USD_PER_M_TOKENS}/1M): ${cost:.2f}")


if __name__ == "__main__":
    main()
