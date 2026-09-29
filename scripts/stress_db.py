"""DB-layer stress test: the retrieval tools alone, no ASGI server and no chat-model calls.

It fires concurrent `semantic_search` / `keyword_search` / `fetch` calls straight at
`rag.tools` through the production pool (`rag.db`, max_size=20) to find the DB and pool ceiling,
which separates "OpenAI-bound" from "our infrastructure-bound".

Query vectors are embedded once up front in ONE embeddings request, and `rag.llm.embed_texts`
is then replaced (in this process only) by a lookup, so the measured calls make no OpenAI calls.

Each level is a closed loop of N workers for `--duration` seconds. Every worker cycles through
the op mix the research agent uses per turn: semantic_search (3 queries -> 3 pooled connections
at once), keyword_search (3 queries), fetch of 5 chunks and fetch of 1 whole article. Pool stats
(`psycopg_pool` get_stats/pop_stats) are sampled every 50 ms to see when requests start queueing
for a connection.

Usage:
    uv run python scripts/stress_db.py --levels 1,2,4,8,16,32,64 --duration 5
    uv run python scripts/stress_db.py --pool-size 40   # compare a bigger pool (script-only)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from psycopg_pool import AsyncConnectionPool

from rag import db, llm
from rag.config import get_settings
from rag.tools.fetch import fetch
from rag.tools.keyword import keyword_search
from rag.tools.search import semantic_search

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stress_test import load_questions, percentile

QUERIES_PER_SEARCH = 3
OPS = ("semantic", "keyword", "fetch_chunks", "fetch_article")
SQL_PER_OP = {"semantic": QUERIES_PER_SEARCH, "keyword": QUERIES_PER_SEARCH}  # others: 1


def install_cached_embedder(table: dict[str, list[float]]) -> None:
    async def cached(texts: list[str], *, max_retries: int | None = None) -> list[list[float]]:
        return [table[t] for t in texts]

    llm.embed_texts = cached  # rag.tools.search calls llm.embed_texts at call time


async def prepare(n_queries: int) -> tuple[list[str], list[int], list[int]]:
    qs = [q["question"].strip() for q in load_questions()][:n_queries]
    vectors = await llm.embed_texts(qs)  # the one real OpenAI call
    install_cached_embedder(dict(zip(qs, vectors)))
    chunk_ids: list[int] = []
    article_ids: list[int] = []
    for group in await semantic_search(qs[:20]):
        for h in group.hits:
            chunk_ids.append(h.chunk_id)
            article_ids.append(h.article_id)
    return qs, list(dict.fromkeys(chunk_ids)), list(dict.fromkeys(article_ids))


async def run_level(
    level: int, duration: float, qs: list[str], cids: list[int], aids: list[int]
) -> dict:
    pool = await db.get_pool()
    pool.pop_stats()
    lat: dict[str, list[float]] = defaultdict(list)
    errors: dict[str, int] = defaultdict(int)
    peak = {"pool_size": 0, "requests_waiting": 0, "in_use": 0}
    stop_at = time.perf_counter() + duration
    counter = 0

    async def one(op: str, i: int) -> None:
        if op == "semantic":
            await semantic_search([qs[(i + k) % len(qs)] for k in range(QUERIES_PER_SEARCH)])
        elif op == "keyword":
            await keyword_search([qs[(i + k) % len(qs)] for k in range(QUERIES_PER_SEARCH)])
        elif op == "fetch_chunks":
            await fetch(chunk_ids=[cids[(i * 5 + k) % len(cids)] for k in range(5)])
        else:
            await fetch(article_ids=[aids[i % len(aids)]])

    async def worker(w: int) -> None:
        nonlocal counter
        j = w
        while time.perf_counter() < stop_at:
            op = OPS[j % len(OPS)]
            counter += 1
            t0 = time.perf_counter()
            try:
                await one(op, counter)
                lat[op].append(time.perf_counter() - t0)
            except Exception as e:  # noqa: BLE001 - count and keep going
                errors[f"{op}:{type(e).__name__}"] += 1
            j += 1

    async def sampler() -> None:
        while True:
            s = pool.get_stats()
            size, avail = s.get("pool_size", 0), s.get("pool_available", 0)
            peak["pool_size"] = max(peak["pool_size"], size)
            peak["in_use"] = max(peak["in_use"], size - avail)
            peak["requests_waiting"] = max(peak["requests_waiting"], s.get("requests_waiting", 0))
            await asyncio.sleep(0.05)

    samp = asyncio.create_task(sampler())
    t0 = time.perf_counter()
    await asyncio.gather(*(worker(w) for w in range(level)))
    wall = time.perf_counter() - t0
    samp.cancel()
    stats = pool.pop_stats()

    all_lat = [x for v in lat.values() for x in v]
    n_ops = len(all_lat)
    n_sql = sum(len(v) * SQL_PER_OP.get(op, 1) for op, v in lat.items())
    req = stats.get("requests_num", 0)

    def ms(p: float, xs: list[float]) -> float | None:
        v = percentile(xs, p)
        return None if v is None else round(v * 1000, 1)

    return {
        "level": level,
        "wall_s": round(wall, 2),
        "ops": n_ops,
        "ops_per_s": round(n_ops / wall, 1),
        "sql_per_s": round(n_sql / wall, 1),
        "latency_ms": {"p50": ms(50, all_lat), "p90": ms(90, all_lat), "p99": ms(99, all_lat)},
        "by_op": {
            op: {
                "n": len(v),
                "p50_ms": ms(50, v),
                "p90_ms": ms(90, v),
                "p99_ms": ms(99, v),
            }
            for op, v in sorted(lat.items())
        },
        "errors": dict(errors),
        "pool": {
            "max_size": pool.max_size,
            "peak_size": peak["pool_size"],
            "peak_in_use": peak["in_use"],
            "peak_waiting": peak["requests_waiting"],
            "conn_requests": req,
            "queued": stats.get("requests_queued", 0),
            "avg_wait_ms": round(stats.get("requests_wait_ms", 0) / req, 2) if req else None,
            "saturated": peak["requests_waiting"] > 0 or stats.get("requests_queued", 0) > 0,
        },
    }


def format_table(rows: list[dict]) -> str:
    head = (
        f"{'N':>4} {'ops/s':>7} {'sql/s':>7} | {'p50ms':>7} {'p90ms':>7} {'p99ms':>7} | "
        f"{'sem50':>6} {'kw50':>6} {'fch50':>6} {'art50':>6} | "
        f"{'poolPk':>6} {'inUse':>5} {'waitPk':>6} {'queued':>6} {'wait_ms':>7} {'err':>4}"
    )
    out = [head, "-" * len(head)]
    for r in rows:
        b, p = r["by_op"], r["pool"]

        def g(op: str, b: dict = b) -> str:
            v = b.get(op, {}).get("p50_ms")
            return "-" if v is None else f"{v:.0f}"

        out.append(
            f"{r['level']:>4} {r['ops_per_s']:>7.1f} {r['sql_per_s']:>7.1f} | "
            f"{r['latency_ms']['p50']:>7.1f} {r['latency_ms']['p90']:>7.1f} "
            f"{r['latency_ms']['p99']:>7.1f} | "
            f"{g('semantic'):>6} {g('keyword'):>6} {g('fetch_chunks'):>6} {g('fetch_article'):>6} | "
            f"{p['peak_size']:>6} {p['peak_in_use']:>5} {p['peak_waiting']:>6} {p['queued']:>6} "
            f"{p['avg_wait_ms'] if p['avg_wait_ms'] is not None else '-':>7} "
            f"{sum(r['errors'].values()):>4}"
        )
    return "\n".join(out)


async def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--levels", default="1,2,4,8,16,32,64")
    p.add_argument("--duration", type=float, default=5.0, help="seconds per level")
    p.add_argument("--queries", type=int, default=60, help="distinct query texts to embed")
    p.add_argument(
        "--pool-size", type=int, default=None, help="override max_size (default: production 20)"
    )
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)
    levels = [int(x) for x in a.levels.split(",") if x.strip()]

    if a.pool_size:
        # Script-only: pre-create the module pool with another max_size; rag.db is unchanged.
        db._pool = AsyncConnectionPool(
            get_settings().database_url,
            min_size=1,
            max_size=a.pool_size,
            configure=db._configure,
            open=False,
        )
        await db._pool.open()
    try:
        qs, cids, aids = await prepare(a.queries)
        print(f"{len(qs)} queries, {len(cids)} chunk ids, {len(aids)} article ids", file=sys.stderr)
        rows = []
        for lv in levels:
            rows.append(await run_level(lv, a.duration, qs, cids, aids))
            print(format_table(rows[-1:]).splitlines()[-1], file=sys.stderr, flush=True)
    finally:
        await db.close_pool()
    print(format_table(rows))
    result = {"pool_size": a.pool_size or 20, "duration_s": a.duration, "levels": rows}
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(result, indent=1))
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
