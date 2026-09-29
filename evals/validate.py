"""Validate the evaluation question set against the database.

Usage:
    uv run python evals/validate.py [path/to/questions.jsonl]

Checks, for every line:
  * it parses as JSON and has the required fields with the right types/values;
  * the `id` is unique;
  * every supporting chunk exists, with the recorded article_id, chunk_index and title
    (section is checked too and reported as a warning on mismatch);
  * every `evidence` string appears in that chunk's text after whitespace normalization.

Prints a summary by type and difficulty. Exits non-zero if any check fails.
Read-only: it only runs SELECTs.
"""

import asyncio
import json
import re
import sys
from collections import Counter
from pathlib import Path

from rag import db

DEFAULT_PATH = Path(__file__).resolve().parent / "questions.jsonl"

TYPES = {"multi_article", "multi_chunk", "comparison", "aggregation", "temporal", "single_hop"}
DIFFICULTIES = {"easy", "medium", "hard"}
REQUIRED = {
    "id": str,
    "question": str,
    "answer": str,
    "answer_aliases": list,
    "type": str,
    "hops": int,
    "difficulty": str,
    "supporting_chunks": list,
    "reasoning": str,
    "notes": str,
}
CHUNK_REQUIRED = {
    "chunk_id": int,
    "article_id": int,
    "title": str,
    "chunk_index": int,
    "evidence": str,
}

_SQL = """
SELECT c.id, c.article_id, c.chunk_index, c.section, c.text, a.title
FROM chunks c JOIN articles a ON a.id = c.article_id
WHERE c.id = ANY(%s)
"""


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def check_shape(q: dict, where: str, errors: list[str]) -> None:
    for key, typ in REQUIRED.items():
        if key not in q:
            errors.append(f"{where}: missing field {key!r}")
        elif not isinstance(q[key], typ) or (typ is int and isinstance(q[key], bool)):
            errors.append(f"{where}: field {key!r} should be {typ.__name__}")
    if q.get("type") not in TYPES:
        errors.append(f"{where}: bad type {q.get('type')!r}")
    if q.get("difficulty") not in DIFFICULTIES:
        errors.append(f"{where}: bad difficulty {q.get('difficulty')!r}")
    if not str(q.get("question", "")).strip() or not str(q.get("answer", "")).strip():
        errors.append(f"{where}: empty question or answer")
    if isinstance(q.get("hops"), int) and q["hops"] < 1:
        errors.append(f"{where}: hops must be >= 1")
    if not all(isinstance(a, str) for a in q.get("answer_aliases", [])):
        errors.append(f"{where}: answer_aliases must be strings")
    sc = q.get("supporting_chunks")
    if isinstance(sc, list):
        if not sc:
            errors.append(f"{where}: no supporting_chunks")
        for j, c in enumerate(sc):
            if not isinstance(c, dict):
                errors.append(f"{where}: supporting_chunks[{j}] is not an object")
                continue
            for key, typ in CHUNK_REQUIRED.items():
                if not isinstance(c.get(key), typ):
                    errors.append(f"{where}: supporting_chunks[{j}].{key} missing or not {typ.__name__}")
            if "section" not in c or not (c["section"] is None or isinstance(c["section"], str)):
                errors.append(f"{where}: supporting_chunks[{j}].section must be str or null")
            if isinstance(c.get("evidence"), str) and not c["evidence"].strip():
                errors.append(f"{where}: supporting_chunks[{j}].evidence is empty")


async def main(path: Path) -> int:
    errors: list[str] = []
    warnings: list[str] = []
    questions: list[dict] = []

    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            q = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"line {lineno}: invalid JSON ({e})")
            continue
        if not isinstance(q, dict):
            errors.append(f"line {lineno}: not a JSON object")
            continue
        check_shape(q, f"line {lineno} ({q.get('id')})", errors)
        questions.append(q)

    ids = Counter(q.get("id") for q in questions)
    errors += [f"duplicate id {i!r} ({n}x)" for i, n in ids.items() if n > 1]
    texts = Counter(norm(str(q.get("question", ""))).lower() for q in questions)
    errors += [f"duplicate question text: {t[:80]!r}" for t, n in texts.items() if n > 1]

    wanted = sorted(
        {
            c["chunk_id"]
            for q in questions
            for c in q.get("supporting_chunks", [])
            if isinstance(c, dict) and isinstance(c.get("chunk_id"), int)
        }
    )
    try:
        async with db.connection() as conn:
            rows = await (await conn.execute(_SQL, (wanted,))).fetchall()
    finally:
        await db.close_pool()
    by_id = {r["id"]: r for r in rows}

    n_chunks = 0
    for q in questions:
        qid = q.get("id")
        for j, c in enumerate(q.get("supporting_chunks", [])):
            if not isinstance(c, dict) or not isinstance(c.get("chunk_id"), int):
                continue
            n_chunks += 1
            where = f"{qid} supporting_chunks[{j}] (chunk {c['chunk_id']})"
            row = by_id.get(c["chunk_id"])
            if row is None:
                errors.append(f"{where}: chunk does not exist")
                continue
            if row["article_id"] != c.get("article_id"):
                errors.append(f"{where}: article_id {c.get('article_id')} != DB {row['article_id']}")
            if row["chunk_index"] != c.get("chunk_index"):
                errors.append(f"{where}: chunk_index {c.get('chunk_index')} != DB {row['chunk_index']}")
            if row["title"] != c.get("title"):
                errors.append(f"{where}: title {c.get('title')!r} != DB {row['title']!r}")
            if row["section"] != c.get("section"):
                warnings.append(f"{where}: section {c.get('section')!r} != DB {row['section']!r}")
            ev = c.get("evidence")
            if isinstance(ev, str) and norm(ev) not in norm(row["text"]):
                errors.append(f"{where}: evidence not found in chunk text: {ev[:100]!r}")

    # Summary
    print(f"{path}: {len(questions)} questions, {n_chunks} supporting chunks "
          f"({len(wanted)} distinct), {sum(len({c.get('article_id') for c in q.get('supporting_chunks', []) if isinstance(c, dict)}) > 1 for q in questions)} span 2+ articles")
    by_type = Counter(q.get("type") for q in questions)
    by_diff = Counter(q.get("difficulty") for q in questions)
    total = max(len(questions), 1)
    print("\nBy type:")
    for t in sorted(TYPES, key=lambda t: -by_type[t]):
        print(f"  {t:<14} {by_type[t]:>3}  ({100 * by_type[t] / total:4.1f}%)")
    print("\nBy difficulty:")
    for d in ("easy", "medium", "hard"):
        print(f"  {d:<14} {by_diff[d]:>3}  ({100 * by_diff[d] / total:4.1f}%)")
    hops = Counter(q.get("hops") for q in questions)
    print("\nBy hops: " + ", ".join(f"{h}: {hops[h]}" for h in sorted(hops, key=lambda x: (not isinstance(x, int), x))))
    print("\nType x difficulty:")
    for t in sorted(TYPES):
        row = Counter(q.get("difficulty") for q in questions if q.get("type") == t)
        print(f"  {t:<14} " + "  ".join(f"{d}={row[d]}" for d in ("easy", "medium", "hard")))

    for w in warnings:
        print(f"WARNING: {w}")
    if errors:
        print(f"\nFAILED: {len(errors)} error(s)")
        for e in errors:
            print(f"  ERROR: {e}")
        return 1
    print("\nOK: all checks passed")
    return 0


if __name__ == "__main__":
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH
    sys.exit(asyncio.run(main(p)))
