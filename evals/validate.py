"""Validate the evaluation question sets against the database.

Usage:
    uv run python evals/validate.py [path.jsonl ...]

With no arguments, validates evals/questions.jsonl, evals/questions_super_hard.jsonl and
evals/questions_premise.jsonl.

Checks, for every line:
  * it parses as JSON and has the required fields with the right types/values;
  * the `id` is unique (across all files validated together);
  * super-hard questions (`tier: "super_hard"`) have the extra fields, with
    1 <= sequential_depth <= hops and breadth == number of distinct supporting article_ids;
  * premise questions (`tier: "premise"`) have a premise `type`, the matching
    `expected_behavior` and a non-empty `premise`; only `unanswerable` questions may have empty
    `supporting_chunks`;
  * every supporting chunk exists, with the recorded article_id, chunk_index and title
    (section is checked too and reported as a warning on mismatch);
  * every `evidence` string appears in that chunk's text after whitespace normalization.

Prints a summary by type and difficulty per file, plus depth/breadth distributions for the
super-hard tier. Exits non-zero if any check fails.
Read-only: it only runs SELECTs.
"""

import asyncio
import json
import re
import sys
from collections import Counter
from pathlib import Path

from rag import db

EVALS_DIR = Path(__file__).resolve().parent
DEFAULT_PATHS = [
    EVALS_DIR / "questions.jsonl",
    EVALS_DIR / "questions_super_hard.jsonl",
    EVALS_DIR / "questions_premise.jsonl",
]

TYPES = {"multi_article", "multi_chunk", "comparison", "aggregation", "temporal", "single_hop"}
DIFFICULTIES = {"easy", "medium", "hard"}
SUPER_TYPES = {"deep_chain", "wide", "deep_wide"}
# premise tier: type -> the expected_behavior it must carry
PREMISE_BEHAVIORS = {
    "false_premise": "correct_premise",
    "ambiguous": "disambiguate",
    "unanswerable": "not_found",
}
SUPER_REQUIRED = {"tier": str, "sequential_depth": int, "breadth": int, "min_turns_estimate": int}
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


def is_super(q: dict) -> bool:
    return q.get("tier") == "super_hard"


def is_premise(q: dict) -> bool:
    return q.get("tier") == "premise"


def check_premise(q: dict, where: str, errors: list[str]) -> None:
    t = q.get("type")
    if t not in PREMISE_BEHAVIORS:
        errors.append(f"{where}: bad premise type {t!r}")
    elif q.get("expected_behavior") != PREMISE_BEHAVIORS[t]:
        errors.append(
            f"{where}: type {t!r} needs expected_behavior {PREMISE_BEHAVIORS[t]!r}, "
            f"got {q.get('expected_behavior')!r}"
        )
    if not isinstance(q.get("premise"), str) or not q["premise"].strip():
        errors.append(f"{where}: premise field missing or empty")
    if q.get("difficulty") not in DIFFICULTIES:
        errors.append(f"{where}: bad difficulty {q.get('difficulty')!r}")
    if not str(q.get("notes", "")).strip():
        errors.append(f"{where}: premise questions need notes (what good vs bad looks like)")


def check_shape(q: dict, where: str, errors: list[str]) -> None:
    for key, typ in REQUIRED.items():
        if key not in q:
            errors.append(f"{where}: missing field {key!r}")
        elif not isinstance(q[key], typ) or (typ is int and isinstance(q[key], bool)):
            errors.append(f"{where}: field {key!r} should be {typ.__name__}")
    if is_super(q):
        for key, typ in SUPER_REQUIRED.items():
            if not isinstance(q.get(key), typ) or isinstance(q.get(key), bool):
                errors.append(f"{where}: super_hard field {key!r} missing or not {typ.__name__}")
        if q.get("type") not in SUPER_TYPES:
            errors.append(f"{where}: bad super_hard type {q.get('type')!r}")
        if q.get("difficulty") != "super_hard":
            errors.append(f"{where}: super_hard tier needs difficulty 'super_hard'")
        d, h, t = q.get("sequential_depth"), q.get("hops"), q.get("min_turns_estimate")
        if isinstance(d, int) and isinstance(h, int) and not 1 <= d <= h:
            errors.append(f"{where}: need 1 <= sequential_depth ({d}) <= hops ({h})")
        if isinstance(d, int) and isinstance(t, int) and t < d:
            errors.append(f"{where}: min_turns_estimate ({t}) < sequential_depth ({d})")
        sc = q.get("supporting_chunks")
        if isinstance(sc, list) and isinstance(q.get("breadth"), int):
            arts = {c.get("article_id") for c in sc if isinstance(c, dict)}
            if q["breadth"] != len(arts):
                errors.append(
                    f"{where}: breadth {q['breadth']} != {len(arts)} distinct supporting articles"
                )
    elif is_premise(q):
        check_premise(q, where, errors)
    else:
        if "tier" in q:
            errors.append(f"{where}: unknown tier {q.get('tier')!r}")
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
        if not sc and not (is_premise(q) and q.get("type") == "unanswerable"):
            errors.append(f"{where}: no supporting_chunks")
        for j, c in enumerate(sc):
            if not isinstance(c, dict):
                errors.append(f"{where}: supporting_chunks[{j}] is not an object")
                continue
            for key, typ in CHUNK_REQUIRED.items():
                if not isinstance(c.get(key), typ):
                    errors.append(
                        f"{where}: supporting_chunks[{j}].{key} missing or not {typ.__name__}"
                    )
            if "section" not in c or not (c["section"] is None or isinstance(c["section"], str)):
                errors.append(f"{where}: supporting_chunks[{j}].section must be str or null")
            if isinstance(c.get("evidence"), str) and not c["evidence"].strip():
                errors.append(f"{where}: supporting_chunks[{j}].evidence is empty")


def load(path: Path, errors: list[str]) -> list[dict]:
    questions: list[dict] = []
    if not path.exists():
        errors.append(f"{path}: file not found")
        return questions
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        where = f"{path.name}:{lineno}"
        try:
            q = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"{where}: invalid JSON ({e})")
            continue
        if not isinstance(q, dict):
            errors.append(f"{where}: not a JSON object")
            continue
        check_shape(q, f"{where} ({q.get('id')})", errors)
        questions.append(q)
    return questions


def dist(values) -> str:
    c = Counter(values)
    return ", ".join(f"{k}: {c[k]}" for k in sorted(c, key=lambda x: (not isinstance(x, int), x)))


def n_articles(q: dict) -> int:
    return len({c.get("article_id") for c in q.get("supporting_chunks", []) if isinstance(c, dict)})


def summarize(path: Path, questions: list[dict]) -> None:
    n_chunks = sum(len(q.get("supporting_chunks", [])) for q in questions)
    distinct = {
        c.get("chunk_id")
        for q in questions
        for c in q.get("supporting_chunks", [])
        if isinstance(c, dict)
    }
    print(
        f"\n=== {path.name}: {len(questions)} questions, {n_chunks} supporting chunks "
        f"({len(distinct)} distinct), {sum(n_articles(q) > 1 for q in questions)} span 2+ articles"
    )
    total = max(len(questions), 1)
    main_qs = [q for q in questions if not is_super(q) and not is_premise(q)]
    premise_qs = [q for q in questions if is_premise(q)]
    super_qs = [q for q in questions if is_super(q)]
    if main_qs:
        by_type = Counter(q.get("type") for q in main_qs)
        by_diff = Counter(q.get("difficulty") for q in main_qs)
        print("By type:")
        for t in sorted(TYPES, key=lambda t: -by_type[t]):
            print(f"  {t:<14} {by_type[t]:>3}  ({100 * by_type[t] / total:4.1f}%)")
        print("By difficulty:")
        for d in ("easy", "medium", "hard"):
            print(f"  {d:<14} {by_diff[d]:>3}  ({100 * by_diff[d] / total:4.1f}%)")
        print("By hops: " + dist(q.get("hops") for q in main_qs))
        print("Type x difficulty:")
        for t in sorted(TYPES):
            row = Counter(q.get("difficulty") for q in main_qs if q.get("type") == t)
            print(f"  {t:<14} " + "  ".join(f"{d}={row[d]}" for d in ("easy", "medium", "hard")))
    if super_qs:
        print("Super-hard tier, by type:")
        by_type = Counter(q.get("type") for q in super_qs)
        for t in ("deep_chain", "wide", "deep_wide"):
            print(f"  {t:<14} {by_type[t]:>3}")
        print("  hops:               " + dist(q.get("hops") for q in super_qs))
        print("  sequential_depth:   " + dist(q.get("sequential_depth") for q in super_qs))
        print("  breadth:            " + dist(q.get("breadth") for q in super_qs))
        print("  min_turns_estimate: " + dist(q.get("min_turns_estimate") for q in super_qs))
        for t in ("deep_chain", "wide", "deep_wide"):
            sub = [q for q in super_qs if q.get("type") == t]
            if sub:
                depth = sorted(q.get("sequential_depth", 0) for q in sub)
                br = sorted(q.get("breadth", 0) for q in sub)
                print(f"  {t:<10} depth {depth[0]}-{depth[-1]}, breadth {br[0]}-{br[-1]}")
        over = sum(q.get("min_turns_estimate", 0) > 7 for q in super_qs)
        print(f"  min_turns_estimate > 7 (production budget): {over}/{len(super_qs)}")
    if premise_qs:
        print("Premise tier, by type (expected_behavior):")
        by_type = Counter(q.get("type") for q in premise_qs)
        for t, b in PREMISE_BEHAVIORS.items():
            print(f"  {t:<14} {by_type[t]:>3}  ({b})")
        print("  difficulty: " + dist(q.get("difficulty") for q in premise_qs))
        print("  hops:       " + dist(q.get("hops") for q in premise_qs))


async def main(paths: list[Path]) -> int:
    errors: list[str] = []
    warnings: list[str] = []
    per_file = [(p, load(p, errors)) for p in paths]
    questions = [q for _, qs in per_file for q in qs]

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

    for q in questions:
        qid = q.get("id")
        for j, c in enumerate(q.get("supporting_chunks", [])):
            if not isinstance(c, dict) or not isinstance(c.get("chunk_id"), int):
                continue
            where = f"{qid} supporting_chunks[{j}] (chunk {c['chunk_id']})"
            row = by_id.get(c["chunk_id"])
            if row is None:
                errors.append(f"{where}: chunk does not exist")
                continue
            if row["article_id"] != c.get("article_id"):
                errors.append(
                    f"{where}: article_id {c.get('article_id')} != DB {row['article_id']}"
                )
            if row["chunk_index"] != c.get("chunk_index"):
                errors.append(
                    f"{where}: chunk_index {c.get('chunk_index')} != DB {row['chunk_index']}"
                )
            if row["title"] != c.get("title"):
                errors.append(f"{where}: title {c.get('title')!r} != DB {row['title']!r}")
            if row["section"] != c.get("section"):
                warnings.append(f"{where}: section {c.get('section')!r} != DB {row['section']!r}")
            ev = c.get("evidence")
            if isinstance(ev, str) and norm(ev) not in norm(row["text"]):
                errors.append(f"{where}: evidence not found in chunk text: {ev[:100]!r}")

    for p, qs in per_file:
        summarize(p, qs)

    print()
    for w in warnings:
        print(f"WARNING: {w}")
    if errors:
        print(f"\nFAILED: {len(errors)} error(s)")
        for e in errors:
            print(f"  ERROR: {e}")
        return 1
    print(f"OK: all checks passed ({len(questions)} questions in {len(paths)} file(s))")
    return 0


if __name__ == "__main__":
    args = [Path(a) for a in sys.argv[1:]] or DEFAULT_PATHS
    sys.exit(asyncio.run(main(args)))
