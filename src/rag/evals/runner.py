"""Eval runner: runs the real pipeline over question sets, timing and grading each question."""

import asyncio
import contextlib
import json
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict

from rag import db
from rag.agents import orchestrator
from rag.agents.model import track_usage
from rag.config import ROOT, get_settings
from rag.evals import grading, report
from rag.tools import fetch as fetch_tool

EVALS_DIR = ROOT / "evals"
SETS: dict[str, Path] = {
    "main": EVALS_DIR / "questions.jsonl",
    "super_hard": EVALS_DIR / "questions_super_hard.jsonl",
    "premise": EVALS_DIR / "questions_premise.jsonl",
}
DEFAULT_OUT = EVALS_DIR / "results"
OUTCOMES = ("supported", "not_found", "out_of_turns", "error", "timeout")


class Pricing(BaseSettings):
    """USD per 1M tokens. Nothing in the repo states chat-model prices, so all default to None
    (tokens only). Set e.g. EVAL_PRICE_INPUT_PER_1M in the environment or .env to get costs."""

    model_config = SettingsConfigDict(
        env_prefix="EVAL_PRICE_", env_file=ROOT / ".env", extra="ignore"
    )

    input_per_1m: float | None = None
    cached_input_per_1m: float | None = None  # falls back to input_per_1m
    output_per_1m: float | None = None
    embedding_per_1m: float | None = None

    @property
    def enabled(self) -> bool:
        return self.input_per_1m is not None and self.output_per_1m is not None

    def cost(self, usage: dict[str, int]) -> float | None:
        if not self.enabled:
            return None
        cached = usage.get("cached_input_tokens", 0)
        cached_price = (
            self.cached_input_per_1m if self.cached_input_per_1m is not None else self.input_per_1m
        )
        usd = (
            (usage.get("input_tokens", 0) - cached) * self.input_per_1m
            + cached * cached_price
            + usage.get("output_tokens", 0) * self.output_per_1m
            + usage.get("embedding_tokens", 0) * (self.embedding_per_1m or 0.0)
        )
        return usd / 1_000_000


# ---- loading ------------------------------------------------------------------------------------


def load_questions(
    set_name: str = "main",
    file: Path | None = None,
    ids: list[str] | None = None,
    limit: int | None = None,
    warn: Callable[[str], None] = lambda m: print(m, file=sys.stderr),
) -> list[dict]:
    """Questions from `file`, or from the named set ("main", "super_hard", "premise" or "all").

    Each question gets `_set` (the set it came from). `tier` defaults to that set name.
    """
    if file is not None:
        known = {v.resolve(): k for k, v in SETS.items()}
        sources = [(known.get(Path(file).resolve(), Path(file).stem), Path(file))]
    elif set_name == "all":
        sources = list(SETS.items())
    elif set_name in SETS:
        sources = [(set_name, SETS[set_name])]
    else:
        raise ValueError(f"unknown set {set_name!r}; use {', '.join(SETS)} or all")

    questions: list[dict] = []
    for name, path in sources:
        if not path.exists():
            if set_name == "all" and file is None:
                warn(f"skipping missing set {name}: {path}")
                continue
            raise FileNotFoundError(path)
        for line in path.read_text().splitlines():
            if line.strip():
                q = json.loads(line)
                q["_set"] = name
                q.setdefault("tier", name)
                questions.append(q)
    if ids:
        wanted = set(ids)
        questions = [q for q in questions if q["id"] in wanted]
        missing = wanted - {q["id"] for q in questions}
        if missing:
            warn(f"unknown ids: {', '.join(sorted(missing))}")
    if limit is not None:
        questions = questions[:limit]
    return questions


# ---- running one question -----------------------------------------------------------------------


@dataclass
class _Trace:
    """Timestamps (seconds since the question started) collected while consuming events."""

    t0: float
    research_turns: list[dict] = field(default_factory=list)
    evaluator_calls: list[dict] = field(default_factory=list)
    first_token_s: float | None = None
    respond_start_s: float | None = None
    last_token_s: float | None = None
    done_s: float | None = None
    # open intervals
    _research: tuple[int, float] | None = None
    _evaluate: tuple[int, float] | None = None
    _model_done: float | None = None  # first tool_call / research_answer of the open turn
    _last_tool: float | None = None  # last tool_result of the open turn

    def now(self) -> float:
        return time.perf_counter() - self.t0

    def open_research(self, turn: int, t: float) -> None:
        self._research, self._model_done, self._last_tool = (turn, t), None, None

    def mark_model_done(self, t: float) -> None:
        if self._research is not None and self._model_done is None:
            self._model_done = t

    def close_research(self, t: float) -> None:
        """A turn = its `status` to the next `status` (or done/error). `model_s` runs to the
        turn's first tool_call/research_answer event (for an answer, that includes checking the
        citations in the DB); `tools_s` from there to the last tool_result."""
        if self._research is not None:
            turn, start = self._research
            md = self._model_done
            self.research_turns.append(
                {
                    "turn": turn,
                    "s": round(t - start, 3),
                    "model_s": None if md is None else round(md - start, 3),
                    "tools_s": (
                        None
                        if md is None or self._last_tool is None
                        else round(self._last_tool - md, 3)
                    ),
                }
            )
            self._research = None


def _r(x: float | None) -> float | None:
    return None if x is None else round(x, 3)


async def run_question(
    q: dict, timeout: float = 600.0, judge: bool = True, pricing: Pricing | None = None
) -> dict:
    """Run one question through the pipeline and return its graded record. Never raises."""
    rec: dict[str, Any] = {
        "id": q["id"],
        "set": q.get("_set"),
        "tier": q.get("tier"),
        "type": q.get("type"),
        "difficulty": q.get("difficulty"),
        "hops": q.get("hops"),
        "sequential_depth": q.get("sequential_depth"),
        "breadth": q.get("breadth"),
        "min_turns_estimate": q.get("min_turns_estimate"),
        "question": q["question"],
        "expected_answer": q.get("answer", ""),
        "answer_aliases": q.get("answer_aliases", []),
    }
    behavior = q.get("expected_behavior")  # premise tier only
    if behavior is not None:
        rec["expected_behavior"] = behavior
        rec["premise"] = q.get("premise")
    events: dict[str, list] = {
        "research_answers": [],
        "evaluations": [],
        "tool_calls": [],
        "errors": [],
    }
    outcome_event: dict | None = None
    citations: list[dict] = []
    tokens: list[str] = []
    timed_out = False
    crash: str | None = None

    with track_usage() as usage:
        tr = _Trace(t0=time.perf_counter())
        try:
            async with asyncio.timeout(timeout):
                gen = orchestrator.run(q["question"], [])
                async with contextlib.aclosing(gen):
                    async for ev in gen:
                        t = tr.now()
                        d = ev.data
                        if ev.type == "status":
                            tr.close_research(t)
                            if d["stage"] == "research":
                                tr.open_research(d["turn"], t)
                            elif d["stage"] == "evaluate":
                                tr._evaluate = (d["turn"], t)
                            elif d["stage"] == "respond":
                                tr.respond_start_s = t
                        elif ev.type == "evaluation":
                            if tr._evaluate is not None:
                                turn, start = tr._evaluate
                                tr.evaluator_calls.append(
                                    {
                                        "turn": turn,
                                        "s": round(t - start, 3),
                                        "verdict": d["verdict"],
                                    }
                                )
                                tr._evaluate = None
                            events["evaluations"].append(d)
                        elif ev.type == "research_answer":
                            tr.mark_model_done(t)
                            events["research_answers"].append(d)
                        elif ev.type == "tool_call":
                            tr.mark_model_done(t)
                            events["tool_calls"].append({"turn": d["turn"], "name": d["name"]})
                        elif ev.type == "tool_result":
                            tr._last_tool = t
                        elif ev.type == "outcome":
                            outcome_event = d
                        elif ev.type == "citations":
                            citations = d
                        elif ev.type == "token":
                            if tr.first_token_s is None:
                                tr.first_token_s = t
                            tr.last_token_s = t
                            tokens.append(d["delta"])
                        elif ev.type == "error":
                            tr.close_research(t)
                            events["errors"].append(d["message"])
                        elif ev.type == "done":
                            tr.close_research(t)
                            tr.done_s = t
        except TimeoutError:
            timed_out = True
        except Exception as exc:  # noqa: BLE001 - isolate failures to this question
            crash = f"{type(exc).__name__}: {exc}"
        end_s = tr.done_s if tr.done_s is not None else tr.now()
        tr.close_research(end_s)
    pipeline_usage = usage.to_dict()

    if timed_out:
        outcome = "timeout"
    elif crash is not None or events["errors"]:
        outcome = "error"
        if crash is not None:
            events["errors"].append(crash)
    elif outcome_event is not None:
        outcome = outcome_event["result"]
    else:
        outcome = "error"
        events["errors"].append("pipeline ended without an outcome event")

    supported = outcome == "supported"
    responder_first = responder_total = None
    if supported and tr.respond_start_s is not None and tr.first_token_s is not None:
        responder_first = tr.first_token_s - tr.respond_start_s
        responder_total = (tr.last_token_s or tr.first_token_s) - tr.respond_start_s

    final_answer = "".join(tokens)
    rec.update(
        {
            "outcome": outcome,
            "turns_used": (
                outcome_event["turns_used"] if outcome_event else len(tr.research_turns)
            ),
            "error": "; ".join(events["errors"]) or None,
            "final_answer": final_answer,
            "research_answers": events["research_answers"],
            "evaluations": events["evaluations"],
            "tool_calls": events["tool_calls"],
            "evaluator_rejections": sum(e["verdict"] != "supported" for e in events["evaluations"]),
            "invalid_answers": sum(a["status"] == "invalid" for a in events["research_answers"]),
            "timing": {
                "total_s": _r(end_s),
                "first_token_s": _r(tr.first_token_s),
                "research_turns": tr.research_turns,
                "research_total_s": _r(sum(x["s"] for x in tr.research_turns)),
                "evaluator_calls": tr.evaluator_calls,
                "evaluator_total_s": _r(sum(x["s"] for x in tr.evaluator_calls)),
                "responder_first_token_s": _r(responder_first),
                "responder_total_s": _r(responder_total),
            },
            "usage": pipeline_usage,
        }
    )

    # Cited chunks with their stable (article_id, chunk_index) keys. Not part of the timing.
    cited: list[dict] = []
    if citations:
        try:
            res = await fetch_tool.fetch(chunk_ids=[c["chunk_id"] for c in citations])
            by_id = {c.chunk_id: c for c in res.chunks}
            for c in citations:
                ch = by_id.get(c["chunk_id"])
                cited.append(
                    {
                        "n": c.get("n"),
                        "chunk_id": c["chunk_id"],
                        "article_id": c["article_id"],
                        "chunk_index": ch.chunk_index if ch else None,
                        "title": c.get("title"),
                        "section": c.get("section"),
                    }
                )
        except Exception as exc:  # noqa: BLE001
            rec["error"] = (rec["error"] + "; " if rec["error"] else "") + f"citation lookup: {exc}"
    rec["citations"] = cited

    # Grading. For the premise tier the outcome alone can decide (a decline is right for
    # not_found and wrong otherwise), exact match is only a diagnostic, and the judge gets the
    # expected behavior and the premise.
    exp_answer, aliases = rec["expected_answer"], rec["answer_aliases"]
    exact = supported and grading.exact_match(exp_answer, aliases, final_answer)
    judgement: dict | None = None
    judge_usage: dict | None = None
    decided = grading.premise_outcome_verdict(behavior, outcome) if behavior else None
    if decided is not None:
        judgement = decided
    elif judge:
        if supported and final_answer.strip():
            with track_usage() as ju:
                try:
                    j = await grading.judge(
                        q["question"],
                        exp_answer,
                        aliases,
                        final_answer,
                        q.get("notes", ""),
                        expected_behavior=behavior,
                        premise=q.get("premise"),
                    )
                    judgement = j.model_dump()
                except Exception as exc:  # noqa: BLE001 - fall back to exact match
                    judgement = {"error": f"{type(exc).__name__}: {exc}"}
            judge_usage = ju.to_dict()
        else:
            judgement = {
                "correct": False,
                "partially_correct": False,
                "reason": f"no answer ({outcome}); not sent to the judge",
            }
    if judgement is not None and "correct" in judgement:
        correct, partial = judgement["correct"], judgement["partially_correct"]
        if judge_usage is not None:
            method = "judge"
        elif decided is not None:
            method = "outcome"
        else:
            method = "no_answer"
    else:
        # no judge (or it failed): exact match, except that a supported answer to a not_found
        # question is never correct
        correct, partial = exact and behavior != "not_found", False
        method = "exact"
    recall, overlap, article_recall, precision = grading.citation_recall(
        [c for c in q.get("supporting_chunks", []) if "chunk_index" in c],
        [c for c in cited if c["chunk_index"] is not None],
    )
    rec["grading"] = {
        "exact": exact,
        "judge": judgement,
        "correct": correct,
        "partially_correct": partial,
        "method": method,
        "citation_recall": _r(recall),
        "citation_any_overlap": overlap,
        "citation_article_recall": _r(article_recall),
        "citation_precision": _r(precision),
    }
    rec["judge_usage"] = judge_usage
    rec["cost_usd"] = pricing.cost(pipeline_usage) if pricing else None
    return rec


# ---- running a set ------------------------------------------------------------------------------


def git_commit() -> str | None:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


@contextlib.contextmanager
def override_max_turns(max_turns: int | None):
    """Temporarily set `research_max_turns` (read by orchestrator.run() at its start)."""
    settings = get_settings()
    old = settings.research_max_turns
    if max_turns is not None:
        settings.research_max_turns = max_turns
    try:
        yield settings.research_max_turns
    finally:
        settings.research_max_turns = old


def _progress_line(i: int, n: int, rec: dict) -> str:
    g = rec["grading"]
    verdict = "OK   " if g["correct"] else ("PART " if g["partially_correct"] else "WRONG")
    return (
        f"[{i:>3}/{n}] {rec['id']:<8} {verdict} {rec['outcome']:<12} "
        f"turns={rec['turns_used']:<2} {rec['timing']['total_s']:>7.1f}s"
    )


async def run_eval(
    questions: list[dict],
    *,
    concurrency: int = 4,
    max_turns: int | None = None,
    timeout: float = 600.0,
    judge: bool = True,
    out_path: Path | None = None,
    pricing: Pricing | None = None,
    log: Callable[[str], None] = print,
) -> list[dict]:
    """Run `questions` concurrently (bounded by a semaphore) and return records in input order.

    Records are appended to `out_path` (JSONL) as they finish, so a crash keeps partial results.
    """
    sem = asyncio.Semaphore(max(1, concurrency))
    results: dict[str, dict] = {}
    n = len(questions)
    done_count = 0
    out = out_path.open("a") if out_path else None

    async def one(q: dict) -> None:
        nonlocal done_count
        async with sem:
            try:
                rec = await run_question(q, timeout=timeout, judge=judge, pricing=pricing)
            except Exception as exc:  # noqa: BLE001 - belt and braces: run_question shouldn't raise
                rec = {"id": q["id"], "outcome": "error", "error": repr(exc)}
        results[q["id"]] = rec
        done_count += 1
        if out:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
        if "grading" in rec:
            log(_progress_line(done_count, n, rec))

    try:
        with override_max_turns(max_turns):
            await asyncio.gather(*(one(q) for q in questions))
    finally:
        if out:
            out.close()
    return [results[q["id"]] for q in questions if q["id"] in results]


def result_paths(out_dir: Path, label: str | None) -> tuple[Path, Path]:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in (label or "run"))
    stem = f"{stamp}_{safe}"
    return out_dir / f"{stem}.jsonl", out_dir / f"{stem}.summary.json"


async def main(
    *,
    set_name: str,
    file: Path | None,
    ids: list[str] | None,
    limit: int | None,
    max_turns: int | None,
    concurrency: int,
    timeout: float,
    judge: bool,
    out_dir: Path,
    label: str | None,
) -> Path:
    questions = load_questions(set_name, file, ids, limit)
    if not questions:
        raise SystemExit("no questions selected")
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl, summary_path = result_paths(out_dir, label)
    settings = get_settings()
    pricing = Pricing()
    config = {
        "label": label,
        "set": set_name if file is None else None,
        "file": str(file) if file else None,
        "ids": ids,
        "limit": limit,
        "n_questions": len(questions),
        "max_turns": max_turns if max_turns is not None else settings.research_max_turns,
        "concurrency": concurrency,
        "timeout_s": timeout,
        "judge": judge,
        "chat_model": settings.chat_model,
        "judge_model": settings.chat_model if judge else None,
        "embedding_model": settings.embedding_model,
        "git_commit": git_commit(),
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "pricing_per_1m": pricing.model_dump() if pricing.enabled else None,
        "results_file": jsonl.name,
    }
    print(
        f"Running {len(questions)} questions (max_turns={config['max_turns']}, "
        f"concurrency={concurrency}, judge={'on' if judge else 'off'}) -> {jsonl}"
    )
    t0 = time.perf_counter()
    try:
        records = await run_eval(
            questions,
            concurrency=concurrency,
            max_turns=max_turns,
            timeout=timeout,
            judge=judge,
            out_path=jsonl,
            pricing=pricing,
        )
    finally:
        await db.close_pool()
    config["wall_clock_s"] = round(time.perf_counter() - t0, 1)
    config["finished_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    summary = report.summarize(records)
    summary_path.write_text(
        json.dumps({"config": config, "summary": summary}, indent=2, ensure_ascii=False) + "\n"
    )
    print()
    print(report.render(records, summary, config))
    print(f"\nResults: {jsonl}\nSummary: {summary_path}")
    return jsonl
