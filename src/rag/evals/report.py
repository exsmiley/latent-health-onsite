"""Summary statistics, the printed report, and run-to-run comparison."""

import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

OUTCOMES = ("supported", "not_found", "out_of_turns", "error", "timeout")
OUTCOME_ABBR = {"supported": "sup", "not_found": "nf", "out_of_turns": "oot", "error": "err",
                "timeout": "to"}  # fmt: skip
USAGE_KEYS = (
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "embedding_tokens",
)


# ---- math ---------------------------------------------------------------------------------------


def percentile(values: list[float], p: float) -> float | None:
    """Linear-interpolated percentile (like numpy's default), p in [0, 100]."""
    xs = sorted(v for v in values if v is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _mean(values: list[Any]) -> float | None:
    xs = [v for v in values if v is not None]
    return mean(xs) if xs else None


def _median(values: list[Any]) -> float | None:
    xs = [v for v in values if v is not None]
    return median(xs) if xs else None


def _rate(flags: list[bool]) -> float | None:
    return sum(bool(f) for f in flags) / len(flags) if flags else None


def _rd(x: float | None, n: int = 3) -> float | None:
    return None if x is None else round(x, n)


def _g(rec: dict) -> dict:
    return rec.get("grading") or {}


def _t(rec: dict) -> dict:
    return rec.get("timing") or {}


def group_stats(recs: list[dict]) -> dict:
    totals = [_t(r).get("total_s") for r in recs]
    usage = [r.get("usage") or {} for r in recs]
    costs = [r.get("cost_usd") for r in recs]
    turns = [x for r in recs for x in _t(r).get("research_turns", [])]
    turn_durs = [x["s"] for x in turns]
    eval_durs = [x["s"] for r in recs for x in _t(r).get("evaluator_calls", [])]
    outcomes = Counter(r.get("outcome", "error") for r in recs)
    return {
        "n": len(recs),
        "accuracy": _rd(_rate([_g(r).get("correct", False) for r in recs])),
        "exact_rate": _rd(_rate([_g(r).get("exact", False) for r in recs])),
        "partial_rate": _rd(_rate([_g(r).get("partially_correct", False) for r in recs])),
        "outcomes": {o: outcomes.get(o, 0) for o in OUTCOMES},
        "turns_mean": _rd(_mean([r.get("turns_used") for r in recs]), 2),
        "turns_p90": _rd(percentile([r.get("turns_used") for r in recs], 90), 1),
        "evaluator_rejections_mean": _rd(_mean([r.get("evaluator_rejections") for r in recs]), 2),
        "time_s": {
            "median": _rd(_median(totals), 1),
            "p90": _rd(percentile(totals, 90), 1),
            "max": _rd(max((t for t in totals if t is not None), default=None), 1),
            "mean": _rd(_mean(totals), 1),
        },
        "first_token_s_median": _rd(_median([_t(r).get("first_token_s") for r in recs]), 1),
        "research_turn_s_mean": _rd(_mean(turn_durs), 2),
        "research_model_s_mean": _rd(_mean([x.get("model_s") for x in turns]), 2),
        "research_tools_s_mean": _rd(_mean([x.get("tools_s") for x in turns]), 2),
        "evaluator_call_s_mean": _rd(_mean(eval_durs), 2),
        "responder_first_token_s_median": _rd(
            _median([_t(r).get("responder_first_token_s") for r in recs]), 2
        ),
        "responder_total_s_median": _rd(_median([_t(r).get("responder_total_s") for r in recs]), 2),
        "tokens_mean": {k: _rd(_mean([u.get(k, 0) for u in usage]), 0) for k in USAGE_KEYS},
        "cost_usd_mean": _rd(_mean(costs), 4),
        "cost_usd_total": _rd(sum(c for c in costs if c is not None), 4)
        if any(c is not None for c in costs)
        else None,
        "citation_recall_mean": _rd(_mean([_g(r).get("citation_recall") for r in recs])),
        "citation_precision_mean": _rd(_mean([_g(r).get("citation_precision") for r in recs])),
        "citation_any_overlap_rate": _rd(
            _rate([_g(r).get("citation_any_overlap", False) for r in recs])
        ),
    }


def _by(recs: list[dict], key: str) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for r in recs:
        groups.setdefault(str(r.get(key)), []).append(r)
    return dict(sorted(groups.items()))


def _by_number(recs: list[dict], key: str) -> dict[str, dict]:
    groups: dict = {}
    for r in recs:
        groups.setdefault(r.get(key), []).append(r)
    ordered = sorted(groups.items(), key=lambda kv: (kv[0] is None, kv[0] or 0))
    return {str(k): group_stats(rs) for k, rs in ordered}


def summarize(records: list[dict]) -> dict:
    by_tier = {}
    for tier, recs in _by(records, "tier").items():
        entry = {
            "all": group_stats(recs),
            "by_type": {t: group_stats(rs) for t, rs in _by(recs, "type").items()},
        }
        if len({r.get("difficulty") for r in recs}) > 1:
            entry["by_difficulty"] = {
                d: group_stats(rs) for d, rs in _by(recs, "difficulty").items()
            }
        for key in ("sequential_depth", "min_turns_estimate"):
            if any(r.get(key) is not None for r in recs):
                entry[f"by_{key}"] = _by_number(recs, key)
        by_tier[tier] = entry
    slowest = sorted(records, key=lambda r: -(_t(r).get("total_s") or 0))[:5]
    wrong = [r for r in records if not _g(r).get("correct")]
    return {
        "overall": group_stats(records),
        "by_tier": by_tier,
        "slowest": [
            {
                "id": r["id"],
                "total_s": _t(r).get("total_s"),
                "turns_used": r.get("turns_used"),
                "outcome": r.get("outcome"),
                "correct": _g(r).get("correct"),
            }
            for r in slowest
        ],
        "wrong": [
            {
                "id": r["id"],
                "tier": r.get("tier"),
                "type": r.get("type"),
                "outcome": r.get("outcome"),
                "partially_correct": _g(r).get("partially_correct"),
                "expected": r.get("expected_answer"),
                "got": _short(r.get("final_answer") or r.get("error") or "", 160),
                "judge_reason": (_g(r).get("judge") or {}).get("reason")
                or (_g(r).get("judge") or {}).get("error"),
            }
            for r in wrong
        ],
    }


# ---- rendering ----------------------------------------------------------------------------------


def _short(text: str, n: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.0f}%"


def _num(x: float | None, fmt: str = ".1f") -> str:
    return "-" if x is None else format(x, fmt)


def _outcomes(o: dict) -> str:
    return " ".join(f"{OUTCOME_ABBR[k]}={v}" for k, v in o.items() if v)


HEADER = (
    f"{'group':<22} {'n':>3} {'acc':>5} {'exact':>5} {'part':>5} {'turns':>5} {'trn90':>5} "
    f"{'t_med':>6} {'t_p90':>6} {'t_max':>6} {'tok_in':>8} {'cached':>8} {'tok_out':>7} "
    f"{'cost':>7}  outcomes"
)


def _row(name: str, s: dict) -> str:
    tm, tk = s["time_s"], s["tokens_mean"]
    cost = "-" if s["cost_usd_mean"] is None else f"${s['cost_usd_mean']:.3f}"
    return (
        f"{_short(name, 22):<22} {s['n']:>3} {_pct(s['accuracy']):>5} {_pct(s['exact_rate']):>5} "
        f"{_pct(s['partial_rate']):>5} {_num(s['turns_mean']):>5} {_num(s['turns_p90']):>5} "
        f"{_num(tm['median']):>6} "
        f"{_num(tm['p90']):>6} {_num(tm['max']):>6} {_num(tk['input_tokens'], '.0f'):>8} "
        f"{_num(tk['cached_input_tokens'], '.0f'):>8} {_num(tk['output_tokens'], '.0f'):>7} "
        f"{cost:>7}  {_outcomes(s['outcomes'])}"
    )


def render(records: list[dict], summary: dict, config: dict | None = None) -> str:
    lines: list[str] = []
    if config:
        lines.append(
            f"Run {config.get('label') or ''}: {config.get('n_questions')} questions, "
            f"max_turns={config.get('max_turns')}, concurrency={config.get('concurrency')}, "
            f"model={config.get('chat_model')}, commit={config.get('git_commit')}, "
            f"wall clock {config.get('wall_clock_s')}s"
        )
    lines.append("(acc = judge verdict, or exact match with --no-judge; times in seconds; "
                 "tokens are means per question)")  # fmt: skip
    for tier, entry in summary["by_tier"].items():
        lines += ["", f"== tier: {tier} ==", HEADER]
        for t, s in entry["by_type"].items():
            lines.append(_row(t, s))
        lines.append(_row("ALL", entry["all"]))
        for key, prefix in (
            ("difficulty", ""),
            ("sequential_depth", "depth="),
            ("min_turns_estimate", "min_turns="),
        ):
            if f"by_{key}" in entry:
                lines += ["", f"  by {key}:", HEADER]
                for d, s in entry[f"by_{key}"].items():
                    lines.append(_row(f"{prefix}{d}", s))
    if len(summary["by_tier"]) > 1:
        lines += ["", HEADER, _row("OVERALL", summary["overall"])]

    o = summary["overall"]
    lines += [
        "",
        (
            "Where the time goes (overall): "
            f"research turn mean {_num(o['research_turn_s_mean'], '.2f')}s "
            f"(model {_num(o['research_model_s_mean'], '.2f')}s + "
            f"tools {_num(o['research_tools_s_mean'], '.2f')}s), "
            f"evaluator call mean {_num(o['evaluator_call_s_mean'], '.2f')}s, "
            f"responder first token median {_num(o['responder_first_token_s_median'], '.2f')}s / "
            f"total median {_num(o['responder_total_s_median'], '.2f')}s, "
            f"first answer token median {_num(o['first_token_s_median'])}s"
        ),
        (
            f"Citations (diagnostic): recall mean {_pct(o['citation_recall_mean'])}, "
            f"precision mean {_pct(o['citation_precision_mean'])}, "
            f"any overlap {_pct(o['citation_any_overlap_rate'])}"
        ),
    ]
    if o["cost_usd_total"] is not None:
        lines.append(f"Estimated cost: ${o['cost_usd_total']:.3f} total")
    lines += ["", "Slowest:"]
    for s in summary["slowest"]:
        lines.append(
            f"  {s['id']:<8} {_num(s['total_s']):>7}s  turns={s['turns_used']}  {s['outcome']}"
            f"  {'correct' if s['correct'] else 'wrong'}"
        )
    lines += ["", f"Wrong answers ({len(summary['wrong'])}):"]
    for w in summary["wrong"]:
        tag = "PARTIAL" if w["partially_correct"] else w["outcome"]
        lines.append(f"  {w['id']:<8} [{w['type']}] {tag}: expected {_short(w['expected'], 60)!r}")
        if w["outcome"] == "supported" or w["outcome"] in ("error", "timeout"):
            lines.append(f"           got: {_short(w['got'], 110)}")
        if w["judge_reason"] and w["outcome"] == "supported":
            lines.append(f"           judge: {_short(w['judge_reason'], 110)}")
    return "\n".join(lines)


# ---- compare ------------------------------------------------------------------------------------


def resolve_run(path: str | Path) -> Path:
    """Accept a results .jsonl, its .summary.json, or the stem without extension."""
    p = Path(path)
    if p.name.endswith(".summary.json"):
        p = p.with_name(p.name[: -len(".summary.json")] + ".jsonl")
    elif p.suffix != ".jsonl":
        p = p.with_name(p.name + ".jsonl")
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def load_run(path: str | Path) -> list[dict]:
    p = resolve_run(path)
    recs = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    latest: dict[str, dict] = {}
    for r in recs:  # a re-run appended to the same file wins
        latest[r["id"]] = r
    return list(latest.values())


def _delta(a: float | None, b: float | None, fmt: str, pct: bool = False) -> str:
    if a is None or b is None:
        return "-"
    if pct:
        return f"{100 * a:.0f}% -> {100 * b:.0f}% ({100 * (b - a):+.0f}pp)"
    return f"{a:{fmt}} -> {b:{fmt}} ({b - a:+{fmt}})"


def compare(a: list[dict], b: list[dict], name_a: str = "A", name_b: str = "B") -> str:
    ids_a, ids_b = {r["id"] for r in a}, {r["id"] for r in b}
    common = ids_a & ids_b
    a_c = [r for r in a if r["id"] in common]
    b_c = [r for r in b if r["id"] in common]
    lines = [f"A = {name_a}", f"B = {name_b}", f"{len(common)} questions in common"]
    if ids_a - ids_b:
        lines.append(f"only in A: {', '.join(sorted(ids_a - ids_b))}")
    if ids_b - ids_a:
        lines.append(f"only in B: {', '.join(sorted(ids_b - ids_a))}")

    def section(title: str, ra: list[dict], rb: list[dict]) -> None:
        sa, sb = group_stats(ra), group_stats(rb)
        lines.append(f"\n{title} (n={sa['n']})")
        lines.append(f"  accuracy     {_delta(sa['accuracy'], sb['accuracy'], '', pct=True)}")
        lines.append(f"  exact        {_delta(sa['exact_rate'], sb['exact_rate'], '', pct=True)}")
        lines.append(f"  turns mean   {_delta(sa['turns_mean'], sb['turns_mean'], '.2f')}")
        for k in ("median", "p90", "max"):
            lines.append(f"  time {k:<7} {_delta(sa['time_s'][k], sb['time_s'][k], '.1f')}s")
        ti_a, ti_b = sa["tokens_mean"]["input_tokens"], sb["tokens_mean"]["input_tokens"]
        lines.append(f"  input tokens {_delta(ti_a, ti_b, '.0f')}")
        lines.append(f"  outcomes     {_outcomes(sa['outcomes'])}  ->  {_outcomes(sb['outcomes'])}")

    section("ALL", a_c, b_c)
    tiers = sorted({str(r.get("tier")) for r in a_c})
    for tier in tiers:
        ra = [r for r in a_c if str(r.get("tier")) == tier]
        rb = [r for r in b_c if str(r.get("tier")) == tier]
        if len(tiers) > 1:
            section(f"tier {tier}", ra, rb)
        for t in sorted({str(r.get("type")) for r in ra}):
            section(
                f"tier {tier} / {t}",
                [r for r in ra if str(r.get("type")) == t],
                [r for r in rb if str(r.get("type")) == t],
            )

    by_b = {r["id"]: r for r in b_c}
    fixed, broken = [], []
    for r in sorted(a_c, key=lambda r: r["id"]):
        rb = by_b[r["id"]]
        ca, cb = bool(_g(r).get("correct")), bool(_g(rb).get("correct"))
        row = (
            f"  {r['id']:<8} [{r.get('type')}] {r.get('outcome')} -> {rb.get('outcome')}, "
            f"{_num(_t(r).get('total_s'))}s -> {_num(_t(rb).get('total_s'))}s, "
            f"turns {r.get('turns_used')} -> {rb.get('turns_used')}"
        )
        if cb and not ca:
            fixed.append(row)
        elif ca and not cb:
            broken.append(row)
    lines += [f"\nFixed (wrong in A, correct in B): {len(fixed)}", *fixed]
    lines += [f"\nBroken (correct in A, wrong in B): {len(broken)}", *broken]

    # Biggest per-question time changes
    diffs = []
    for r in a_c:
        ta, tb = _t(r).get("total_s"), _t(by_b[r["id"]]).get("total_s")
        if ta is not None and tb is not None:
            diffs.append((tb - ta, r["id"], ta, tb))
    diffs.sort()
    if diffs:
        lines.append("\nLargest time changes (B - A):")
        for d, qid, ta, tb in diffs[:5] + [x for x in diffs[-5:] if x not in diffs[:5]]:
            lines.append(f"  {qid:<8} {ta:7.1f}s -> {tb:7.1f}s ({d:+.1f}s)")
    return "\n".join(lines)
