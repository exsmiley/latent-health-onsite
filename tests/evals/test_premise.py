"""Premise tier: loading, outcome-based grading, the premise-aware judge, reporting, validation."""

import importlib.util
import json

from fakes import Stream, answer, last_input_text, msg, not_found, resp, search, verdict

from rag.config import ROOT
from rag.evals.grading import (
    JUDGE_PROMPT,
    PREMISE_JUDGE_PROMPT,
    judge_input,
    premise_outcome_verdict,
)
from rag.evals.report import render, summarize
from rag.evals.runner import load_questions, run_eval

_spec = importlib.util.spec_from_file_location("evals_validate", ROOT / "evals" / "validate.py")
validate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(validate)

Q_FP = "Who is the king of Switzerland?"
Q_UN = "What was Einstein's goldfish called?"


def pq(qid, text, behavior, type_):
    return {
        "id": qid,
        "question": text,
        "answer": "No king; Switzerland is a republic run by the Federal Council",
        "answer_aliases": ["republic; Federal Council"],
        "tier": "premise",
        "type": type_,
        "expected_behavior": behavior,
        "premise": "Switzerland has a king.",
        "difficulty": "easy",
        "hops": 1,
        "notes": f"good vs bad for {qid}",
        "supporting_chunks": [],
    }


def router(scripts: dict[str, list], judge_calls: list, judge_reply: dict):
    def step(kwargs):
        if kwargs.get("instructions") in (JUDGE_PROMPT, PREMISE_JUDGE_PROMPT):
            judge_calls.append(kwargs)
            return resp(msg(json.dumps(judge_reply)))
        text = last_input_text(kwargs)
        for q, script in scripts.items():
            if q in text:
                return script.pop(0)
        raise AssertionError(f"unrouted call: {text[:80]}")

    return step


def test_premise_set_loads_and_all_includes_it():
    qs = load_questions("premise")
    assert len(qs) >= 25
    assert {q["tier"] for q in qs} == {"premise"}
    assert {q["expected_behavior"] for q in qs} == {"correct_premise", "disambiguate", "not_found"}
    azer = [q for q in qs if q["question"] == "who is the prince of Azerbaijian?"]
    assert len(azer) == 1 and azer[0]["expected_behavior"] == "correct_premise"
    assert {q["_set"] for q in load_questions("all")} >= {"main", "super_hard", "premise"}


def test_premise_outcome_verdict():
    ok = premise_outcome_verdict("not_found", "not_found")
    assert ok["correct"] is True and ok["partially_correct"] is False
    assert premise_outcome_verdict("not_found", "out_of_turns")["correct"] is True
    for behavior in ("correct_premise", "disambiguate"):
        v = premise_outcome_verdict(behavior, "not_found")
        assert v["correct"] is False and "couldn't find" in v["reason"]
        assert premise_outcome_verdict(behavior, "out_of_turns")["correct"] is False
    for behavior in ("correct_premise", "disambiguate", "not_found"):
        assert premise_outcome_verdict(behavior, "supported") is None  # goes to the judge
        assert premise_outcome_verdict(behavior, "error")["correct"] is False
        assert premise_outcome_verdict(behavior, "timeout")["correct"] is False


def test_judge_input_carries_behavior_and_premise():
    text = judge_input("Q?", "A", [], "resp", "the notes", "correct_premise", "X has a king.")
    assert "EXPECTED BEHAVIOR:\ncorrect_premise" in text
    assert "PREMISE:\nX has a king." in text
    assert "NOTES:\nthe notes" in text
    plain = judge_input("Q?", "A", [], "resp", "n")
    assert "EXPECTED BEHAVIOR" not in plain and "PREMISE" not in plain


async def test_decline_grading_without_the_judge(fake_env):
    scripts = {
        Q_FP: [search("swiss king"), not_found("No king found.")],
        Q_UN: [search("goldfish"), not_found("No goldfish mentioned.")],
    }
    judge_calls: list = []
    reply = {"reason": "r", "correct": True, "partially_correct": False}
    fake_env([router(scripts, judge_calls, reply)] * 10)
    recs = await run_eval(
        [
            pq("fp", Q_FP, "correct_premise", "false_premise"),
            pq("un", Q_UN, "not_found", "unanswerable"),
        ],
        log=lambda _: None,
    )
    by_id = {r["id"]: r for r in recs}
    assert judge_calls == []
    fp, un = by_id["fp"], by_id["un"]
    assert fp["outcome"] == un["outcome"] == "not_found"
    assert (
        fp["expected_behavior"] == "correct_premise" and fp["premise"] == "Switzerland has a king."
    )
    assert fp["grading"]["correct"] is False and fp["grading"]["method"] == "outcome"
    assert un["grading"]["correct"] is True and un["grading"]["method"] == "outcome"


async def test_supported_answer_goes_to_premise_judge_and_skips_exact(fake_env):
    scripts = {
        Q_FP: [
            answer("Switzerland has no monarch.", [101]),
            verdict("supported"),
            Stream(["There is no king: Switzerland is governed collectively [1]."]),
        ],
    }
    judge_calls: list = []
    reply = {"reason": "corrects the premise", "correct": True, "partially_correct": False}
    fake_env([router(scripts, judge_calls, reply)] * 10)
    (rec,) = await run_eval(
        [pq("fp", Q_FP, "correct_premise", "false_premise")], log=lambda _: None
    )
    g = rec["grading"]
    assert g["exact"] is False  # kept as a diagnostic only
    assert g["correct"] is True and g["method"] == "judge"
    (call,) = judge_calls
    assert call["instructions"] == PREMISE_JUDGE_PROMPT
    text = last_input_text(call)
    assert "EXPECTED BEHAVIOR:\ncorrect_premise" in text
    assert "PREMISE:\nSwitzerland has a king." in text
    assert "NOTES:\ngood vs bad for fp" in text


async def test_invented_answer_to_unanswerable_is_wrong_without_judge(fake_env):
    scripts = {
        Q_UN: [
            answer("It was called Goldie.", [101]),
            verdict("supported"),
            Stream(["republic; Federal Council"]),  # would even match the exact aliases
        ],
    }
    fake_env([router(scripts, [], {})] * 10)
    (rec,) = await run_eval(
        [pq("un", Q_UN, "not_found", "unanswerable")], judge=False, log=lambda _: None
    )
    assert rec["outcome"] == "supported"
    assert rec["grading"]["exact"] is True
    assert rec["grading"]["correct"] is False and rec["grading"]["method"] == "exact"


def _rec(qid, behavior, correct):
    return {
        "id": qid,
        "tier": "premise",
        "type": {"correct_premise": "false_premise", "not_found": "unanswerable"}[behavior],
        "expected_behavior": behavior,
        "outcome": "not_found",
        "turns_used": 2,
        "expected_answer": "x",
        "final_answer": "",
        "timing": {"total_s": 3.0, "research_turns": [], "evaluator_calls": []},
        "usage": {},
        "grading": {"correct": correct, "exact": False, "partially_correct": False},
    }


def test_report_shows_accuracy_per_expected_behavior():
    recs = [
        _rec("a", "correct_premise", False),
        _rec("b", "correct_premise", True),
        _rec("c", "not_found", True),
    ]
    s = summarize(recs)
    by_b = s["by_tier"]["premise"]["by_expected_behavior"]
    assert by_b["correct_premise"]["accuracy"] == 0.5
    assert by_b["not_found"]["accuracy"] == 1.0
    assert [w["expected_behavior"] for w in s["wrong"]] == ["correct_premise"]
    out = render(recs, s)
    assert "by expected_behavior:" in out
    assert "expected behavior: correct_premise" in out
    plain = [{k: v for k, v in r.items() if k != "expected_behavior"} for r in recs]
    assert "by_expected_behavior" not in summarize(plain)["by_tier"]["premise"]


def _vq(**over):
    q = {
        "id": "p1",
        "question": "Who is the king of X?",
        "answer": "No king",
        "answer_aliases": [],
        "tier": "premise",
        "type": "false_premise",
        "expected_behavior": "correct_premise",
        "premise": "X has a king.",
        "hops": 1,
        "difficulty": "easy",
        "supporting_chunks": [
            {
                "chunk_id": 1,
                "article_id": 1,
                "title": "X",
                "section": None,
                "chunk_index": 0,
                "evidence": "X is a republic.",
            }
        ],
        "reasoning": "r",
        "notes": "n",
    }
    q.update(over)
    return q


def test_validator_checks_premise_fields():
    errors: list[str] = []
    validate.check_shape(_vq(), "ok", errors)
    assert errors == []
    validate.check_shape(
        _vq(type="unanswerable", expected_behavior="not_found", supporting_chunks=[]), "un", errors
    )
    assert errors == []

    for bad, needle in (
        (_vq(expected_behavior="not_found"), "needs expected_behavior"),
        (_vq(premise=""), "premise field"),
        (_vq(type="deep_chain"), "bad premise type"),
        (_vq(supporting_chunks=[]), "no supporting_chunks"),
        (_vq(difficulty="super_hard"), "bad difficulty"),
    ):
        errs: list[str] = []
        validate.check_shape(bad, "bad", errs)
        assert any(needle in e for e in errs), (needle, errs)
