import pytest

from rag.evals.report import compare, group_stats, percentile, render, summarize


def rec(qid, *, correct, total, outcome="supported", turns=3, tier="main", type_="multi_article",
        depth=None, exact=None, partial=False, usage=None):  # fmt: skip
    return {
        "id": qid,
        "tier": tier,
        "type": type_,
        "sequential_depth": depth,
        "outcome": outcome,
        "turns_used": turns,
        "expected_answer": "x",
        "final_answer": "y",
        "timing": {
            "total_s": total,
            "first_token_s": total - 1,
            "research_turns": [{"turn": 1, "s": 2.0}, {"turn": 2, "s": 4.0}],
            "evaluator_calls": [{"turn": 2, "s": 1.0, "verdict": "supported"}],
            "responder_first_token_s": 0.5,
            "responder_total_s": 1.5,
        },
        "usage": usage or {"input_tokens": 100, "output_tokens": 10},
        "grading": {
            "correct": correct,
            "exact": correct if exact is None else exact,
            "partially_correct": partial,
            "citation_recall": 0.5,
            "citation_any_overlap": True,
            "judge": {"correct": correct, "partially_correct": partial, "reason": "r"},
        },
        "cost_usd": None,
    }


def test_percentile_linear_interpolation():
    assert percentile([], 90) is None
    assert percentile([5.0], 90) == 5.0
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 90) == pytest.approx(9.1)
    assert percentile([10, 1, 5], 50) == 5
    assert percentile([1, None, 3], 100) == 3


def test_group_stats():
    recs = [
        rec("a", correct=True, total=10.0, turns=2),
        rec("b", correct=False, total=30.0, outcome="not_found", turns=4, partial=True),
        rec(
            "c",
            correct=True,
            total=20.0,
            exact=False,
            turns=3,
            usage={"input_tokens": 400, "output_tokens": 40},
        ),
        rec("d", correct=False, total=40.0, outcome="error", turns=1),
    ]
    s = group_stats(recs)
    assert s["n"] == 4
    assert s["accuracy"] == 0.5
    assert s["exact_rate"] == 0.25
    assert s["partial_rate"] == 0.25
    assert s["outcomes"] == {
        "supported": 2, "not_found": 1, "out_of_turns": 0, "error": 1, "timeout": 0,
    }  # fmt: skip
    assert s["turns_mean"] == 2.5
    assert s["time_s"] == {"median": 25.0, "p90": 37.0, "max": 40.0, "mean": 25.0}
    assert s["tokens_mean"]["input_tokens"] == 175
    assert s["tokens_mean"]["cached_input_tokens"] == 0
    assert s["research_turn_s_mean"] == 3.0
    assert s["evaluator_call_s_mean"] == 1.0
    assert s["cost_usd_mean"] is None and s["cost_usd_total"] is None


def test_summarize_groups_and_lists():
    recs = [
        rec("m1", correct=True, total=5.0),
        rec("m2", correct=False, total=50.0, type_="temporal"),
        rec("s1", correct=True, total=60.0, tier="super_hard", type_="deep_chain", depth=3),
        rec(
            "s2",
            correct=False,
            total=90.0,
            tier="super_hard",
            type_="wide",
            depth=1,
            outcome="out_of_turns",
        ),
        rec("s3", correct=False, total=70.0, tier="super_hard", type_="deep_chain", depth=3),
    ]
    s = summarize(recs)
    assert set(s["by_tier"]) == {"main", "super_hard"}
    assert set(s["by_tier"]["main"]["by_type"]) == {"multi_article", "temporal"}
    assert "by_sequential_depth" not in s["by_tier"]["main"]
    depth = s["by_tier"]["super_hard"]["by_sequential_depth"]
    assert list(depth) == ["1", "3"]
    assert depth["3"]["n"] == 2 and depth["3"]["accuracy"] == 0.5
    assert "by_min_turns_estimate" not in s["by_tier"]["super_hard"]  # none recorded
    assert [x["id"] for x in s["slowest"]] == ["s2", "s3", "s1", "m2", "m1"]
    assert [w["id"] for w in s["wrong"]] == ["m2", "s2", "s3"]
    assert s["overall"]["n"] == 5
    text = render(recs, s, {"label": "t", "n_questions": 5})
    assert "== tier: super_hard ==" in text and "depth=3" in text and "OVERALL" in text


def test_compare_fixed_and_broken():
    a = [rec("q1", correct=False, total=10.0), rec("q2", correct=True, total=10.0),
         rec("q3", correct=True, total=10.0)]  # fmt: skip
    b = [rec("q1", correct=True, total=20.0), rec("q2", correct=False, total=5.0),
         rec("q4", correct=True, total=1.0)]  # fmt: skip
    out = compare(a, b)
    assert "2 questions in common" in out
    assert "only in A: q3" in out and "only in B: q4" in out
    assert "Fixed (wrong in A, correct in B): 1\n  q1" in out
    assert "Broken (correct in A, wrong in B): 1\n  q2" in out
    assert "50% -> 50% (+0pp)" in out
