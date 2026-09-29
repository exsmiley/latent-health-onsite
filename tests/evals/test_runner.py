"""Runner tests: the real orchestrator with the scripted fake client from tests/agents/fakes.py."""

import json

from fakes import (
    Stream,
    ToolRecorder,
    answer,
    last_input_text,
    msg,
    not_found,
    resp,
    search,
    verdict,
)
from openai.types.responses import ResponseUsage
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails

from rag.config import get_settings
from rag.evals import runner
from rag.evals.grading import JUDGE_PROMPT
from rag.evals.runner import load_questions, run_eval

Q_OK = "When was Einstein born?"
Q_NF = "What was the name of Einstein's pet?"
Q_ERR = "Which ship did Einstein sail on?"


def question(qid, text, answer_="1879", aliases=None):
    return {
        "id": qid,
        "question": text,
        "answer": answer_,
        "answer_aliases": aliases or [],
        "type": "single_hop",
        "difficulty": "easy",
        "hops": 1,
        "tier": "main",
        "supporting_chunks": [
            {"chunk_id": 999, "article_id": 1, "chunk_index": 0},
            {"chunk_id": 998, "article_id": 1, "chunk_index": 5},
        ],
    }


def _usage(n: int) -> ResponseUsage:
    return ResponseUsage(
        input_tokens=n,
        input_tokens_details=InputTokensDetails(cached_tokens=n // 4, cache_write_tokens=0),
        output_tokens=n // 10,
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
        total_tokens=n + n // 10,
    )


def router(scripts: dict[str, list], judge_calls: list):
    """A script step that picks the next response by which question the call is about."""

    def step(kwargs):
        if kwargs.get("instructions") == JUDGE_PROMPT:
            judge_calls.append(kwargs)
            return resp(msg(json.dumps({"reason": "same year", "correct": True,
                                        "partially_correct": False})))  # fmt: skip
        text = last_input_text(kwargs)
        for q, script in scripts.items():
            if q in text:
                out = script.pop(0)
                if not isinstance(out, (Stream, BaseException)):
                    out.usage = _usage(1000)
                return out
        raise AssertionError(f"unrouted call: {text[:80]}")

    return step


async def test_runs_questions_concurrently_and_records(fake_env, tmp_path):
    scripts = {
        Q_OK: [
            search("einstein birth"),
            answer("Einstein was born in 1879.", [101]),
            verdict("supported", "14 March 1879"),
            Stream(["Einstein was born ", "on 14 March 1879 [1]."]),
        ],
        Q_NF: [search("einstein pet"), not_found("No pet is mentioned.")],
        Q_ERR: [RuntimeError("model exploded")],
    }
    judge_calls: list = []
    step = router(scripts, judge_calls)
    fake_env([step] * 20)

    questions = [question("q1", Q_OK), question("q2", Q_NF), question("q3", Q_ERR)]
    out = tmp_path / "run.jsonl"
    before = get_settings().research_max_turns
    lines: list[str] = []
    recs = await run_eval(questions, concurrency=3, max_turns=5, out_path=out, log=lines.append)
    assert get_settings().research_max_turns == before  # restored
    assert all(not s for s in scripts.values())  # every scripted step was used
    assert len(lines) == 3

    by_id = {r["id"]: r for r in recs}
    assert [r["id"] for r in recs] == ["q1", "q2", "q3"]
    assert {json.loads(x)["id"] for x in out.read_text().splitlines()} == {"q1", "q2", "q3"}

    ok = by_id["q1"]
    assert ok["outcome"] == "supported"
    assert ok["turns_used"] == 2
    assert ok["final_answer"] == "Einstein was born on 14 March 1879 [1]."
    assert [a["status"] for a in ok["research_answers"]] == ["answered"]
    assert [e["verdict"] for e in ok["evaluations"]] == ["supported"]
    assert ok["citations"] == [
        {"n": 1, "chunk_id": 101, "article_id": 1, "chunk_index": 0,
         "title": "Albert Einstein", "section": None},
    ]  # fmt: skip
    t = ok["timing"]
    assert [x["turn"] for x in t["research_turns"]] == [1, 2]
    assert all(x["s"] >= 0 for x in t["research_turns"])
    assert [(x["turn"], x["verdict"]) for x in t["evaluator_calls"]] == [(2, "supported")]
    assert 0 <= t["responder_first_token_s"] <= t["responder_total_s"]
    assert 0 < t["first_token_s"] <= t["total_s"]
    assert t["research_total_s"] <= t["total_s"]
    # 3 non-streamed calls (2 research + 1 evaluator) reported usage; the judge is separate
    assert ok["usage"]["calls"] == 3
    assert ok["usage"]["input_tokens"] == 3000
    assert ok["usage"]["cached_input_tokens"] == 750
    assert ok["judge_usage"]["calls"] == 0  # the judge reply has no usage attached
    g = ok["grading"]
    assert g["exact"] is True and g["correct"] is True
    assert g["judge"]["reason"] == "same year"
    assert g["citation_recall"] == 0.5 and g["citation_any_overlap"] is True
    assert len(judge_calls) == 1 and Q_OK in last_input_text(judge_calls[0])

    nf = by_id["q2"]
    assert nf["outcome"] == "not_found"
    assert nf["turns_used"] == 2
    assert nf["timing"]["evaluator_calls"] == []
    assert nf["timing"]["responder_first_token_s"] is None
    assert nf["timing"]["first_token_s"] is not None  # the fixed "couldn't find" reply
    assert nf["grading"]["correct"] is False and nf["grading"]["exact"] is False
    assert nf["citations"] == [] and nf["grading"]["citation_recall"] == 0.0

    err = by_id["q3"]
    assert err["outcome"] == "error"
    assert "RuntimeError: model exploded" in err["error"]
    assert err["grading"]["correct"] is False
    assert err["timing"]["total_s"] >= 0
    assert len(err["timing"]["research_turns"]) == 1


async def test_timeout_is_isolated(fake_env):
    scripts = {
        Q_OK: [search("slow")],
        Q_NF: [answer("x", [101]), verdict("supported"), Stream(["x"])],  # no tool calls
    }
    fake_env([router(scripts, [])] * 10, tools=ToolRecorder(delay=0.3))
    recs = await run_eval(
        [question("slow", Q_OK), question("fine", Q_NF)],
        concurrency=2,
        timeout=0.1,
        judge=False,
        log=lambda _: None,
    )
    by_id = {r["id"]: r for r in recs}
    assert by_id["slow"]["outcome"] == "timeout"
    assert 0.1 <= by_id["slow"]["timing"]["total_s"] < 0.3
    assert by_id["slow"]["timing"]["research_turns"][0]["turn"] == 1
    assert by_id["fine"]["outcome"] == "supported"


async def test_max_turns_override_reaches_the_pipeline(fake_env):
    scripts = {Q_OK: [search("a"), search("b")]}
    fake_env([router(scripts, [])] * 5)
    (rec,) = await run_eval([question("q", Q_OK)], max_turns=2, judge=False, log=lambda _: None)
    assert rec["outcome"] == "out_of_turns"
    assert rec["turns_used"] == 2
    assert [x["turn"] for x in rec["timing"]["research_turns"]] == [1, 2]


def test_load_questions_filters_and_accepts_super_hard_fields(tmp_path):
    p = tmp_path / "sh.jsonl"
    rows = [
        {
            "id": "s1",
            "question": "a",
            "answer": "x",
            "type": "deep_chain",
            "tier": "super_hard",
            "sequential_depth": 4,
            "breadth": 1,
            "min_turns_estimate": 6,
        },
        {"id": "s2", "question": "b", "answer": "y", "type": "wide"},
        {"id": "s3", "question": "c", "answer": "z", "type": "deep_wide"},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    qs = load_questions(file=p)
    assert [q["id"] for q in qs] == ["s1", "s2", "s3"]
    assert qs[1]["tier"] == "sh"  # defaults to the file/set name
    assert [q["id"] for q in load_questions(file=p, ids=["s3", "s1"])] == ["s1", "s3"]
    assert [q["id"] for q in load_questions(file=p, limit=1)] == ["s1"]
    main = load_questions("main", limit=2)
    assert len(main) == 2 and main[0]["tier"] == "main"


def test_pricing_defaults_to_none(monkeypatch):
    for k in ("INPUT", "CACHED_INPUT", "OUTPUT", "EMBEDDING"):
        monkeypatch.delenv(f"EVAL_PRICE_{k}_PER_1M", raising=False)
    p = runner.Pricing(_env_file=None)
    assert p.cost({"input_tokens": 100}) is None
    p = runner.Pricing(_env_file=None, input_per_1m=1.0, cached_input_per_1m=0.1, output_per_1m=10)
    cost = p.cost(
        {"input_tokens": 1_000_000, "cached_input_tokens": 500_000, "output_tokens": 1000}
    )
    assert abs(cost - (0.5 + 0.05 + 0.01)) < 1e-9
