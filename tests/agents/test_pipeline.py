import json

import pytest
from fakes import Stream, ToolRecorder, fc, last_input_text, msg, resp, submit, verdict

from rag.agents import orchestrator
from rag.agents.evaluator import evaluate
from rag.agents.orchestrator import NOT_FOUND_MESSAGE
from rag.agents.research import SUBMIT_ANSWER
from rag.config import get_settings


async def collect(question="When was Einstein born?", history=None):
    return [ev async for ev in orchestrator.run(question, history or [])]


def types_of(evs):
    return [e.type for e in evs]


def only(evs, t):
    return [e.data for e in evs if e.type == t]


async def test_happy_path(fake_env):
    client, tools = fake_env(
        [
            resp(
                fc("semantic_search", {"queries": ["Einstein birth date"], "top_k": None}, "c1"),
                fc("keyword_search", {"queries": ["Einstein born 1879"], "top_k": None}, "c2"),
            ),
            resp(fc("fetch", {"chunk_ids": [101], "article_ids": []}, "c3")),
            submit("Einstein was born on 14 March 1879.", [101]),
            verdict("supported", "14 March 1879", "Passage [1] states it."),
            Stream(["Einstein was born ", "on 14 March 1879 [1]."]),
        ]
    )
    evs = await collect()

    assert types_of(evs) == [
        "status", "tool_call", "tool_call", "tool_result", "tool_result",
        "status", "tool_call", "tool_result",
        "status", "tool_call", "tool_result",
        "research_answer", "status", "evaluation", "status", "citations",
        "token", "token", "done",
    ]  # fmt: skip
    assert evs[0].data == {
        "stage": "research", "round": 1, "turn": 1, "message": evs[0].data["message"],
    }  # fmt: skip
    call = evs[1].data
    assert call == {
        "id": "c1",
        "name": "semantic_search",
        "arguments": {"queries": ["Einstein birth date"], "top_k": None},
        "round": 1,
        "turn": 1,
    }
    assert only(evs, "tool_result")[:2] == [
        {"id": "c1", "name": "semantic_search", "summary": "1 queries, 1 hits"},
        {"id": "c2", "name": "keyword_search", "summary": "1 queries, 1 hits"},
    ]
    assert only(evs, "research_answer") == [
        {"round": 1, "answer": "Einstein was born on 14 March 1879.", "citations": [101]}
    ]
    assert only(evs, "evaluation") == [
        {
            "round": 1,
            "verdict": "supported",
            "independent_answer": "14 March 1879",
            "feedback": "Passage [1] states it.",
        }
    ]
    assert only(evs, "citations") == [
        [
            {
                "n": 1,
                "chunk_id": 101,
                "article_id": 1,
                "title": "Albert Einstein",
                "section": None,
                "url": "https://simple.wikipedia.org/wiki/Albert_Einstein",
            }
        ]
    ]
    assert (
        "".join(d["delta"] for d in only(evs, "token")) == "Einstein was born on 14 March 1879 [1]."
    )
    assert [s["stage"] for s in only(evs, "status")] == [
        "research", "research", "research", "evaluate", "respond",
    ]  # fmt: skip

    # Both searches in turn 1 ran concurrently.
    assert tools.max_active == 2

    calls = client.calls
    assert len(calls) == 5
    model = get_settings().chat_model
    assert all(c["model"] == model for c in calls)
    # Research calls: tools include submit_answer, tool_choice auto before the last turn.
    assert [t["name"] for t in calls[0]["tools"]][-1] == SUBMIT_ANSWER
    assert calls[0]["tool_choice"] == "auto"
    # Turn 2's input carries turn 1's function calls and outputs, in call order.
    items = calls[1]["input"]
    outs = [i for i in items if i.get("type") == "function_call_output"]
    assert [o["call_id"] for o in outs] == ["c1", "c2"]
    assert [i["call_id"] for i in items if i.get("type") == "function_call"] == ["c1", "c2"]
    assert "Turn 2 of 7" in last_input_text(calls[1])

    # Evaluator sees only the question, the cited chunk text and the answer, with a schema.
    ev_call = calls[3]
    assert ev_call["text"]["format"]["type"] == "json_schema"
    assert ev_call["text"]["format"]["strict"] is True
    assert "tools" not in ev_call
    ev_text = last_input_text(ev_call)
    assert "[1] Albert Einstein\nAlbert Einstein was a German-born physicist." in ev_text
    assert ev_text.index("SOURCE PASSAGES") < ev_text.index("CANDIDATE ANSWER")
    assert "Nobel" not in ev_text  # uncited chunk not shown
    assert len(ev_call["input"]) == 1

    # Responder streams.
    assert calls[4]["stream"] is True
    assert "tools" not in calls[4]


async def test_forced_submit_on_last_turn(fake_env):
    max_turns = get_settings().research_max_turns
    search = lambda: resp(fc("semantic_search", {"queries": ["q"], "top_k": None}))
    client, _ = fake_env(
        [search() for _ in range(max_turns - 1)]
        + [
            submit("Born 1879.", [101]),
            verdict("supported"),
            Stream(["ok"]),
        ]
    )
    evs = await collect()
    research_calls = client.calls[:max_turns]
    assert all(c["tool_choice"] == "auto" for c in research_calls[:-1])
    assert research_calls[-1]["tool_choice"] == {"type": "function", "name": SUBMIT_ANSWER}
    assert "FINAL turn" in research_calls[-1]["input"][-1]["content"]
    turns = [s["turn"] for s in only(evs, "status") if s["stage"] == "research"]
    assert turns == list(range(1, max_turns + 1))
    assert only(evs, "research_answer")[0]["citations"] == [101]
    assert evs[-1].type == "done"


@pytest.mark.parametrize(
    "citations, expected",
    [
        ([999], "not valid chunk ids: [999]"),
        ([], "non-empty list"),
        (["101"], "not an integer"),
    ],
)
async def test_invalid_citations_cost_a_turn(fake_env, citations, expected):
    client, _ = fake_env(
        [
            resp(fc("submit_answer", {"answer": "x", "citations": citations}, "bad")),
            submit("Born 1879.", [101, 101]),
            verdict("supported"),
            Stream(["ok"]),
        ]
    )
    evs = await collect()
    results = only(evs, "tool_result")
    assert results[0]["id"] == "bad" and results[0]["summary"].startswith("rejected")
    # The error went back to the model as the function output, and the next call was turn 2.
    out = next(i for i in client.calls[1]["input"] if i.get("type") == "function_call_output")
    assert out["call_id"] == "bad"
    assert expected in json.loads(out["output"])["error"]
    assert [s["turn"] for s in only(evs, "status") if s["stage"] == "research"] == [1, 2]
    # Duplicate ids are collapsed.
    assert only(evs, "research_answer") == [
        {"round": 1, "answer": "Born 1879.", "citations": [101]}
    ]


async def test_unsupported_then_retry_round_supported(fake_env):
    client, _ = fake_env(
        [
            submit("Born 1879 and won the Nobel in 1921.", [101]),
            verdict("unsupported", "Born 14 March 1879.", "Nothing about a Nobel Prize."),
            resp(fc("fetch", {"chunk_ids": [102], "article_ids": []})),
            submit("Born 1879 and won the Nobel in 1921.", [101, 102]),
            verdict("supported"),
            Stream(["Born 1879 [1], Nobel 1921 [2]."]),
        ]
    )
    evs = await collect("When was Einstein born and when did he win the Nobel?")
    assert [e["verdict"] for e in only(evs, "evaluation")] == ["unsupported", "supported"]
    assert [a["round"] for a in only(evs, "research_answer")] == [1, 2]
    research_status = [
        (s["round"], s["turn"]) for s in only(evs, "status") if s["stage"] == "research"
    ]
    assert research_status == [(1, 1), (2, 1), (2, 2)]  # fresh budget in round 2
    # Round 2 continues the same conversation with the feedback injected.
    r2 = client.calls[2]
    text = last_input_text(r2)
    assert "When was Einstein born" in text
    assert "Nothing about a Nobel Prize." in text
    assert "fresh budget of 7 turns" in text
    assert any(i.get("type") == "function_call" for i in r2["input"])
    assert "Turn 1 of 7" in r2["input"][-1]["content"]
    assert [c["chunk_id"] for c in only(evs, "citations")[0]] == [101, 102]


async def test_max_rounds_exhausted_says_not_found(fake_env):
    rounds = get_settings().max_research_rounds
    script = []
    for _ in range(rounds):
        script += [
            submit("The sources don't say.", [101]),
            verdict("unsupported", "Not stated.", "Not-found answer; try other queries."),
        ]
    client, _ = fake_env(script)  # no responder Stream: the not-found reply is fixed text
    evs = await collect()
    assert [e["verdict"] for e in only(evs, "evaluation")] == ["unsupported"] * rounds
    respond_status = next(s for s in only(evs, "status") if s["stage"] == "respond")
    assert "could not be found" in respond_status["message"]
    assert only(evs, "citations") == [[]]
    assert "".join(t["delta"] for t in only(evs, "token")) == NOT_FOUND_MESSAGE
    assert types_of(evs)[-3:] == ["citations", "token", "done"]
    assert len(client.calls) == rounds * 2


async def test_round_without_submission_counts_as_unsupported(fake_env, monkeypatch):
    monkeypatch.setattr(get_settings(), "research_max_turns", 2)
    client, _ = fake_env(
        [
            resp(msg("I think it's 1879.")),  # no tool call: wastes a turn
            resp(fc("submit_answer", {"answer": "1879", "citations": [5]})),  # forced, invalid
            submit("1879", [101]),
            verdict("supported"),
            Stream(["1879 [1]"]),
        ]
    )
    evs = await collect()
    first_eval = only(evs, "evaluation")[0]
    assert first_eval["verdict"] == "unsupported" and first_eval["round"] == 1
    assert "did not submit a valid answer" in first_eval["feedback"]
    assert "without calling a tool" in client.calls[1]["input"][-2]["content"]
    assert only(evs, "research_answer") == [{"round": 2, "answer": "1879", "citations": [101]}]


async def test_history_is_passed_to_research_and_responder(fake_env):
    history = [
        {"role": "user", "content": "Who was Albert Einstein?"},
        {"role": "assistant", "content": "A German-born physicist [1]."},
    ]
    client, _ = fake_env([submit("1879", [101]), verdict("supported"), Stream(["1879 [1]"])])
    await collect("When was he born?", history)
    research_input = client.calls[0]["input"]
    assert research_input[:3] == [*history, {"role": "user", "content": "When was he born?"}]
    assert client.calls[2]["input"][:2] == history
    # The evaluator gets only the question, no history.
    assert "Who was Albert Einstein" not in last_input_text(client.calls[1])


async def test_exception_becomes_error_then_done(fake_env):
    fake_env([RuntimeError("boom")])
    evs = await collect()
    assert types_of(evs) == ["status", "error", "done"]
    assert "boom" in evs[1].data["message"]


async def test_tool_error_is_returned_to_model(fake_env):
    tools = ToolRecorder()

    async def failing(name, args):
        raise ValueError("db down")

    tools.dispatch = failing
    client, _ = fake_env(
        [
            resp(
                fc("semantic_search", {"queries": ["q"], "top_k": None}, "s1"),
                fc("nonexistent", {}, "n1"),
                fc("fetch", "{not json", "f1"),
            ),
            submit("1879", [101]),
            verdict("supported"),
            Stream(["ok"]),
        ],
        tools,
    )
    evs = await collect()
    outs = {
        i["call_id"]: json.loads(i["output"])
        for i in client.calls[1]["input"]
        if i.get("type") == "function_call_output"
    }
    assert "db down" in outs["s1"]["error"]
    assert "Unknown tool" in outs["n1"]["error"]
    assert "not valid JSON" in outs["f1"]["error"]
    assert only(evs, "tool_call")[2]["arguments"] == {"_raw": "{not json"}


async def test_evaluator_parses_structured_output(fake_env):
    from fakes import CHUNKS

    fake_env([verdict("unsupported", "1879", "Missing Nobel.")])
    result = await evaluate("q", "a", [CHUNKS[101]])
    assert result.verdict == "unsupported" and result.feedback == "Missing Nobel."


async def test_run_cli_prints_trace(fake_env, monkeypatch, capsys):
    import rag.db

    async def noop():
        return None

    monkeypatch.setattr(rag.db, "close_pool", noop)
    fake_env([submit("1879", [101]), verdict("supported"), Stream(["Born ", "1879 [1]."])])
    await orchestrator.run_cli("When was Einstein born?")
    out = capsys.readouterr().out
    assert "-> submit_answer" in out
    assert "verdict: SUPPORTED" in out
    assert "[1] Albert Einstein (chunk 101)" in out
    assert out.rstrip().endswith("Born 1879 [1].")
