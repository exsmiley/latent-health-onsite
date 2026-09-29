import json

import pytest
from fakes import (
    Stream,
    ToolRecorder,
    answer,
    fc,
    final,
    last_input_text,
    msg,
    not_found,
    resp,
    search,
    verdict,
)

from rag.agents import orchestrator
from rag.agents.evaluator import evaluate
from rag.agents.orchestrator import NOT_FOUND_MESSAGE
from rag.agents.research import ANSWER_SCHEMA
from rag.config import get_settings


async def collect(question="When was Einstein born?", history=None):
    return [ev async for ev in orchestrator.run(question, history or [])]


def types_of(evs):
    return [e.type for e in evs]


def only(evs, t):
    return [e.data for e in evs if e.type == t]


def research_turns(evs):
    return [s["turn"] for s in only(evs, "status") if s["stage"] == "research"]


def reply_text(evs):
    return "".join(t["delta"] for t in only(evs, "token"))


def research_calls(client):
    return [c for c in client.calls if "tools" in c]


async def test_happy_path_answered_on_turn_3(fake_env):
    client, tools = fake_env(
        [
            resp(
                fc("semantic_search", {"queries": ["Einstein birth date"], "top_k": None}, "c1"),
                fc("keyword_search", {"queries": ["Einstein born 1879"], "top_k": None}, "c2"),
            ),
            resp(fc("fetch", {"chunk_ids": [101], "article_ids": []}, "c3")),
            answer("Einstein was born on 14 March 1879.", [101]),
            verdict("supported", "14 March 1879", "Passage [1] states it."),
            Stream(["Einstein was born ", "on 14 March 1879 [1]."]),
        ]
    )
    evs = await collect()
    max_turns = get_settings().research_max_turns

    assert types_of(evs) == [
        "status", "tool_call", "tool_call", "tool_result", "tool_result",
        "status", "tool_call", "tool_result",
        "status", "research_answer",
        "status", "evaluation",
        "status", "outcome", "citations", "token", "token", "done",
    ]  # fmt: skip
    msg0 = evs[0].data["message"]
    assert evs[0].data == {"stage": "research", "turn": 1, "max_turns": max_turns, "message": msg0}
    assert evs[1].data == {
        "id": "c1",
        "name": "semantic_search",
        "arguments": {"queries": ["Einstein birth date"], "top_k": None},
        "turn": 1,
    }
    assert only(evs, "tool_result")[:2] == [
        {"id": "c1", "name": "semantic_search", "summary": "1 queries, 1 hits"},
        {"id": "c2", "name": "keyword_search", "summary": "1 queries, 1 hits"},
    ]
    assert only(evs, "research_answer") == [
        {
            "turn": 3,
            "status": "answered",
            "answer": "Einstein was born on 14 March 1879.",
            "citations": [101],
            "reason": "",
        }
    ]
    assert only(evs, "evaluation") == [
        {
            "turn": 3,
            "verdict": "supported",
            "independent_answer": "14 March 1879",
            "feedback": "Passage [1] states it.",
        }
    ]
    assert only(evs, "outcome") == [{"result": "supported", "turns_used": 3}]
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
    assert reply_text(evs) == "Einstein was born on 14 March 1879 [1]."
    statuses = [(s["stage"], s["turn"]) for s in only(evs, "status")]
    assert statuses == [
        ("research", 1), ("research", 2), ("research", 3), ("evaluate", 3), ("respond", None),
    ]  # fmt: skip
    assert all(s["max_turns"] == max_turns for s in only(evs, "status"))

    # Both searches in turn 1 ran concurrently.
    assert tools.max_active == 2

    calls = client.calls
    assert len(calls) == 5
    model = get_settings().chat_model
    assert all(c["model"] == model for c in calls)
    # Research calls: only the retrieval tools, auto tool choice, structured final answer.
    for c in calls[:3]:
        assert [t["name"] for t in c["tools"]] == ["semantic_search", "keyword_search", "fetch"]
        assert c["tool_choice"] == "auto"
        fmt = c["text"]["format"]
        assert fmt["type"] == "json_schema" and fmt["strict"] is True
        assert fmt["schema"] == ANSWER_SCHEMA
    # Turn 2's input carries turn 1's function calls and outputs, in call order.
    items = calls[1]["input"]
    outs = [i for i in items if i.get("type") == "function_call_output"]
    assert [o["call_id"] for o in outs] == ["c1", "c2"]
    assert [i["call_id"] for i in items if i.get("type") == "function_call"] == ["c1", "c2"]
    assert f"Turn 2 of {max_turns}" in last_input_text(calls[1])

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


async def test_answer_after_evaluator_rejects_uses_same_budget(fake_env):
    max_turns = get_settings().research_max_turns
    client, _ = fake_env(
        [
            search(),
            answer("Born 1879 and won the Nobel in 1921.", [101]),
            verdict("unsupported", "Born 14 March 1879.", "Nothing about a Nobel Prize."),
            resp(fc("fetch", {"chunk_ids": [102], "article_ids": []})),
            answer("Born 1879 and won the Nobel in 1921.", [101, 102]),
            verdict("supported"),
            Stream(["Born 1879 [1], Nobel 1921 [2]."]),
        ]
    )
    evs = await collect("When was Einstein born and when did he win the Nobel?")
    assert [e["verdict"] for e in only(evs, "evaluation")] == ["unsupported", "supported"]
    assert [e["turn"] for e in only(evs, "evaluation")] == [2, 4]
    assert [a["turn"] for a in only(evs, "research_answer")] == [2, 4]
    assert research_turns(evs) == [1, 2, 3, 4]  # one budget, no reset
    assert only(evs, "outcome") == [{"result": "supported", "turns_used": 4}]
    # Turn 3 continues the same conversation with the feedback and the turns left.
    t3 = research_calls(client)[2]
    text = last_input_text(t3)
    assert "When was Einstein born" in text
    assert "Nothing about a Nobel Prize." in text
    assert f"You have {max_turns - 2} turns left." in text
    assert any(i.get("type") == "function_call" for i in t3["input"])
    assert f"Turn 3 of {max_turns}" in t3["input"][-1]["content"]
    assert [c["chunk_id"] for c in only(evs, "citations")[0]] == [101, 102]


async def test_not_found_ends_immediately_without_evaluation(fake_env):
    client, _ = fake_env([search("Einstein pet"), not_found("No pet is mentioned.")])
    evs = await collect("What was Einstein's goldfish called?")
    assert len(client.calls) == 2  # two research turns; no evaluator, no responder
    assert only(evs, "evaluation") == []
    assert only(evs, "research_answer") == [
        {
            "turn": 2,
            "status": "not_found",
            "answer": "",
            "citations": [],
            "reason": "No pet is mentioned.",
        }
    ]
    assert only(evs, "outcome") == [{"result": "not_found", "turns_used": 2}]
    assert only(evs, "citations") == [[]]
    assert reply_text(evs) == NOT_FOUND_MESSAGE
    assert types_of(evs)[-5:] == ["status", "outcome", "citations", "token", "done"]
    respond_status = only(evs, "status")[-1]
    assert respond_status["stage"] == "respond" and respond_status["turn"] is None


async def test_not_found_rejected_before_any_search(fake_env):
    client, _ = fake_env([not_found(), search("Einstein pet"), not_found()])
    evs = await collect()
    answers = only(evs, "research_answer")
    assert [a["status"] for a in answers] == ["invalid", "not_found"]
    assert "without searching" in answers[0]["reason"]
    assert [a["turn"] for a in answers] == [1, 3]
    # The rejection went back to the model before turn 2.
    assert "without searching" in last_input_text(client.calls[1])
    assert len(client.calls) == 3
    assert only(evs, "outcome") == [{"result": "not_found", "turns_used": 3}]
    assert reply_text(evs) == NOT_FOUND_MESSAGE


@pytest.mark.parametrize(
    "reply, expected, shown_citations",
    [
        ({"status": "answered", "answer": "x", "citations": [999], "reason": ""},
         "not valid chunk ids: [999]", [999]),
        ({"status": "answered", "answer": "x", "citations": [], "reason": ""},
         "non-empty list", []),
        ({"status": "answered", "answer": "x", "citations": ["101"], "reason": ""},
         "not an integer", []),
        ({"status": "answered", "answer": "", "citations": [101], "reason": ""},
         "non-empty string", [101]),
        ("I think it's 1879.", "not valid JSON", []),
        ({"status": "maybe"}, '"answered" or "not_found"', []),
        ("", "empty", []),
    ],
)  # fmt: skip
async def test_invalid_answer_costs_a_turn(fake_env, reply, expected, shown_citations):
    client, _ = fake_env(
        [
            final(reply),
            answer("Born 1879.", [101, 101]),
            verdict("supported"),
            Stream(["ok"]),
        ]
    )
    evs = await collect()
    answers = only(evs, "research_answer")
    assert answers[0]["turn"] == 1 and answers[0]["status"] == "invalid"
    assert expected in answers[0]["reason"]
    assert answers[0]["citations"] == shown_citations
    # The error went back to the model, and the next call was turn 2.
    feedback = client.calls[1]["input"][-2]
    assert feedback["role"] == "developer" and expected in feedback["content"]
    assert "rejected" in feedback["content"]
    assert research_turns(evs) == [1, 2]
    # Duplicate ids are collapsed.
    assert answers[1] == {
        "turn": 2,
        "status": "answered",
        "answer": "Born 1879.",
        "citations": [101],
        "reason": "",
    }
    assert only(evs, "outcome") == [{"result": "supported", "turns_used": 2}]


async def test_out_of_turns_gives_not_found_and_last_turn_is_not_forced(fake_env):
    max_turns = get_settings().research_max_turns
    client, tools = fake_env([search() for _ in range(max_turns)])
    evs = await collect()
    assert len(client.calls) == max_turns
    assert all(c["tool_choice"] == "auto" for c in client.calls)
    last_note = client.calls[-1]["input"][-1]["content"]
    assert "LAST turn" in last_note and "couldn't be found" in last_note
    assert research_turns(evs) == list(range(1, max_turns + 1))
    # Tool calls on the last turn can't be followed up, so they aren't run.
    assert len(tools.calls) == max_turns - 1
    assert only(evs, "tool_result")[-1]["summary"] == "not run: no turns left"
    assert only(evs, "outcome") == [{"result": "out_of_turns", "turns_used": max_turns}]
    assert only(evs, "citations") == [[]]
    assert reply_text(evs) == NOT_FOUND_MESSAGE
    assert types_of(evs)[-5:] == ["status", "outcome", "citations", "token", "done"]


async def test_rejected_on_last_turn_is_out_of_turns(fake_env, monkeypatch):
    monkeypatch.setattr(get_settings(), "research_max_turns", 2)
    client, _ = fake_env(
        [
            search(),
            answer("Goldie.", [101]),
            verdict("unsupported", "Not stated.", "No passage names a goldfish."),
        ]
    )
    evs = await collect()
    assert only(evs, "outcome") == [{"result": "out_of_turns", "turns_used": 2}]
    assert reply_text(evs) == NOT_FOUND_MESSAGE
    assert len(client.calls) == 3


async def test_message_with_function_calls_is_a_tool_turn(fake_env):
    client, tools = fake_env(
        [
            resp(
                msg("I'll search for Einstein's birth date first.", phase="commentary"),
                fc("semantic_search", {"queries": ["Einstein birth"], "top_k": None}, "s1"),
            ),
            # Even a JSON answer next to a function call is ignored: it's still a tool turn.
            resp(
                msg(json.dumps({"status": "answered", "answer": "x", "citations": [101]})),
                fc("fetch", {"chunk_ids": [101], "article_ids": []}, "f1"),
            ),
            resp(
                msg("Reading done; answering now.", phase="commentary"),
                msg(
                    json.dumps(
                        {"status": "answered", "answer": "1879", "citations": [101], "reason": ""}
                    ),
                    phase="final_answer",
                ),
            ),
            verdict("supported"),
            Stream(["1879 [1]"]),
        ]
    )
    evs = await collect()
    assert [n for n, _ in tools.calls] == ["semantic_search", "fetch"]
    answers = only(evs, "research_answer")
    assert [(a["turn"], a["status"], a["answer"]) for a in answers] == [(3, "answered", "1879")]
    # The commentary message is kept in the conversation (with its phase) for the next turn.
    kept = [i for i in client.calls[1]["input"] if i.get("type") == "message"]
    assert kept and kept[0]["phase"] == "commentary"
    assert only(evs, "outcome") == [{"result": "supported", "turns_used": 3}]


async def test_history_is_passed_to_research_and_responder(fake_env):
    history = [
        {"role": "user", "content": "Who was Albert Einstein?"},
        {"role": "assistant", "content": "A German-born physicist [1]."},
    ]
    client, _ = fake_env([answer("1879", [101]), verdict("supported"), Stream(["1879 [1]"])])
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
            answer("1879", [101]),
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
    fake_env(
        [
            search("Einstein born"),
            answer("1879", [101]),
            verdict("supported"),
            Stream(["Born ", "1879 [1]."]),
        ]
    )
    await orchestrator.run_cli("When was Einstein born?")
    out = capsys.readouterr().out
    assert "[research 1/7]" in out
    assert "-> semantic_search" in out
    assert "answered (turn 2)" in out
    assert "verdict: SUPPORTED" in out
    assert "[outcome] supported after 2 turn(s)" in out
    assert "[1] Albert Einstein (chunk 101)" in out
    assert out.rstrip().endswith("Born 1879 [1].")
    assert "round" not in out


async def test_evaluator_gets_standalone_question_for_follow_ups(fake_env):
    history = [
        {"role": "user", "content": "Who was Albert Einstein?"},
        {"role": "assistant", "content": "A German-born physicist [1]."},
    ]
    client, _ = fake_env(
        [
            final(
                {
                    "question": "When was Albert Einstein born?",
                    "status": "answered",
                    "answer": "14 March 1879.",
                    "citations": [101],
                    "reason": "",
                }
            ),
            verdict("supported"),
            Stream(["14 March 1879 [1]."]),
        ]
    )
    evs = await collect("When was he born?", history)
    ev_text = last_input_text(client.calls[1])
    assert "QUESTION:\nWhen was Albert Einstein born?" in ev_text
    assert "When was he born?" not in ev_text
    assert "Who was Albert Einstein" not in ev_text  # still no history for the evaluator
    assert only(evs, "outcome")[0]["result"] == "supported"
    assert "question" in ANSWER_SCHEMA["required"]


async def test_citation_check_failure_is_retryable_not_fatal(fake_env, monkeypatch):
    from fakes import fake_fetch

    from rag.tools import fetch as fetch_tool

    fake_env(
        [
            search(),
            answer("1879", [101]),
            answer("1879", [101]),
            verdict("supported"),
            Stream(["ok"]),
        ]
    )
    calls = {"n": 0}

    async def flaky_fetch(chunk_ids=(), article_ids=()):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("pool timeout")
        return await fake_fetch(list(chunk_ids), list(article_ids))

    monkeypatch.setattr(fetch_tool, "fetch", flaky_fetch)
    evs = await collect()
    answers = only(evs, "research_answer")
    assert [a["status"] for a in answers] == ["invalid", "answered"]
    assert "temporary error (ConnectionError)" in answers[0]["reason"]
    assert only(evs, "error") == []
    assert only(evs, "outcome")[0] == {"result": "supported", "turns_used": 3}


@pytest.mark.parametrize("raw", ["", "not json", '{"verdict": "maybe"}'])
async def test_unusable_evaluator_output_counts_as_unsupported(fake_env, raw):
    client, _ = fake_env(
        [
            answer("1879", [101]),
            resp(msg(raw)),  # evaluator: empty / refusal-like / malformed
            answer("1879", [101]),
            verdict("supported"),
            Stream(["ok"]),
        ]
    )
    evs = await collect()
    evals = only(evs, "evaluation")
    assert [e["verdict"] for e in evals] == ["unsupported", "supported"]
    assert "could not be verified" in evals[0]["feedback"]
    assert only(evs, "error") == []
    assert "could not be verified" in last_input_text(client.calls[2])
