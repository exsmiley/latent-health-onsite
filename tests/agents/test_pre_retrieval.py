import json

import pytest
from fakes import CHUNKS, Stream, ToolRecorder, answer, not_found, search, verdict

from rag.agents import orchestrator, pre_retrieval
from rag.config import get_settings


async def collect(question="When was Einstein born?", history=None):
    return [ev async for ev in orchestrator.run(question, history or [])]


def only(evs, t):
    return [e.data for e in evs if e.type == t]


async def test_pre_retrieval_runs_before_turn_1_as_turn_0(fake_env):
    client, tools = fake_env(
        [answer("14 March 1879.", [101]), verdict("supported"), Stream(["1879 [1]"])],
        pre_retrieve=True,
    )
    evs = await collect("When was Albert Einstein born in 1879?")
    max_turns = get_settings().research_max_turns

    assert [e.type for e in evs[:7]] == [
        "status", "tool_call", "tool_call", "tool_result", "tool_result", "tool_call",
        "tool_result",
    ]  # fmt: skip
    assert evs[0].data["stage"] == "research" and evs[0].data["turn"] == 0
    calls = only(evs, "tool_call")
    assert [(c["name"], c["turn"]) for c in calls] == [
        ("semantic_search", 0), ("keyword_search", 0), ("fetch", 0),
    ]  # fmt: skip
    assert calls[0]["arguments"] == {
        "queries": ["When was Albert Einstein born in 1879?"],
        "top_k": pre_retrieval.SEMANTIC_TOP_K,
    }
    assert calls[1]["arguments"]["queries"] == ['"Albert Einstein" 1879']
    assert calls[2]["arguments"] == {"chunk_ids": [101], "article_ids": []}
    assert [n for n, _ in tools.calls] == ["semantic_search", "keyword_search", "fetch"]

    # Not a turn: the model answered on turn 1 and one turn was used.
    assert only(evs, "research_answer")[0]["turn"] == 1
    assert only(evs, "outcome") == [{"result": "supported", "turns_used": 1}]
    research = [s["turn"] for s in only(evs, "status") if s["stage"] == "research"]
    assert research == [0, 1]

    # Turn 1 sees the question, then the three calls as function_call/output pairs.
    items = client.calls[0]["input"]
    assert items[0] == {"role": "user", "content": "When was Albert Einstein born in 1879?"}
    pairs = items[1:7]
    assert [i["type"] for i in pairs] == ["function_call", "function_call_output"] * 3
    assert [i["call_id"] for i in pairs[::2]] == [i["call_id"] for i in pairs[1::2]]
    assert json.loads(pairs[4]["arguments"]) == calls[2]["arguments"]
    assert CHUNKS[101].text in pairs[5]["output"]
    assert f"Turn 1 of {max_turns}" in items[7]["content"]


async def test_pre_retrieved_chunks_are_shown_once(fake_env):
    client, _ = fake_env(
        [search(), answer("14 March 1879.", [101]), verdict("supported"), Stream(["1879 [1]"])],
        pre_retrieve=True,
    )
    await collect("When was Albert Einstein born in 1879?")

    # The pre-retrieval searches show a blurb (then a reference to the repeat); the full text
    # comes once, from its fetch.
    pairs = client.calls[0]["input"][1:7]
    semantic_hit = json.loads(pairs[1]["output"])[0]["hits"][0]
    keyword_hit = json.loads(pairs[3]["output"])[0]["hits"][0]
    assert "text" not in semantic_hit and semantic_hit["blurb"]
    assert keyword_hit["seen"] is True and "text" not in keyword_hit
    assert CHUNKS[101].text in pairs[5]["output"]

    # A search on turn 1 only references the fetched chunk.
    turn_1_output = client.calls[1]["input"][-2]["output"]
    assert json.loads(turn_1_output)[0]["hits"][0]["seen"] is True


async def test_not_found_allowed_on_turn_1_after_pre_retrieval(fake_env):
    fake_env([not_found("No goldfish.")], pre_retrieve=True)
    evs = await collect("What was Einstein's goldfish called?")
    assert [a["status"] for a in only(evs, "research_answer")] == ["not_found"]
    assert only(evs, "outcome") == [{"result": "not_found", "turns_used": 1}]


async def test_failed_pre_retrieval_does_not_count_as_searched(fake_env):
    tools = ToolRecorder()

    async def failing(name, args):
        return json.dumps({"error": f"{name} failed: db down"})

    tools.dispatch = failing
    tools.summarize = lambda name, result: "error"
    fake_env([not_found(), search(), not_found()], tools, pre_retrieve=True)
    evs = await collect("What was Einstein's goldfish called?")
    # Nothing to fetch, and a "not_found" still needs a real search first.
    assert [c["name"] for c in only(evs, "tool_call") if c["turn"] == 0] == [
        "semantic_search",
        "keyword_search",
    ]
    assert [a["status"] for a in only(evs, "research_answer")] == ["invalid", "not_found"]


async def test_follow_up_searches_with_the_previous_exchange(fake_env):
    history = [
        {"role": "user", "content": "Who discovered radium?"},
        {"role": "assistant", "content": "Marie Curie and Pierre Curie discovered it [1]."},
    ]
    fake_env([answer("1934", [101]), verdict("supported"), Stream(["ok"])], pre_retrieve=True)
    evs = await collect("When did she die?", history)
    semantic, keyword = only(evs, "tool_call")[:2]
    assert semantic["arguments"]["queries"] == [
        "Who discovered radium? Marie Curie and Pierre Curie discovered it . When did she die?"
    ]
    # The question has no names, so the keyword query comes from the previous exchange.
    assert keyword["arguments"]["queries"] == ['"Marie Curie" "Pierre Curie"']


@pytest.mark.parametrize(
    "question, expected",
    [
        ("When was Albert Einstein born?", '"Albert Einstein"'),
        ("How old was Martin Luther when he married Katherine von Bora?",
         '"Martin Luther" Katherine Bora'),
        ("Put Mahatma Gandhi, Muhammad Ali Jinnah and Jawaharlal Nehru in order of birth.",
         '"Mahatma Gandhi" "Muhammad Ali Jinnah" "Jawaharlal Nehru"'),
        ("Who wrote Newton's Principia in 1687?", "Newton Principia 1687"),
        ("Which volcano erupted most recently?", ""),
        ("Is Mars, or Mars, Venus, Jupiter, Saturn or Uranus bigger?",
         "Mars Venus Jupiter Saturn"),
    ],
)  # fmt: skip
def test_keyword_query(question, expected):
    assert pre_retrieval.keyword_query(question) == expected


def test_full_text_ids_and_long_chunks():
    sem = json.dumps([{"query": "q", "hits": [{"chunk_id": i} for i in (1, 2, 3, 4, 5, 6, 7)]}])
    kw = json.dumps([{"query": "q", "hits": [{"chunk_id": i} for i in (2, 8, 9)]}])
    assert pre_retrieval.full_text_ids(sem, kw) == [1, 2, 3, 4, 5, 8]
    assert pre_retrieval.full_text_ids('{"error": "x"}') == []

    long_text = "x" * (pre_retrieval.FULL_TEXT_MAX_CHARS + 1)
    out = json.dumps(
        {
            "chunks": [{"chunk_id": 1, "text": "short"}, {"chunk_id": 2, "text": long_text}],
            "articles": [],
            "missing_chunk_ids": [],
            "missing_article_ids": [],
        }
    )
    ids, trimmed = pre_retrieval.drop_long_chunks(out)
    assert ids == [1]
    assert [c["chunk_id"] for c in json.loads(trimmed)["chunks"]] == [1]
