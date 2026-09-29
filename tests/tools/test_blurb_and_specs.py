"""Pure tests (no database): blurbs, TOOL_SPECS schema, dispatch validation, summaries."""

import json

import pytest

from rag.tools.blurb import make_blurb, normalize_queries
from rag.tools.models import FetchResult, QueryResults, SearchHit
from rag.tools.registry import TOOL_SPECS, _SearchArgs, dispatch, summarize


def test_blurb_short_text_is_normalized_without_ellipsis():
    assert make_blurb("  a\n\nb\tc  ", 5) == "a b c"


def test_blurb_truncates_with_ellipsis():
    text = " ".join(str(i) for i in range(50))
    assert make_blurb(text, 3) == "0 1 2…"
    assert make_blurb(text).split() == [str(i) for i in range(39)] + ["39…"]  # default 40


def test_blurb_exact_length_no_ellipsis():
    assert make_blurb("a b c", 3) == "a b c"


def test_normalize_queries():
    assert normalize_queries([" a ", "a", "", "  ", "b  c", "b c"]) == ["a", "b c"]


# ---- TOOL_SPECS --------------------------------------------------------------------------


def _check_strict(schema: dict, path: str) -> None:
    t = schema.get("type")
    types = t if isinstance(t, list) else [t]
    if "object" in types:
        props = schema.get("properties", {})
        assert schema.get("additionalProperties") is False, f"{path}: additionalProperties"
        assert sorted(schema.get("required", [])) == sorted(props), f"{path}: required"
        for k, sub in props.items():
            _check_strict(sub, f"{path}.{k}")
    if "array" in types:
        assert "items" in schema, f"{path}: items"
        _check_strict(schema["items"], f"{path}[]")
    assert t is not None, f"{path}: missing type"


def test_tool_specs_are_strict_responses_api_tools():
    names = [s["name"] for s in TOOL_SPECS]
    assert names == ["semantic_search", "keyword_search", "fetch"]
    for spec in TOOL_SPECS:
        assert set(spec) == {"type", "name", "description", "parameters", "strict"}
        assert spec["type"] == "function"
        assert spec["strict"] is True
        assert len(spec["description"]) > 50
        assert spec["parameters"]["type"] == "object"
        _check_strict(spec["parameters"], spec["name"])
    json.dumps(TOOL_SPECS)  # serializable


def test_fetch_description_warns_about_article_citations():
    desc = next(s for s in TOOL_SPECS if s["name"] == "fetch")["description"]
    assert "chunk id" in desc and "will not accept whole articles" in desc


def test_top_k_clamped_and_defaulted():
    assert _SearchArgs(queries=["x"], top_k=50).k() == 10
    assert _SearchArgs(queries=["x"], top_k=0).k() == 1
    assert _SearchArgs(queries=["x"], top_k=None).k() == 5


# ---- dispatch errors (validation happens before any DB/embedding call) -------------------


@pytest.mark.parametrize(
    "name,args,fragment",
    [
        ("nope", {}, "unknown tool"),
        ("semantic_search", {}, "queries"),
        ("semantic_search", {"queries": []}, "at least one"),
        ("semantic_search", {"queries": ["  ", ""]}, "at least one"),
        ("semantic_search", {"queries": ["a", "b", "c", "d", "e", "f"]}, "at most 5"),
        ("keyword_search", {"queries": "just a string"}, "queries"),
        ("keyword_search", {"queries": ["a"], "top_k": "many"}, "top_k"),
        ("keyword_search", {"queries": ["a"], "bogus": 1}, "bogus"),
        ("fetch", {"chunk_ids": [], "article_ids": []}, "at least one"),
        ("fetch", {"chunk_ids": ["abc"], "article_ids": []}, "chunk_ids"),
        ("fetch", {"chunk_ids": list(range(51)), "article_ids": []}, "at most 50"),
        ("fetch", "{not json", "not valid JSON"),
        ("fetch", [1, 2], "JSON object"),
    ],
)
async def test_dispatch_errors_return_json(name, args, fragment):
    out = await dispatch(name, args)
    err = json.loads(out)
    assert set(err) == {"error"}
    assert fragment in err["error"]


async def test_dispatch_tool_failure_is_reported(monkeypatch):
    async def boom(texts):
        raise RuntimeError("embeddings down")

    monkeypatch.setattr("rag.llm.embed_texts", boom)
    err = json.loads(await dispatch("semantic_search", {"queries": ["x"], "top_k": None}))
    assert "embeddings down" in err["error"]


# ---- summarize -------------------------------------------------------------------------


def test_summarize_search_and_fetch():
    hit = SearchHit(chunk_id=1, article_id=1, title="t", section=None, score=1.0, blurb="b")
    res = [QueryResults(query="a", hits=[hit, hit]), QueryResults(query="b", hits=[hit])]
    assert summarize("semantic_search", res) == "2 queries, 3 hits"
    one = json.dumps([{"query": "a", "hits": [hit.model_dump()]}])
    assert summarize("keyword_search", one) == "1 query, 1 hit"
    fr = FetchResult(chunks=[], articles=[], missing_chunk_ids=[5], missing_article_ids=[])
    assert summarize("fetch", fr) == "fetched 0 chunks, 0 articles, 1 missing"
    assert summarize("fetch", '{"error":"boom"}') == "error: boom"
    assert summarize("fetch", "not json") == "fetch done"
