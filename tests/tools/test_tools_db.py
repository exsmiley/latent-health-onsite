"""Tool tests against a dedicated, seeded `rag_test_tools` database."""

import json

import pytest

from rag.tools.fetch import fetch
from rag.tools.keyword import keyword_search
from rag.tools.registry import dispatch
from rag.tools.search import semantic_search

from .conftest import ARTICLES, CHUNKS, LONG_TEXT

pytestmark = [pytest.mark.db, pytest.mark.usefixtures("tools_db")]


def ids(qr):
    return [h.chunk_id for h in qr.hits]


# ---- semantic ------------------------------------------------------------------------------


async def test_semantic_ordering_scores_and_join(fake_embed):
    [res] = await semantic_search(["nobel prize winners"], top_k=2)
    assert res.query == "nobel prize winners"
    assert ids(res) == [102, 104]
    assert res.hits[0].score == pytest.approx(1.0)
    assert res.hits[1].score == pytest.approx(0.8)
    assert res.hits[0].title == "Albert Einstein"
    assert res.hits[0].section == "Nobel Prize"
    assert res.hits[1].title == "Marie Curie" and res.hits[1].section is None


async def test_semantic_top_k_and_null_embeddings_excluded(fake_embed):
    [res] = await semantic_search(["relativity"], top_k=10)
    assert ids(res)[0] == 101
    assert sorted(ids(res)) == [101, 102, 104, 105]  # 103 has no embedding
    [res1] = await semantic_search(["relativity"], top_k=1)
    assert ids(res1) == [101]


async def test_semantic_one_embed_call_deduped_and_ordered(fake_embed):
    results = await semantic_search(
        [" how did curie die ", "radioactivity chemist", "how did curie die", ""], top_k=1
    )
    assert fake_embed.calls == [["how did curie die", "radioactivity chemist"]]
    assert [r.query for r in results] == ["how did curie die", "radioactivity chemist"]
    assert [ids(r) for r in results] == [[105], [104]]


async def test_semantic_empty_queries_skip_embedding(fake_embed):
    assert await semantic_search(["  "]) == []
    assert fake_embed.calls == []


# ---- keyword -------------------------------------------------------------------------------


async def test_keyword_ranking():
    [res] = await keyword_search(["Nobel Prize"], top_k=5)
    # 102 mentions it in the section heading and the text; 104 only has "Nobel".
    assert ids(res) == [102]
    [res] = await keyword_search(["nobel"], top_k=5)
    assert ids(res) == [102, 104]
    assert res.hits[0].score > res.hits[1].score > 0


async def test_keyword_phrase_or_exclude():
    [phrase] = await keyword_search(['"theory of relativity"'])
    assert ids(phrase) == [101]
    [wrong_order] = await keyword_search(['"relativity theory"'])
    assert ids(wrong_order) == []
    [unquoted] = await keyword_search(["relativity theory"])
    assert ids(unquoted) == [101]
    [excl] = await keyword_search(["physicist -Curie"])
    assert ids(excl) == [101]
    [either] = await keyword_search(["anaemia OR relativity"])
    assert sorted(ids(either)) == [101, 105]


async def test_keyword_top_k_stopwords_and_unembedded_chunks():
    [res] = await keyword_search(["nobel"], top_k=1)
    assert ids(res) == [102]
    [stop] = await keyword_search(["the of and"])
    assert stop.hits == []
    # keyword search sees chunks that have no embedding yet, and blurbs are truncated
    [q] = await keyword_search(["quantum mechanics"])
    assert ids(q) == [103]
    assert q.hits[0].blurb == " ".join(LONG_TEXT.split()[:40]) + "…"


async def test_keyword_concurrent_multiple_queries():
    results = await keyword_search(["anaemia", "relativity", "anaemia"])
    assert [r.query for r in results] == ["anaemia", "relativity"]
    assert [ids(r) for r in results] == [[105], [101]]


# ---- fetch ---------------------------------------------------------------------------------


async def test_fetch_chunks_order_dedupe_missing():
    res = await fetch(chunk_ids=[105, 101, 9999, 101])
    assert [c.chunk_id for c in res.chunks] == [105, 101]
    assert res.missing_chunk_ids == [9999]
    c = res.chunks[0]
    assert (c.article_id, c.title, c.section, c.chunk_index) == (2, "Marie Curie", "Death", 1)
    assert c.text == "She died in 1934 from aplastic anaemia."
    assert c.url.endswith("Marie%20Curie")
    assert res.articles == [] and res.missing_article_ids == []


async def test_fetch_articles_full_text_and_chunk_ids():
    res = await fetch(article_ids=[2, 1, 777, 3])
    assert [a.article_id for a in res.articles] == [2, 1, 3]
    assert res.missing_article_ids == [777]
    by_id = {a.article_id: a for a in res.articles}
    assert by_id[1].chunk_ids == [101, 102, 103]
    assert by_id[2].chunk_ids == [104, 105]
    assert by_id[3].chunk_ids == []
    assert by_id[1].text == ARTICLES[0][3]


async def test_fetch_empty():
    res = await fetch()
    assert res.chunks == res.articles == res.missing_chunk_ids == res.missing_article_ids == []


# ---- dispatch ------------------------------------------------------------------------------


async def test_dispatch_semantic_compact_json(fake_embed):
    out = await dispatch("semantic_search", {"queries": ["nobel prize winners"], "top_k": 2})
    assert "\n" not in out and ": " not in out.replace('": "', "")  # compact
    data = json.loads(out)
    assert [h["chunk_id"] for h in data[0]["hits"]] == [102, 104]
    assert "section" not in data[0]["hits"][1]  # exclude_none drops null sections
    assert set(data[0]["hits"][0]) == {
        "chunk_id",
        "article_id",
        "title",
        "section",
        "score",
        "blurb",
        "text",
        "token_count",
    }
    hit = data[0]["hits"][0]
    assert hit["text"] == CHUNKS[1][4]  # full text for the agent layer to show or trim
    assert hit["token_count"] == len(CHUNKS[1][4].split())


async def test_dispatch_keyword_default_top_k_and_string_args():
    data = json.loads(await dispatch("keyword_search", '{"queries": ["nobel"], "top_k": null}'))
    assert [h["chunk_id"] for h in data[0]["hits"]] == [102, 104]


async def test_dispatch_fetch():
    data = json.loads(await dispatch("fetch", {"chunk_ids": [102, 5000], "article_ids": [3]}))
    assert [c["chunk_id"] for c in data["chunks"]] == [102]
    assert data["articles"][0]["title"] == "Paris"
    assert data["missing_chunk_ids"] == [5000]
    assert data["missing_article_ids"] == []
