"""Pure unit tests for rag.ingest.chunker (no database)."""

import random
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from rag.ingest import chunker
from rag.ingest.chunker import (
    MAX_PARAGRAPH_TOKENS,
    chunk_article,
    count_tokens,
    is_heading,
    split_blocks,
)

DATASET = Path(__file__).resolve().parents[2] / "data" / "simplewiki-20220301.parquet"


def para(n_words: int, seed: str = "w") -> str:
    """A one-line paragraph of roughly n_words tokens, ending in a period."""
    return " ".join(f"{seed}{i}" for i in range(n_words)) + "."


# --- heading heuristic -----------------------------------------------------------------------


HEADINGS = [
    "The Month ",
    "Types of art ",
    "References",
    "Events in April",
    'What "art" means ',
    "Early life",
    "1) Cognitive function",
    "Conquest by Edward I and brief independence under Owain Glyndŵr ",  # 9 words + trailing space
]


@pytest.mark.parametrize("line", HEADINGS)
def test_is_heading_true(line):
    assert is_heading(line)


@pytest.mark.parametrize(
    "line",
    [
        " Plastic art",  # list item
        " April 1 - April Fools' Day",  # list item
        "Angola is divided into eighteen provinces.",  # short sentence
        "There are three regions in Belgium:",  # list intro
        "Is philosophy good or bad? ",
        "Fire is both hot and dry;",
        'In some cultures, a person has a one-part name, such as "ShiningWater." ',
        "This line is definitely much too long to be any kind of section heading at all",
        "|-",
        "   ",
        "",
        "and then it continued",  # lowercase fragment
    ],
)
def test_is_heading_false(line):
    assert not is_heading(line)


def test_heading_as_first_line_of_paragraph_is_peeled():
    blocks = split_blocks("Lead text.\n\nApril in poetry \nPoets use April.\nMore text here.")
    assert blocks == [
        ("para", "Lead text."),
        ("heading", "April in poetry"),
        ("para", "Poets use April.\nMore text here."),
    ]


def test_category_block_is_not_peeled():
    text = "Body.\n\n1930 births\nLiving people\nAmerican movie actors"
    assert split_blocks(text) == [
        ("para", "Body."),
        ("para", "1930 births\nLiving people\nAmerican movie actors"),
    ]


def test_list_items_keep_leading_space_and_are_not_headings():
    blocks = split_blocks("Fixed Events \n\n April 1 - Fools Day\n April 2 - Other day")
    assert blocks == [
        ("heading", "Fixed Events"),
        ("para", " April 1 - Fools Day\n April 2 - Other day"),
    ]


# --- packing rules ---------------------------------------------------------------------------


def test_empty_article():
    assert chunk_article("T", "") == []
    assert chunk_article("T", "  \n\n \n") == []


def test_no_mid_paragraph_split_and_greedy_packing():
    paras = [para(100, f"p{i}_") for i in range(7)]  # ~200 tokens each
    chunks = chunk_article("T", "\n\n".join(paras), target_tokens=350, min_tokens=100)
    joined = [p for c in chunks for p in c.text.split("\n\n")]
    assert joined == paras
    for c in chunks:
        assert c.token_count <= 350 or "\n\n" not in c.text
        assert c.token_count == count_tokens(c.text)


def test_greedy_packs_up_to_target():
    paras = [para(20, f"p{i}_") for i in range(30)]  # ~45 tokens each
    chunks = chunk_article("T", "\n\n".join(paras), target_tokens=350, min_tokens=100)
    assert len(chunks) > 1
    for c in chunks[:-1]:
        assert c.token_count <= 350
        # the next paragraph would not have fit
        nxt = chunks[c.chunk_index + 1].text.split("\n\n")[0]
        assert count_tokens(c.text + "\n\n" + nxt) > 350


def test_heading_starts_new_chunk_when_current_is_big_enough():
    lead = para(80, "lead")  # >100 tokens
    text = f"{lead}\n\nHistory \n\n{para(10, 'h')}\n\nLater life\n{para(10, 'l')}"
    chunks = chunk_article("Person", text, target_tokens=350, min_tokens=100)
    assert [c.section for c in chunks] == [None, "History"]
    # "Later life" follows a chunk below min_tokens, so it keeps packing into "History"
    assert chunks[1].text == f"{para(10, 'h')}\n\n{para(10, 'l')}"
    assert "History" not in chunks[1].text and "Later life" not in chunks[1].text


def test_heading_below_min_keeps_packing_and_section_is_first_paragraphs():
    text = f"{para(10, 'a')}\n\nHistory \n\n{para(10, 'b')}"
    chunks = chunk_article("T", text, target_tokens=350, min_tokens=100)
    assert len(chunks) == 1
    assert chunks[0].section is None
    assert chunks[0].text == f"{para(10, 'a')}\n\n{para(10, 'b')}"


def test_section_assignment_and_consecutive_headings():
    big = para(80, "x")
    text = (
        f"{big}\n\nEvents in April\n\nFixed Events \n\n{big}\n\n"
        f"Moveable Events \n{big}\n\nReferences\n\nOther websites \n\n{big}"
    )
    chunks = chunk_article("April", text, target_tokens=350, min_tokens=100)
    assert [c.section for c in chunks] == [
        None,
        "Events in April > Fixed Events",
        "Moveable Events",
        "Other websites",  # "References" is a known-empty section: replaced, not joined
    ]


def test_embed_text_format():
    big = para(80, "x")
    chunks = chunk_article("Art", f"{big}\n\nHistory of art \n{big}", min_tokens=100)
    assert chunks[0].embed_text == f"Art\n\n{big}"
    assert chunks[1].embed_text == f"Art > History of art\n\n{big}"
    assert [c.chunk_index for c in chunks] == [0, 1]


def test_oversize_paragraph_is_its_own_chunk_as_is():
    small, huge = para(10, "s"), para(400, "h")  # huge is ~800 tokens
    chunks = chunk_article("T", f"{small}\n\n{huge}\n\n{small}", target_tokens=350, min_tokens=100)
    assert [c.text for c in chunks] == [small, huge, small]
    assert chunks[1].token_count > 350


def test_paragraph_over_embedding_limit_is_split_on_sentences():
    sentences = [f"Sentence number {i} has a few words in it." for i in range(1500)]
    huge = " ".join(sentences)
    assert count_tokens(huge) > MAX_PARAGRAPH_TOKENS
    chunks = chunk_article("T", huge, target_tokens=350, min_tokens=100)
    assert len(chunks) > 1
    assert all(c.token_count <= 350 for c in chunks)
    assert " ".join(c.text for c in chunks) == huge  # split only between sentences
    assert all(c.text.endswith(".") for c in chunks)


def test_article_of_only_heading_like_lines_is_kept():
    chunks = chunk_article("Flagstaff", "Flagstaff is a station in Melbourne\n\nRailway stations")
    assert [(c.section, c.text) for c in chunks] == [
        (None, "Flagstaff is a station in Melbourne\n\nRailway stations")
    ]


def test_uses_settings_defaults(monkeypatch):
    class S:
        chunk_target_tokens = 50
        chunk_min_tokens = 10

    monkeypatch.setattr(chunker, "get_settings", lambda: S)
    chunks = chunk_article("T", "\n\n".join(para(20, f"p{i}") for i in range(6)))
    assert all(c.token_count <= 50 for c in chunks)
    assert len(chunks) >= 3


# --- real data ---------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_articles():
    if not DATASET.exists():
        pytest.skip("dataset not downloaded")
    pf = pq.ParquetFile(DATASET)
    rng = random.Random(42)
    rows = pf.read_row_group(0, columns=["title", "text"]).to_pylist()
    for g in rng.sample(range(1, pf.num_row_groups), 3):
        rows += rng.sample(pf.read_row_group(g, columns=["title", "text"]).to_pylist(), 300)
    return rows


def test_real_articles_roundtrip(real_articles):
    """Joining the chunk texts reproduces every non-heading paragraph, verbatim and in order."""
    for art in real_articles:
        chunks = chunk_article(art["title"], art["text"])
        paras = [b for kind, b in split_blocks(art["text"]) if kind == "para"]
        assert "\n\n".join(c.text for c in chunks) == "\n\n".join(paras), art["title"]
        for p in paras:  # each paragraph is a verbatim slice of the source
            assert p in art["text"]
        for c in chunks:
            assert c.token_count == count_tokens(c.text)
            assert c.text.strip()
            head = f"{art['title']} > {c.section}" if c.section else art["title"]
            assert c.embed_text == f"{head}\n\n{c.text}"


def test_real_articles_known_structure(real_articles):
    by_title = {a["title"]: a for a in real_articles}
    april = chunk_article("April", by_title["April"]["text"])
    sections = [c.section for c in april]
    assert sections[0] is None
    assert "The Month" in sections
    assert "Events in April > Fixed Events" in sections
    assert all(s is None or not s.startswith(" ") for s in sections)
    art = chunk_article("Art", by_title["Art"]["text"])
    assert {"Types of art", 'What "art" means', "History of art"} <= {c.section for c in art}
    types = next(c for c in art if c.section == "Types of art")
    assert " Plastic art" in types.text  # list items stay in the body
