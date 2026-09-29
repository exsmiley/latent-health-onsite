from rag.evals.grading import (
    candidate_matches,
    citation_recall,
    exact_match,
    expected_candidates,
    normalize,
)


def test_normalize_casefold_punctuation_articles_accents():
    assert normalize("The Beatles!") == "beatles"
    assert normalize("  An apple, a DAY  ") == "apple day"
    assert normalize("Pelé") == "pele"
    assert normalize("50,000 people") == "50000 people"
    assert normalize("Tolkien's") == "tolkien s"
    # articles are only removed as whole words
    assert normalize("Theodore the Great") == "theodore great"


def test_candidate_must_match_whole_words():
    assert candidate_matches("38", "It took 38 years [1].")
    assert not candidate_matches("38", "It took 1938 years.")
    assert not candidate_matches("Chain", "Chainsaw inventors")
    assert candidate_matches("Berlin", "He was born in berlin.")


def test_multi_part_requires_every_part():
    assert candidate_matches("Chain; Berlin", "Ernst Chain was the youngest; born in Berlin.")
    assert not candidate_matches("Chain; Berlin", "Ernst Chain was born in Frankfurt.")
    assert not candidate_matches(" ; ", "anything")


def test_expected_candidates_strip_parenthetical():
    c = expected_candidates("38 years (Ph.D. 1927, Chancellor 1965)", ["38"])
    assert c == ["38 years (Ph.D. 1927, Chancellor 1965)", "38 years", "38"]


def test_exact_match_uses_answer_and_aliases():
    assert exact_match("38 years (Ph.D. 1927, Chancellor 1965)", [], "That is 38 years later.")
    assert exact_match("Ernst Chain, born in Berlin", ["Chain; Berlin"], "Chain, from Berlin [2]")
    assert not exact_match("Ernst Chain, born in Berlin", ["Chain; Berlin"], "Florey, Adelaide")
    assert exact_match("Unguja", ["Unguja Island"], "The island of UNGUJA.")


def test_citation_recall():
    expected = [
        {"article_id": 1, "chunk_index": 0},
        {"article_id": 1, "chunk_index": 2},
        {"article_id": 5, "chunk_index": 1},
        {"article_id": 5, "chunk_index": 1},  # duplicate entry counts once
    ]
    cited = [{"article_id": 1, "chunk_index": 2}, {"article_id": 9, "chunk_index": 0}]
    recall, overlap, art, precision = citation_recall(expected, cited)
    assert recall == 1 / 3
    assert overlap is True
    assert art == 1 / 2
    assert precision == 1 / 2
    assert citation_recall(expected, []) == (0.0, False, 0.0, None)
    assert citation_recall([], cited) == (None, False, None, None)
