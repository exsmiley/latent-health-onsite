"""Pre-retrieval: the searches the harness runs on the question before the first research turn.

No model call is involved. The question (with the previous exchange, for follow-ups) goes to
semantic_search as-is, and a keyword query is derived from its capitalized names and numbers.
The top hits of both are then fetched in full, so the model can often answer on turn 1 without
spending a turn on searching and another on reading.
"""

import json
import re

SEMANTIC_TOP_K = 8
KEYWORD_TOP_K = 5
KEYWORD_MAX_TERMS = 4
FULL_TEXT_SEMANTIC = 5  # the top semantic hits fetched in full
FULL_TEXT_KEYWORD = 2  # the top keyword hits fetched in full
FULL_TEXT_MAX_CHARS = 4000  # longer chunks (big lists) stay as blurbs
HISTORY_MAX_CHARS = 400

# Words that are capitalized only because they start a sentence, plus question words, so they
# never become keyword terms.
_STOP_TEXT = """
a an the of in on at to for from by with and or but is are was were be been being do does did
what which who whom whose when where why how that this these those it its he she they his her
their them as into than then there after before about according many much old year years later
first name give had has have not also can could would should will if so one two three put order
list roughly
"""
_STOP_WORDS = frozenset(_STOP_TEXT.split())
_TOKEN = re.compile(r"[^\W_][\w'’.-]*|[^\w\s]")
_CITATION_MARK = re.compile(r"\[\d+\]")


def retrieval_text(question: str, history: list[dict]) -> str:
    """The text to search for: the question, prefixed with the previous user/assistant turn
    when there is history, so follow-ups ("When did she die?") keep their subject."""
    previous = []
    for msg in history[-2:]:
        content = msg.get("content")
        if msg.get("role") in ("user", "assistant") and isinstance(content, str):
            text = " ".join(_CITATION_MARK.sub("", content).split())
            previous.append(text[:HISTORY_MAX_CHARS])
    return " ".join([*previous, question]).strip()


def keyword_query(text: str, max_terms: int = KEYWORD_MAX_TERMS) -> str:
    """A short web-search-syntax query from `text`: its capitalized names (consecutive ones as
    a quoted phrase) and numbers, at most `max_terms`, or "" when there are none.

    Plain words are ANDed by keyword_search, so a whole question usually matches nothing.
    """
    terms: list[str] = []
    run: list[str] = []

    def close_run() -> None:
        if run:
            terms.append(f'"{" ".join(run)}"' if len(run) > 1 else run[0])
            run.clear()

    for token in _TOKEN.findall(text):
        word = re.sub(r"(['’]s?|[.-])$", "", token)
        if len(word) > 1 and word[0].isupper() and word.lower() not in _STOP_WORDS:
            run.append(word)
            if word != token:
                close_run()  # a possessive ("Newton's") or trailing dot ends the name
            continue
        close_run()
        if word.isdigit():
            terms.append(word)
    close_run()
    return " ".join(list(dict.fromkeys(terms))[:max_terms])


def full_text_ids(semantic_output: str, keyword_output: str = "") -> list[int]:
    """Chunk ids to fetch in full: the top semantic hits, then the top keyword hits, deduped.

    Takes the JSON strings the search tools returned; an error output contributes nothing.
    """
    ids: list[int] = []
    for output, n in ((semantic_output, FULL_TEXT_SEMANTIC), (keyword_output, FULL_TEXT_KEYWORD)):
        for hit in _hits(output)[:n]:
            if hit.get("chunk_id") not in ids:
                ids.append(hit["chunk_id"])
    return ids


def drop_long_chunks(fetch_output: str) -> tuple[list[int], str]:
    """Drop chunks longer than FULL_TEXT_MAX_CHARS (big lists) from a fetch output.

    Returns the kept chunk ids and the output rewritten with only those chunks, so the
    fetch call shown to the model matches what it got. An error output comes back unchanged.
    """
    try:
        data = json.loads(fetch_output)
        chunks = [c for c in data["chunks"] if len(c.get("text", "")) <= FULL_TEXT_MAX_CHARS]
    except (ValueError, KeyError, TypeError):
        return [], fetch_output
    data["chunks"] = chunks
    data["missing_chunk_ids"] = []
    output = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return [c["chunk_id"] for c in chunks], output


def succeeded(output: str) -> bool:
    try:
        return not (isinstance(data := json.loads(output), dict) and "error" in data)
    except ValueError:
        return False


def _hits(output: str) -> list[dict]:
    try:
        results = json.loads(output)
    except ValueError:
        return []
    if not isinstance(results, list):
        return []
    return [h for r in results for h in r.get("hits", []) if isinstance(h.get("chunk_id"), int)]
