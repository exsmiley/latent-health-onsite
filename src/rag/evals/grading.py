"""Grading: normalized exact match, an LLM judge, and citation recall."""

import re
import unicodedata
from typing import Any

from pydantic import BaseModel, ValidationError

from rag.agents import model

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")
_PUNCT = re.compile(r"[^\w\s]|_")
_PAREN = re.compile(r"\s*\([^)]*\)")


def normalize(text: str) -> str:
    """Casefold, drop accents, punctuation and English articles, collapse whitespace.

    Digit-group commas are removed first so "50,000" and "50000" compare equal.
    """
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = _THOUSANDS.sub("", text.casefold())
    text = _PUNCT.sub(" ", text)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def _contains(haystack_norm: str, needle: str) -> bool:
    n = normalize(needle)
    return bool(n) and f" {n} " in f" {haystack_norm} "


def candidate_matches(candidate: str, response: str) -> bool:
    """True if every ";"-separated part of `candidate` occurs (normalized) in `response`."""
    parts = [p for p in candidate.split(";") if normalize(p)]
    if not parts:
        return False
    hay = normalize(response)
    return all(_contains(hay, p) for p in parts)


def expected_candidates(answer: str, aliases: list[str]) -> list[str]:
    """The canonical answer, the answer without its parenthetical notes, and every alias."""
    out: list[str] = []
    for c in [answer, _PAREN.sub("", answer), *aliases]:
        c = c.strip()
        if c and c not in out:
            out.append(c)
    return out


def exact_match(answer: str, aliases: list[str], response: str) -> bool:
    return any(candidate_matches(c, response) for c in expected_candidates(answer, aliases))


def citation_recall(
    expected: list[dict], cited: list[dict]
) -> tuple[float | None, bool, float | None, float | None]:
    """(chunk recall, any chunk overlap, article recall, chunk precision) of cited vs expected
    supporting chunks.

    Chunks match on (article_id, chunk_index), which survives re-ingest; chunk_id doesn't.
    """
    exp = {(c["article_id"], c["chunk_index"]) for c in expected}
    got = {(c["article_id"], c["chunk_index"]) for c in cited}
    if not exp:
        return None, False, None, None
    exp_articles = {a for a, _ in exp}
    got_articles = {a for a, _ in got}
    return (
        len(exp & got) / len(exp),
        bool(exp & got),
        len(exp_articles & got_articles) / len(exp_articles),
        len(exp & got) / len(got) if got else None,
    )


# ---- LLM judge ----------------------------------------------------------------------------------

JUDGE_PROMPT = """\
You grade answers from a question-answering system against a reference answer.

You get a QUESTION, the EXPECTED ANSWER (with acceptable ALIASES), and the SYSTEM ANSWER. \
Decide whether the system answer agrees factually with the expected answer.

Rules:
- Judge ONLY factual agreement with the expected answer. Ignore style, length, citation \
markers like [1], extra context and hedging, unless the extra content contradicts the expected \
answer or gives several conflicting final answers.
- The expected answer is the ground truth for this task, even if you believe the real-world \
facts differ. Don't use outside knowledge to override it.
- Equivalent forms count: different spellings, a fuller or shorter name, equivalent units or \
rounding, a date in another format. Any alias is acceptable.
- If the expected answer has several parts (e.g. a person AND a place, or a list), "correct" \
needs every part. If some but not all parts are right, set partially_correct to true and \
correct to false.
- A small numeric difference that the expected answer notes as approximate is fine. A \
different number is wrong.
- A reply saying the answer couldn't be found, or refusing, is incorrect (correct=false, \
partially_correct=false).
- The NOTES (when given) are the question author's notes: accepted approximations, pinned \
answers and caveats where the corpus differs from reality. Use them to interpret the expected \
answer.
- correct=true implies partially_correct=false.
Give a one or two sentence reason."""

PREMISE_JUDGE_PROMPT = """\
You grade answers from a question-answering system on questions that test how it handles a \
false, ambiguous or unanswerable premise.

You get a QUESTION, the EXPECTED BEHAVIOR, the PREMISE the question rests on, the EXPECTED \
ANSWER (the key content of an ideal reply, with acceptable ALIASES), the question author's \
NOTES (what a good vs bad response looks like) and the SYSTEM ANSWER.

EXPECTED BEHAVIOR is one of:
- correct_premise: the PREMISE is false. A correct reply says so (explicitly or by clearly \
stating the contradicting fact) and gives the corrective facts in the expected answer, plus the \
closest true answer where the expected answer has one. A reply that answers as if the premise \
were true is WRONG, even if its other facts are right. A reply that only says the answer \
couldn't be found, or refuses, is WRONG.
- disambiguate: the question has several plausible referents. A correct reply covers the main \
readings in the expected answer, or clearly picks the most likely one and names the \
alternative. Answering a single reading with no sign of the other is partially correct unless \
the NOTES say the dominant reading alone is enough. Mixing facts across referents is WRONG. A \
reply that only says the answer couldn't be found is WRONG.
- not_found: the sources do not contain the answer and the premise is NOT known to be false. \
A correct reply says the information isn't available (it may add clearly related context, as \
long as it doesn't present it as the answer). Any concrete answer is WRONG, and so is an \
invented premise correction (e.g. claiming the thing never existed).

Rules:
- The expected answer and notes are the ground truth for this task, even if you believe the \
real-world facts differ. Don't use outside knowledge to override them.
- Ignore style, length, citation markers like [1] and hedging. Equivalent forms, spellings and \
date formats count.
- If the expected answer has several parts, correct needs every essential part (the NOTES say \
which parts are optional). If some but not all essential parts are right, set \
partially_correct to true and correct to false.
- correct=true implies partially_correct=false.
Give a one or two sentence reason."""

PREMISE_BEHAVIORS = ("correct_premise", "disambiguate", "not_found")
DECLINED_OUTCOMES = ("not_found", "out_of_turns")  # both end in the fixed "couldn't find" reply


def premise_outcome_verdict(expected_behavior: str, outcome: str) -> dict | None:
    """The verdict for a premise-tier question that needs no judge, or None to ask the judge.

    A decline (not_found, or out_of_turns, which shows the same "couldn't find" reply) is
    correct for `not_found` and wrong for `correct_premise`/`disambiguate`. Errors and timeouts
    are always wrong. A supported answer always goes to the judge.
    """
    if outcome == "supported":
        return None
    if outcome in DECLINED_OUTCOMES and expected_behavior == "not_found":
        return {
            "correct": True,
            "partially_correct": False,
            "reason": f"declined ({outcome}) as expected; not sent to the judge",
        }
    why = (
        f"bare 'couldn't find' ({outcome}) but the expected behavior is {expected_behavior}"
        if outcome in DECLINED_OUTCOMES
        else f"no answer ({outcome}); not sent to the judge"
    )
    return {"correct": False, "partially_correct": False, "reason": why}


JUDGE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "correct": {"type": "boolean"},
        "partially_correct": {"type": "boolean"},
    },
    "required": ["reason", "correct", "partially_correct"],
    "additionalProperties": False,
}


class Judgement(BaseModel):
    correct: bool
    partially_correct: bool
    reason: str


def judge_input(
    question: str,
    answer: str,
    aliases: list[str],
    response: str,
    notes: str = "",
    expected_behavior: str | None = None,
    premise: str | None = None,
) -> str:
    alias_text = "\n".join(f"- {a}" for a in aliases) or "(none)"
    behavior = (
        f"EXPECTED BEHAVIOR:\n{expected_behavior}\n\nPREMISE:\n{premise or '(none)'}\n\n"
        if expected_behavior
        else ""
    )
    return (
        f"QUESTION:\n{question}\n\n"
        f"{behavior}"
        f"EXPECTED ANSWER:\n{answer}\n\n"
        f"ALIASES:\n{alias_text}\n\n"
        f"NOTES:\n{notes.strip() or '(none)'}\n\n"
        f"SYSTEM ANSWER:\n{response}"
    )


async def judge(
    question: str,
    answer: str,
    aliases: list[str],
    response: str,
    notes: str = "",
    expected_behavior: str | None = None,
    premise: str | None = None,
) -> Judgement:
    """One strict-JSON call with `settings.chat_model`. Raises ValueError on an unusable reply.

    With `expected_behavior` (premise tier) the judge uses PREMISE_JUDGE_PROMPT and also sees
    the behavior and the premise.
    """
    content = judge_input(question, answer, aliases, response, notes, expected_behavior, premise)
    result: Any = await model.create(
        instructions=PREMISE_JUDGE_PROMPT if expected_behavior else JUDGE_PROMPT,
        input=[{"role": "user", "content": content}],
        text={
            "format": {
                "type": "json_schema",
                "name": "judgement",
                "schema": JUDGE_SCHEMA,
                "strict": True,
            }
        },
    )
    try:
        j = Judgement.model_validate_json(result.output_text)
    except ValidationError as exc:
        raise ValueError(f"unusable judge reply: {result.output_text[:200]!r}") from exc
    if j.correct:
        j.partially_correct = False
    return j
