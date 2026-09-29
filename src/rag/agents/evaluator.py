"""Evaluator: checks whether the cited chunk texts alone support the research answer.

It makes one non-streaming call with a strict JSON-schema output. The input is ONLY the
user's original latest question (plus, for follow-ups in a conversation, the research agent's
version with references resolved), the cited chunk texts (numbered, with title/section) and,
after a clear separator, the research answer. It never sees the chat history.
"""

from typing import Literal

from pydantic import BaseModel, ValidationError

from rag.agents import model
from rag.tools.models import Chunk

SYSTEM_PROMPT = """\
You are a strict, fair fact-checking evaluator. You receive the user's QUESTION, a set of \
numbered SOURCE PASSAGES, and (after a separator) a CANDIDATE ANSWER written by a research agent.

The source passages are your ONLY evidence. Don't use outside knowledge, even if you are sure \
of a fact. If something isn't stated in (or directly implied by) the passages, treat it as \
unknown. The candidate answer is not evidence either.

## The question
Judge against the user's question exactly as given. For a follow-up in a conversation you may \
also get a version with references resolved (pronouns and "what about..." replaced by what they \
refer to). That version may only replace pronouns and references; it must never drop a \
constraint, a step or a compared entity. Where the two differ in anything else, the user's \
question wins.

## Work in this order
1. requirements: list what the question requires before answering it. Break it into every \
link of the chain it describes (e.g. "the teacher of the philosopher who taught Alexander" is \
two links: who taught Alexander, then who taught that person), every constraint (dates, places, \
"first", "Italian", ...), every entity being compared or counted, and finally the exact thing \
the question ASKS for (which entity, at which hop, which attribute).
2. independent_answer: answer the question yourself using ONLY the source passages, as if the \
candidate answer did not exist, following your requirements one by one. For each link, say \
which passage establishes it, or that no passage does. Be specific (names, numbers, dates). If \
the passages only partly answer the question, say which parts they answer and which they \
don't. If they don't answer it at all, say so.
3. verdict: now compare your independent answer with the candidate answer.
   - "supported": the passages alone establish EVERY requirement and yield the same answer. \
Every substantive claim in the candidate answer is backed by the passages and nothing \
contradicts it. Different wording is fine. Harmless framing or hedging is fine.
   - "unsupported" if any of these holds:
     - any link of the chain, constraint or compared entity is not established by the \
passages, even if the candidate asserts it or you know it to be true. A final fact backed by \
the passages is NOT enough if the passages don't show it belongs to the thing the question \
describes (e.g. a passage gives a river's source, but no passage shows it is the river the \
question leads to);
     - for comparisons, superlatives, counts or "which of these" questions, the passages don't \
give the needed fact for EVERY entity compared, so the winner or count can't be confirmed;
     - the answer names the wrong thing: stops a hop short or goes a hop too far (e.g. names \
the teacher when the question asks for the teacher's teacher), or gives a different \
attribute than the one asked for;
     - the candidate has a substantive claim the passages don't back, contradicts the \
passages, answers a different question, or misses a part of the question.
   - A "not found" answer (the candidate says the answer can't be found, isn't in the \
sources, or can't be determined) is ALWAYS "unsupported", even when the passages really don't \
answer the question. The research agent must keep looking.
4. feedback: for "unsupported", say exactly what's missing, unbacked or contradicted, and name \
each missing link, so the research agent knows what evidence to find or what to remove (e.g. \
"No passage shows that the composer who conducted the premiere is Liszt's son-in-law"; "No \
passage gives the birth date of X, so 'oldest' can't be checked"; "Passage [2] says Y, not Z"; \
"The question asks for the teacher of the teacher; the answer names only the first teacher"). \
For a "not found" answer, suggest different search angles (other wordings, related articles, \
exact names for keyword search). For "supported", briefly confirm which passages back each link.

Judge only support by the passages. Don't judge style, length or real-world truth."""

SCHEMA: dict = {
    "type": "object",
    "properties": {
        "requirements": {
            "type": "string",
            "description": (
                "Every link of the chain the question describes, every constraint and compared "
                "entity, and the exact thing the question asks for."
            ),
        },
        "independent_answer": {
            "type": "string",
            "description": "Your own answer to the question from the source passages alone.",
        },
        "verdict": {"type": "string", "enum": ["supported", "unsupported"]},
        "feedback": {
            "type": "string",
            "description": "What is missing, unbacked or contradicted, or why it is supported.",
        },
    },
    "required": ["requirements", "independent_answer", "verdict", "feedback"],
    "additionalProperties": False,
}


class Evaluation(BaseModel):
    independent_answer: str
    requirements: str = ""
    verdict: Literal["supported", "unsupported"]
    feedback: str

    @property
    def supported(self) -> bool:
        return self.verdict == "supported"


def format_sources(chunks: list[Chunk]) -> str:
    blocks = []
    for n, c in enumerate(chunks, start=1):
        heading = f"{c.title} > {c.section}" if c.section else c.title
        blocks.append(f"[{n}] {heading}\n{c.text.strip()}")
    return "\n\n".join(blocks)


def format_question(question: str, resolved_question: str | None = None) -> str:
    """The question block: the user's own words, plus the resolved version for follow-ups."""
    if resolved_question is None:
        return f"QUESTION:\n{question}"
    return (
        f"QUESTION (the user's latest message, verbatim; this is what must be answered):\n"
        f"{question}\n\n"
        "SAME QUESTION WITH REFERENCES RESOLVED (written by the research agent from the earlier "
        "conversation, which you don't see; it may only replace pronouns and references, never "
        "drop a constraint or step, so where it differs otherwise, follow the user's words):\n"
        f"{resolved_question}"
    )


def build_input(
    question: str, answer: str, chunks: list[Chunk], resolved_question: str | None = None
) -> str:
    return (
        f"{format_question(question, resolved_question)}\n\n"
        f"SOURCE PASSAGES ({len(chunks)}):\n\n{format_sources(chunks)}\n\n"
        "==================== END OF SOURCES ====================\n\n"
        "List the question's requirements and write your independent answer from the sources "
        "above BEFORE you consider the candidate answer below. The candidate answer is NOT "
        "evidence.\n\n"
        f"CANDIDATE ANSWER (from the research agent):\n{answer}"
    )


async def evaluate(
    question: str, answer: str, chunks: list[Chunk], resolved_question: str | None = None
) -> Evaluation:
    """Judge `answer` against the user's own `question`.

    `resolved_question` is the research agent's standalone rewrite. Pass it only for follow-ups
    (when there is chat history); without history the rewrite adds nothing and could drop steps.
    """
    content = build_input(question, answer, chunks, resolved_question)
    response = await model.create(
        instructions=SYSTEM_PROMPT,
        input=[{"role": "user", "content": content}],
        text={
            "format": {
                "type": "json_schema",
                "name": "evaluation",
                "schema": SCHEMA,
                "strict": True,
            }
        },
    )
    raw = response.output_text
    try:
        return Evaluation.model_validate_json(raw)
    except ValidationError:
        # Empty output, a refusal or malformed JSON: count it as unverified rather than failing
        # the whole request, so the research agent can re-check and reply again.
        return Evaluation(
            independent_answer="",
            verdict="unsupported",
            feedback=(
                "The answer could not be verified (the checker returned no usable verdict). "
                "Make sure your cited chunks state the answer explicitly, then reply again."
            ),
        )
