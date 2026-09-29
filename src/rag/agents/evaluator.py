"""Evaluator: checks whether the cited chunk texts alone support the research answer.

It makes one non-streaming call with a strict JSON-schema output. The input is ONLY the
question, the cited chunk texts (numbered, with title/section) and, after a clear separator,
the research answer.
"""

from typing import Literal

from pydantic import BaseModel

from rag.agents import model
from rag.tools.models import Chunk

SYSTEM_PROMPT = """\
You are a strict, fair fact-checking evaluator. You receive a QUESTION, a set of numbered SOURCE \
PASSAGES, and (after a separator) a CANDIDATE ANSWER written by a research agent.

The source passages are your ONLY evidence. Don't use outside knowledge, even if you are sure \
of a fact. If something isn't stated in (or directly implied by) the passages, treat it as \
unknown.

Work in this order:
1. independent_answer: answer the question yourself using ONLY the source passages, as if the \
candidate answer did not exist. Be specific (names, numbers, dates). If the passages only \
partly answer the question, say which part they answer and which they don't. If they don't \
answer it at all, say so.
2. verdict: now compare your independent answer with the candidate answer.
   - "supported": the passages alone yield the same answer. Every substantive claim in the \
candidate answer is backed by the passages and nothing contradicts it. Different wording is \
fine. Harmless framing or hedging is fine.
   - "unsupported": the candidate answer has a substantive claim the passages don't back, \
contradicts the passages, answers a different question, or misses a part of the question \
that the passages don't cover either.
   - A "not found" answer (the candidate says the answer can't be found, isn't in the \
sources, or can't be determined) is ALWAYS "unsupported", even when the passages really don't \
answer the question. The research agent must keep looking.
3. feedback: for "unsupported", say exactly what's missing, unbacked or contradicted, so the \
research agent knows what evidence to find or what to remove (e.g. "No passage states the year \
X happened"; "Passage [2] says Y, not Z"; "Passages never name who 'he' refers to"). For a \
"not found" answer, suggest different search angles (other wordings, related articles, exact \
names for keyword search). For "supported", briefly confirm which passages back the key \
claims.

Judge only support by the passages. Don't judge style, length or real-world truth."""

SCHEMA: dict = {
    "type": "object",
    "properties": {
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
    "required": ["independent_answer", "verdict", "feedback"],
    "additionalProperties": False,
}


class Evaluation(BaseModel):
    independent_answer: str
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


def build_input(question: str, answer: str, chunks: list[Chunk]) -> str:
    return (
        f"QUESTION:\n{question}\n\n"
        f"SOURCE PASSAGES ({len(chunks)}):\n\n{format_sources(chunks)}\n\n"
        "==================== END OF SOURCES ====================\n\n"
        "Write your independent answer from the sources above BEFORE you consider the candidate "
        "answer below. The candidate answer is NOT evidence.\n\n"
        f"CANDIDATE ANSWER (from the research agent):\n{answer}"
    )


async def evaluate(question: str, answer: str, chunks: list[Chunk]) -> Evaluation:
    response = await model.create(
        instructions=SYSTEM_PROMPT,
        input=[{"role": "user", "content": build_input(question, answer, chunks)}],
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
    if not raw:
        raise model.ModelError("The evaluator returned no output (possibly a refusal).")
    return Evaluation.model_validate_json(raw)
