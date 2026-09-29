"""Responder: streams the final user-facing answer with [n] citation markers."""

from collections.abc import AsyncIterator

from rag.agents import model
from rag.agents.evaluator import Evaluation, format_sources
from rag.tools.models import Chunk

SYSTEM_PROMPT = """\
You write the final answer shown to the user of a question-answering app over Simple English \
Wikipedia. You get the user's question, numbered SOURCES, a draft answer from a research agent, \
and a fact-checker's answer written from the sources alone.

Rules:
- Be concise and direct. Lead with the answer in the first sentence. Usually 1 to 3 short \
paragraphs, or a short list if the question asks for several items. No preamble, and don't \
mention the research agent or the fact-checker.
- Only state what the SOURCES support. Where the draft and the fact-checker disagree, follow \
the sources.
- Add inline citation markers like [1] or [2][3] right after the claims they support. n is \
the source number. Use only numbers that exist in SOURCES, and cite every factual claim.
- Plain text or light Markdown. No reference list at the end (the app shows the sources).
- Keep EVERY reading, interpretation, item and part that the draft covers and the SOURCES \
support. Never drop one, even if it looks secondary: shorten the wording, not the content.
- If the input says the question's PREMISE IS FALSE, open with a short, polite correction of \
the premise (one sentence, with citations), then give the closest true answer(s) from the \
draft. Don't lecture, and don't pretend the premise holds.
- If the question is ambiguous and the draft covers several readings, keep each one, short and \
clearly separated (or the likeliest first and the alternative in one sentence)."""


def build_input(
    question: str,
    answer: str,
    evaluation: Evaluation | None,
    chunks: list[Chunk],
    premise_false: bool = False,
) -> str:
    sources = format_sources(chunks) if chunks else "(none)"
    checker = evaluation.independent_answer if evaluation else "(not available)"
    note = (
        "NOTE: the question's PREMISE IS FALSE (verified against the sources). Correct it "
        "briefly, then give the closest true answer.\n\n"
        if premise_false
        else ""
    )
    return (
        f"QUESTION:\n{question}\n\n"
        f"{note}"
        f"SOURCES:\n\n{sources}\n\n"
        f"DRAFT ANSWER:\n{answer or '(none)'}\n\n"
        f"FACT-CHECKER'S ANSWER FROM THE SOURCES ALONE:\n{checker}"
    )


async def respond(
    question: str,
    history: list[dict],
    answer: str,
    evaluation: Evaluation | None,
    chunks: list[Chunk],
    premise_false: bool = False,
) -> AsyncIterator[str]:
    """Only called for evaluator-supported answers; not-found is handled by the orchestrator."""
    items: list[dict] = [
        {"role": m["role"], "content": m["content"]}
        for m in history
        if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)
    ]
    items.append(
        {
            "role": "user",
            "content": build_input(question, answer, evaluation, chunks, premise_false),
        }
    )
    async for delta in model.stream_text(instructions=SYSTEM_PROMPT, input=items):
        yield delta
