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
- Plain text or light Markdown. No reference list at the end (the app shows the sources)."""


def build_input(
    question: str,
    answer: str,
    evaluation: Evaluation | None,
    chunks: list[Chunk],
) -> str:
    sources = format_sources(chunks) if chunks else "(none)"
    checker = evaluation.independent_answer if evaluation else "(not available)"
    return (
        f"QUESTION:\n{question}\n\n"
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
) -> AsyncIterator[str]:
    """Only called for evaluator-supported answers; not-found is handled by the orchestrator."""
    items: list[dict] = [
        {"role": m["role"], "content": m["content"]}
        for m in history
        if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)
    ]
    items.append({"role": "user", "content": build_input(question, answer, evaluation, chunks)})
    async for delta in model.stream_text(instructions=SYSTEM_PROMPT, input=items):
        yield delta
