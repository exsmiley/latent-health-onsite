"""Orchestrator: research -> evaluate (-> retry research) -> respond, as a stream of events."""

import json
import sys
from collections.abc import AsyncIterator

from rag.agents import events
from rag.agents.evaluator import Evaluation, evaluate
from rag.agents.events import Event
from rag.agents.research import (
    ResearchAgent,
    Submission,
    no_submission_feedback,
    retry_feedback,
)
from rag.agents.responder import respond
from rag.config import get_settings

NOT_FOUND_MESSAGE = "Sorry, I couldn't find the answer to that in Simple English Wikipedia."


def _citation_items(submission: Submission) -> list[dict]:
    return [
        {
            "n": n,
            "chunk_id": c.chunk_id,
            "article_id": c.article_id,
            "title": c.title,
            "section": c.section,
            "url": c.url,
        }
        for n, c in enumerate(submission.chunks, start=1)
    ]


async def run(question: str, history: list[dict] | None = None) -> AsyncIterator[Event]:
    """Run the whole pipeline for one question, yielding the SSE-contract events.

    Always ends with `done`. Any exception becomes an `error` event first.
    """
    history = history or []
    settings = get_settings()
    max_rounds = max(1, settings.max_research_rounds)
    max_turns = settings.research_max_turns
    try:
        agent = ResearchAgent(question=question, history=history)
        final: Submission | None = None  # latest valid submission
        final_eval: Evaluation | None = None
        supported = False
        feedback: str | None = None

        for round_no in range(1, max_rounds + 1):
            async for ev in agent.run_round(round_no, feedback):
                yield ev
            sub = agent.submission

            if sub is None:
                reason = agent.last_error or "no valid submission"
                yield events.evaluation(
                    round_no,
                    "unsupported",
                    "",
                    f"The research agent did not submit a valid answer. {reason}",
                )
                feedback = no_submission_feedback(reason, max_turns)
                continue

            final, final_eval = sub, None
            yield events.research_answer(round_no, sub.answer, sub.citations)
            yield events.status(
                "evaluate", round_no, None, "Checking the answer against the cited sources"
            )
            final_eval = await evaluate(question, sub.answer, sub.chunks)
            yield events.evaluation(
                round_no, final_eval.verdict, final_eval.independent_answer, final_eval.feedback
            )
            if final_eval.supported:
                supported = True
                break
            feedback = retry_feedback(final_eval.feedback, final_eval.independent_answer, max_turns)

        if not supported or final is None:
            yield events.status("respond", round_no, None, "The answer could not be found")
            yield events.citations([])
            yield events.token(NOT_FOUND_MESSAGE)
        else:
            yield events.status("respond", round_no, None, "Writing the final answer")
            yield events.citations(_citation_items(final))
            async for delta in respond(
                question=question,
                history=history,
                answer=final.answer,
                evaluation=final_eval,
                chunks=final.chunks,
            ):
                yield events.token(delta)
    except Exception as exc:  # noqa: BLE001 - reported to the client as an error event
        yield events.error(f"{type(exc).__name__}: {exc}")
    yield events.done()


def _short(obj: object, limit: int = 160) -> str:
    text = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def format_event(ev: Event) -> str | None:
    """Readable one-line (or short block) trace of an event. Tokens are handled by the caller."""
    d = ev.data
    match ev.type:
        case "status":
            turn = f" t{d['turn']}" if d["turn"] is not None else ""
            return f"\n[{d['stage']} r{d['round']}{turn}] {d['message']}"
        case "tool_call":
            return f"  -> {d['name']} {_short(d['arguments'])}"
        case "tool_result":
            return f"  <- {d['name']}: {d['summary']}"
        case "research_answer":
            return f"  answer (round {d['round']}): {_short(d['answer'], 400)}\n  citations: {d['citations']}"
        case "evaluation":
            return (
                f"  verdict: {d['verdict'].upper()}\n"
                f"  independent answer: {_short(d['independent_answer'], 400)}\n"
                f"  feedback: {_short(d['feedback'], 400)}"
            )
        case "citations":
            lines = ["\nSources:"] + [
                f"  [{c['n']}] {c['title']}"
                + (f" > {c['section']}" if c["section"] else "")
                + f" (chunk {c['chunk_id']}) {c['url']}"
                for c in d
            ]
            return "\n".join(lines) + "\n\nAnswer:"
        case "error":
            return f"\nERROR: {d['message']}"
        case _:
            return None


async def run_cli(question: str) -> None:
    from rag.db import close_pool

    out = sys.stdout
    try:
        async for ev in run(question, []):
            if ev.type == "token":
                out.write(ev.data["delta"])
                out.flush()
                continue
            if ev.type == "done":
                out.write("\n")
                break
            line = format_event(ev)
            if line is not None:
                print(line, file=out, flush=True)
    finally:
        await close_pool()
