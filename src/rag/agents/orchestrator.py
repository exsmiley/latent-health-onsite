"""Orchestrator: research (<-> evaluate) -> respond, as a stream of events."""

import json
import sys
from collections.abc import AsyncIterator

from rag import db
from rag.agents import events
from rag.agents.evaluator import Evaluation, evaluate
from rag.agents.events import Event, OutcomeResult
from rag.agents.research import ResearchAgent, Submission, retry_feedback
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

    Unless `settings.research_pre_retrieve` is off, the harness first searches for the
    question itself (turn 0, not a research turn). Then one budget of `settings.research_max_turns` research turns. The agent may answer after any
    turn; an answer the evaluator rejects sends its feedback back into the same conversation
    and research continues with the turns left. Always ends with `done`. Any exception
    becomes an `error` event first.
    """
    history = history or []
    settings = get_settings()
    max_turns = settings.research_max_turns
    try:
        agent = ResearchAgent(question=question, history=history)
        result: OutcomeResult = "out_of_turns"
        final: Submission | None = None
        final_eval: Evaluation | None = None

        if settings.research_pre_retrieve:
            async for ev in agent.pre_retrieve(max_turns):
                yield ev

        for turn in range(1, max_turns + 1):
            async for ev in agent.run_turn(turn, max_turns):
                yield ev
            if agent.not_found is not None:
                result = "not_found"  # no evaluation: the agent says the index lacks it
                break
            sub = agent.submission
            if sub is None:
                continue  # tool turn, or an invalid answer that was fed back
            yield events.status(
                "evaluate", turn, max_turns, "Checking the answer against the cited sources"
            )
            # Judge against the user's own words. The agent's standalone rewrite only helps to
            # resolve references to earlier turns, so it is passed only for follow-ups.
            evaluation = await evaluate(
                question,
                sub.answer,
                sub.chunks,
                resolved_question=sub.question if agent.has_history else None,
            )
            yield events.evaluation(
                turn, evaluation.verdict, evaluation.independent_answer, evaluation.feedback
            )
            if evaluation.supported:
                result, final, final_eval = "supported", sub, evaluation
                break
            if turn < max_turns:
                agent.add_feedback(
                    retry_feedback(
                        evaluation.feedback, evaluation.independent_answer, max_turns - turn
                    )
                )

        if final is None:
            yield events.status("respond", None, max_turns, "The answer could not be found")
            yield events.outcome(result, agent.turns_used)
            yield events.citations([])
            yield events.token(NOT_FOUND_MESSAGE)
        else:
            yield events.status("respond", None, max_turns, "Writing the final answer")
            yield events.outcome(result, agent.turns_used)
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
            turn = f" {d['turn']}/{d['max_turns']}" if d["turn"] is not None else ""
            return f"\n[{d['stage']}{turn}] {d['message']}"
        case "tool_call":
            return f"  -> {d['name']} {_short(d['arguments'])}"
        case "tool_result":
            return f"  <- {d['name']}: {d['summary']}"
        case "research_answer":
            lines = [f"  {d['status']} (turn {d['turn']})"]
            if d["answer"]:
                lines.append(f"  answer: {_short(d['answer'], 400)}")
            if d["citations"]:
                lines.append(f"  citations: {d['citations']}")
            if d["reason"]:
                lines.append(f"  reason: {_short(d['reason'], 400)}")
            return "\n".join(lines)
        case "evaluation":
            return (
                f"  verdict: {d['verdict'].upper()}\n"
                f"  independent answer: {_short(d['independent_answer'], 400)}\n"
                f"  feedback: {_short(d['feedback'], 400)}"
            )
        case "outcome":
            return f"\n[outcome] {d['result']} after {d['turns_used']} turn(s)"
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
        await db.close_pool()
