"""Research agent: searches the corpus with tools and submits an answer with chunk citations.

A *turn* is one model call. All function calls from one turn run concurrently. A *round* is at
most `settings.research_max_turns` turns and ends when a valid `submit_answer` arrives or the
budget runs out. On the last turn of a round `tool_choice` is forced to `submit_answer`. On a
retry round the conversation continues, with the evaluator's feedback added as a user message
and a fresh turn budget.
"""

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from rag.agents import events, model
from rag.agents.events import Event
from rag.config import get_settings
from rag.tools.models import Chunk

SUBMIT_ANSWER = "submit_answer"

SUBMIT_ANSWER_SPEC: dict = {
    "type": "function",
    "name": SUBMIT_ANSWER,
    "description": (
        "Submit your final answer and the chunk ids that support it. This ends the research "
        "round. An independent evaluator will then read ONLY the text of the cited chunks and "
        "check that it supports your answer. Citations must be chunk ids (never article ids)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "A complete, direct answer to the user's question.",
            },
            "citations": {
                "type": "array",
                "items": {"type": "integer"},
                "description": (
                    "The minimal set of chunk ids whose text, read on its own, fully supports "
                    "every claim in the answer."
                ),
            },
        },
        "required": ["answer", "citations"],
        "additionalProperties": False,
    },
    "strict": True,
}

SYSTEM_PROMPT = """\
You are a meticulous research agent answering questions from a search index over Simple English \
Wikipedia (a snapshot from March 2022). You answer by finding evidence, not by memory. Every \
claim you submit must be backed by text you have actually read in the index.

## The corpus and tools
- Articles are split into chunks of about 350 tokens made of whole paragraphs. Each chunk has a \
chunk_id, its article_id, the article title and the section heading (if any).
- `semantic_search(queries)`: vector search. Good for paraphrases, descriptions and concepts. \
Pass SEVERAL queries in one call (different phrasings, sub-questions, likely article titles).
- `keyword_search(queries)`: Postgres full-text search. Good for exact names, rare terms, \
numbers, dates and quoted phrases. Use it alongside semantic search, not instead of it.
- Search hits only show a short blurb (the first ~40 words of the chunk). A blurb is NOT enough \
to cite. Read the chunk first.
- `fetch(chunk_ids, article_ids)`: reads full text. PREFER fetching specific chunks (several in \
one call) over whole articles. Fetch a whole article only when you need to scan it to find \
the right passage. It returns the article's chunk_ids, so you can then fetch or cite the \
exact chunks.
- `submit_answer(answer, citations)`: ends the round.
- Each search call takes at most 5 queries. Semantic scores (cosine similarity) and keyword \
scores (ts_rank_cd) are on different scales, so don't compare them with each other. Rank hits \
within one tool's results only, and judge relevance by reading.
- If a tool output is `{"error": ...}`, it's a recoverable mistake (bad arguments, too many \
queries, unknown ids, a temporary failure). Read the message, fix the call and retry, \
preferably in the same turn as other useful calls. Don't give up because of an error.

## How to work
1. Plan first. Break the question into the facts you need. For multi-part or comparison \
questions, cover each part.
2. Search in parallel. In a single turn, call semantic_search and keyword_search together, each \
with several queries. You can call several tools in the same turn and they run concurrently. \
Turns are the scarce resource, not tool calls.
3. Read early. As soon as a blurb looks relevant, fetch it (and other promising chunks) in \
the next turn, together with any follow-up searches. Don't re-run searches just to confirm a \
blurb. Reading the chunk is the confirmation. Follow up with targeted searches only if \
something is missing or the sources conflict.
4. Submit as soon as the evidence is sufficient. Don't spend turns you don't need.

## Citations (read carefully)
- Citations MUST be chunk ids. Article ids are rejected.
- The evaluator sees ONLY the question and the text of the chunks you cite. It does not see \
your searches, other chunks, whole articles, or your reasoning. If a fact is not in a cited \
chunk's text, then as far as the evaluator knows it is unsupported.
- Cite the MINIMAL set of chunks that TOGETHER fully support the answer: every claim covered, \
no padding. Usually 1 to 4 chunks. Drop chunks that add nothing.
- Chunks don't carry context from neighbouring chunks. If a chunk says "He was born in 1879" \
without naming the person, also cite a chunk that establishes who "he" is, or pick a better \
chunk.
- Only put in the answer what the cited text supports.
- The evaluator ALWAYS rejects "not found" answers ("the sources don't say", "can't be \
determined"). If you haven't found the answer, keep searching: try other wordings, related or \
broader articles, and exact names with keyword_search. Only submit a partial or "not found" \
answer when you are out of turns. If the answer truly isn't in the index, the user will be told \
it couldn't be found.

## Turn budget
Each round gives you a fixed number of turns (model calls). Before each turn you are told how \
many remain. On the final turn you must call submit_answer, so make sure you have fetched \
and read your evidence before then. An invalid submission (e.g. unknown or non-chunk ids) is \
returned to you as an error and costs a turn.

If the evaluator rejects an answer, you'll get its feedback and a fresh turn budget. Fix \
exactly what it points out: find the missing evidence, cite the right chunks, or narrow the \
answer to what the sources support.

The conversation may include earlier user/assistant exchanges. Use them to resolve \
follow-up questions (pronouns, "what about...", etc.), but still ground this answer in fresh \
citations."""


@dataclass
class Submission:
    answer: str
    citations: list[int]
    chunks: list[Chunk]  # same order as `citations`


@dataclass
class _CallOutcome:
    output: str
    summary: str
    submission: Submission | None = None


@dataclass
class ResearchAgent:
    question: str
    history: list[dict] = field(default_factory=list)
    items: list[dict] = field(default_factory=list)
    submission: Submission | None = None
    last_error: str | None = None

    def __post_init__(self) -> None:
        for msg in self.history:
            role, content = msg.get("role"), msg.get("content")
            if role in ("user", "assistant") and isinstance(content, str) and content:
                self.items.append({"role": role, "content": content})
        self.items.append({"role": "user", "content": self.question})

    def tools(self) -> list[dict]:
        from rag.tools import registry

        return [*registry.TOOL_SPECS, SUBMIT_ANSWER_SPEC]

    async def run_round(self, round_no: int, feedback: str | None = None) -> AsyncIterator[Event]:
        """Run one research round, yielding status/tool_call/tool_result events.

        Afterwards `self.submission` holds the accepted submission, or None if the budget ran
        out without a valid one (then `self.last_error` explains why).
        """
        settings = get_settings()
        max_turns = settings.research_max_turns
        self.submission = None
        self.last_error = None
        if feedback:
            self.items.append({"role": "user", "content": feedback})

        tools = self.tools()
        for turn in range(1, max_turns + 1):
            remaining = max_turns - turn + 1
            last = turn == max_turns
            yield events.status(
                "research",
                round_no,
                turn,
                f"Researching (round {round_no}, turn {turn}/{max_turns})",
            )
            self.items.append({"role": "developer", "content": _budget_note(turn, max_turns)})
            response = await model.create(
                instructions=SYSTEM_PROMPT,
                input=self.items,
                tools=tools,
                tool_choice=({"type": "function", "name": SUBMIT_ANSWER} if last else "auto"),
                parallel_tool_calls=True,
            )
            self.items.extend(model.dump_output_item(o) for o in response.output)
            calls = [o for o in response.output if getattr(o, "type", None) == "function_call"]

            if not calls:
                self.last_error = "The model replied without calling a tool."
                if not last:
                    self.items.append(
                        {
                            "role": "developer",
                            "content": (
                                "You replied without calling a tool. Keep researching with the "
                                "tools, or call submit_answer with chunk citations. "
                                f"{remaining - 1} turn(s) left."
                            ),
                        }
                    )
                continue

            async for ev in self._run_calls(calls, round_no, turn):
                yield ev
            if self.submission is not None:
                return

        if self.last_error is None:
            self.last_error = f"No valid answer was submitted within {max_turns} turns."

    async def _run_calls(self, calls: list[Any], round_no: int, turn: int) -> AsyncIterator[Event]:
        parsed: list[tuple[Any, dict | None, str | None]] = []
        for call in calls:
            args, err = _parse_args(call.arguments)
            parsed.append((call, args, err))
            yield events.tool_call(
                call.call_id,
                call.name,
                args if args is not None else {"_raw": call.arguments},
                round_no,
                turn,
            )

        tasks = {
            asyncio.ensure_future(self._execute(call, args, err)): i
            for i, (call, args, err) in enumerate(parsed)
        }
        outcomes: list[_CallOutcome | None] = [None] * len(parsed)
        try:
            pending = set(tasks)
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in sorted(done, key=lambda t: tasks[t]):
                    i = tasks[task]
                    outcomes[i] = task.result()
                    call = parsed[i][0]
                    yield events.tool_result(call.call_id, call.name, outcomes[i].summary)
        finally:
            for task in tasks:
                task.cancel()

        # Every function call must get an output (in call order) so the conversation stays
        # valid for later turns and retry rounds.
        accepted: Submission | None = None
        for (call, _, _), outcome in zip(parsed, outcomes):
            assert outcome is not None
            output = outcome.output
            if outcome.submission is not None:
                if accepted is None:
                    accepted = outcome.submission
                else:
                    output = "Ignored: an earlier submit_answer call in this turn was accepted."
            self.items.append(
                {"type": "function_call_output", "call_id": call.call_id, "output": output}
            )
        if accepted is not None:
            self.submission = accepted

    async def _execute(self, call: Any, args: dict | None, parse_error: str | None) -> _CallOutcome:
        if parse_error is not None:
            return _CallOutcome(_error_json(parse_error), "error: invalid arguments")
        if call.name == SUBMIT_ANSWER:
            return await self._submit(args)
        from rag.tools import registry

        known = {spec.get("name") for spec in registry.TOOL_SPECS}
        if call.name not in known:
            return _CallOutcome(_error_json(f"Unknown tool {call.name!r}."), "error: unknown tool")
        try:
            result = await registry.dispatch(call.name, args)
        except Exception as exc:  # noqa: BLE001 - the model gets the error and can adapt
            return _CallOutcome(
                _error_json(f"{type(exc).__name__}: {exc}"), f"error: {type(exc).__name__}"
            )
        return _CallOutcome(result, _summarize(call.name, result))

    async def _submit(self, args: dict) -> _CallOutcome:
        submission, error = await validate_submission(args)
        if submission is None:
            self.last_error = f"Invalid submission: {error}"
            return _CallOutcome(
                _error_json(f"Submission rejected: {error} Fix it and call submit_answer again."),
                f"rejected: {error}",
            )
        n = len(submission.citations)
        return _CallOutcome(
            json.dumps({"ok": True, "message": "Answer submitted for evaluation."}),
            f"accepted ({n} citation{'s' if n != 1 else ''})",
            submission,
        )


async def validate_submission(args: dict) -> tuple[Submission | None, str | None]:
    """Check a submit_answer payload. Returns (submission, None) or (None, error message)."""
    answer = args.get("answer")
    citations = args.get("citations")
    if not isinstance(answer, str) or not answer.strip():
        return None, "`answer` must be a non-empty string."
    if not isinstance(citations, list) or not citations:
        return None, "`citations` must be a non-empty list of chunk ids."
    ids: list[int] = []
    for c in citations:
        if isinstance(c, bool) or not isinstance(c, int):
            return None, f"Citation {c!r} is not an integer chunk id."
        if c not in ids:
            ids.append(c)

    from rag.tools.fetch import fetch

    result = await fetch(chunk_ids=ids, article_ids=[])
    if result.missing_chunk_ids:
        return None, (
            f"These citations are not valid chunk ids: {sorted(result.missing_chunk_ids)}. "
            "Cite chunk_id values from search hits or from an article's chunk_ids; article ids "
            "are not accepted."
        )
    by_id = {c.chunk_id: c for c in result.chunks}
    missing = [i for i in ids if i not in by_id]
    if missing:
        return None, f"These citations are not valid chunk ids: {missing}."
    return Submission(answer=answer.strip(), citations=ids, chunks=[by_id[i] for i in ids]), None


def retry_feedback(evaluator_feedback: str, independent_answer: str, max_turns: int) -> str:
    """The user message that opens a retry round after the evaluator rejects an answer."""
    parts = [
        (
            "The evaluator REJECTED your answer. It only saw the question and the text of the "
            "chunks you cited."
        ),
        f"Evaluator feedback: {evaluator_feedback or '(none)'}",
    ]
    if independent_answer:
        parts.append(
            f"What the evaluator could conclude from your cited chunks alone: {independent_answer}"
        )
    parts.append(
        f"You have a fresh budget of {max_turns} turns. Find the missing evidence (or narrow "
        "the answer to what the sources support), then call submit_answer again with the "
        "minimal set of chunk ids that fully supports it."
    )
    return "\n\n".join(parts)


def no_submission_feedback(reason: str, max_turns: int) -> str:
    return (
        f"Your previous research round ended without a valid submission ({reason}). You have "
        f"a fresh budget of {max_turns} turns. Make sure you fetch and read the evidence, then "
        "call submit_answer with chunk ids (not article ids) before the budget runs out."
    )


def _budget_note(turn: int, max_turns: int) -> str:
    remaining = max_turns - turn + 1
    if remaining == 1:
        return (
            f"Turn {turn} of {max_turns}. This is your FINAL turn: you must call submit_answer "
            "now with your best answer and the chunk ids that support it."
        )
    note = f"Turn {turn} of {max_turns}: {remaining} turns left in this round, including this one."
    if remaining == 2:
        note += " Next turn is your last and must be submit_answer, so read what you need now."
    return note


def _parse_args(raw: str) -> tuple[dict | None, str | None]:
    try:
        args = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        return None, f"Arguments are not valid JSON: {exc}"
    if not isinstance(args, dict):
        return None, "Arguments must be a JSON object."
    return args, None


def _error_json(message: str) -> str:
    return json.dumps({"error": message}, ensure_ascii=False)


def _summarize(name: str, result: str) -> str:
    from rag.tools import registry

    summarize = getattr(registry, "summarize", None)
    if summarize is not None:
        try:
            return str(summarize(name, result))
        except Exception:  # noqa: BLE001, S110 - a summary is cosmetic; fall back below
            pass
    return _fallback_summary(result)


def _fallback_summary(result: str) -> str:
    try:
        data = json.loads(result)
    except (TypeError, json.JSONDecodeError):
        return f"{len(result or '')} chars"
    if isinstance(data, dict) and "error" in data:
        return f"error: {data['error']}"[:120]
    if isinstance(data, list):
        hits = sum(len(q.get("hits", [])) for q in data if isinstance(q, dict))
        return f"{len(data)} queries, {hits} hits"
    if isinstance(data, dict) and ("chunks" in data or "articles" in data):
        return f"{len(data.get('chunks', []))} chunks, {len(data.get('articles', []))} articles"
    return f"{len(result)} chars"
