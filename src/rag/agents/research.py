"""Research agent: searches the corpus with tools and answers with chunk citations.

A *turn* is one research model call; the orchestrator gives each question a single budget of
`settings.research_max_turns` turns. The model always has `tool_choice="auto"`. A response
with any function call is a tool turn: the calls run concurrently and their outputs go back to
the model (any message text in that response, e.g. a commentary preamble, is ignored). A
response with no function call is the agent's final answer, a JSON object matching
`ANSWER_SCHEMA` (set via `text.format`). The orchestrator evaluates "answered" answers and, if
the evaluator rejects one, adds its feedback to the same conversation and keeps going.
"""

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from rag.agents import events, model, pre_retrieval
from rag.agents.events import Event
from rag.tools import fetch as fetch_tool
from rag.tools import registry
from rag.tools.models import Chunk

SEARCH_TOOLS = {"semantic_search", "keyword_search"}

ANSWER_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "question": {
            "type": "string",
            "description": (
                "The user's latest question rewritten to stand alone: resolve pronouns and "
                "references to earlier messages. Repeat it unchanged if it already stands alone."
            ),
        },
        "status": {
            "type": "string",
            "enum": ["answered", "not_found"],
            "description": '"answered" if your cited chunks state the answer, else "not_found".',
        },
        "answer": {
            "type": "string",
            "description": 'A complete, direct answer to the question ("" for not_found).',
        },
        "citations": {
            "type": "array",
            "items": {"type": "integer"},
            "description": (
                "The minimal set of chunk ids whose text, read on its own, fully supports every "
                "claim in the answer ([] for not_found)."
            ),
        },
        "reason": {
            "type": "string",
            "description": (
                'For not_found: what you searched for and what was missing. "" when answered.'
            ),
        },
    },
    "required": ["question", "status", "answer", "citations", "reason"],
    "additionalProperties": False,
}

ANSWER_FORMAT: dict = {
    "format": {
        "type": "json_schema",
        "name": "research_answer",
        "schema": ANSWER_SCHEMA,
        "strict": True,
    }
}

SYSTEM_PROMPT = """\
You are a meticulous research agent answering questions from a search index over Simple English \
Wikipedia (a snapshot from March 2022). You answer by finding evidence, not by memory. Every \
claim in your answer must be backed by text you have actually read in the index.

## The corpus and tools
- Articles are split into chunks of about 350 tokens made of whole paragraphs. Each chunk has a \
chunk_id, its article_id, the article title and the section heading (if any).
- `semantic_search(queries)`: vector search. Good for paraphrases, descriptions and concepts. \
Pass SEVERAL queries in one call (different phrasings, sub-questions, likely article titles).
- `keyword_search(queries)`: Postgres full-text search. Good for exact names, rare terms, \
numbers, dates and quoted phrases. Use it alongside semantic search, not instead of it.
- Search hits come in three forms. The top few new hits of each query include `text`: the \
chunk's FULL text, which you can read and cite directly. Lower-ranked hits only show a `blurb` \
(the first ~40 words). A blurb is NOT enough to cite: fetch the chunk first. A hit with \
`"seen": true` was already shown to you (above in the same turn, or in full in an earlier \
turn), so its text isn't repeated; look it up there.
- `fetch(chunk_ids, article_ids)`: reads full text. Use it for blurb-only hits and for chunks \
you haven't seen. PREFER fetching specific chunks (several in one call) over whole articles. \
Fetch a whole article only when you need to scan it to find the right passage. It returns the \
article's chunk_ids, so you can then fetch or cite the exact chunks.
- Each search call takes at most 5 queries. Semantic scores (cosine similarity) and keyword \
scores (ts_rank_cd) are on different scales, so don't compare them with each other. Rank hits \
within one tool's results only, and judge relevance by reading.
- If a tool output is `{"error": ...}`, it's a recoverable mistake (bad arguments, too many \
queries, unknown ids, a temporary failure). Read the message, fix the call and retry, \
preferably in the same turn as other useful calls. Don't give up because of an error.

## How to answer
You finish by replying WITHOUT calling any tool. That reply is your final answer and must be \
a JSON object with exactly these fields:
- `question`: the user's latest question rewritten to stand alone, e.g. "When did she die?" \
after a question about Marie Curie becomes "When did Marie Curie die?". The evaluator sees only \
this question and your cited chunks, never the conversation, so it must be self-contained.
- `status`: "answered" or "not_found".
- `answer`: for "answered", a complete, direct answer to the question. "" for "not_found".
- `citations`: for "answered", the chunk ids that support the answer. [] for "not_found".
- `reason`: for "not_found", one or two sentences on what you searched for and what was \
missing. "" for "answered".
You can reply with your answer after any turn, as soon as you're ready. While you still want \
to research, call tools instead; a reply without tool calls always ends your research.

## Starting evidence
Before your first turn, the system usually searches the index for the question itself: the \
first tool calls after the question (semantic_search on the question, keyword_search on its \
names and numbers, and a fetch of the top hits' full text) were run for you and cost no turn. \
Read those chunks first. If they already state the answer, reply with it on your first turn. \
Otherwise use them to plan: the fetched chunks often settle the first step of a multi-step \
question, so search for the next step, and fetch other promising hits, right away. Don't \
repeat those searches.

## How to work
1. Plan first. Break the question into the facts you need. For multi-part or comparison \
questions, cover each part.
2. Search in parallel. In a single turn, call semantic_search and keyword_search together, each \
with several queries. You can call several tools in the same turn and they run concurrently. \
Turns are the scarce resource, not tool calls.
3. Read the hits. If full-text hits already state the answer, answer right away and cite \
them; there is no need to fetch them again. Otherwise fetch the promising blurb-only chunks in \
the next turn, together with any follow-up searches. Don't re-run searches just to confirm a \
blurb. Reading the chunk is the confirmation. Follow up with targeted searches only if \
something is missing or the sources conflict.
4. Answer as soon as the evidence is sufficient. Don't spend turns you don't need.
5. Give up quickly when it isn't there. If about two turns of focused searching (semantic and \
keyword, a few phrasings, plus reading the most relevant chunks, e.g. the subject's own \
article) find nothing that answers the question, reply with status "not_found". Don't keep \
rephrasing the same search, and never search for guessed answers (e.g. candidate names).

## Never guess
Never answer with anything that the chunks you have read don't state. A made-up or inferred \
answer will be rejected and wastes the user's time. "not_found" is always better than a guess. \
Never write "not found" or "the sources don't say" as an "answered" answer; use status \
"not_found". You can only reply "not_found" after you have searched.

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

## Turn budget
You have a fixed number of turns (model calls) for this question. Before each turn you are \
told how many remain. An invalid answer (bad JSON, missing, unknown or non-chunk citations) is \
returned to you as an error and costs a turn. If you run out of turns, the user is told the \
answer couldn't be found.

If the evaluator rejects an answer, you'll get its feedback and can keep researching with the \
turns you have left. Fix exactly what it points out: find the missing evidence, cite the right \
chunks, or narrow the answer to what the sources support. If the evidence isn't there, reply \
"not_found".

The conversation may include earlier user/assistant exchanges. Use them to resolve \
follow-up questions (pronouns, "what about...", etc.), but still ground this answer in fresh \
citations."""


@dataclass
class Submission:
    question: str  # the standalone question the evaluator checks the answer against
    answer: str
    citations: list[int]
    chunks: list[Chunk]  # same order as `citations`


@dataclass
class _CallOutcome:
    output: str
    summary: str


@dataclass
class ResearchAgent:
    question: str
    history: list[dict] = field(default_factory=list)
    items: list[dict] = field(default_factory=list)
    # Set by `run_turn` when this turn's reply was an accepted final answer.
    submission: Submission | None = None
    not_found: str | None = None  # the reason, for an accepted "not_found"
    searched: bool = False  # a search tool has run in this conversation
    shown: set[int] = field(default_factory=set)  # chunk ids whose full text the model has seen
    turns_used: int = 0

    def __post_init__(self) -> None:
        for msg in self.history:
            role, content = msg.get("role"), msg.get("content")
            if role in ("user", "assistant") and isinstance(content, str) and content:
                self.items.append({"role": role, "content": content})
        self.items.append({"role": "user", "content": self.question})

    async def pre_retrieve(self, max_turns: int) -> AsyncIterator[Event]:
        """Search the index for the question before turn 1, without a model call.

        Runs semantic_search on the question (with the previous exchange, for follow-ups) and
        keyword_search on its names and numbers, then fetches the top hits in full. The calls
        go into the conversation as ordinary function_call/function_call_output pairs, so
        turn 1 starts as if the model had already searched and read. This is not a turn: its
        events carry turn 0.
        """
        yield events.status("research", 0, max_turns, "Searching the index for the question")
        text = pre_retrieval.retrieval_text(self.question, self.history)
        keywords = pre_retrieval.keyword_query(self.question) or pre_retrieval.keyword_query(text)
        calls = [
            (
                "call_pre_semantic",
                "semantic_search",
                {"queries": [text], "top_k": pre_retrieval.SEMANTIC_TOP_K},
            )
        ]
        if keywords:
            calls.append(
                (
                    "call_pre_keyword",
                    "keyword_search",
                    {"queries": [keywords], "top_k": pre_retrieval.KEYWORD_TOP_K},
                )
            )
        for call_id, name, args in calls:
            yield events.tool_call(call_id, name, args, 0)
        outputs = list(await asyncio.gather(*(registry.dispatch(n, a) for _, n, a in calls)))
        for (call_id, name, _), output in zip(calls, outputs):
            yield events.tool_result(call_id, name, registry.summarize(name, output))
        self.searched = any(pre_retrieval.succeeded(o) for o in outputs)

        ids = pre_retrieval.full_text_ids(*outputs)
        if ids:
            output = await registry.dispatch("fetch", {"chunk_ids": ids, "article_ids": []})
            ids, output = pre_retrieval.drop_long_chunks(output)
            if ids:
                args = {"chunk_ids": ids, "article_ids": []}
                calls.append(("call_pre_fetch", "fetch", args))
                outputs.append(output)
                yield events.tool_call("call_pre_fetch", "fetch", args, 0)
                yield events.tool_result(
                    "call_pre_fetch", "fetch", registry.summarize("fetch", output)
                )

        # Search hits normally carry full text (see registry.present); here the tuned fetch
        # above supplies it, so the searches show blurbs only and the fetched chunks count as
        # shown for later turns.
        searches = len(outputs) - (1 if calls[-1][0] == "call_pre_fetch" else 0)
        outputs[:searches] = registry.present(
            [(name, out) for (_, name, _), out in zip(calls, outputs[:searches])],
            set(),
            full_text=0,
        )
        self.shown.update(ids)
        for (call_id, name, args), output in zip(calls, outputs):
            self.items.append(
                {
                    "type": "function_call",
                    "call_id": call_id,
                    "name": name,
                    "arguments": json.dumps(args),
                }
            )
            self.items.append(
                {"type": "function_call_output", "call_id": call_id, "output": output}
            )

    def add_feedback(self, text: str) -> None:
        """Add a message (e.g. evaluator feedback) for the model to see on its next turn."""
        self.items.append({"role": "user", "content": text})

    async def run_turn(self, turn: int, max_turns: int) -> AsyncIterator[Event]:
        """Run one research turn, yielding status/tool_call/tool_result/research_answer events.

        Afterwards `self.submission` holds an accepted "answered" reply, `self.not_found` the
        reason of an accepted "not_found" reply, or both are None (a tool turn, or an invalid
        answer that was fed back to the model).
        """
        self.submission = None
        self.not_found = None
        yield events.status("research", turn, max_turns, f"Researching (turn {turn}/{max_turns})")
        self.items.append({"role": "developer", "content": _budget_note(turn, max_turns)})
        response = await model.create(
            instructions=SYSTEM_PROMPT,
            input=self.items,
            tools=registry.TOOL_SPECS,
            tool_choice="auto",
            parallel_tool_calls=True,
            text=ANSWER_FORMAT,
        )
        self.turns_used += 1
        self.items.extend(model.dump_output_item(o) for o in response.output)
        calls = [o for o in response.output if getattr(o, "type", None) == "function_call"]

        if calls:
            # A tool turn. Any message text alongside the calls is commentary and is ignored.
            async for ev in self._run_calls(calls, turn, run=turn < max_turns):
                yield ev
            return

        async for ev in self._handle_answer(final_text(response), turn, max_turns):
            yield ev

    async def _handle_answer(self, raw: str, turn: int, max_turns: int) -> AsyncIterator[Event]:
        data, error = _parse_answer(raw)
        answer = str(data.get("answer") or "").strip() if data else ""
        cited = data.get("citations") if data else None
        citations = [c for c in cited if _is_int(c)] if isinstance(cited, list) else []

        if error is None and data["status"] == "not_found":
            if self.searched:
                reason = str(data.get("reason") or "").strip() or "(no reason given)"
                self.not_found = reason
                yield events.research_answer(turn, "not_found", "", [], reason)
                return
            error = (
                'You replied "not_found" without searching. Search the index '
                "(semantic_search and keyword_search) first."
            )
        elif error is None:
            submission, error = await validate_submission(data, fallback_question=self.question)
            if submission is not None:
                self.submission = submission
                yield events.research_answer(
                    turn, "answered", submission.answer, submission.citations, ""
                )
                return

        yield events.research_answer(turn, "invalid", answer, citations, error)
        left = max_turns - turn
        self.items.append(
            {
                "role": "developer",
                "content": (
                    f"Your answer was rejected: {error} {_turns_left(left)} Fix it and reply "
                    "again, or keep researching with the tools."
                ),
            }
        )

    async def _run_calls(self, calls: list[Any], turn: int, run: bool) -> AsyncIterator[Event]:
        parsed: list[tuple[Any, dict | None, str | None]] = []
        for call in calls:
            args, err = _parse_args(call.arguments)
            parsed.append((call, args, err))
            yield events.tool_call(
                call.call_id,
                call.name,
                args if args is not None else {"_raw": call.arguments},
                turn,
            )

        if not run:
            # Last turn: the model can't use the results, so don't spend time running them.
            for call, _, _ in parsed:
                yield events.tool_result(call.call_id, call.name, "not run: no turns left")
            return

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
        # valid for later turns. Search hits are trimmed across the whole turn (full text for
        # the top hits, no repeats), so that happens once every call has finished.
        outputs = registry.present(
            [(call.name, outcome.output) for (call, _, _), outcome in zip(parsed, outcomes)],
            self.shown,
        )
        for (call, _, _), output in zip(parsed, outputs):
            self.items.append(
                {"type": "function_call_output", "call_id": call.call_id, "output": output}
            )

    async def _execute(self, call: Any, args: dict | None, parse_error: str | None) -> _CallOutcome:
        if parse_error is not None:
            return _CallOutcome(_error_json(parse_error), "error: invalid arguments")
        known = {spec.get("name") for spec in registry.TOOL_SPECS}
        if call.name not in known:
            return _CallOutcome(_error_json(f"Unknown tool {call.name!r}."), "error: unknown tool")
        if call.name in SEARCH_TOOLS:
            self.searched = True
        try:
            result = await registry.dispatch(call.name, args)
        except Exception as exc:  # noqa: BLE001 - the model gets the error and can adapt
            return _CallOutcome(
                _error_json(f"{type(exc).__name__}: {exc}"), f"error: {type(exc).__name__}"
            )
        return _CallOutcome(result, registry.summarize(call.name, result))


def final_text(response: Any) -> str:
    """The text of a response's final answer: its `final_answer`-phase message if there is
    one, otherwise its last message. Only used for responses without function calls."""
    messages = [o for o in response.output if getattr(o, "type", None) == "message"]
    finals = [m for m in messages if getattr(m, "phase", None) == "final_answer"]
    message = finals[-1] if finals else (messages[-1] if messages else None)
    if message is None:
        return ""
    parts = []
    for content in message.content:
        if getattr(content, "type", None) == "output_text" and content.text:
            parts.append(content.text)
        elif getattr(content, "type", None) == "refusal":
            parts.append(content.refusal)
    return "".join(parts)


def _parse_answer(raw: str) -> tuple[dict | None, str | None]:
    if not raw.strip():
        return None, "Your reply was empty. Reply with the JSON answer object."
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None, (
            "Your reply was not valid JSON. A reply without tool calls must be the JSON answer "
            "object with status, answer, citations and reason."
        )
    if not isinstance(data, dict) or data.get("status") not in ("answered", "not_found"):
        return None, 'The answer must be a JSON object whose status is "answered" or "not_found".'
    return data, None


async def validate_submission(
    args: dict, fallback_question: str
) -> tuple[Submission | None, str | None]:
    """Check an "answered" reply. Returns (submission, None) or (None, error message).

    `fallback_question` is used when the reply has no standalone `question`.
    """
    question = str(args.get("question") or "").strip() or fallback_question
    answer = args.get("answer")
    citations = args.get("citations")
    if not isinstance(answer, str) or not answer.strip():
        return None, "`answer` must be a non-empty string."
    if not isinstance(citations, list) or not citations:
        return None, "`citations` must be a non-empty list of chunk ids."
    ids: list[int] = []
    for c in citations:
        if not _is_int(c):
            return None, f"Citation {c!r} is not an integer chunk id."
        if c not in ids:
            ids.append(c)

    try:
        result = await fetch_tool.fetch(chunk_ids=ids, article_ids=[])
    except Exception as exc:  # noqa: BLE001 - e.g. a DB blip; the agent can simply reply again
        return None, (
            f"Your citations couldn't be checked because of a temporary error "
            f"({type(exc).__name__}). Reply again with the same answer."
        )
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
    return (
        Submission(
            question=question,
            answer=answer.strip(),
            citations=ids,
            chunks=[by_id[i] for i in ids],
        ),
        None,
    )


def retry_feedback(evaluator_feedback: str, independent_answer: str, turns_left: int) -> str:
    """The message added to the conversation after the evaluator rejects an answer."""
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
        f"{_turns_left(turns_left)} Find the missing evidence (or narrow the answer to what the "
        "sources support), then reply again with the minimal set of chunk ids that fully "
        'supports it, or reply "not_found" if the index doesn\'t have it.'
    )
    return "\n\n".join(parts)


def _turns_left(n: int) -> str:
    return f"You have {n} turn{'s' if n != 1 else ''} left."


def _budget_note(turn: int, max_turns: int) -> str:
    remaining = max_turns - turn + 1
    if remaining == 1:
        return (
            f"Turn {turn} of {max_turns}. This is your LAST turn. Tool calls made now can't be "
            "followed up: their results would never reach you. If you don't reply with your "
            "answer now, the user will be told the answer couldn't be found."
        )
    note = f"Turn {turn} of {max_turns}: {remaining} turns left, including this one."
    if remaining == 2:
        note += " After this one you get one more turn, and tool calls made then can't be used."
    return note


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


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
