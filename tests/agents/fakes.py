"""Fakes for agent/API tests: a scripted Responses client plus fake tool modules."""

import asyncio
import copy
import json
from collections.abc import Callable
from typing import Any

from openai.types.responses import (
    Response,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseTextDeltaEvent,
)

from rag.tools.models import Chunk, FetchResult

# ---- scripted model outputs -------------------------------------------------------------------

_counter = 0


def _next_id(prefix: str) -> str:
    global _counter
    _counter += 1
    return f"{prefix}_{_counter}"


def fc(name: str, args: dict | str, call_id: str | None = None) -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(
        type="function_call",
        id=_next_id("fc"),
        call_id=call_id or _next_id("call"),
        name=name,
        arguments=args if isinstance(args, str) else json.dumps(args),
        status="completed",
    )


def msg(text: str, phase: str | None = None) -> ResponseOutputMessage:
    return ResponseOutputMessage(
        type="message",
        id=_next_id("msg"),
        role="assistant",
        status="completed",
        phase=phase,
        content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
    )


def resp(*items: Any) -> Response:
    return Response.model_construct(id=_next_id("resp"), object="response", output=list(items))


def final(payload: dict | str) -> Response:
    """A research reply without tool calls: the structured final answer."""
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return resp(msg(text, phase="final_answer"))


def answer(text: str, citations: list) -> Response:
    return final({"status": "answered", "answer": text, "citations": citations, "reason": ""})


def not_found(reason: str = "Searched Einstein's article; no pet is mentioned.") -> Response:
    return final({"status": "not_found", "answer": "", "citations": [], "reason": reason})


def search(query: str = "q", call_id: str | None = None) -> Response:
    return resp(fc("semantic_search", {"queries": [query], "top_k": None}, call_id))


def verdict(verdict: str, independent: str = "indep", feedback: str = "fb") -> Response:
    payload = {"independent_answer": independent, "verdict": verdict, "feedback": feedback}
    return resp(msg(json.dumps(payload)))


class Stream(list):
    """Marks a script step as a streamed text response (list of deltas)."""


class FakeStream:
    def __init__(self, deltas: list[str]):
        self.events = [
            ResponseTextDeltaEvent(
                type="response.output_text.delta",
                content_index=0,
                delta=d,
                item_id="msg_stream",
                logprobs=[],
                output_index=0,
                sequence_number=i,
            )
            for i, d in enumerate(deltas)
        ]
        self.closed = False

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for ev in self.events:
            await asyncio.sleep(0)
            yield ev

    async def close(self) -> None:
        self.closed = True


class FakeResponses:
    def __init__(self, script: list[Any]):
        self.script = list(script)
        self.calls: list[dict] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(copy.deepcopy(kwargs))
        if not self.script:
            raise AssertionError(
                f"Unexpected extra model call: {kwargs.get('instructions', '')[:60]}"
            )
        step = self.script.pop(0)
        if callable(step) and not isinstance(step, Stream):
            step = step(kwargs)
        if isinstance(step, BaseException):
            raise step
        if kwargs.get("stream"):
            assert isinstance(step, Stream), "streaming call but script step is not a Stream"
            return FakeStream(list(step))
        assert not isinstance(step, Stream), "non-streaming call but script step is a Stream"
        return step


class FakeClient:
    def __init__(self, script: list[Any]):
        self.responses = FakeResponses(script)

    @property
    def calls(self) -> list[dict]:
        return self.responses.calls


# ---- fake tools ---------------------------------------------------------------------------------

CHUNKS: dict[int, Chunk] = {
    101: Chunk(
        chunk_id=101,
        article_id=1,
        title="Albert Einstein",
        url="https://simple.wikipedia.org/wiki/Albert_Einstein",
        section=None,
        chunk_index=0,
        text="Albert Einstein was a German-born physicist. He was born on 14 March 1879 in Ulm.",
    ),
    102: Chunk(
        chunk_id=102,
        article_id=1,
        title="Albert Einstein",
        url="https://simple.wikipedia.org/wiki/Albert_Einstein",
        section="Nobel Prize",
        chunk_index=3,
        text="Einstein won the Nobel Prize in Physics in 1921 for the photoelectric effect.",
    ),
}

TOOL_SPECS = [
    {"type": "function", "name": n, "parameters": {"type": "object"}, "strict": False}
    for n in ("semantic_search", "keyword_search", "fetch")
]


class ToolRecorder:
    def __init__(self, delay: float = 0.02):
        self.delay = delay
        self.calls: list[tuple[str, dict]] = []
        self.active = 0
        self.max_active = 0

    async def dispatch(self, name: str, args: dict | str) -> str:
        self.calls.append((name, args))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(self.delay)
        finally:
            self.active -= 1
        if name == "fetch":
            found = [CHUNKS[i].model_dump() for i in args.get("chunk_ids", []) if i in CHUNKS]
            return json.dumps({"chunks": found, "articles": []})
        return json.dumps(
            [
                {"query": q, "hits": [{"chunk_id": 101, "title": "Albert Einstein"}]}
                for q in args.get("queries", [])
            ]
        )

    def summarize(self, name: str, result: str) -> str:
        data = json.loads(result)
        if name == "fetch":
            return f"fetched {len(data['chunks'])} chunks"
        return f"{len(data)} queries, {sum(len(q['hits']) for q in data)} hits"


async def fake_fetch(chunk_ids: list[int] = [], article_ids: list[int] = []) -> FetchResult:  # noqa: B006
    return FetchResult(
        chunks=[CHUNKS[i] for i in chunk_ids if i in CHUNKS],
        articles=[],
        missing_chunk_ids=[i for i in chunk_ids if i not in CHUNKS],
        missing_article_ids=list(article_ids),
    )


def install(
    monkeypatch, script: list[Any], tools: ToolRecorder | None = None
) -> tuple[FakeClient, ToolRecorder]:
    """Swap in the fake model client and fake tools (patched on the real tool modules)."""
    import rag.agents.model
    from rag.tools import fetch, registry

    client = FakeClient(script)
    tools = tools or ToolRecorder()
    monkeypatch.setattr(rag.agents.model, "get_client", lambda: client)
    monkeypatch.setattr(registry, "TOOL_SPECS", TOOL_SPECS)
    monkeypatch.setattr(registry, "dispatch", tools.dispatch)
    monkeypatch.setattr(registry, "summarize", tools.summarize)
    monkeypatch.setattr(fetch, "fetch", fake_fetch)
    return client, tools


def last_input_text(call: dict) -> str:
    """All text content of a recorded call's input, joined (for assertions)."""
    parts = []
    inp = call.get("input")
    if isinstance(inp, str):
        return inp
    for item in inp or []:
        if isinstance(item.get("content"), str):
            parts.append(item["content"])
        if "output" in item:
            parts.append(str(item["output"]))
    return "\n".join(parts)


Step = Response | Stream | BaseException | Callable[[dict], Any]
