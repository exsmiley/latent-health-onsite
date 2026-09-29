import asyncio
from types import SimpleNamespace

from fakes import msg, resp
from openai.types.responses import ResponseUsage
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails

import rag.agents.model
from rag import llm
from rag.agents import model
from rag.usage import track_usage


def usage(inp: int, out: int, cached: int = 0, reasoning: int = 0) -> ResponseUsage:
    return ResponseUsage(
        input_tokens=inp,
        input_tokens_details=InputTokensDetails(cached_tokens=cached, cache_write_tokens=0),
        output_tokens=out,
        output_tokens_details=OutputTokensDetails(reasoning_tokens=reasoning),
        total_tokens=inp + out,
    )


class Stream:
    def __init__(self, u):
        self.events = [
            SimpleNamespace(type="response.output_text.delta", item_id="m", delta="hi"),
            SimpleNamespace(type="response.completed", response=SimpleNamespace(usage=u)),
        ]

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for e in self.events:
            yield e

    async def close(self):
        pass


class Client:
    def __init__(self):
        self.responses = self

    async def create(self, stream=False, **kwargs):
        await asyncio.sleep(0.01)
        u = usage(kwargs["input"], 10, cached=kwargs["input"] // 2, reasoning=3)
        if stream:
            return Stream(u)
        r = resp(msg("ok"))
        r.usage = u
        return r


def install(monkeypatch):
    monkeypatch.setattr(rag.agents.model, "get_client", lambda: Client())


async def test_track_usage_sums_create_and_stream(monkeypatch):
    install(monkeypatch)
    with model.track_usage() as u:
        await model.create(input=100)
        assert [d async for d in model.stream_text(input=40)] == ["hi"]
    assert u.to_dict() == {
        "calls": 2,
        "input_tokens": 140,
        "cached_input_tokens": 70,
        "output_tokens": 20,
        "reasoning_tokens": 6,
        "embedding_calls": 0,
        "embedding_tokens": 0,
    }
    # after the block, calls are no longer recorded (and nothing breaks)
    await model.create(input=5)
    assert u.calls == 2


async def test_no_tracker_is_a_no_op(monkeypatch):
    install(monkeypatch)
    r = await model.create(input=1)
    assert r.output_text == "ok"


async def test_trackers_are_isolated_per_task(monkeypatch):
    install(monkeypatch)

    async def job(n: int):
        with track_usage() as u:
            # spawned sub-tasks (like concurrent tool calls) share the parent's tracker
            await asyncio.gather(*(model.create(input=n) for _ in range(3)))
        return u

    a, b = await asyncio.gather(job(10), job(1000))
    assert (a.calls, a.input_tokens) == (3, 30)
    assert (b.calls, b.input_tokens) == (3, 3000)


async def test_embedding_calls_are_counted(monkeypatch):
    class Emb:
        async def create(self, model, input):
            return SimpleNamespace(
                data=[SimpleNamespace(index=i, embedding=[0.0]) for i in range(len(input))],
                usage=SimpleNamespace(prompt_tokens=7 * len(input), total_tokens=7 * len(input)),
            )

    monkeypatch.setattr(llm, "get_client", lambda: SimpleNamespace(embeddings=Emb()))
    with track_usage() as u:
        await llm.embed_texts(["a", "b"])
        await llm.embed_texts(["c"])
    assert (u.embedding_calls, u.embedding_tokens) == (2, 21)
    await llm.embed_texts(["d"])  # no tracker: fine
