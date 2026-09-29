import asyncio
import json

import httpx
import pytest
from fakes import Stream, fc, resp, submit, verdict

from rag.agents import events
from rag.api import app as app_module
from rag.api.app import app, sse_stream


def client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def parse_sse(body: str) -> tuple[list[tuple[str, object]], int]:
    """Returns ([(event, data)], number of ping comments)."""
    out, pings = [], 0
    for block in body.split("\n\n"):
        if not block.strip():
            continue
        if block.startswith(":"):
            pings += 1
            continue
        lines = dict(line.split(": ", 1) for line in block.split("\n"))
        out.append((lines["event"], json.loads(lines["data"])))
    return out, pings


async def test_health():
    async with client() as c:
        r = await c.get("/api/health")
    assert r.status_code == 200 and r.json() == {"ok": True}


async def test_chunk_lookup(fake_env):
    fake_env([])
    async with client() as c:
        ok = await c.get("/api/chunks/101")
        missing = await c.get("/api/chunks/999")
    assert ok.status_code == 200
    assert ok.json()["chunk_id"] == 101 and ok.json()["title"] == "Albert Einstein"
    assert set(ok.json()) == {
        "chunk_id", "article_id", "title", "url", "section", "chunk_index", "text",
    }  # fmt: skip
    assert missing.status_code == 404


async def test_cors_preflight():
    async with client() as c:
        r = await c.options(
            "/api/chat",
            headers={
                "Origin": "http://localhost:5180",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
    assert r.headers["access-control-allow-origin"] == "http://localhost:5180"


async def test_chat_streams_contract_events(fake_env):
    client_, _ = fake_env(
        [
            resp(fc("semantic_search", {"queries": ["einstein born"], "top_k": None}, "c1")),
            submit("Born 1879.", [101]),
            verdict("supported", "14 March 1879", "ok"),
            Stream(["Born ", "1879 [1]."]),
        ]
    )
    messages = [
        {"role": "user", "content": "Who was Einstein?"},
        {"role": "assistant", "content": "A physicist."},
        {"role": "user", "content": "When was he born?"},
    ]
    async with client() as c:
        r = await c.post("/api/chat", json={"messages": messages})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    evs, _ = parse_sse(r.text)
    assert [e for e, _ in evs] == [
        "status", "tool_call", "tool_result",
        "status", "tool_call", "tool_result",
        "research_answer", "status", "evaluation", "status", "citations",
        "token", "token", "done",
    ]  # fmt: skip
    data = dict(evs)  # last of each type
    assert set(evs[0][1]) == {"stage", "round", "turn", "message"}
    assert evs[1][1] == {
        "id": "c1",
        "name": "semantic_search",
        "arguments": {"queries": ["einstein born"], "top_k": None},
        "round": 1,
        "turn": 1,
    }
    assert evs[2][1] == {"id": "c1", "name": "semantic_search", "summary": "1 queries, 1 hits"}
    assert data["research_answer"] == {"round": 1, "answer": "Born 1879.", "citations": [101]}
    assert data["evaluation"] == {
        "round": 1, "verdict": "supported", "independent_answer": "14 March 1879", "feedback": "ok",
    }  # fmt: skip
    assert data["citations"] == [
        {
            "n": 1,
            "chunk_id": 101,
            "article_id": 1,
            "title": "Albert Einstein",
            "section": None,
            "url": "https://simple.wikipedia.org/wiki/Albert_Einstein",
        }
    ]
    assert [d["delta"] for e, d in evs if e == "token"] == ["Born ", "1879 [1]."]
    assert data["done"] == {}
    # History reached the research agent; the last message is the question.
    assert client_.calls[0]["input"][:3] == messages


async def test_chat_error_event_then_done(fake_env):
    fake_env([RuntimeError("model unavailable")])
    async with client() as c:
        r = await c.post("/api/chat", json={"messages": [{"role": "user", "content": "q"}]})
    evs, _ = parse_sse(r.text)
    assert [e for e, _ in evs] == ["status", "error", "done"]
    assert "model unavailable" in evs[1][1]["message"]


@pytest.mark.parametrize(
    "body",
    [
        {"messages": []},
        {"messages": [{"role": "assistant", "content": "hi"}]},
        {"messages": [{"role": "system", "content": "hi"}]},
    ],
)
async def test_chat_rejects_bad_requests(body):
    async with client() as c:
        r = await c.post("/api/chat", json=body)
    assert r.status_code == 422


async def test_heartbeat_while_waiting(monkeypatch):
    monkeypatch.setattr(app_module, "HEARTBEAT_SECONDS", 0.02)

    async def slow_run(question, history):
        yield events.status("research", 1, 1, "thinking")
        await asyncio.sleep(0.1)
        yield events.done()

    monkeypatch.setattr(app_module.orchestrator, "run", slow_run)
    async with client() as c:
        r = await c.post("/api/chat", json={"messages": [{"role": "user", "content": "q"}]})
    evs, pings = parse_sse(r.text)
    assert [e for e, _ in evs] == ["status", "done"]
    assert pings >= 2
    assert ": ping\n\n" in r.text


class FakeRequest:
    def __init__(self, disconnect_after: float):
        self.disconnect_after = disconnect_after

    async def receive(self):
        await asyncio.sleep(self.disconnect_after)
        return {"type": "http.disconnect"}


async def test_disconnect_stops_orchestrator():
    state = {"closed": False, "cancelled": False}

    async def endless():
        try:
            yield events.status("research", 1, 1, "start")
            await asyncio.sleep(10)  # waiting on the model
            yield events.done()
        except asyncio.CancelledError:
            state["cancelled"] = True
            raise
        finally:
            state["closed"] = True

    out = [chunk async for chunk in sse_stream(FakeRequest(0.05), endless())]
    assert len(out) == 1 and out[0].startswith("event: status")
    assert state["closed"] and state["cancelled"]
