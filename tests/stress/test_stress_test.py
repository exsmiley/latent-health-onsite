"""Unit tests for scripts/stress_test.py against a tiny fake SSE ASGI app (no server, no model)."""

import asyncio
import json

import httpx
import pytest
from stress_test import (
    LogWatcher,
    RequestResult,
    RoundRobin,
    Sample,
    SSEParser,
    _parse_cputime,
    count_log,
    percentile,
    pool_from_log,
    run_one,
    run_ramp,
    should_stop,
    summarize_level,
    summarize_samples,
)


def sse(event: str, data) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


class FakeChat:
    """ASGI app for POST /api/chat. The question text picks the behaviour:
    ok / notfound / error / http500 / slow / incomplete. `delay_per_inflight` makes latency grow
    with the number of requests in flight (to exercise the p50 early stop)."""

    def __init__(self, delay_per_inflight: float = 0.0, fail_all: bool = False) -> None:
        self.delay_per_inflight = delay_per_inflight
        self.fail_all = fail_all
        self.inflight = 0
        self.max_inflight = 0
        self.calls = 0

    async def __call__(self, scope, receive, send):
        assert scope["type"] == "http"
        body = b""
        while True:
            msg = await receive()
            body += msg.get("body", b"")
            if not msg.get("more_body"):
                break
        q = json.loads(body)["messages"][-1]["content"]
        self.calls += 1
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        try:
            await self._respond(q, send)
        finally:
            self.inflight -= 1

    async def _respond(self, q: str, send) -> None:
        if self.delay_per_inflight:
            await asyncio.sleep(self.delay_per_inflight * self.inflight**2)
        if q == "http500" or self.fail_all:
            await send({"type": "http.response.start", "status": 500, "headers": []})
            await send({"type": "http.response.body", "body": b"boom"})
            return
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/event-stream")],
            }
        )

        async def out(chunk: bytes, more: bool = True) -> None:
            await send({"type": "http.response.body", "body": chunk, "more_body": more})

        await out(sse("status", {"stage": "research", "turn": 1, "max_turns": 7, "message": ""}))
        await out(b": ping\n\n")
        if q == "slow":
            await asyncio.sleep(5)
        if q == "error":
            await out(sse("error", {"message": "RateLimitError: 429"}))
            await out(sse("done", {}), more=False)
            return
        if q == "incomplete":
            await out(b"", more=False)
            return
        result = "not_found" if q == "notfound" else "supported"
        await out(sse("outcome", {"result": result, "turns_used": 3}))
        await out(sse("citations", []))
        # split one event across two body chunks, and use CRLF line endings for another
        await out(b'event: token\ndata: {"del')
        await out(b'ta": "Hello"}\n\n')
        await out(b'event: token\r\ndata: {"delta": " world"}\r\n\r\n')
        await out(sse("done", {}), more=False)


def client_for(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def one(app, question: str, timeout: float = 5.0) -> RequestResult:
    async with client_for(app) as c:
        return await run_one(
            c,
            "http://test",
            {"id": "q", "question": question},
            level=1,
            seq=0,
            timeout=timeout,
            run_t0=0.0,
        )


# ---- SSE parsing ---------------------------------------------------------------------------------


def test_parser_events_comments_and_multiline_data():
    p = SSEParser()
    events = p.feed(
        ': ping\n\nevent: status\ndata: {"turn": 1}\n\n'
        "event: note\ndata: line1\ndata: line2\n\n"
        'data: {"x": 2}\n\n'
    )
    assert events == [("status", {"turn": 1}), ("note", "line1\nline2"), ("message", {"x": 2})]
    assert p.comments == 1


def test_parser_line_by_line_with_crlf_and_no_space():
    p = SSEParser()
    out = [p.feed_line(x) for x in ["event:token\r", 'data:{"delta":"a"}\r', "\r"]]
    assert out == [None, None, ("token", {"delta": "a"})]
    assert p.feed_line("") is None  # stray blank line is not an event


# ---- one request against the fake app ------------------------------------------------------------


async def test_run_one_ok_parses_outcome_and_tokens():
    r = await one(FakeChat(), "ok")
    assert r.kind == "ok" and r.http_status == 200
    assert r.result == "supported" and r.turns_used == 3
    assert r.tokens == 2 and r.pings == 1
    assert r.events == 6  # status, outcome, citations, 2 tokens, done
    assert 0 <= r.first_event_s <= r.first_token_s <= r.total_s


async def test_run_one_not_found():
    r = await one(FakeChat(), "notfound")
    assert r.kind == "ok" and r.result == "not_found"


async def test_run_one_error_event_then_done_is_an_error():
    r = await one(FakeChat(), "error")
    assert r.kind == "error_event" and "429" in r.error and r.first_token_s is None


async def test_run_one_http_error():
    r = await one(FakeChat(), "http500")
    assert r.kind == "http_error" and r.http_status == 500 and "boom" in r.error


async def test_run_one_stream_without_done_is_incomplete():
    r = await one(FakeChat(), "incomplete")
    assert r.kind == "incomplete"


async def test_run_one_timeout():
    r = await one(FakeChat(), "slow", timeout=0.2)
    assert r.kind == "timeout" and r.total_s < 2


async def test_run_one_transport_error():
    def boom(request):
        raise httpx.ConnectError("refused")

    async with httpx.AsyncClient(transport=httpx.MockTransport(boom)) as c:
        r = await run_one(
            c, "http://x", {"id": "q", "question": "ok"}, level=1, seq=0, timeout=1, run_t0=0.0
        )
    assert r.kind == "transport" and "refused" in r.error


# ---- stats ---------------------------------------------------------------------------------------


def test_percentile_interpolates():
    assert percentile([], 50) is None
    assert percentile([5.0], 99) == 5.0
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile(range(1, 101), 90) == pytest.approx(90.1)
    assert percentile([3, 1, 2], 100) == 3


def _r(kind="ok", total=1.0, ttft=0.5, result="supported", turns=2):
    return RequestResult(
        level=2,
        seq=0,
        question_id="q",
        started_s=0,
        kind=kind,
        total_s=total,
        first_event_s=0.1,
        first_token_s=ttft if kind == "ok" else None,
        result=result if kind == "ok" else None,
        turns_used=turns if kind == "ok" else None,
        error=None if kind == "ok" else kind,
    )


def test_summarize_level():
    rs = [
        _r(total=1.0),
        _r(total=3.0, result="not_found", ttft=2.0),
        _r(total=2.0, result="out_of_turns", turns=7),
        _r(kind="timeout", total=300.0),
        _r(kind="error_event", total=0.3),
    ]
    s = summarize_level(2, rs, wall_s=30.0)
    assert s["requests"] == 5 and s["completed"] == 3
    assert s["throughput_per_min"] == 6.0
    assert s["total_s"]["p50"] == 2.0  # errors and timeouts excluded from latency
    assert s["ttft_s"]["p50"] == 0.5
    assert s["error_rate"] == 0.4
    assert s["errors"] == {
        "error_event": 1,
        "http_error": 0,
        "timeout": 1,
        "transport": 0,
        "incomplete": 0,
    }
    assert s["outcomes"] == {"supported": 1, "not_found": 1, "out_of_turns": 1}
    assert s["turns_mean"] == pytest.approx(3.67)


def test_should_stop():
    def summ(err, p50):
        return {"error_rate": err, "total_s": {"p50": p50}}

    assert should_stop(summ(0.0, 10.0), 10.0) is None
    assert should_stop(summ(0.2, 10.0), 10.0) is None  # exactly 20% is allowed
    assert "error rate" in should_stop(summ(0.25, 10.0), 10.0)
    assert should_stop(summ(0.0, 40.0), 10.0) is None
    assert "p50" in should_stop(summ(0.0, 40.1), 10.0)
    assert should_stop(summ(0.0, None), 10.0) is None  # nothing completed: error rule decides
    assert should_stop(summ(0.0, 99.0), None) is None
    assert should_stop(summ(0.0, 30.0), 10.0, p50_factor=2) is not None


def test_round_robin_is_seeded_and_wraps():
    qs = [{"id": str(i)} for i in range(5)]
    a, b = RoundRobin(qs, 1), RoundRobin(qs, 1)
    seq_a = [a.next()["id"] for _ in range(10)]
    assert seq_a == [b.next()["id"] for _ in range(10)]
    assert sorted(seq_a[:5]) == ["0", "1", "2", "3", "4"] and seq_a[:5] == seq_a[5:]


# ---- ramp ----------------------------------------------------------------------------------------


async def test_ramp_closed_loop_keeps_n_in_flight():
    app = FakeChat(delay_per_inflight=0.01)
    qs = RoundRobin([{"id": "a", "question": "ok"}], 0)
    run = await run_ramp(
        "http://test",
        [1, 4],
        qs,
        requests_per_level=8,
        transport=httpx.ASGITransport(app=app),
        verbose=False,
        p50_factor=1000,
    )
    assert [lv["requests"] for lv in run["levels"]] == [8, 8]
    assert app.calls == 16 and app.max_inflight == 4
    assert run["stopped_early"] is None and len(run["requests"]) == 16
    assert run["levels"][1]["outcomes"]["supported"] == 8


async def test_ramp_default_requests_per_level():
    app = FakeChat()
    qs = RoundRobin([{"id": "a", "question": "ok"}], 0)
    run = await run_ramp(
        "http://test", [1, 6], qs, transport=httpx.ASGITransport(app=app), verbose=False
    )
    assert [lv["requests"] for lv in run["levels"]] == [8, 12]


async def test_ramp_stops_on_error_rate():
    app = FakeChat(fail_all=True)
    qs = RoundRobin([{"id": "a", "question": "ok"}], 0)
    run = await run_ramp(
        "http://test",
        [1, 2, 4],
        qs,
        requests_per_level=4,
        transport=httpx.ASGITransport(app=app),
        verbose=False,
    )
    assert len(run["levels"]) == 1 and "error rate" in run["stopped_early"]
    assert run["levels"][0]["errors"]["http_error"] == 4


async def test_ramp_stops_when_p50_degrades():
    # latency = 0.004 * inflight^2: ~4 ms alone, ~260 ms with 8 in flight
    app = FakeChat(delay_per_inflight=0.004)
    qs = RoundRobin([{"id": "a", "question": "ok"}], 0)
    run = await run_ramp(
        "http://test",
        [1, 8, 16],
        qs,
        requests_per_level=8,
        transport=httpx.ASGITransport(app=app),
        verbose=False,
    )
    assert len(run["levels"]) == 2 and "p50" in run["stopped_early"]


# ---- server-side signal helpers ------------------------------------------------------------------

LOG = """\
10:00:00.001 INFO httpx: HTTP Request: POST https://api.openai.com/v1/responses "HTTP/1.1 200 OK"
10:00:00.002 INFO httpx: HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"
10:00:01.000 INFO httpx: HTTP Request: POST https://api.openai.com/v1/responses "HTTP/1.1 429 Too Many Requests"
10:00:01.001 INFO openai._base_client: Retrying request to /responses in 0.400000 seconds
10:00:02.000 INFO httpx: HTTP Request: POST https://api.openai.com/v1/responses "HTTP/1.1 503 Service Unavailable"
10:00:03.000 INFO httpx: HTTP Request: GET http://127.0.0.1:8300/api/health "HTTP/1.1 200 OK"
Traceback (most recent call last):
"""


def test_count_log():
    c = count_log(LOG)
    assert c["openai_requests"] == 4
    assert c["openai_by_endpoint"] == {"responses": 3, "embeddings": 1}
    assert c["http_429"] == 1 and c["http_5xx"] == 1 and c["retries"] == 1
    assert c["tracebacks"] == 1


def test_log_watcher_counts_only_the_slice(tmp_path):
    path = tmp_path / "server.log"
    path.write_text(LOG)
    w = LogWatcher(path)
    start = w.offset()
    with path.open("a") as f:
        f.write(LOG.splitlines()[2] + "\n")
    assert w.count(start, w.offset())["http_429"] == 1
    assert LogWatcher(None).count(0, 10) is None


def test_pool_from_log_tracks_in_use_and_queueing():
    req, give, ret = (
        "1 INFO psycopg.pool: connection requested from 'pool-1'",
        "1 INFO psycopg.pool: connection given by 'pool-1'",
        "1 INFO psycopg.pool: returning connection to 'pool-1'",
    )
    # 3 connections taken, then a 4th request while all 3 (pool_max) are busy, then releases
    text = "\n".join([req, give] * 3 + [req, ret, give, ret, ret, ret])
    p = pool_from_log(text, pool_max=3)
    assert p["requests"] == 4 and p["peak_in_use"] == 3 and p["queued"] == 1 and p["saturated"]
    assert pool_from_log("\n".join([req, give, ret] * 5), pool_max=3)["saturated"] is False


def test_parse_cputime():
    assert _parse_cputime("0:01.50") == 1.5
    assert _parse_cputime("2:03.00") == 123.0
    assert _parse_cputime("1:00:00") == 3600.0
    assert _parse_cputime("1-00:00:01") == 86401.0


def test_summarize_samples():
    ss = [
        Sample(t=0, pg_total=3, pg_active=1, server_db_conns=2, cpu_s=10.0, rss_mb=100),
        Sample(t=1, pg_total=22, pg_active=9, server_db_conns=20, cpu_s=10.5, rss_mb=120),
        Sample(t=2, pg_total=5, pg_active=2, server_db_conns=20, cpu_s=10.7, rss_mb=110),
    ]
    s = summarize_samples(ss)
    assert s["pg_total_peak"] == 22 and s["pg_active_peak"] == 9
    assert s["server_db_conns_peak"] == 20 and s["pool_saturated"] is True
    assert s["server_cpu_pct_avg"] == 35.0 and s["server_cpu_pct_peak"] == 50.0
    assert s["server_rss_mb_peak"] == 120
    assert summarize_samples([])["pool_saturated"] is None
