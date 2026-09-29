"""Closed-loop HTTP load generator for `POST /api/chat` (SSE).

At each concurrency level N it keeps N requests in flight until `--requests-per-level` requests
(default max(8, 2N)) have been sent, recording per request the time to the first SSE event, to
the first `token`, and to `done`, plus the outcome. After each level it prints and stores
throughput, latency percentiles, error rates and outcome counts, and stops ramping early when the
error rate goes over `--max-error-rate` or p50 goes over `--p50-factor` x the first level's p50.

While it runs it samples server-side signals without touching the server's code:

- Postgres connections for the DB (`pg_stat_activity`, all clients, every second);
- the server process: CPU (from cumulative CPU time), RSS and its own TCP connections to the DB
  port (`ps` / `lsof`, found from the listening port or `--server-pid`);
- the server log (`--server-log`): OpenAI HTTP status codes, 429s and SDK "Retrying request"
  lines, counted per level. That needs the `httpx`/`openai` loggers at INFO; start the server with
  `scripts/stress_logging.json` (see docs/perf/STRESS.md).

Usage:
    uv run python scripts/stress_test.py --levels 1,2,4 --requests-per-level 4 \
        --server-log /path/to/server.log
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import psycopg

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = ROOT / "evals" / "questions.jsonl"
DEFAULT_OUT_DIR = ROOT / "evals" / "results" / "stress"
DEFAULT_DB_URL = "postgresql://rag:rag@localhost:5433/rag"
POOL_MAX_SIZE = 20  # rag.db.get_pool(max_size=20)

ERROR_KINDS = ("error_event", "http_error", "timeout", "transport", "incomplete")
RESULTS = ("supported", "not_found", "out_of_turns")


# ---- SSE parsing ---------------------------------------------------------------------------------


class SSEParser:
    """Incremental text/event-stream parser. Feed it lines (without the trailing newline); it
    returns `(event, data)` when a blank line completes an event. `data` is JSON-decoded when it
    parses, else the raw string. Comment lines (`: ping`) are counted, not returned."""

    def __init__(self) -> None:
        self.comments = 0
        self._event: str | None = None
        self._data: list[str] = []

    def feed_line(self, line: str) -> tuple[str, Any] | None:
        line = line.rstrip("\r")
        if line == "":
            if self._event is None and not self._data:
                return None
            name, raw = self._event or "message", "\n".join(self._data)
            self._event, self._data = None, []
            try:
                return name, json.loads(raw)
            except ValueError:
                return name, raw
        if line.startswith(":"):
            self.comments += 1
            return None
        key, _, value = line.partition(":")
        value = value.removeprefix(" ")
        if key == "event":
            self._event = value
        elif key == "data":
            self._data.append(value)
        return None

    def feed(self, text: str) -> list[tuple[str, Any]]:
        """Parse a whole chunk of text (for tests and offline use)."""
        out = []
        for line in text.split("\n"):
            ev = self.feed_line(line)
            if ev is not None:
                out.append(ev)
        return out


# ---- one request ---------------------------------------------------------------------------------


@dataclass
class RequestResult:
    level: int
    seq: int
    question_id: str
    started_s: float  # since the run started
    kind: str = "incomplete"  # "ok" or one of ERROR_KINDS
    http_status: int | None = None
    first_event_s: float | None = None
    first_token_s: float | None = None
    total_s: float | None = None
    result: str | None = None  # outcome.result
    turns_used: int | None = None
    events: int = 0
    tokens: int = 0
    pings: int = 0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.kind == "ok"


async def run_one(
    client: httpx.AsyncClient,
    url: str,
    question: dict,
    *,
    level: int,
    seq: int,
    timeout: float,
    run_t0: float,
) -> RequestResult:
    t0 = time.perf_counter()
    r = RequestResult(level=level, seq=seq, question_id=question["id"], started_s=t0 - run_t0)
    body = {"messages": [{"role": "user", "content": question["question"]}]}
    parser = SSEParser()
    saw_error = False
    try:
        async with asyncio.timeout(timeout):
            async with client.stream("POST", f"{url}/api/chat", json=body) as resp:
                r.http_status = resp.status_code
                if resp.status_code != 200:
                    await resp.aread()
                    r.kind, r.error = "http_error", f"HTTP {resp.status_code}: {resp.text[:200]}"
                    return r
                async for line in resp.aiter_lines():
                    ev = parser.feed_line(line)
                    if ev is None:
                        continue
                    now = time.perf_counter() - t0
                    name, data = ev
                    r.events += 1
                    if r.first_event_s is None:
                        r.first_event_s = now
                    if name == "token":
                        r.tokens += 1
                        if r.first_token_s is None:
                            r.first_token_s = now
                    elif name == "outcome" and isinstance(data, dict):
                        r.result = data.get("result")
                        r.turns_used = data.get("turns_used")
                    elif name == "error":
                        saw_error = True
                        r.error = data.get("message") if isinstance(data, dict) else str(data)
                    elif name == "done":
                        r.kind = "error_event" if saw_error else "ok"
                        break
                else:
                    r.kind = "error_event" if saw_error else "incomplete"
    except (TimeoutError, httpx.TimeoutException):
        r.kind, r.error = "timeout", f"no done after {timeout:g}s"
    except httpx.HTTPError as e:
        r.kind, r.error = "transport", f"{type(e).__name__}: {e}"
    finally:
        r.total_s = time.perf_counter() - t0
        r.pings = parser.comments
    return r


# ---- questions -----------------------------------------------------------------------------------


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class RoundRobin:
    """Questions shuffled once with `seed`, then handed out in order, wrapping around. The cursor
    carries over between levels so each level gets fresh questions until the set wraps."""

    def __init__(self, questions: Sequence[dict], seed: int) -> None:
        if not questions:
            raise ValueError("no questions")
        self.items = list(questions)
        random.Random(seed).shuffle(self.items)
        self.i = 0

    def next(self) -> dict:
        q = self.items[self.i % len(self.items)]
        self.i += 1
        return q


# ---- stats ---------------------------------------------------------------------------------------


def percentile(values: Iterable[float], p: float) -> float | None:
    """Linear-interpolated percentile (numpy's default); None for no values."""
    xs = sorted(values)
    if not xs:
        return None
    k = (len(xs) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _pcts(values: list[float]) -> dict[str, float | None]:
    return {f"p{p}": _round(percentile(values, p)) for p in (50, 90, 99)}


def _round(x: float | None, nd: int = 3) -> float | None:
    return None if x is None else round(x, nd)


def summarize_level(level: int, results: list[RequestResult], wall_s: float) -> dict:
    ok = [r for r in results if r.ok]
    n = len(results)
    errors = Counter(r.kind for r in results if not r.ok)
    outcomes = Counter(r.result for r in ok if r.result)
    turns = [r.turns_used for r in ok if r.turns_used is not None]
    return {
        "level": level,
        "requests": n,
        "completed": len(ok),
        "wall_s": round(wall_s, 3),
        "throughput_per_min": round(len(ok) / wall_s * 60, 2) if wall_s > 0 else 0.0,
        "total_s": _pcts([r.total_s for r in ok if r.total_s is not None]),
        "ttft_s": _pcts([r.first_token_s for r in ok if r.first_token_s is not None]),
        "first_event_s": _pcts([r.first_event_s for r in ok if r.first_event_s is not None]),
        "error_rate": round(sum(errors.values()) / n, 4) if n else 0.0,
        "errors": {k: errors.get(k, 0) for k in ERROR_KINDS},
        "outcomes": {k: outcomes.get(k, 0) for k in RESULTS},
        "turns_mean": round(sum(turns) / len(turns), 2) if turns else None,
        "error_samples": sorted({r.error for r in results if r.error})[:5],
    }


def should_stop(
    summary: dict,
    baseline_p50: float | None,
    *,
    max_error_rate: float = 0.20,
    p50_factor: float = 4.0,
) -> str | None:
    """Why ramping should stop after this level, or None to keep going."""
    if summary["error_rate"] > max_error_rate:
        return f"error rate {summary['error_rate']:.0%} > {max_error_rate:.0%}"
    p50 = summary["total_s"]["p50"]
    if baseline_p50 and p50 is not None and p50 > p50_factor * baseline_p50:
        return f"p50 {p50:.1f}s > {p50_factor:g}x baseline p50 {baseline_p50:.1f}s"
    return None


# ---- server-side signals -------------------------------------------------------------------------

_HTTPX_LINE = re.compile(r'HTTP Request: (\w+) (\S+) "HTTP/[\d.]+ (\d{3})')


def count_log(text: str) -> dict:
    """Counts OpenAI HTTP calls (by endpoint and status), 429s, SDK retries and tracebacks in a
    slice of server log written with the httpx/openai loggers at INFO."""
    statuses: Counter[str] = Counter()
    endpoints: Counter[str] = Counter()
    for m in _HTTPX_LINE.finditer(text):
        _, u, status = m.groups()
        if "api.openai.com" not in u:
            continue
        statuses[status] += 1
        endpoints[urlparse(u).path.rsplit("/", 1)[-1]] += 1
    return {
        "openai_requests": sum(statuses.values()),
        "openai_by_endpoint": dict(endpoints),
        "openai_status": dict(statuses),
        "http_429": statuses.get("429", 0),
        "http_5xx": sum(v for k, v in statuses.items() if k.startswith("5")),
        "retries": text.count("Retrying request"),
        "tracebacks": text.count("Traceback (most recent call last)"),
        "pool": pool_from_log(text),
    }


def pool_from_log(text: str, pool_max: int = POOL_MAX_SIZE) -> dict:
    """Replays psycopg.pool INFO lines ("connection requested" / "given" / "returning") to get
    the pool's peak in-use connections and peak waiters, and how many connection requests arrived
    while all `pool_max` connections were in use (i.e. had to queue). Assumes the slice starts
    with the pool idle, which holds between closed-loop levels."""
    waiting = in_use = peak_wait = peak_use = queued = requests = 0
    for line in text.splitlines():
        if "psycopg.pool" not in line:
            continue
        if "connection requested from" in line:
            requests += 1
            if in_use >= pool_max:
                queued += 1
            waiting += 1
            peak_wait = max(peak_wait, waiting)
        elif "connection given by" in line:
            waiting = max(0, waiting - 1)
            in_use += 1
            peak_use = max(peak_use, in_use)
        elif "returning connection to" in line:
            in_use = max(0, in_use - 1)
    return {
        "requests": requests,
        "peak_in_use": peak_use,
        "peak_waiting": peak_wait,
        "queued": queued,
        "saturated": queued > 0,
    }


class LogWatcher:
    def __init__(self, path: Path | None) -> None:
        self.path = path

    def offset(self) -> int:
        if self.path is None or not self.path.exists():
            return 0
        return self.path.stat().st_size

    def count(self, start: int, end: int) -> dict | None:
        if self.path is None or not self.path.exists():
            return None
        with self.path.open("rb") as f:
            f.seek(start)
            text = f.read(max(0, end - start)).decode("utf-8", "replace")
        return count_log(text)


def _parse_cputime(s: str) -> float:
    """ps `time` column: [[dd-]hh:]mm:ss[.ss] -> seconds."""
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
    parts = [float(p) for p in s.split(":")]
    secs = 0.0
    for p in parts:
        secs = secs * 60 + p
    return days * 86400 + secs


def find_listening_pid(port: int) -> int | None:
    try:
        out = subprocess.run(
            ["lsof", "-nP", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"],
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        ).stdout.split()
    except (OSError, subprocess.SubprocessError):
        return None
    return int(out[0]) if out else None


async def _run(*cmd: str) -> str:
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
    )
    out, _ = await proc.communicate()
    return out.decode()


@dataclass
class Sample:
    t: float
    pg_total: int | None = None
    pg_active: int | None = None
    server_db_conns: int | None = None
    cpu_s: float | None = None
    rss_mb: float | None = None


class Monitor:
    """Samples Postgres and the server process every `interval` seconds in the background."""

    def __init__(
        self,
        run_t0: float,
        db_url: str | None,
        server_pid: int | None,
        db_port: int,
        interval: float = 1.0,
    ) -> None:
        self.run_t0, self.db_url, self.pid, self.db_port = run_t0, db_url, server_pid, db_port
        self.interval = interval
        self.samples: list[Sample] = []
        self.db_name = urlparse(db_url).path.lstrip("/") if db_url else None
        self._task: asyncio.Task | None = None
        self._conn: psycopg.AsyncConnection | None = None
        self.notes: list[str] = []

    async def start(self) -> None:
        if self.db_url:
            try:
                self._conn = await psycopg.AsyncConnection.connect(self.db_url, autocommit=True)
            except psycopg.Error as e:
                self.notes.append(f"pg sampling off: {e}")
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._conn is not None:
            await self._conn.close()

    async def _loop(self) -> None:
        while True:
            try:
                self.samples.append(await self.sample())
            except Exception as e:  # noqa: BLE001 - sampling must never kill the run
                self.notes.append(f"sample failed: {type(e).__name__}: {e}")
            await asyncio.sleep(self.interval)

    async def sample(self) -> Sample:
        s = Sample(t=time.perf_counter() - self.run_t0)
        if self._conn is not None:
            cur = await self._conn.execute(
                "SELECT count(*), count(*) FILTER (WHERE state = 'active') FROM pg_stat_activity"
                " WHERE datname = %s AND pid <> pg_backend_pid()",
                (self.db_name,),
            )
            s.pg_total, s.pg_active = await cur.fetchone()
        if self.pid:
            out = (await _run("ps", "-o", "time=,rss=", "-p", str(self.pid))).split()
            if len(out) >= 2:
                s.cpu_s, s.rss_mb = _parse_cputime(out[0]), round(int(out[1]) / 1024, 1)
            lsof = await _run(
                "lsof", "-nP", "-a", "-p", str(self.pid), f"-iTCP:{self.db_port}", "-F", "n"
            )
            s.server_db_conns = sum(1 for ln in lsof.splitlines() if ln.startswith("n"))
        return s

    def window(self, t_start: float, t_end: float) -> dict:
        return summarize_samples([s for s in self.samples if t_start <= s.t <= t_end])


def summarize_samples(samples: list[Sample], pool_max: int = POOL_MAX_SIZE) -> dict:
    def peak(attr: str) -> Any:
        vals = [getattr(s, attr) for s in samples if getattr(s, attr) is not None]
        return max(vals) if vals else None

    cpu = [s for s in samples if s.cpu_s is not None]
    cpu_avg = cpu_peak = None
    if len(cpu) >= 2 and cpu[-1].t > cpu[0].t:
        cpu_avg = round((cpu[-1].cpu_s - cpu[0].cpu_s) / (cpu[-1].t - cpu[0].t) * 100, 1)
        steps = [(b.cpu_s - a.cpu_s) / (b.t - a.t) * 100 for a, b in pairwise(cpu) if b.t > a.t]
        cpu_peak = round(max(steps), 1) if steps else None
    conns = peak("server_db_conns")
    return {
        "samples": len(samples),
        "pg_total_peak": peak("pg_total"),
        "pg_active_peak": peak("pg_active"),
        "server_db_conns_peak": conns,
        "pool_saturated": None if conns is None else conns >= pool_max,
        "server_cpu_pct_avg": cpu_avg,
        "server_cpu_pct_peak": cpu_peak,
        "server_rss_mb_peak": peak("rss_mb"),
    }


# ---- driver --------------------------------------------------------------------------------------


async def run_level(
    client: httpx.AsyncClient,
    url: str,
    questions: RoundRobin,
    level: int,
    n_requests: int,
    timeout: float,
    run_t0: float,
    on_result=None,
) -> tuple[list[RequestResult], float]:
    """Closed loop: `level` workers each send a request as soon as their previous one ends,
    until `n_requests` have been sent."""
    results: list[RequestResult] = []
    sent = 0

    async def worker() -> None:
        nonlocal sent
        while sent < n_requests:
            seq, q = sent, questions.next()
            sent += 1
            r = await run_one(client, url, q, level=level, seq=seq, timeout=timeout, run_t0=run_t0)
            results.append(r)
            if on_result:
                on_result(r)

    t0 = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(min(level, n_requests))))
    return results, time.perf_counter() - t0


def requests_for(level: int, fixed: int | None) -> int:
    return fixed if fixed else max(8, 2 * level)


async def run_ramp(
    url: str,
    levels: list[int],
    questions: RoundRobin,
    *,
    requests_per_level: int | None = None,
    timeout: float = 300.0,
    max_error_rate: float = 0.20,
    p50_factor: float = 4.0,
    monitor: Monitor | None = None,
    logs: LogWatcher | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    run_t0: float | None = None,
    verbose: bool = True,
) -> dict:
    run_t0 = time.perf_counter() if run_t0 is None else run_t0
    logs = logs or LogWatcher(None)
    limits = httpx.Limits(max_connections=max(levels) + 4, max_keepalive_connections=max(levels))
    http_timeout = httpx.Timeout(timeout, connect=10.0)
    levels_out: list[dict] = []
    all_results: list[RequestResult] = []
    stopped: str | None = None
    baseline: float | None = None

    def progress(r: RequestResult) -> None:
        if verbose:
            tt = f"{r.total_s:6.1f}s" if r.total_s is not None else "   -  "
            print(
                f"  [N={r.level:>3} #{r.seq:>3}] {r.question_id} {r.kind:<11} {tt} "
                f"{r.result or '':<12} turns={r.turns_used}"
                + (f"  {r.error[:80]}" if r.error else ""),
                file=sys.stderr,
                flush=True,
            )

    async with httpx.AsyncClient(timeout=http_timeout, limits=limits, transport=transport) as c:
        for level in levels:
            n = requests_for(level, requests_per_level)
            if verbose:
                print(f"level {level}: {n} requests", file=sys.stderr, flush=True)
            off0, t_start = logs.offset(), time.perf_counter() - run_t0
            results, wall = await run_level(c, url, questions, level, n, timeout, run_t0, progress)
            t_end, off1 = time.perf_counter() - run_t0, logs.offset()
            summary = summarize_level(level, results, wall)
            summary["t_start_s"], summary["t_end_s"] = round(t_start, 2), round(t_end, 2)
            summary["server_log"] = logs.count(off0, off1)
            summary["server"] = monitor.window(t_start, t_end) if monitor else None
            levels_out.append(summary)
            all_results.extend(results)
            if verbose:
                print(format_table([summary], header=False), file=sys.stderr, flush=True)
            if baseline is None:
                baseline = summary["total_s"]["p50"]
            stopped = should_stop(
                summary, baseline, max_error_rate=max_error_rate, p50_factor=p50_factor
            )
            if stopped:
                stopped = f"after level {level}: {stopped}"
                break
    return {
        "levels": levels_out,
        "stopped_early": stopped,
        "baseline_p50_s": baseline,
        "requests": [asdict(r) for r in all_results],
    }


def _f(x: float | None, nd: int = 1) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def format_table(levels: list[dict], header: bool = True) -> str:
    cols = (
        f"{'N':>3} {'req':>4} {'ok':>4} {'ans/min':>7} | {'p50':>6} {'p90':>6} {'p99':>6} | "
        f"{'ttft50':>6} {'ttft90':>6} {'ttft99':>6} | {'err%':>5} {'errors':<14} | "
        f"{'sup':>3} {'nf':>3} {'oot':>3} {'turns':>5} | {'oai':>4} {'429':>4} {'retry':>5} | "
        f"{'dbc':>3} {'use':>3} {'plQ':>4} {'pgA':>3} {'cpu%':>5} {'cpuPk':>5}"
    )
    lines = [cols, "-" * len(cols)] if header else []
    for s in levels:
        errs = ",".join(f"{k[:4]}={v}" for k, v in s["errors"].items() if v) or "-"
        lg = s.get("server_log") or {}
        sv = s.get("server") or {}
        pl = lg.get("pool") or {}
        lines.append(
            f"{s['level']:>3} {s['requests']:>4} {s['completed']:>4} "
            f"{s['throughput_per_min']:>7.2f} | "
            f"{_f(s['total_s']['p50']):>6} {_f(s['total_s']['p90']):>6} "
            f"{_f(s['total_s']['p99']):>6} | "
            f"{_f(s['ttft_s']['p50']):>6} {_f(s['ttft_s']['p90']):>6} "
            f"{_f(s['ttft_s']['p99']):>6} | "
            f"{s['error_rate'] * 100:>5.1f} {errs:<14} | "
            f"{s['outcomes']['supported']:>3} {s['outcomes']['not_found']:>3} "
            f"{s['outcomes']['out_of_turns']:>3} {_f(s['turns_mean'], 2):>5} | "
            f"{lg.get('openai_requests', '-'):>4} {lg.get('http_429', '-'):>4} "
            f"{lg.get('retries', '-'):>5} | "
            f"{_s(sv.get('server_db_conns_peak')):>3} {_s(pl.get('peak_in_use')):>3} "
            f"{_s(pl.get('queued')):>4} {_s(sv.get('pg_active_peak')):>3} "
            f"{_f(sv.get('server_cpu_pct_avg')):>5} {_f(sv.get('server_cpu_pct_peak')):>5}"
        )
    return "\n".join(lines)


def _s(x: Any) -> str:
    return "-" if x is None else str(x)


def summary_only(run: dict) -> dict:
    return {k: v for k, v in run.items() if k != "requests"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8300")
    p.add_argument("--levels", default="1,2,4,8,16,32,64", help="comma-separated concurrency")
    p.add_argument("--requests-per-level", type=int, default=None, help="default max(8, 2N)")
    p.add_argument("--timeout", type=float, default=300.0, help="per-request seconds")
    p.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-error-rate", type=float, default=0.20)
    p.add_argument("--p50-factor", type=float, default=4.0)
    p.add_argument("--max-requests", type=int, default=None, help="refuse plans above this")
    p.add_argument("--server-log", type=Path, default=None, help="server log to count 429s in")
    p.add_argument("--server-pid", type=int, default=None, help="default: pid listening on --url")
    p.add_argument("--db-url", default=os.environ.get("DATABASE_URL", DEFAULT_DB_URL))
    p.add_argument("--no-monitor", action="store_true", help="no pg/ps/lsof sampling")
    p.add_argument("--out", type=Path, default=None, help="full results JSON path")
    p.add_argument("--label", default=None)
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    return p.parse_args(argv)


async def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    levels = [int(x) for x in a.levels.split(",") if x.strip()]
    plan = [(lv, requests_for(lv, a.requests_per_level)) for lv in levels]
    total = sum(n for _, n in plan)
    print(f"plan: {plan} -> at most {total} questions against {a.url}", file=sys.stderr)
    if a.max_requests is not None and total > a.max_requests:
        print(f"refusing: {total} > --max-requests {a.max_requests}", file=sys.stderr)
        return 2
    if a.dry_run:
        return 0

    questions = RoundRobin(load_questions(a.questions), a.seed)
    run_t0 = time.perf_counter()
    monitor = None
    if not a.no_monitor:
        pid = a.server_pid or find_listening_pid(urlparse(a.url).port or 80)
        db_port = urlparse(a.db_url).port or 5432
        monitor = Monitor(run_t0, a.db_url, pid, db_port)
        await monitor.start()
        print(f"monitoring server pid {pid}, db port {db_port}", file=sys.stderr)
    started = datetime.now(UTC)
    try:
        run = await run_ramp(
            a.url,
            levels,
            questions,
            requests_per_level=a.requests_per_level,
            timeout=a.timeout,
            max_error_rate=a.max_error_rate,
            p50_factor=a.p50_factor,
            monitor=monitor,
            logs=LogWatcher(a.server_log),
            run_t0=run_t0,
        )
    finally:
        if monitor:
            await monitor.stop()
    run = {
        "started": started.isoformat(timespec="seconds"),
        "url": a.url,
        "label": a.label,
        "config": {
            "levels": levels,
            "requests_per_level": a.requests_per_level,
            "timeout": a.timeout,
            "seed": a.seed,
            "max_error_rate": a.max_error_rate,
            "p50_factor": a.p50_factor,
            "questions": os.path.relpath(a.questions, ROOT),
            "server_log": str(a.server_log) if a.server_log else None,
        },
        **run,
        "monitor_notes": monitor.notes[:10] if monitor else [],
        "server_samples": [asdict(s) for s in monitor.samples] if monitor else [],
    }
    out = a.out or DEFAULT_OUT_DIR / (
        f"stress-{started:%Y%m%d-%H%M%S}{'-' + a.label if a.label else ''}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(run, indent=1))
    summary_path = out.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps({k: v for k, v in summary_only(run).items() if k != "server_samples"}, indent=1)
    )
    print(format_table(run["levels"]))
    if run["stopped_early"]:
        print(f"stopped early {run['stopped_early']}")
    print(f"wrote {out} and {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
