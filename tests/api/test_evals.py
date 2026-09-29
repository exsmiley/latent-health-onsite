import json

import httpx
import pytest

from rag.api.app import app
from rag.config import get_settings


def client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def record(qid: str, correct: bool, total_s: float) -> dict:
    return {
        "id": qid,
        "tier": "main",
        "type": "single_hop",
        "difficulty": "easy",
        "outcome": "supported",
        "turns_used": 2,
        "timing": {"total_s": total_s, "research_turns": [], "evaluator_calls": []},
        "usage": {},
        "grading": {"correct": correct, "exact": correct},
    }


@pytest.fixture
def results(tmp_path, monkeypatch):
    monkeypatch.setattr(get_settings(), "eval_results_dir", tmp_path)
    return tmp_path


def write_run(d, stem: str, records: list[dict], summary: dict | None = None) -> None:
    (d / f"{stem}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    if summary is not None:
        (d / f"{stem}.summary.json").write_text(json.dumps(summary))


async def test_lists_runs_newest_first_including_unfinished(results):
    write_run(
        results,
        "20260101T000000Z_old",
        [record("q1", True, 3.0)],
        {"config": {"label": "old", "max_turns": 7}, "summary": {"overall": {"n": 1}}},
    )
    # No summary file yet: still running, or interrupted. Summarized from its records.
    write_run(results, "20260102T030405Z_live", [record("q1", True, 2.0), record("q2", False, 4.0)])
    async with client() as c:
        r = await c.get("/api/evals")
    assert r.status_code == 200
    runs = r.json()["runs"]
    assert [x["id"] for x in runs] == ["20260102T030405Z_live", "20260101T000000Z_old"]
    live, old = runs
    assert live["complete"] is False
    assert live["config"]["label"] == "live"
    assert live["config"]["started_at"] == "2026-01-02T03:04:05+00:00"
    assert live["overall"]["n"] == 2 and live["overall"]["accuracy"] == 0.5
    assert old["complete"] is True and old["config"]["max_turns"] == 7


async def test_get_run_returns_records_and_summary(results):
    write_run(
        results,
        "20260101T000000Z_a",
        [record("q1", True, 3.0), record("q1", False, 5.0)],  # a re-run of q1 wins
        {"config": {"label": "a"}, "summary": {"overall": {"n": 1}}},
    )
    async with client() as c:
        r = await c.get("/api/evals/20260101T000000Z_a")
    body = r.json()
    assert body["complete"] is True
    assert body["summary"] == {"overall": {"n": 1}}
    assert [(x["id"], x["grading"]["correct"]) for x in body["records"]] == [("q1", False)]


@pytest.mark.parametrize("run_id", ["missing", "..%2Fsecret", ".hidden"])
async def test_get_run_404s_on_unknown_or_unsafe_ids(results, run_id):
    (results.parent / "secret.jsonl").write_text("{}\n")
    async with client() as c:
        r = await c.get(f"/api/evals/{run_id}")
    assert r.status_code == 404
