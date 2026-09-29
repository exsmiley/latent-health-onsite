"""Eval runs on disk, for the API: each run is a results `.jsonl` plus, once it finishes, a
`.summary.json` beside it (see evals/README.md, "Output")."""

import json
import re
from pathlib import Path

from rag.evals import report

# Run ids are result-file stems (`<UTC stamp>_<label>`); anything else is rejected, so an id can
# never name a path outside the results directory.
RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _summary_file(jsonl: Path) -> Path:
    return jsonl.with_name(jsonl.stem + ".summary.json")


def _config_from_stem(stem: str) -> dict:
    """What an unfinished run's file name tells us: its start time and label."""
    stamp, _, label = stem.partition("_")
    m = re.fullmatch(r"(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z", stamp)
    started = f"{m[1]}-{m[2]}-{m[3]}T{m[4]}:{m[5]}:{m[6]}+00:00" if m else None
    return {"label": label or None, "started_at": started, "results_file": stem + ".jsonl"}


def _load(jsonl: Path) -> tuple[dict, dict | None, list[dict] | None]:
    """(config, summary, records). A finished run's summary is read from its file and its
    records are not loaded; an unfinished one (running or interrupted) is summarized from the
    records so far."""
    summary_file = _summary_file(jsonl)
    if summary_file.exists():
        data = json.loads(summary_file.read_text())
        return data.get("config") or {}, data.get("summary"), None
    records = report.load_run(jsonl)
    return _config_from_stem(jsonl.stem), report.summarize(records), records


def list_runs(results_dir: Path) -> list[dict]:
    """Every run in `results_dir`, newest first, with its config and overall stats."""
    runs = []
    for jsonl in sorted(results_dir.glob("*.jsonl"), reverse=True):
        try:
            config, summary, _ = _load(jsonl)
        except (OSError, ValueError, KeyError):
            continue  # unreadable or half-written: skip it rather than fail the list
        runs.append(
            {
                "id": jsonl.stem,
                "complete": _summary_file(jsonl).exists(),
                "config": config,
                "overall": (summary or {}).get("overall"),
            }
        )
    return runs


def get_run(results_dir: Path, run_id: str) -> dict | None:
    """One run with its config, summary and per-question records; None if there is no such run."""
    jsonl = results_dir / f"{run_id}.jsonl"
    if not RUN_ID.match(run_id) or not jsonl.is_file():
        return None
    config, summary, records = _load(jsonl)
    if records is None:
        records = report.load_run(jsonl)
    return {
        "id": run_id,
        "complete": _summary_file(jsonl).exists(),
        "config": config,
        "summary": summary,
        "records": records,
    }
