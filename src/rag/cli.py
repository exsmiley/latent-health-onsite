import asyncio
from pathlib import Path

import typer
import uvicorn

from rag.agents.orchestrator import run_cli
from rag.db import apply_schema
from rag.embed.pool import run_embed
from rag.evals.report import compare, load_run, resolve_run
from rag.evals.runner import main as run_eval
from rag.ingest.pipeline import run_ingest

app = typer.Typer(no_args_is_help=True)


@app.command()
def init_db() -> None:
    """Apply db/schema.sql (idempotent)."""
    asyncio.run(apply_schema())


@app.command()
def ingest(
    limit: int | None = typer.Option(None, help="Only ingest the first N articles"),
    batch_size: int = typer.Option(500, help="Articles per DB transaction"),
) -> None:
    """Download (if needed), chunk and load articles into Postgres."""
    asyncio.run(run_ingest(limit=limit, batch_size=batch_size))


@app.command()
def embed(
    workers: int | None = typer.Option(None, help="Concurrent embedding workers"),
    batch_size: int | None = typer.Option(None, help="Chunks per embeddings request"),
    limit: int | None = typer.Option(None, help="Stop after embedding N chunks"),
) -> None:
    """Embed every chunk whose embedding is NULL, using a worker pool. Resumable."""
    asyncio.run(run_embed(workers=workers, batch_size=batch_size, limit=limit))


@app.command()
def ask(question: str) -> None:
    """Run the full research -> evaluate -> respond pipeline and print events."""
    asyncio.run(run_cli(question))


@app.command("eval")
def eval_(
    set_name: str = typer.Option(
        "main", "--set", help="Question set: main, super_hard, premise or all"
    ),
    file: str | None = typer.Option(None, help="Question JSONL file (overrides --set)"),
    ids: str | None = typer.Option(None, help="Comma-separated question ids, e.g. q001,q002"),
    limit: int | None = typer.Option(None, help="Only the first N selected questions"),
    max_turns: int | None = typer.Option(None, help="Override research_max_turns for the run"),
    concurrency: int = typer.Option(4, help="Questions run at the same time"),
    timeout: float = typer.Option(600.0, help="Per-question timeout in seconds"),
    judge: bool = typer.Option(True, "--judge/--no-judge", help="Grade with the LLM judge"),
    out: str = typer.Option("evals/results", help="Directory for result files"),
    label: str | None = typer.Option(None, help="Label added to the result file name"),
) -> None:
    """Benchmark the pipeline on the eval questions: accuracy, time, turns and tokens."""
    asyncio.run(
        run_eval(
            set_name=set_name,
            file=Path(file) if file else None,
            ids=[i.strip() for i in ids.split(",") if i.strip()] if ids else None,
            limit=limit,
            max_turns=max_turns,
            concurrency=concurrency,
            timeout=timeout,
            judge=judge,
            out_dir=Path(out),
            label=label,
        )
    )


@app.command("eval-compare")
def eval_compare(
    run_a: str = typer.Argument(..., help="Baseline results .jsonl (or its .summary.json)"),
    run_b: str = typer.Argument(..., help="New results .jsonl (or its .summary.json)"),
) -> None:
    """Compare two eval runs: accuracy/time/turn deltas per tier and type, fixed and broken."""
    print(compare(load_run(run_a), load_run(run_b), resolve_run(run_a).name,
                  resolve_run(run_b).name))  # fmt: skip


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8100, reload: bool = False) -> None:
    """Run the streaming API server."""
    uvicorn.run("rag.api.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
