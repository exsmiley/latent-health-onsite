import asyncio

import typer
import uvicorn

from rag.agents.orchestrator import run_cli
from rag.db import apply_schema
from rag.embed.pool import run_embed
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


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8100, reload: bool = False) -> None:
    """Run the streaming API server."""
    uvicorn.run("rag.api.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
