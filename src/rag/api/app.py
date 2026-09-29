"""FastAPI backend: health, chunk lookup and the streaming chat endpoint (SSE)."""

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from rag import db
from rag.agents import orchestrator
from rag.agents.events import Event
from rag.tools import fetch as fetch_tool
from rag.tools.models import Chunk

log = logging.getLogger(__name__)

HEARTBEAT_SECONDS = 15.0
PING = ": ping\n\n"


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    await db.get_pool()  # the pool opens without waiting, so the API starts even if the DB is down
    try:
        yield
    finally:
        await db.close_pool()


app = FastAPI(title="RAG over Simple English Wikipedia", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5180"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1)


@app.get("/api/health")
async def health() -> dict:
    return {"ok": True}


@app.get("/api/chunks/{chunk_id}", response_model=Chunk)
async def get_chunk(chunk_id: int) -> Chunk:
    result = await fetch_tool.fetch(chunk_ids=[chunk_id], article_ids=[])
    if not result.chunks:
        raise HTTPException(status_code=404, detail=f"Chunk {chunk_id} not found")
    return result.chunks[0]


async def _wait_for_disconnect(request: Request) -> None:
    # The request body has already been read, so the next ASGI message is http.disconnect.
    # Starlette only watches for this itself on ASGI spec < 2.4.
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            return


async def sse_stream(request: Request, events: AsyncIterator[Event]) -> AsyncIterator[str]:
    """Serialize events as SSE, send `: ping` while idle, and stop when the client leaves."""
    it = aiter(events)
    disconnect = asyncio.ensure_future(_wait_for_disconnect(request))
    pending: asyncio.Future | None = None
    try:
        while True:
            if pending is None:
                pending = asyncio.ensure_future(anext(it))
            done, _ = await asyncio.wait(
                {pending, disconnect},
                timeout=HEARTBEAT_SECONDS,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if disconnect in done:
                log.info("client disconnected; stopping the pipeline")
                return
            if pending not in done:
                yield PING
                continue
            task, pending = pending, None
            try:
                event = task.result()
            except StopAsyncIteration:
                return
            yield event.to_sse()
    finally:
        disconnect.cancel()
        if pending is not None:
            pending.cancel()
            with contextlib.suppress(BaseException):
                await pending
        aclose = getattr(it, "aclose", None)
        if aclose is not None:
            with contextlib.suppress(BaseException):
                await aclose()


@app.post("/api/chat")
async def chat(body: ChatRequest, request: Request) -> StreamingResponse:
    *prior, last = body.messages
    if last.role != "user" or not last.content.strip():
        raise HTTPException(
            status_code=422, detail="The last message must be a non-empty user message"
        )
    history = [m.model_dump() for m in prior]
    events = orchestrator.run(last.content, history)
    return StreamingResponse(
        sse_stream(request, events),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
