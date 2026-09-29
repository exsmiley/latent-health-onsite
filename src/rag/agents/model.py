"""Thin wrapper around the OpenAI Responses API.

Every model call in rag.agents goes through `create` or `stream_text`, and both get the client
from `get_client`, so tests only have to replace `get_client` with a fake whose
`responses.create(**kwargs)` mimics the SDK.
"""

from collections.abc import AsyncIterator
from typing import Any

from openai.types.responses import Response

from rag import llm
from rag.config import get_settings


class ModelError(RuntimeError):
    pass


def get_client() -> Any:
    return llm.get_client()


async def create(**kwargs: Any) -> Response:
    """One non-streaming Responses API call using `settings.chat_model`."""
    kwargs.setdefault("model", get_settings().chat_model)
    return await get_client().responses.create(**kwargs)


async def stream_text(**kwargs: Any) -> AsyncIterator[str]:
    """Stream a Responses API call and yield only final-answer text deltas.

    Messages the model labels `phase="commentary"` are skipped. Newer models may emit those
    before the final answer.
    """
    kwargs.setdefault("model", get_settings().chat_model)
    stream = await get_client().responses.create(stream=True, **kwargs)
    skip_items: set[str] = set()
    try:
        async for event in stream:
            etype = getattr(event, "type", None)
            if etype == "response.output_item.added":
                item = event.item
                if getattr(item, "type", None) == "message" and getattr(item, "phase", None) == (
                    "commentary"
                ):
                    skip_items.add(item.id)
            elif etype == "response.output_text.delta":
                if event.item_id not in skip_items and event.delta:
                    yield event.delta
            elif etype == "error":
                raise ModelError(f"Model stream error: {event.message}")
            elif etype == "response.failed":
                err = getattr(event.response, "error", None)
                raise ModelError(f"Model response failed: {getattr(err, 'message', err)}")
    finally:
        close = getattr(stream, "close", None)
        if close is not None:
            await close()


def dump_output_item(item: Any) -> dict:
    """Turn a response output item back into an input item for the next request.

    This keeps `phase` on assistant messages and reasoning items as-is, which the SDK
    recommends for multi-turn with newer models.
    """
    if isinstance(item, dict):
        return item
    return item.model_dump(mode="json", by_alias=True, exclude_none=True)
