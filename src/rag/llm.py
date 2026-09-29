from functools import lru_cache

from openai import AsyncOpenAI

from rag.config import get_settings


@lru_cache
def get_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=get_settings().openai_api_key or None)


async def embed_texts(texts: list[str], *, max_retries: int | None = None) -> list[list[float]]:
    """Embed texts in one request. Callers own batching; `max_retries` overrides the SDK's
    built-in retries (pass 0 when the caller runs its own retry/back-off loop)."""
    settings = get_settings()
    client = get_client()
    if max_retries is not None:
        client = client.with_options(max_retries=max_retries)
    resp = await client.embeddings.create(model=settings.embedding_model, input=texts)
    return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]
