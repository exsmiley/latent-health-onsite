from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    openai_api_key: str = ""
    database_url: str = "postgresql://rag:rag@localhost:5433/rag"

    # Models
    chat_model: str = "gpt-6-luna"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    # Data
    dataset_path: Path = ROOT / "data" / "simplewiki-20220301.parquet"
    dataset_url: str = (
        "https://huggingface.co/datasets/legacy-datasets/wikipedia/resolve/main/"
        "data/20220301.simple/train-00000-of-00001.parquet"
    )

    # Chunking (tokens, cl100k_base). Chunks are whole paragraphs only.
    chunk_target_tokens: int = 350
    chunk_min_tokens: int = 100

    # Embedding worker pool
    embed_workers: int = 8
    embed_batch_size: int = 128

    # Agents
    research_max_turns: int = 7  # model calls per research round
    max_research_rounds: int = 3  # initial round + evaluator-requested retries
    search_top_k: int = 5
    blurb_words: int = 40


@lru_cache
def get_settings() -> Settings:
    return Settings()
