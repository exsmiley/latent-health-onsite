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
    research_max_turns: int = 7  # total research-agent model calls per question
    research_pre_retrieve: bool = True  # search + fetch the question before turn 1 (turn 0)
    search_top_k: int = 5
    blurb_words: int = 40
    hit_full_text: int = 3  # new hits per query shown with full text (the rest get blurbs)
    hit_text_budget_tokens: int = 5000  # cap on full hit text per research turn


@lru_cache
def get_settings() -> Settings:
    return Settings()
