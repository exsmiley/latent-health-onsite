"""Result shapes shared by the tools (rag.tools) and the agents (rag.agents)."""

from pydantic import BaseModel


class SearchHit(BaseModel):
    chunk_id: int
    article_id: int
    title: str
    section: str | None
    score: float  # higher is better (cosine similarity for semantic, ts_rank_cd for keyword)
    blurb: str  # first ~N words of the chunk text


class QueryResults(BaseModel):
    query: str
    hits: list[SearchHit]


class Chunk(BaseModel):
    chunk_id: int
    article_id: int
    title: str
    url: str
    section: str | None
    chunk_index: int
    text: str


class Article(BaseModel):
    article_id: int
    title: str
    url: str
    chunk_ids: list[int]  # so agents can cite specific chunks after reading the whole article
    text: str


class FetchResult(BaseModel):
    chunks: list[Chunk]
    articles: list[Article]
    missing_chunk_ids: list[int]
    missing_article_ids: list[int]
