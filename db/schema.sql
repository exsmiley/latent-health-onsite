-- Applied automatically on first `docker compose up`, and idempotently by `rag.db.apply_schema()`.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS articles (
    id    BIGINT PRIMARY KEY,          -- Wikipedia page id from the dataset
    title TEXT   NOT NULL,
    url   TEXT   NOT NULL,
    text  TEXT   NOT NULL
);

CREATE INDEX IF NOT EXISTS articles_title_idx ON articles (lower(title));

CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    article_id  BIGINT  NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
    chunk_index INT     NOT NULL,      -- 0-based position within the article
    section     TEXT,                  -- nearest section heading; NULL for the lead section
    text        TEXT    NOT NULL,      -- chunk body: whole paragraphs only, returned to agents
    embed_text  TEXT    NOT NULL,      -- "Title > Section\n\n" + text; what gets embedded and keyword-indexed
    token_count INT     NOT NULL,      -- tiktoken cl100k_base count of `text`
    embedding   vector(1536),          -- text-embedding-3-small; NULL until the embed pool fills it
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', embed_text)) STORED,
    UNIQUE (article_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_unembedded_idx ON chunks (id) WHERE embedding IS NULL;
