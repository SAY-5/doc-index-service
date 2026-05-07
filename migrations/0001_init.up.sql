-- Hybrid retrieval needs both a TF-IDF text representation and a dense
-- vector column on each chunk so the two ranked lists can be fused at
-- query time. The full-text column is GENERATED so we never have to
-- remember to keep it in sync, and the HNSW index is built lazily because
-- it's the slowest part of bulk loads.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE docs (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source       TEXT NOT NULL,
    title        TEXT NOT NULL,
    body         TEXT NOT NULL,
    content_hash TEXT NOT NULL UNIQUE,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX docs_created_at_id_idx ON docs (created_at DESC, id DESC);

CREATE TABLE doc_chunks (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id      UUID NOT NULL REFERENCES docs(id) ON DELETE CASCADE,
    chunk_index INT  NOT NULL,
    text        TEXT NOT NULL,
    embedding   vector(384),
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    UNIQUE (doc_id, chunk_index)
);

-- GIN supports the tsvector @@ tsquery operator; this is the keyword path.
CREATE INDEX doc_chunks_tsv_idx ON doc_chunks USING GIN (tsv);

-- HNSW with cosine distance matches the all-MiniLM-L6-v2 normalised output.
-- m and ef_construction defaults are reasonable for ~50k vectors.
CREATE INDEX doc_chunks_embedding_idx
    ON doc_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX doc_chunks_doc_id_idx ON doc_chunks (doc_id);
