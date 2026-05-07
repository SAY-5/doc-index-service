-- Soft-delete plumbing for incremental indexing.
--
-- Two columns are added so search queries can filter out deleted docs
-- without losing the ability to compact them later:
--
--   docs.deleted_at — populated on DELETE /v1/docs/{id}; NULL for live
--   docs. Search WHERE clauses must include `deleted_at IS NULL`.
--
-- A separate doc_tombstones table records the deletion event with its
-- own id so the admin compact endpoint can iterate them and reclaim the
-- HNSW slots in batches without scanning the docs table linearly.

ALTER TABLE docs ADD COLUMN deleted_at TIMESTAMPTZ;

-- Partial index keeps the live-only path fast: BM25 / vector queries
-- only touch this index, not the deleted rows.
CREATE INDEX docs_live_idx ON docs (created_at DESC, id DESC) WHERE deleted_at IS NULL;

CREATE TABLE doc_tombstones (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id      UUID NOT NULL UNIQUE REFERENCES docs(id) ON DELETE CASCADE,
    deleted_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    compacted_at TIMESTAMPTZ
);

-- Look-up by doc_id is the hot path during compaction; we already have
-- a unique constraint there, but the explicit index makes the intent
-- obvious in EXPLAIN output.
CREATE INDEX doc_tombstones_pending_idx
    ON doc_tombstones (deleted_at)
    WHERE compacted_at IS NULL;
