DROP INDEX IF EXISTS doc_tombstones_pending_idx;
DROP TABLE IF EXISTS doc_tombstones;
DROP INDEX IF EXISTS docs_live_idx;
ALTER TABLE docs DROP COLUMN IF EXISTS deleted_at;
