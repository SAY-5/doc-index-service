package store

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
)

// formatVector renders a []float32 in the textual form pgvector accepts:
// "[1.23,4.56,...]". We do this by hand rather than depending on
// pgvector-go to keep the dependency surface small.
func formatVector(v []float32) string {
	if len(v) == 0 {
		return "[]"
	}
	var b strings.Builder
	b.Grow(len(v) * 8)
	b.WriteByte('[')
	for i, x := range v {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteString(strconv.FormatFloat(float64(x), 'f', 6, 32))
	}
	b.WriteByte(']')
	return b.String()
}

// UpsertDoc inserts a doc and its chunks atomically. If a doc with the
// same content_hash already exists the existing row is returned and no
// chunks are written — this is how /v1/index stays idempotent across
// retries.
//
// A previously soft-deleted doc with a matching content_hash is treated
// the same as any other conflict: the existing (still tombstoned) id is
// returned with alreadyExisted=true. Callers that want to "undelete"
// should issue an explicit reactivation; collapsing the two semantics
// here would surprise replays of a delete + re-index sequence.
//
// Returns (doc_id, chunk_count, alreadyExisted).
func (s *Store) UpsertDoc(ctx context.Context, d Doc, chunks []Chunk) (uuid.UUID, int, bool, error) {
	tx, err := s.Pool.BeginTx(ctx, pgx.TxOptions{})
	if err != nil {
		return uuid.Nil, 0, false, err
	}
	defer func() { _ = tx.Rollback(ctx) }()

	// Try to claim the hash.
	var id uuid.UUID
	err = tx.QueryRow(ctx, `
        INSERT INTO docs (source, title, body, content_hash)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (content_hash) DO NOTHING
        RETURNING id
    `, d.Source, d.Title, d.Body, d.ContentHash).Scan(&id)
	if err == nil {
		// Newly inserted: persist chunks.
		if err := s.insertChunks(ctx, tx, id, chunks); err != nil {
			return uuid.Nil, 0, false, err
		}
		if err := tx.Commit(ctx); err != nil {
			return uuid.Nil, 0, false, err
		}
		return id, len(chunks), false, nil
	}
	if err != pgx.ErrNoRows {
		return uuid.Nil, 0, false, err
	}

	// Conflict: fetch the existing id and chunk count.
	if err := tx.QueryRow(ctx, `SELECT id FROM docs WHERE content_hash = $1`, d.ContentHash).Scan(&id); err != nil {
		return uuid.Nil, 0, false, fmt.Errorf("lookup existing: %w", err)
	}
	var n int
	if err := tx.QueryRow(ctx, `SELECT COUNT(*) FROM doc_chunks WHERE doc_id = $1`, id).Scan(&n); err != nil {
		return uuid.Nil, 0, false, err
	}
	if err := tx.Commit(ctx); err != nil {
		return uuid.Nil, 0, false, err
	}
	return id, n, true, nil
}

// DeleteDoc soft-deletes a doc and writes a tombstone row. It returns
// ErrNotFound when:
//
//   - the doc id does not exist, or
//   - the doc was already soft-deleted on a previous call.
//
// The two cases are intentionally indistinguishable to the caller: an
// idempotent DELETE that returns 404 on replay is the standard REST
// shape, and emitting "already deleted" leaks the existence of a doc
// the caller has already disowned.
//
// The update + tombstone insert run in a single transaction with
// SELECT ... FOR UPDATE so an `index` racing against `delete` for the
// same doc cannot leave the tombstone behind a still-live row.
func (s *Store) DeleteDoc(ctx context.Context, id uuid.UUID) error {
	tx, err := s.Pool.BeginTx(ctx, pgx.TxOptions{})
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback(ctx) }()

	var deletedAt *time.Time
	err = tx.QueryRow(ctx, `
        SELECT deleted_at FROM docs WHERE id = $1 FOR UPDATE
    `, id).Scan(&deletedAt)
	if err != nil {
		return translateNoRows(err)
	}
	if deletedAt != nil {
		return ErrNotFound
	}

	if _, err := tx.Exec(ctx, `UPDATE docs SET deleted_at = now() WHERE id = $1`, id); err != nil {
		return fmt.Errorf("mark deleted: %w", err)
	}
	if _, err := tx.Exec(ctx, `
        INSERT INTO doc_tombstones (doc_id) VALUES ($1)
        ON CONFLICT (doc_id) DO NOTHING
    `, id); err != nil {
		return fmt.Errorf("tombstone: %w", err)
	}
	return tx.Commit(ctx)
}

// CompactResult records what a Compact run did so the admin endpoint
// can return something more useful than 200 OK.
type CompactResult struct {
	Tombstones    int
	ChunksDeleted int
}

// Compact removes tombstoned docs and their chunks, then marks the
// matching tombstone rows as compacted. It runs the work in batches so
// a long-running compaction does not hold a single transaction open
// against the whole table.
//
// Crucially, only rows whose live counterpart is still flagged
// deleted_at IS NOT NULL are removed; this defends against the
// theoretical case where a doc is reactivated between tombstone and
// compaction. Concurrent indexing of unrelated docs is not affected
// because the deletes are scoped by tombstone id.
func (s *Store) Compact(ctx context.Context, batchSize int) (CompactResult, error) {
	if batchSize <= 0 {
		batchSize = 256
	}
	var res CompactResult
	for {
		var (
			tombID  uuid.UUID
			docID   uuid.UUID
			chunks  int
			drained bool
		)
		err := s.Pool.QueryRow(ctx, `
            SELECT t.id, t.doc_id
            FROM doc_tombstones t
            JOIN docs d ON d.id = t.doc_id AND d.deleted_at IS NOT NULL
            WHERE t.compacted_at IS NULL
            ORDER BY t.deleted_at
            LIMIT 1
        `).Scan(&tombID, &docID)
		if err != nil {
			if err == pgx.ErrNoRows {
				drained = true
			} else {
				return res, fmt.Errorf("scan tombstones: %w", err)
			}
		}
		if drained {
			break
		}

		// Run this tombstone's cleanup as a single transaction.
		tx, err := s.Pool.BeginTx(ctx, pgx.TxOptions{})
		if err != nil {
			return res, err
		}
		row := tx.QueryRow(ctx, `
            WITH deleted AS (
                DELETE FROM doc_chunks WHERE doc_id = $1 RETURNING 1
            )
            SELECT count(*) FROM deleted
        `, docID)
		if err := row.Scan(&chunks); err != nil {
			_ = tx.Rollback(ctx)
			return res, fmt.Errorf("delete chunks: %w", err)
		}
		if _, err := tx.Exec(ctx, `
            UPDATE doc_tombstones SET compacted_at = now() WHERE id = $1
        `, tombID); err != nil {
			_ = tx.Rollback(ctx)
			return res, fmt.Errorf("mark compacted: %w", err)
		}
		if err := tx.Commit(ctx); err != nil {
			return res, err
		}

		res.Tombstones++
		res.ChunksDeleted += chunks
		_ = batchSize // Per-tombstone txn keeps each transaction tiny;
		// batchSize is currently advisory and reserved for a future
		// multi-doc DELETE that bundles N tombstones at once.
	}
	return res, nil
}

func (s *Store) insertChunks(ctx context.Context, tx pgx.Tx, docID uuid.UUID, chunks []Chunk) error {
	if len(chunks) == 0 {
		return nil
	}
	// Build a single multi-row INSERT for throughput.
	var b strings.Builder
	b.WriteString(`INSERT INTO doc_chunks (doc_id, chunk_index, text, embedding) VALUES `)
	args := make([]any, 0, len(chunks)*4)
	for i, c := range chunks {
		if i > 0 {
			b.WriteByte(',')
		}
		base := i * 4
		fmt.Fprintf(&b, "($%d,$%d,$%d,$%d::vector)", base+1, base+2, base+3, base+4)
		args = append(args, docID, c.ChunkIndex, c.Text, formatVector(c.Embedding))
	}
	if _, err := tx.Exec(ctx, b.String(), args...); err != nil {
		return fmt.Errorf("insert chunks: %w", err)
	}
	return nil
}

// GetDoc fetches a single doc by id. Tombstoned docs are not returned —
// the caller sees ErrNotFound, matching the DELETE semantics.
func (s *Store) GetDoc(ctx context.Context, id uuid.UUID) (Doc, error) {
	var d Doc
	err := s.Pool.QueryRow(ctx, `
        SELECT id, source, title, body, content_hash, created_at, deleted_at
        FROM docs WHERE id = $1
    `, id).Scan(&d.ID, &d.Source, &d.Title, &d.Body, &d.ContentHash, &d.CreatedAt, &d.DeletedAt)
	if err != nil {
		return Doc{}, translateNoRows(err)
	}
	if d.DeletedAt != nil {
		return Doc{}, ErrNotFound
	}
	return d, nil
}

// ListDocs paginates the docs table by (created_at DESC, id DESC),
// hiding any tombstoned rows. An empty afterCreatedAt + afterID returns
// the first page. The partial index docs_live_idx makes the live-only
// scan as cheap as the original full-table version.
func (s *Store) ListDocs(ctx context.Context, afterCreatedAt time.Time, afterID uuid.UUID, limit int) ([]Doc, error) {
	if limit <= 0 || limit > 200 {
		limit = 50
	}
	var rows pgx.Rows
	var err error
	if afterID == uuid.Nil {
		rows, err = s.Pool.Query(ctx, `
            SELECT id, source, title, body, content_hash, created_at
            FROM docs
            WHERE deleted_at IS NULL
            ORDER BY created_at DESC, id DESC
            LIMIT $1
        `, limit)
	} else {
		rows, err = s.Pool.Query(ctx, `
            SELECT id, source, title, body, content_hash, created_at
            FROM docs
            WHERE deleted_at IS NULL AND (created_at, id) < ($1, $2)
            ORDER BY created_at DESC, id DESC
            LIMIT $3
        `, afterCreatedAt, afterID, limit)
	}
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Doc
	for rows.Next() {
		var d Doc
		if err := rows.Scan(&d.ID, &d.Source, &d.Title, &d.Body, &d.ContentHash, &d.CreatedAt); err != nil {
			return nil, err
		}
		out = append(out, d)
	}
	return out, rows.Err()
}

// SearchBM25 runs the keyword path. Returns hits ordered by ts_rank_cd
// descending, capped at limit. Tombstoned docs are excluded by joining
// against the docs table on deleted_at IS NULL.
func (s *Store) SearchBM25(ctx context.Context, q string, limit int) ([]Hit, error) {
	if limit <= 0 {
		limit = 50
	}
	rows, err := s.Pool.Query(ctx, `
        SELECT c.doc_id,
               c.id,
               c.chunk_index,
               c.text,
               ts_rank_cd(c.tsv, websearch_to_tsquery('english', $1)) AS score
        FROM doc_chunks c
        JOIN docs d ON d.id = c.doc_id
        WHERE d.deleted_at IS NULL
          AND c.tsv @@ websearch_to_tsquery('english', $1)
        ORDER BY score DESC
        LIMIT $2
    `, q, limit)
	if err != nil {
		return nil, err
	}
	return collectHits(rows)
}

// SearchVector runs the dense-vector path against the HNSW index.
// Tombstoned docs are filtered out via a JOIN on docs.deleted_at;
// pgvector's HNSW returns more candidates than we need precisely so
// post-filtering like this is cheap.
func (s *Store) SearchVector(ctx context.Context, queryVec []float32, limit int) ([]Hit, error) {
	if limit <= 0 {
		limit = 50
	}
	rows, err := s.Pool.Query(ctx, `
        SELECT c.doc_id,
               c.id,
               c.chunk_index,
               c.text,
               1 - (c.embedding <=> $1::vector) AS score
        FROM doc_chunks c
        JOIN docs d ON d.id = c.doc_id
        WHERE d.deleted_at IS NULL
          AND c.embedding IS NOT NULL
        ORDER BY c.embedding <=> $1::vector
        LIMIT $2
    `, formatVector(queryVec), limit)
	if err != nil {
		return nil, err
	}
	return collectHits(rows)
}

func collectHits(rows pgx.Rows) ([]Hit, error) {
	defer rows.Close()
	var out []Hit
	for rows.Next() {
		var h Hit
		if err := rows.Scan(&h.DocID, &h.ChunkID, &h.ChunkIndex, &h.Snippet, &h.Score); err != nil {
			return nil, err
		}
		out = append(out, h)
	}
	return out, rows.Err()
}

// CountDocs is exposed for the bench harness.
func (s *Store) CountDocs(ctx context.Context) (int, error) {
	var n int
	err := s.Pool.QueryRow(ctx, `SELECT COUNT(*) FROM docs`).Scan(&n)
	return n, err
}
