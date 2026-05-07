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
// chunks are written — this is how /v1/index becomes idempotent.
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

// GetDoc fetches a single doc by id.
func (s *Store) GetDoc(ctx context.Context, id uuid.UUID) (Doc, error) {
	var d Doc
	err := s.Pool.QueryRow(ctx, `
        SELECT id, source, title, body, content_hash, created_at
        FROM docs WHERE id = $1
    `, id).Scan(&d.ID, &d.Source, &d.Title, &d.Body, &d.ContentHash, &d.CreatedAt)
	if err != nil {
		return Doc{}, translateNoRows(err)
	}
	return d, nil
}

// ListDocs paginates the docs table by (created_at DESC, id DESC). An
// empty afterCreatedAt + afterID returns the first page.
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
            ORDER BY created_at DESC, id DESC
            LIMIT $1
        `, limit)
	} else {
		rows, err = s.Pool.Query(ctx, `
            SELECT id, source, title, body, content_hash, created_at
            FROM docs
            WHERE (created_at, id) < ($1, $2)
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
// descending, capped at limit.
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
        WHERE c.tsv @@ websearch_to_tsquery('english', $1)
        ORDER BY score DESC
        LIMIT $2
    `, q, limit)
	if err != nil {
		return nil, err
	}
	return collectHits(rows)
}

// SearchVector runs the dense-vector path against the HNSW index.
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
        WHERE c.embedding IS NOT NULL
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
