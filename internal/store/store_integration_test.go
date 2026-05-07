//go:build integration

// Run with: go test -tags=integration ./internal/store/...
//
// Requires DATABASE_URL pointing at a Postgres with pgvector +
// migrations applied. CI's bench-regress / bench-smoke jobs already
// stand up that environment, so the tag-gated test piggybacks on the
// same containers without standing up its own.
package store

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"sync"
	"testing"

	"github.com/google/uuid"
)

func openTestStore(t *testing.T) *Store {
	t.Helper()
	dsn := os.Getenv("DATABASE_URL")
	if dsn == "" {
		dsn = "postgres://postgres:postgres@localhost:5432/docindex?sslmode=disable"
	}
	st, err := Open(context.Background(), dsn)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	t.Cleanup(st.Close)
	return st
}

func mkDoc(t *testing.T, body string) (Doc, []Chunk) {
	t.Helper()
	h := sha256.Sum256([]byte(body))
	return Doc{
		Source:      "https://example.test/" + body[:8],
		Title:       "title-" + body[:8],
		Body:        body,
		ContentHash: hex.EncodeToString(h[:]),
	}, []Chunk{{ChunkIndex: 0, Text: body, Embedding: makeVec(384)}}
}

func makeVec(dim int) []float32 {
	v := make([]float32, dim)
	v[0] = 1.0
	return v
}

func TestIntegration_DeleteHidesFromSearch(t *testing.T) {
	st := openTestStore(t)
	ctx := context.Background()

	body := fmt.Sprintf("integration-target-%s the kafka topic rebalance", uuid.NewString())
	d, ch := mkDoc(t, body)
	id, _, _, err := st.UpsertDoc(ctx, d, ch)
	if err != nil {
		t.Fatal(err)
	}

	// Confirm the doc is searchable before delete.
	hits, err := st.SearchBM25(ctx, "rebalance", 10)
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, h := range hits {
		if h.DocID == id {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("pre-delete search should find doc %s", id)
	}

	if err := st.DeleteDoc(ctx, id); err != nil {
		t.Fatalf("delete: %v", err)
	}

	// Post-delete: doc must not appear in keyword search.
	hits, err = st.SearchBM25(ctx, "rebalance", 10)
	if err != nil {
		t.Fatal(err)
	}
	for _, h := range hits {
		if h.DocID == id {
			t.Fatalf("deleted doc still in search results")
		}
	}

	// GetDoc must return ErrNotFound.
	if _, err := st.GetDoc(ctx, id); !errors.Is(err, ErrNotFound) {
		t.Fatalf("GetDoc on deleted: %v", err)
	}
}

func TestIntegration_DoubleDeleteIsNotFound(t *testing.T) {
	st := openTestStore(t)
	ctx := context.Background()

	body := fmt.Sprintf("double-delete-%s", uuid.NewString())
	d, ch := mkDoc(t, body)
	id, _, _, err := st.UpsertDoc(ctx, d, ch)
	if err != nil {
		t.Fatal(err)
	}

	if err := st.DeleteDoc(ctx, id); err != nil {
		t.Fatal(err)
	}
	if err := st.DeleteDoc(ctx, id); !errors.Is(err, ErrNotFound) {
		t.Fatalf("second delete should be ErrNotFound, got %v", err)
	}
}

func TestIntegration_UpsertReplayIsIdempotent(t *testing.T) {
	st := openTestStore(t)
	ctx := context.Background()
	body := fmt.Sprintf("idem-%s", uuid.NewString())
	d, ch := mkDoc(t, body)

	id1, _, existed1, err := st.UpsertDoc(ctx, d, ch)
	if err != nil || existed1 {
		t.Fatalf("first upsert: id=%v existed=%v err=%v", id1, existed1, err)
	}
	id2, _, existed2, err := st.UpsertDoc(ctx, d, ch)
	if err != nil || !existed2 || id1 != id2 {
		t.Fatalf("second upsert should return same id with existed=true: id=%v existed=%v err=%v", id2, existed2, err)
	}
}

func TestIntegration_CompactRemovesChunksAndKeepsLiveData(t *testing.T) {
	st := openTestStore(t)
	ctx := context.Background()

	// Live doc: stays.
	liveBody := fmt.Sprintf("live-%s the postgres index", uuid.NewString())
	liveDoc, liveCh := mkDoc(t, liveBody)
	liveID, _, _, err := st.UpsertDoc(ctx, liveDoc, liveCh)
	if err != nil {
		t.Fatal(err)
	}

	// Doomed doc: indexed, then deleted, then compacted.
	doomedBody := fmt.Sprintf("doomed-%s the cassandra rebalance", uuid.NewString())
	doomedDoc, doomedCh := mkDoc(t, doomedBody)
	doomedID, _, _, err := st.UpsertDoc(ctx, doomedDoc, doomedCh)
	if err != nil {
		t.Fatal(err)
	}
	if err := st.DeleteDoc(ctx, doomedID); err != nil {
		t.Fatal(err)
	}

	res, err := st.Compact(ctx, 0)
	if err != nil {
		t.Fatalf("compact: %v", err)
	}
	if res.Tombstones < 1 || res.ChunksDeleted < 1 {
		t.Fatalf("compact did nothing: %+v", res)
	}

	// Live doc still searchable after compact.
	hits, err := st.SearchBM25(ctx, "postgres", 50)
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, h := range hits {
		if h.DocID == liveID {
			found = true
		}
	}
	if !found {
		t.Fatalf("compact dropped live doc %s", liveID)
	}

	// Compacted doomed doc has zero chunks left.
	var n int
	if err := st.Pool.QueryRow(ctx, `SELECT COUNT(*) FROM doc_chunks WHERE doc_id = $1`, doomedID).Scan(&n); err != nil {
		t.Fatal(err)
	}
	if n != 0 {
		t.Fatalf("compacted doc still has %d chunks", n)
	}
}

func TestIntegration_ConcurrentIndexAndDelete(t *testing.T) {
	st := openTestStore(t)
	ctx := context.Background()

	body := fmt.Sprintf("race-%s lorem ipsum dolor", uuid.NewString())
	d, ch := mkDoc(t, body)
	id, _, _, err := st.UpsertDoc(ctx, d, ch)
	if err != nil {
		t.Fatal(err)
	}

	var wg sync.WaitGroup
	wg.Add(2)
	// Goroutine 1: replay the upsert (idempotent on hash).
	go func() {
		defer wg.Done()
		if _, _, _, err := st.UpsertDoc(ctx, d, ch); err != nil {
			t.Errorf("concurrent upsert: %v", err)
		}
	}()
	// Goroutine 2: delete the same doc.
	go func() {
		defer wg.Done()
		if err := st.DeleteDoc(ctx, id); err != nil && !errors.Is(err, ErrNotFound) {
			t.Errorf("concurrent delete: %v", err)
		}
	}()
	wg.Wait()

	// After both finish, the doc should be in *some* consistent state:
	// either deleted (search hides it, GetDoc returns ErrNotFound) or
	// still alive. The key guarantee is that we don't corrupt the data
	// — there should never be a tombstone row pointing at a doc that
	// is still flagged live.
	var deletedAt *string
	var hasTomb bool
	if err := st.Pool.QueryRow(ctx, `
        SELECT d.deleted_at::text, EXISTS(SELECT 1 FROM doc_tombstones t WHERE t.doc_id = d.id)
        FROM docs d WHERE d.id = $1
    `, id).Scan(&deletedAt, &hasTomb); err != nil {
		t.Fatal(err)
	}
	if hasTomb && deletedAt == nil {
		t.Fatalf("invariant violated: tombstone exists but doc is still live")
	}
}
