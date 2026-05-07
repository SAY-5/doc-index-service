package search

import (
	"context"
	"errors"
	"testing"

	"github.com/google/uuid"

	"github.com/SAY-5/doc-index-service/internal/store"
)

type fakeBackend struct {
	bm     []store.Hit
	bmErr  error
	vec    []store.Hit
	vecErr error
}

func (f *fakeBackend) SearchBM25(_ context.Context, _ string, _ int) ([]store.Hit, error) {
	return f.bm, f.bmErr
}
func (f *fakeBackend) SearchVector(_ context.Context, _ []float32, _ int) ([]store.Hit, error) {
	return f.vec, f.vecErr
}

type fakeEmbedder struct {
	out [][]float32
	err error
}

func (f *fakeEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	if f.err != nil {
		return nil, f.err
	}
	if f.out != nil {
		return f.out, nil
	}
	out := make([][]float32, len(texts))
	for i := range texts {
		out[i] = make([]float32, 384)
	}
	return out, nil
}

func mkHit(score float64, snippet string) store.Hit {
	return store.Hit{DocID: uuid.New(), ChunkID: uuid.New(), ChunkIndex: 0, Snippet: snippet, Score: score}
}

func TestEngineQuery_Hybrid(t *testing.T) {
	bm := []store.Hit{mkHit(2, "alpha"), mkHit(1, "beta")}
	vec := []store.Hit{mkHit(0.9, "gamma"), mkHit(0.8, "alpha")}
	be := &fakeBackend{bm: bm, vec: vec}
	eng := NewEngine(be, &fakeEmbedder{})

	res, err := eng.Query(context.Background(), "x", 10, ModeHybrid)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) == 0 {
		t.Fatal("no results")
	}
	for _, r := range res {
		if r.DocID == "" || r.ChunkID == "" {
			t.Fatalf("missing ids: %+v", r)
		}
	}
}

func TestEngineQuery_KeywordOnlyDoesNotEmbed(t *testing.T) {
	be := &fakeBackend{bm: []store.Hit{mkHit(1, "snippet")}}
	emb := &fakeEmbedder{err: errors.New("should not be called")}
	eng := NewEngine(be, emb)
	res, err := eng.Query(context.Background(), "x", 5, ModeKeyword)
	if err != nil {
		t.Fatalf("keyword path should not call embedder: %v", err)
	}
	if len(res) != 1 {
		t.Fatalf("got %d results", len(res))
	}
	if _, ok := res[0].Signals["bm25"]; !ok {
		t.Fatalf("missing bm25 signal: %+v", res[0])
	}
	if _, ok := res[0].Signals["vector"]; ok {
		t.Fatalf("unexpected vector signal: %+v", res[0])
	}
}

func TestEngineQuery_VectorOnly(t *testing.T) {
	be := &fakeBackend{vec: []store.Hit{mkHit(0.9, "snippet")}}
	eng := NewEngine(be, &fakeEmbedder{})
	res, err := eng.Query(context.Background(), "x", 5, ModeVector)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 1 || res[0].Signals["vector"] == 0 {
		t.Fatalf("vector path failed: %+v", res)
	}
}

func TestEngineQuery_KSnippetTruncation(t *testing.T) {
	long := make([]byte, 1024)
	for i := range long {
		long[i] = 'a'
	}
	be := &fakeBackend{bm: []store.Hit{mkHit(1, string(long))}}
	eng := NewEngine(be, &fakeEmbedder{})
	res, err := eng.Query(context.Background(), "x", 1, ModeKeyword)
	if err != nil {
		t.Fatal(err)
	}
	if len([]rune(res[0].Snippet)) > 250 {
		t.Fatalf("snippet not truncated: %d", len([]rune(res[0].Snippet)))
	}
}

func TestEngineQuery_BackendErrorBubblesUp(t *testing.T) {
	be := &fakeBackend{bmErr: errors.New("db dead")}
	eng := NewEngine(be, &fakeEmbedder{})
	if _, err := eng.Query(context.Background(), "x", 1, ModeHybrid); err == nil {
		t.Fatal("expected error")
	}
}

func TestEngineQuery_EmbedErrorBubblesUp(t *testing.T) {
	be := &fakeBackend{}
	eng := NewEngine(be, &fakeEmbedder{err: errors.New("nope")})
	if _, err := eng.Query(context.Background(), "x", 1, ModeVector); err == nil {
		t.Fatal("expected error")
	}
}
