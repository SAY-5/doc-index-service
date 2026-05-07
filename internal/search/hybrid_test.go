package search

import (
	"context"
	"errors"
	"testing"

	"github.com/google/uuid"

	"github.com/SAY-5/doc-index-service/internal/rerank"
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

func TestEngineQuery_VectorBackendErrorBubblesUp(t *testing.T) {
	be := &fakeBackend{vecErr: errors.New("hnsw down")}
	eng := NewEngine(be, &fakeEmbedder{})
	if _, err := eng.Query(context.Background(), "x", 1, ModeVector); err == nil {
		t.Fatal("expected error")
	}
}

func TestEngineQuery_EmbedReturnsEmpty(t *testing.T) {
	be := &fakeBackend{}
	eng := NewEngine(be, &fakeEmbedder{out: [][]float32{}})
	if _, err := eng.Query(context.Background(), "x", 1, ModeVector); err == nil {
		t.Fatal("expected error from empty embed result")
	}
}

// fixedReranker assigns scores by ChunkID lookup, so tests can pin the
// exact top-1 outcome.
type fixedReranker struct {
	byID map[string]float64
	err  error
}

func (f *fixedReranker) Rerank(_ context.Context, _ string, cands []rerank.Candidate, topN int) ([]rerank.Scored, error) {
	if f.err != nil {
		return nil, f.err
	}
	out := make([]rerank.Scored, 0, len(cands))
	for _, c := range cands {
		score := f.byID[c.ChunkID]
		out = append(out, rerank.Scored{Candidate: c, RerankScore: score})
	}
	// Stable bubble: highest score first, prior-rank tie break.
	for i := 1; i < len(out); i++ {
		for j := i; j > 0 && out[j].RerankScore > out[j-1].RerankScore; j-- {
			out[j], out[j-1] = out[j-1], out[j]
		}
	}
	if topN > 0 && len(out) > topN {
		out = out[:topN]
	}
	return out, nil
}

// TestEngineQuery_RerankChangesTop1 asserts that on a synthetic case
// where the reranker scores the second fused candidate highest, the
// returned top-1 doc id flips.
func TestEngineQuery_RerankChangesTop1(t *testing.T) {
	id1, id2 := uuid.New(), uuid.New()
	bm := []store.Hit{
		{ChunkID: id1, DocID: uuid.New(), Snippet: "first"},
		{ChunkID: id2, DocID: uuid.New(), Snippet: "second"},
	}
	be := &fakeBackend{bm: bm}
	eng := NewEngine(be, &fakeEmbedder{}).WithReranker(&fixedReranker{
		byID: map[string]float64{
			id1.String(): 0.1,
			id2.String(): 0.9, // promote the underdog
		},
	})

	// Without rerank, id1 would win because RRF respects the BM25 order.
	plain, err := eng.Query(context.Background(), "x", 10, ModeHybrid)
	if err != nil {
		t.Fatal(err)
	}
	if plain[0].ChunkID != id1.String() {
		t.Fatalf("plain hybrid: want %s, got %s", id1, plain[0].ChunkID)
	}

	// With rerank, the order flips.
	reranked, err := eng.Query(context.Background(), "x", 10, ModeHybridRerank)
	if err != nil {
		t.Fatal(err)
	}
	if reranked[0].ChunkID != id2.String() {
		t.Fatalf("reranked: want %s, got %s", id2, reranked[0].ChunkID)
	}
	// The signals envelope should expose both the prior RRF score and
	// the rerank score so a downstream client can debug.
	if _, ok := reranked[0].Signals["rerank"]; !ok {
		t.Fatalf("missing rerank signal: %+v", reranked[0].Signals)
	}
	if _, ok := reranked[0].Signals["rrf"]; !ok {
		t.Fatalf("missing rrf signal: %+v", reranked[0].Signals)
	}
}

// TestEngineQuery_RerankFallsBackWhenUnconfigured asserts that asking
// for rerank without a Reranker installed degrades to plain hybrid
// rather than failing the request.
func TestEngineQuery_RerankFallsBackWhenUnconfigured(t *testing.T) {
	id1 := uuid.New()
	be := &fakeBackend{bm: []store.Hit{{ChunkID: id1, DocID: uuid.New(), Snippet: "x"}}}
	eng := NewEngine(be, &fakeEmbedder{})
	res, err := eng.Query(context.Background(), "x", 10, ModeHybridRerank)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 1 || res[0].ChunkID != id1.String() {
		t.Fatalf("expected fallback to plain hybrid, got %+v", res)
	}
}

func TestEngineQuery_RerankErrorBubblesUp(t *testing.T) {
	id1 := uuid.New()
	be := &fakeBackend{bm: []store.Hit{{ChunkID: id1, DocID: uuid.New(), Snippet: "x"}}}
	eng := NewEngine(be, &fakeEmbedder{}).WithReranker(&fixedReranker{err: errors.New("model offline")})
	if _, err := eng.Query(context.Background(), "x", 10, ModeHybridRerank); err == nil {
		t.Fatal("expected error from reranker")
	}
}

func TestEngineQuery_DefaultsApplied(t *testing.T) {
	be := &fakeBackend{bm: []store.Hit{mkHit(1, "x")}}
	eng := &Engine{Store: be, Embedder: &fakeEmbedder{}}
	// k <= 0 and PerListLimit <= 0 should fall back to defaults.
	res, err := eng.Query(context.Background(), "x", 0, ModeKeyword)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) == 0 {
		t.Fatal("expected results")
	}
}
