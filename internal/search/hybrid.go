package search

import (
	"context"
	"fmt"

	"github.com/SAY-5/doc-index-service/internal/rerank"
	"github.com/SAY-5/doc-index-service/internal/store"
	"github.com/SAY-5/doc-index-service/pkg/embed"
)

// Backend is the slice of *store.Store the hybrid engine actually uses.
// Splitting it out lets engine tests run against an in-memory fake.
type Backend interface {
	SearchBM25(ctx context.Context, q string, limit int) ([]store.Hit, error)
	SearchVector(ctx context.Context, queryVec []float32, limit int) ([]store.Hit, error)
}

// Mode picks which retrievers to consult.
type Mode string

// Retrieval modes accepted by the /v1/query endpoint.
const (
	ModeHybrid       Mode = "hybrid"
	ModeKeyword      Mode = "keyword"
	ModeVector       Mode = "vector"
	ModeHybridRerank Mode = "hybrid+rerank"
)

// ParseMode validates a string from the request body.
func ParseMode(s string) (Mode, error) {
	switch Mode(s) {
	case "":
		return ModeHybrid, nil
	case ModeHybrid, ModeKeyword, ModeVector, ModeHybridRerank:
		return Mode(s), nil
	default:
		return "", fmt.Errorf("unknown mode %q", s)
	}
}

// Result is a single ranked entry returned to clients.
type Result struct {
	DocID   string             `json:"doc_id"`
	ChunkID string             `json:"chunk_id"`
	Score   float64            `json:"score"`
	Snippet string             `json:"snippet"`
	Signals map[string]float64 `json:"signals"`
}

// Engine fuses the BM25 and vector paths against a Backend.
type Engine struct {
	Store    Backend
	Embedder embed.Embedder
	// PerListLimit is how many candidates each retriever returns before
	// fusion; the default of 50 is enough headroom for k=10 hybrid output
	// without paying for a deeper vector probe.
	PerListLimit int
	// Reranker, when set, is invoked for ModeHybridRerank queries against
	// the top-RerankInputN fused candidates. nil means rerank requests
	// transparently fall back to plain hybrid (so a /v1/query with the
	// flag set never errors out just because the sidecar is offline).
	Reranker rerank.Reranker
	// RerankInputN is how many fused candidates feed the reranker. 20 is
	// the standard "small enough to score with a cross-encoder, large
	// enough to give the ranker something to choose from" number.
	RerankInputN int
}

// NewEngine returns an Engine with sensible defaults.
func NewEngine(s Backend, e embed.Embedder) *Engine {
	return &Engine{
		Store:        s,
		Embedder:     e,
		PerListLimit: 50,
		RerankInputN: 20,
	}
}

// WithReranker installs a Reranker. The receiver is returned so callers
// can chain construction.
func (e *Engine) WithReranker(r rerank.Reranker) *Engine {
	e.Reranker = r
	return e
}

// Query runs the requested retrieval mode and returns up to k results.
func (e *Engine) Query(ctx context.Context, q string, k int, mode Mode) ([]Result, error) {
	if k <= 0 {
		k = 10
	}
	if e.PerListLimit <= 0 {
		e.PerListLimit = 50
	}
	if e.RerankInputN <= 0 {
		e.RerankInputN = 20
	}

	snippets := make(map[string]string)
	chunkToDoc := make(map[string]string)
	chunkIdx := make(map[string]int)

	lists := make(map[string][]Hit)

	useBM25 := mode == ModeHybrid || mode == ModeKeyword || mode == ModeHybridRerank
	useVec := mode == ModeHybrid || mode == ModeVector || mode == ModeHybridRerank

	if useBM25 {
		bm, err := e.Store.SearchBM25(ctx, SanitizeBM25Query(q), e.PerListLimit)
		if err != nil {
			return nil, fmt.Errorf("bm25: %w", err)
		}
		hits := make([]Hit, 0, len(bm))
		for _, h := range bm {
			id := h.ChunkID.String()
			snippets[id] = h.Snippet
			chunkToDoc[id] = h.DocID.String()
			chunkIdx[id] = h.ChunkIndex
			hits = append(hits, Hit{ID: id, Score: h.Score})
		}
		lists["bm25"] = hits
	}

	if useVec {
		vecs, err := e.Embedder.Embed(ctx, []string{q})
		if err != nil {
			return nil, fmt.Errorf("embed: %w", err)
		}
		if len(vecs) != 1 {
			return nil, fmt.Errorf("embed: empty result for query")
		}
		vh, err := e.Store.SearchVector(ctx, vecs[0], e.PerListLimit)
		if err != nil {
			return nil, fmt.Errorf("vector: %w", err)
		}
		hits := make([]Hit, 0, len(vh))
		for _, h := range vh {
			id := h.ChunkID.String()
			if _, ok := snippets[id]; !ok {
				snippets[id] = h.Snippet
				chunkToDoc[id] = h.DocID.String()
				chunkIdx[id] = h.ChunkIndex
			}
			hits = append(hits, Hit{ID: id, Score: h.Score})
		}
		lists["vector"] = hits
	}

	// For the rerank path we pull more candidates out of fusion than we
	// will ultimately return, so the reranker has something to choose
	// from. For every other mode the fusion top-k is the final answer.
	fuseLimit := k
	if mode == ModeHybridRerank && e.Reranker != nil {
		fuseLimit = e.RerankInputN
	}
	fused := FuseRRF(DefaultRRFK, fuseLimit, lists)

	if mode == ModeHybridRerank && e.Reranker != nil && len(fused) > 0 {
		cands := make([]rerank.Candidate, len(fused))
		for i, f := range fused {
			cands[i] = rerank.Candidate{
				ChunkID:    f.ID,
				DocID:      chunkToDoc[f.ID],
				Snippet:    snippets[f.ID],
				PriorScore: f.RRFScore,
				PriorRank:  i,
			}
		}
		scored, err := e.Reranker.Rerank(ctx, q, cands, k)
		if err != nil {
			return nil, fmt.Errorf("rerank: %w", err)
		}
		out := make([]Result, 0, len(scored))
		for _, s := range scored {
			signals := map[string]float64{
				"rrf":    s.PriorScore,
				"rerank": s.RerankScore,
			}
			out = append(out, Result{
				DocID:   s.DocID,
				ChunkID: s.ChunkID,
				Score:   s.RerankScore,
				Snippet: truncate(s.Snippet, 240),
				Signals: signals,
			})
		}
		return out, nil
	}

	out := make([]Result, 0, len(fused))
	for _, f := range fused {
		out = append(out, Result{
			DocID:   chunkToDoc[f.ID],
			ChunkID: f.ID,
			Score:   f.RRFScore,
			Snippet: truncate(snippets[f.ID], 240),
			Signals: f.PerSignal,
		})
	}
	return out, nil
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}
