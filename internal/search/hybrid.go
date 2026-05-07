package search

import (
	"context"
	"fmt"

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
	ModeHybrid  Mode = "hybrid"
	ModeKeyword Mode = "keyword"
	ModeVector  Mode = "vector"
)

// ParseMode validates a string from the request body.
func ParseMode(s string) (Mode, error) {
	switch Mode(s) {
	case "":
		return ModeHybrid, nil
	case ModeHybrid, ModeKeyword, ModeVector:
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
}

// NewEngine returns an Engine with sensible defaults.
func NewEngine(s Backend, e embed.Embedder) *Engine {
	return &Engine{Store: s, Embedder: e, PerListLimit: 50}
}

// Query runs the requested retrieval mode and returns up to k results.
func (e *Engine) Query(ctx context.Context, q string, k int, mode Mode) ([]Result, error) {
	if k <= 0 {
		k = 10
	}
	if e.PerListLimit <= 0 {
		e.PerListLimit = 50
	}

	snippets := make(map[string]string)
	chunkToDoc := make(map[string]string)
	chunkIdx := make(map[string]int)

	lists := make(map[string][]Hit)

	if mode == ModeHybrid || mode == ModeKeyword {
		bm, err := e.Store.SearchBM25(ctx, q, e.PerListLimit)
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

	if mode == ModeHybrid || mode == ModeVector {
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

	fused := FuseRRF(DefaultRRFK, k, lists)
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
