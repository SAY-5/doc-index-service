// Package rerank reorders the top-N candidates produced by RRF using a
// secondary scoring function. Two implementations are provided:
//
//   - HeuristicReranker is a pure-Go token-overlap scorer with a length
//     penalty. It needs no external services so it is the default in CI
//     and makes a sensible fallback when the sidecar is offline.
//   - CrossEncoderReranker calls the embed sidecar's /rerank endpoint,
//     which loads a small cross-encoder (cross-encoder/ms-marco-
//     MiniLM-L-6-v2) and scores (query, passage) pairs jointly. This is
//     the production path; it adds ~50–200 ms of latency in exchange for
//     a measurable top-1 precision lift on hybrid queries where the RRF
//     fusion is right "in spirit" but ranks the wrong chunk first.
//
// Both implementations satisfy the Reranker interface so callers can
// swap them at config time.
package rerank

import (
	"context"
	"math"
	"sort"
	"strings"
	"unicode"
)

// Candidate is a single (chunk, snippet, prior-score) row passed in from
// the fusion stage. The original RRF position is preserved so a stable
// tie-break can fall back to it.
type Candidate struct {
	ChunkID    string
	DocID      string
	Snippet    string
	PriorScore float64
	PriorRank  int // 0-based position in the input list
}

// Scored is a candidate with its rerank score attached. Higher is better.
type Scored struct {
	Candidate
	RerankScore float64
}

// Reranker reorders candidates against a free-form query and returns the
// top topN entries. Implementations must be deterministic for fixed
// inputs so the decision-table tests are repeatable.
type Reranker interface {
	Rerank(ctx context.Context, query string, candidates []Candidate, topN int) ([]Scored, error)
}

// HeuristicReranker scores by lexical overlap between the query and the
// snippet. It is intentionally simple — the goal is a sensible fallback,
// not a state-of-the-art ranker. The scoring function is:
//
//	overlap = |query_tokens ∩ snippet_tokens|
//	denom   = max(1, sqrt(|query_tokens| * |snippet_tokens|))
//	length_penalty = 1 / (1 + max(0, |snippet_tokens| - target) / target)
//	score = (overlap / denom) * length_penalty
//
// where target is the snippet length we'd ideally see (60 tokens). The
// length penalty discourages the ranker from preferring very long
// snippets just because they happen to contain the query terms.
type HeuristicReranker struct {
	// TargetSnippetTokens is the ideal snippet length for the length
	// penalty. Zero means "use the default of 60".
	TargetSnippetTokens int
}

// NewHeuristicReranker returns a reranker with sensible defaults.
func NewHeuristicReranker() *HeuristicReranker {
	return &HeuristicReranker{TargetSnippetTokens: 60}
}

// Rerank implements Reranker.
func (r *HeuristicReranker) Rerank(_ context.Context, query string, candidates []Candidate, topN int) ([]Scored, error) {
	if topN <= 0 {
		topN = len(candidates)
	}
	target := r.TargetSnippetTokens
	if target <= 0 {
		target = 60
	}

	qTokens := tokenSet(query)
	out := make([]Scored, 0, len(candidates))
	for _, c := range candidates {
		sTokens := tokenSet(c.Snippet)
		overlap := 0
		for tok := range qTokens {
			if _, ok := sTokens[tok]; ok {
				overlap++
			}
		}
		denom := 1.0
		if len(qTokens) > 0 && len(sTokens) > 0 {
			denom = math.Sqrt(float64(len(qTokens) * len(sTokens)))
		}
		base := float64(overlap) / denom

		excess := len(sTokens) - target
		if excess < 0 {
			excess = 0
		}
		penalty := 1.0 / (1.0 + float64(excess)/float64(target))

		out = append(out, Scored{Candidate: c, RerankScore: base * penalty})
	}

	// Stable sort: rerank-desc, prior-rank-asc as the tie-break.
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].RerankScore != out[j].RerankScore {
			return out[i].RerankScore > out[j].RerankScore
		}
		return out[i].PriorRank < out[j].PriorRank
	})

	if len(out) > topN {
		out = out[:topN]
	}
	return out, nil
}

// tokenSet lowercases input and splits on non-letter / non-digit runes.
// Stop-word handling is intentionally absent — this is a heuristic, not a
// linguistic model, and dropping common terms hurts as often as it helps
// on short snippets.
func tokenSet(s string) map[string]struct{} {
	if s == "" {
		return nil
	}
	out := make(map[string]struct{})
	var b strings.Builder
	flush := func() {
		if b.Len() == 0 {
			return
		}
		out[b.String()] = struct{}{}
		b.Reset()
	}
	for _, r := range s {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			b.WriteRune(unicode.ToLower(r))
		} else {
			flush()
		}
	}
	flush()
	return out
}
