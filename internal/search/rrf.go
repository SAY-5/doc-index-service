// Package search fuses keyword and vector retrieval into a single ranked
// result set.
//
// The fusion strategy is reciprocal rank fusion (RRF). Both retrievers
// return ordered lists of identifiers; for each id we sum 1 / (k + rank)
// over every list it appears in, with rank starting at 1. The top-N items
// by total score are returned.
//
// RRF is used here in preference to a weighted sum of raw scores because
// BM25 and cosine similarity live on different scales: a small change in
// the BM25 coefficient changes the relative weight of the two signals
// dramatically. RRF only sees ordinal positions, so it is invariant to
// rescaling and degrades gracefully when one retriever returns a noisy
// list. The cost is throwing away score magnitude inside each list — but
// for the top-K reranking step that information is mostly redundant.
package search

import "sort"

// DefaultRRFK is the standard constant from Cormack et al. (2009). Larger
// values flatten the contribution of high ranks; 60 is well-tested.
const DefaultRRFK = 60

// Hit is one retrieval candidate. Score is the underlying retriever's raw
// score and is not used by the fusion math; it is preserved so callers can
// surface it as a per-signal "explanation" alongside the fused score.
type Hit struct {
	ID    string
	Score float64
}

// Fused is the merged result. PerSignal["bm25"] and PerSignal["vector"]
// expose the raw scores for the response envelope.
type Fused struct {
	ID        string
	RRFScore  float64
	PerSignal map[string]float64
}

// FuseRRF merges any number of ranked lists into a single descending list
// of length up to topN. The k parameter dampens contributions from low
// ranks; pass DefaultRRFK if you have no opinion. Lists may be empty.
func FuseRRF(k int, topN int, lists map[string][]Hit) []Fused {
	if k <= 0 {
		k = DefaultRRFK
	}
	if topN <= 0 {
		topN = 10
	}

	scores := make(map[string]*Fused)
	for signal, hits := range lists {
		for rank, h := range hits {
			f, ok := scores[h.ID]
			if !ok {
				f = &Fused{
					ID:        h.ID,
					PerSignal: make(map[string]float64),
				}
				scores[h.ID] = f
			}
			f.RRFScore += 1.0 / float64(k+rank+1)
			// Keep the best (highest) raw score per signal in case the same
			// id appears more than once in a single list.
			if cur, seen := f.PerSignal[signal]; !seen || h.Score > cur {
				f.PerSignal[signal] = h.Score
			}
		}
	}

	out := make([]Fused, 0, len(scores))
	for _, f := range scores {
		out = append(out, *f)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].RRFScore != out[j].RRFScore {
			return out[i].RRFScore > out[j].RRFScore
		}
		// Stable tie-break by id for deterministic output.
		return out[i].ID < out[j].ID
	})
	if len(out) > topN {
		out = out[:topN]
	}
	return out
}
