package rerank

import (
	"context"
	"testing"
)

// TestHeuristic_DecisionTable is the canonical proof that the heuristic
// reranker scores in the order we want. Each row is a (query, snippets,
// expected-top-id) triple covering one decision the ranker has to make.
//
// The cases cover:
//   - exact-match wins over near-miss
//   - long irrelevant snippet loses to short relevant snippet (length
//     penalty)
//   - identical overlap → stable tie-break by prior rank
//   - empty query / empty snippet doesn't crash
//   - case- and punctuation-insensitive token matching
func TestHeuristic_DecisionTable(t *testing.T) {
	cases := []struct {
		name      string
		query     string
		cands     []Candidate
		wantTopID string
	}{
		{
			name:  "exact_match_beats_near_miss",
			query: "redis cluster failover",
			cands: []Candidate{
				{ChunkID: "a", Snippet: "memcached cluster eviction policy", PriorRank: 0},
				{ChunkID: "b", Snippet: "redis cluster failover diagnostics", PriorRank: 1},
			},
			wantTopID: "b",
		},
		{
			name:  "length_penalty_punishes_long_snippet",
			query: "raft leader election",
			cands: []Candidate{
				{ChunkID: "short", Snippet: "raft leader election timing", PriorRank: 0},
				// Distinct fillers so the set grows; a repeated word would
				// dedupe and dodge the penalty (intentional property of
				// set-based overlap, but not what we're testing here).
				{ChunkID: "long", Snippet: "raft leader election " + distinctFiller(200), PriorRank: 1},
			},
			wantTopID: "short",
		},
		{
			name:  "tie_break_by_prior_rank",
			query: "kafka",
			cands: []Candidate{
				{ChunkID: "a", Snippet: "kafka", PriorRank: 0},
				{ChunkID: "b", Snippet: "kafka", PriorRank: 1},
			},
			wantTopID: "a",
		},
		{
			name:  "case_and_punctuation_insensitive",
			query: "GraphQL Schema",
			cands: []Candidate{
				{ChunkID: "a", Snippet: "documentation, generated", PriorRank: 0},
				{ChunkID: "b", Snippet: "GRAPHQL.SCHEMA  validation", PriorRank: 1},
			},
			wantTopID: "b",
		},
		{
			name:  "empty_snippet_does_not_crash",
			query: "anything",
			cands: []Candidate{
				{ChunkID: "a", Snippet: "", PriorRank: 0},
				{ChunkID: "b", Snippet: "anything goes here", PriorRank: 1},
			},
			wantTopID: "b",
		},
		{
			name:  "empty_query_keeps_prior_order",
			query: "",
			cands: []Candidate{
				{ChunkID: "a", Snippet: "alpha", PriorRank: 0},
				{ChunkID: "b", Snippet: "beta", PriorRank: 1},
			},
			wantTopID: "a",
		},
	}

	r := NewHeuristicReranker()
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			out, err := r.Rerank(context.Background(), c.query, c.cands, 10)
			if err != nil {
				t.Fatal(err)
			}
			if len(out) == 0 || out[0].ChunkID != c.wantTopID {
				t.Fatalf("want top=%s, got %+v", c.wantTopID, out)
			}
		})
	}
}

func TestHeuristic_TopNTruncates(t *testing.T) {
	cands := []Candidate{
		{ChunkID: "a", Snippet: "alpha", PriorRank: 0},
		{ChunkID: "b", Snippet: "beta", PriorRank: 1},
		{ChunkID: "c", Snippet: "gamma", PriorRank: 2},
	}
	out, err := NewHeuristicReranker().Rerank(context.Background(), "x", cands, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 2 {
		t.Fatalf("want 2, got %d", len(out))
	}
}

func TestHeuristic_TopNZeroMeansAll(t *testing.T) {
	cands := []Candidate{
		{ChunkID: "a", Snippet: "x", PriorRank: 0},
		{ChunkID: "b", Snippet: "y", PriorRank: 1},
	}
	out, err := NewHeuristicReranker().Rerank(context.Background(), "x", cands, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 2 {
		t.Fatalf("want 2 (all), got %d", len(out))
	}
}

func TestHeuristic_EmptyCandidates(t *testing.T) {
	out, err := NewHeuristicReranker().Rerank(context.Background(), "x", nil, 10)
	if err != nil || len(out) != 0 {
		t.Fatalf("expected empty result, got %d err=%v", len(out), err)
	}
}

func TestTokenSet_Determinism(t *testing.T) {
	// Same input -> same tokens (the reranker depends on this).
	a := tokenSet("Hello, world! WORLD.")
	b := tokenSet("hello world world")
	if len(a) != len(b) {
		t.Fatalf("token sets differ: %v vs %v", a, b)
	}
	for k := range a {
		if _, ok := b[k]; !ok {
			t.Fatalf("token %q missing from canonical set", k)
		}
	}
}

func distinctFiller(n int) string {
	out := make([]byte, 0, n*8)
	for i := 0; i < n; i++ {
		if i > 0 {
			out = append(out, ' ')
		}
		out = append(out, []byte("filler")...)
		out = append(out, []byte(fmtInt(i))...)
	}
	return string(out)
}

func fmtInt(n int) string {
	if n == 0 {
		return "0"
	}
	var b []byte
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	return string(b)
}
