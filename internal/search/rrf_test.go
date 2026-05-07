package search

import (
	"math"
	"testing"
)

const eps = 1e-9

func approx(a, b float64) bool { return math.Abs(a-b) < eps }

func TestFuseRRF_EmptyInputs(t *testing.T) {
	got := FuseRRF(60, 5, nil)
	if len(got) != 0 {
		t.Fatalf("nil input should produce empty result, got %d", len(got))
	}
	got = FuseRRF(60, 5, map[string][]Hit{
		"bm25":   {},
		"vector": {},
	})
	if len(got) != 0 {
		t.Fatalf("empty lists should produce empty result, got %d", len(got))
	}
}

func TestFuseRRF_SingleListIdentity(t *testing.T) {
	hits := []Hit{{ID: "a", Score: 9}, {ID: "b", Score: 8}, {ID: "c", Score: 7}}
	got := FuseRRF(60, 10, map[string][]Hit{"bm25": hits})
	if len(got) != 3 {
		t.Fatalf("expected 3 hits, got %d", len(got))
	}
	wantOrder := []string{"a", "b", "c"}
	for i, w := range wantOrder {
		if got[i].ID != w {
			t.Fatalf("position %d: want %s, got %s", i, w, got[i].ID)
		}
	}
	if !approx(got[0].RRFScore, 1.0/61.0) {
		t.Fatalf("rrf score wrong for top hit: %v", got[0].RRFScore)
	}
	if got[0].PerSignal["bm25"] != 9 {
		t.Fatalf("per-signal score lost: %v", got[0].PerSignal)
	}
}

func TestFuseRRF_BothListsBoostsCommon(t *testing.T) {
	bm := []Hit{{ID: "x", Score: 1}, {ID: "y", Score: 1}, {ID: "z", Score: 1}}
	vec := []Hit{{ID: "y", Score: 1}, {ID: "x", Score: 1}, {ID: "w", Score: 1}}
	got := FuseRRF(60, 10, map[string][]Hit{
		"bm25":   bm,
		"vector": vec,
	})
	if len(got) != 4 {
		t.Fatalf("expected 4 unique ids, got %d", len(got))
	}
	// y appears at ranks 2 (bm25) and 1 (vector); x at 1 and 2.
	// Both should sum to 1/61 + 1/62 and tie-break alphabetically -> x before y.
	if got[0].ID != "x" {
		t.Fatalf("expected x first by alphabetical tie-break, got %s", got[0].ID)
	}
	wantTop := 1.0/61.0 + 1.0/62.0
	if !approx(got[0].RRFScore, wantTop) {
		t.Fatalf("top rrf score: want %v got %v", wantTop, got[0].RRFScore)
	}
	if got[0].PerSignal["bm25"] == 0 || got[0].PerSignal["vector"] == 0 {
		t.Fatalf("expected both signals populated: %v", got[0].PerSignal)
	}
}

func TestFuseRRF_TopNTruncation(t *testing.T) {
	bm := make([]Hit, 0, 50)
	for i := 0; i < 50; i++ {
		bm = append(bm, Hit{ID: string(rune('a' + i%26)), Score: float64(50 - i)})
	}
	got := FuseRRF(60, 5, map[string][]Hit{"bm25": bm})
	if len(got) != 5 {
		t.Fatalf("expected 5 results, got %d", len(got))
	}
}

func TestFuseRRF_ZeroDefaults(t *testing.T) {
	hits := []Hit{{ID: "a", Score: 1}}
	got := FuseRRF(0, 0, map[string][]Hit{"bm25": hits})
	if len(got) != 1 {
		t.Fatalf("expected default topN to admit at least one result")
	}
	// k defaulted to 60 -> 1/(60+1)
	if !approx(got[0].RRFScore, 1.0/61.0) {
		t.Fatalf("default k: %v", got[0].RRFScore)
	}
}
