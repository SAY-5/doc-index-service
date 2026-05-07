package seed

import (
	"testing"
)

func TestGenerator_Deterministic(t *testing.T) {
	g1 := NewGenerator(42)
	g2 := NewGenerator(42)
	for i := 0; i < 50; i++ {
		a := g1.Next(i, 200, 600)
		b := g2.Next(i, 200, 600)
		if a.Body != b.Body || a.Hash != b.Hash || a.Title != b.Title {
			t.Fatalf("non-deterministic at i=%d:\n%+v\n%+v", i, a, b)
		}
	}
}

func TestGenerator_RespectsBounds(t *testing.T) {
	g := NewGenerator(7)
	for i := 0; i < 100; i++ {
		d := g.Next(i, 200, 600)
		// We pad whole sentences so we may slightly overshoot, but never under.
		if len(d.Body) < 200 {
			t.Fatalf("body too short at i=%d: %d", i, len(d.Body))
		}
		if d.Title == "" || d.Source == "" || d.Hash == "" {
			t.Fatalf("missing fields: %+v", d)
		}
	}
}

func TestQueryWorkload_Size(t *testing.T) {
	g := NewGenerator(1)
	q := g.QueryWorkload(100)
	if len(q) != 100 {
		t.Fatalf("expected 100 queries, got %d", len(q))
	}
	for _, s := range q {
		if s == "" {
			t.Fatalf("empty query")
		}
	}
}

func TestGenerator_DifferentSeedsDiffer(t *testing.T) {
	g1 := NewGenerator(1)
	g2 := NewGenerator(2)
	a := g1.Next(0, 200, 600)
	b := g2.Next(0, 200, 600)
	if a.Body == b.Body {
		t.Fatalf("different seeds produced identical output")
	}
}
