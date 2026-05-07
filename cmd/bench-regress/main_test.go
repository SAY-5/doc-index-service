package main

import "testing"

func TestCompare_DetectsLatencyRegression(t *testing.T) {
	b := benchResult{
		Docs: 5000,
		Modes: []modeStats{
			{Mode: "hybrid", P50Ms: 10, P95Ms: 20, P99Ms: 30, P999: 40},
		},
	}
	f := benchResult{
		Docs: 5000,
		Modes: []modeStats{
			{Mode: "hybrid", P50Ms: 14, P95Ms: 21, P99Ms: 31, P999: 41},
		},
	}
	got := compare(b, f, 0.5)
	var p50 *drift
	for i := range got {
		if got[i].metric == "hybrid/p50" {
			p50 = &got[i]
		}
	}
	if p50 == nil || p50.delta < 0.39 || p50.delta > 0.41 {
		t.Fatalf("expected ~0.4 regression on p50, got %+v", p50)
	}
}

func TestCompare_DetectsThroughputRegression(t *testing.T) {
	b := benchResult{Docs: 5000, IndexDocsS: 100}
	f := benchResult{Docs: 5000, IndexDocsS: 60} // 40% slower
	got := compare(b, f, 0.5)
	if len(got) != 1 || got[0].delta < 0.39 || got[0].delta > 0.41 {
		t.Fatalf("expected ~0.4 regression on throughput, got %+v", got)
	}
}

func TestCompare_FasterFreshDoesNotRegress(t *testing.T) {
	b := benchResult{Docs: 5000, IndexDocsS: 50}
	f := benchResult{Docs: 5000, IndexDocsS: 80}
	got := compare(b, f, 0.5)
	if len(got) != 1 || got[0].delta > 0 {
		t.Fatalf("faster fresh should be negative delta, got %+v", got)
	}
}

func TestCompare_TinyLatenciesIgnored(t *testing.T) {
	b := benchResult{
		Docs:  5000,
		Modes: []modeStats{{Mode: "vector", P50Ms: 0.1, P95Ms: 0.2, P99Ms: 0.3, P999: 0.4}},
	}
	f := benchResult{
		Docs:  5000,
		Modes: []modeStats{{Mode: "vector", P50Ms: 5.0, P95Ms: 5.0, P99Ms: 5.0, P999: 5.0}},
	}
	got := compare(b, f, 0.5)
	for _, d := range got {
		t.Logf("%s -> %.2f", d.metric, d.delta)
	}
	if len(got) != 0 {
		t.Fatalf("tiny latencies should be ignored, got %d drifts", len(got))
	}
}

func TestCompare_MissingModeFlagged(t *testing.T) {
	b := benchResult{Docs: 5000, Modes: []modeStats{{Mode: "hybrid", P50Ms: 10}}}
	f := benchResult{Docs: 5000, Modes: nil}
	got := compare(b, f, 0.5)
	if len(got) != 1 || got[0].delta < 1e9 {
		t.Fatalf("missing mode should produce +Inf drift, got %+v", got)
	}
}
