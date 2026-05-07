// Command bench-regress diffs two bench JSON artefacts and exits non-zero
// when any tracked metric drifts beyond the configured tolerance.
//
// The intent is a CI-friendly regression gate: a baseline JSON is
// committed at a small, repeatable scale, and each push runs a fresh
// bench at the same scale and compares the two. Drift is measured per
// metric so a single noisy outlier in one mode doesn't mask a real
// regression in another.
//
// "Drift" is defined relative to the baseline:
//
//	delta = (fresh - baseline) / baseline
//
// For latency metrics (p50/p95/p99/p999/max), positive delta means the
// fresh run is *slower* — that's the regression we want to catch.
// For throughput metrics (index_docs_per_sec, index_chunks_per_sec),
// positive delta means the fresh run is *faster*; a regression here is
// negative drift, so the gate flips the sign before comparing.
//
// Both runs must declare the same `docs` count or the comparison is
// refused outright; comparing 1k against 5k is a category error.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
)

type modeStats struct {
	Mode  string  `json:"mode"`
	N     int     `json:"n"`
	P50Ms float64 `json:"p50_ms"`
	P95Ms float64 `json:"p95_ms"`
	P99Ms float64 `json:"p99_ms"`
	P999  float64 `json:"p999_ms"`
	MaxMs float64 `json:"max_ms"`
}

type benchResult struct {
	Docs        int         `json:"docs"`
	Queries     int         `json:"queries"`
	IndexDocsS  float64     `json:"index_docs_per_sec"`
	IndexChnksS float64     `json:"index_chunks_per_sec"`
	Modes       []modeStats `json:"modes"`
}

type drift struct {
	metric string
	delta  float64 // positive = regression
	base   float64
	fresh  float64
}

func main() {
	var (
		baselinePath = flag.String("baseline", "", "path to baseline JSON")
		freshPath    = flag.String("fresh", "", "path to fresh JSON")
		tol          = flag.Float64("tol", 0.30, "tolerance as a fraction (0.30 = 30%)")
		minLatencyMs = flag.Float64("min-latency-ms", 0.5, "ignore drift on latencies below this floor")
	)
	flag.Parse()

	if *baselinePath == "" || *freshPath == "" {
		fmt.Fprintln(os.Stderr, "usage: bench-regress -baseline <path> -fresh <path> [-tol 0.30]")
		os.Exit(2)
	}

	baseline, err := loadResult(*baselinePath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "baseline: %v\n", err)
		os.Exit(2)
	}
	fresh, err := loadResult(*freshPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fresh: %v\n", err)
		os.Exit(2)
	}

	if baseline.Docs != fresh.Docs {
		fmt.Fprintf(os.Stderr, "scale mismatch: baseline docs=%d, fresh docs=%d\n", baseline.Docs, fresh.Docs)
		os.Exit(2)
	}

	drifts := compare(baseline, fresh, *minLatencyMs)
	report(drifts, *tol)

	regressed := false
	for _, d := range drifts {
		if d.delta > *tol {
			regressed = true
		}
	}
	if regressed {
		fmt.Fprintf(os.Stderr, "\nFAIL: one or more metrics regressed beyond %.0f%%\n", *tol*100)
		os.Exit(1)
	}
	fmt.Println("\nOK: all metrics within tolerance")
}

func loadResult(path string) (benchResult, error) {
	f, err := os.Open(path)
	if err != nil {
		return benchResult{}, err
	}
	defer func() { _ = f.Close() }()
	var r benchResult
	if err := json.NewDecoder(f).Decode(&r); err != nil {
		return benchResult{}, fmt.Errorf("decode %s: %w", path, err)
	}
	return r, nil
}

// compare returns one drift per (metric, mode) combination. Throughput
// metrics' deltas are sign-flipped so "positive delta == regression"
// holds uniformly.
func compare(b, f benchResult, minLat float64) []drift {
	var out []drift

	// Throughput.
	if b.IndexDocsS > 0 {
		out = append(out, drift{
			metric: "index_docs_per_sec",
			delta:  -((f.IndexDocsS - b.IndexDocsS) / b.IndexDocsS),
			base:   b.IndexDocsS,
			fresh:  f.IndexDocsS,
		})
	}
	if b.IndexChnksS > 0 {
		out = append(out, drift{
			metric: "index_chunks_per_sec",
			delta:  -((f.IndexChnksS - b.IndexChnksS) / b.IndexChnksS),
			base:   b.IndexChnksS,
			fresh:  f.IndexChnksS,
		})
	}

	// Latency. Fold by mode name.
	freshByMode := make(map[string]modeStats, len(f.Modes))
	for _, m := range f.Modes {
		freshByMode[m.Mode] = m
	}
	for _, bm := range b.Modes {
		fm, ok := freshByMode[bm.Mode]
		if !ok {
			out = append(out, drift{metric: bm.Mode + "/missing", delta: math.Inf(1), base: 1, fresh: 0})
			continue
		}
		latencies := []struct {
			name string
			b, f float64
		}{
			{"p50", bm.P50Ms, fm.P50Ms},
			{"p95", bm.P95Ms, fm.P95Ms},
			{"p99", bm.P99Ms, fm.P99Ms},
			{"p999", bm.P999, fm.P999},
		}
		for _, lat := range latencies {
			if lat.b < minLat {
				// Tiny numbers — even a 30% relative shift may be noise.
				continue
			}
			out = append(out, drift{
				metric: bm.Mode + "/" + lat.name,
				delta:  (lat.f - lat.b) / lat.b,
				base:   lat.b,
				fresh:  lat.f,
			})
		}
	}
	return out
}

func report(drifts []drift, tol float64) {
	fmt.Printf("%-32s %-10s %-10s %-8s %s\n", "metric", "baseline", "fresh", "delta", "verdict")
	for _, d := range drifts {
		v := "ok"
		if d.delta > tol {
			v = "REGRESS"
		}
		fmt.Printf("%-32s %-10.3f %-10.3f %+7.1f%% %s\n", d.metric, d.base, d.fresh, d.delta*100, v)
	}
}
