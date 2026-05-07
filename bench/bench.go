// Command bench measures index throughput and per-mode query latency for
// the doc-index-service.
//
// The harness is intentionally separate from `go test`. Tests should be
// fast and run on every push; this program seeds tens of thousands of
// rows, hits each retrieval mode a thousand times, and writes a JSON
// artefact under bench/results/.
package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/SAY-5/doc-index-service/internal/chunker"
	"github.com/SAY-5/doc-index-service/internal/search"
	"github.com/SAY-5/doc-index-service/internal/seed"
	"github.com/SAY-5/doc-index-service/internal/store"
	"github.com/SAY-5/doc-index-service/pkg/embed"
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

type result struct {
	StartedAt   time.Time   `json:"started_at"`
	CompletedAt time.Time   `json:"completed_at"`
	Docs        int         `json:"docs"`
	Queries     int         `json:"queries"`
	Workers     int         `json:"workers"`
	Embedder    string      `json:"embedder"`
	IndexDocsS  float64     `json:"index_docs_per_sec"`
	IndexChnksS float64     `json:"index_chunks_per_sec"`
	IndexWallS  float64     `json:"index_wall_seconds"`
	Modes       []modeStats `json:"modes"`
	Host        string      `json:"host"`
	GoVersion   string      `json:"go_version"`
	GOOS        string      `json:"goos"`
	GOARCH      string      `json:"goarch"`
}

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	var (
		docs      = flag.Int("docs", 200000, "number of synthetic docs to index")
		queries   = flag.Int("queries", 1000, "number of queries per mode")
		workers   = flag.Int("workers", 8, "concurrent indexers")
		batch     = flag.Int("batch", 32, "embed batch size")
		seedV     = flag.Int64("seed", 42, "RNG seed")
		dsnFlag   = flag.String("dsn", "", "Postgres DSN ($DATABASE_URL)")
		embedFlag = flag.String("embed", "", "embed sidecar URL ($EMBED_URL)")
		outDir    = flag.String("out", "bench/results", "JSON output directory")
		smoke     = flag.Bool("smoke", false, "run as a CI smoke check (skip index, ignore empty results)")
	)
	flag.Parse()

	dsn := *dsnFlag
	if dsn == "" {
		dsn = envOr("DATABASE_URL", "postgres://postgres:postgres@localhost:5432/docindex?sslmode=disable")
	}
	embedURL := *embedFlag
	if embedURL == "" {
		embedURL = envOr("EMBED_URL", "http://localhost:8088")
	}

	ctx := context.Background()
	st, err := store.Open(ctx, dsn)
	if err != nil {
		return fmt.Errorf("store: %w", err)
	}
	defer st.Close()

	emb := embed.New(embedURL)
	health, err := emb.Healthz(ctx)
	if err != nil {
		return fmt.Errorf("embed healthz: %w", err)
	}

	startedAt := time.Now().UTC()
	res := result{
		StartedAt: startedAt,
		Docs:      *docs,
		Queries:   *queries,
		Workers:   *workers,
		Embedder:  health.Model,
		Host:      hostname(),
		GoVersion: runtime.Version(),
		GOOS:      runtime.GOOS,
		GOARCH:    runtime.GOARCH,
	}

	// Index phase.
	indexStart := time.Now()
	chunkCount := runIndex(ctx, st, emb, *docs, *seedV, *workers, *batch)
	res.IndexWallS = time.Since(indexStart).Seconds()
	if res.IndexWallS > 0 {
		res.IndexDocsS = float64(*docs) / res.IndexWallS
		res.IndexChnksS = float64(chunkCount) / res.IndexWallS
	}

	// Query phase.
	gen := seed.NewGenerator(*seedV + 1)
	q := gen.QueryWorkload(*queries)
	engine := search.NewEngine(st, emb)

	for _, m := range []search.Mode{search.ModeKeyword, search.ModeVector, search.ModeHybrid} {
		stats := runQueries(ctx, engine, q, m, *smoke)
		res.Modes = append(res.Modes, stats)
	}

	res.CompletedAt = time.Now().UTC()
	printTable(res)

	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		return fmt.Errorf("mkdir: %w", err)
	}
	path := filepath.Join(*outDir, fmt.Sprintf("bench-%s.json", startedAt.Format("20060102-150405")))
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create: %w", err)
	}
	defer func() { _ = f.Close() }()
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	if err := enc.Encode(res); err != nil {
		return fmt.Errorf("encode: %w", err)
	}
	fmt.Printf("\nwrote %s\n", path)
	return nil
}

func runIndex(ctx context.Context, st *store.Store, emb embed.Embedder, n int, seedV int64, workers, batch int) int64 {
	gen := seed.NewGenerator(seedV)
	docs := make([]seed.Doc, n)
	for i := 0; i < n; i++ {
		docs[i] = gen.Next(i, 200, 2000)
	}

	jobs := make(chan seed.Doc, workers*2)
	var (
		chunks int64
		wg     sync.WaitGroup
	)
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for d := range jobs {
				cs := chunker.Split(d.Body, chunker.Options{})
				texts := make([]string, len(cs))
				for i, c := range cs {
					texts[i] = c.Text
				}
				vecs, err := embedBatched(ctx, emb, texts, batch)
				if err != nil {
					log.Printf("embed: %v", err)
					continue
				}
				h := sha256.Sum256([]byte(d.Body))
				sd := store.Doc{
					Source: d.Source, Title: d.Title, Body: d.Body,
					ContentHash: hex.EncodeToString(h[:]),
				}
				sc := make([]store.Chunk, len(cs))
				for i, c := range cs {
					sc[i] = store.Chunk{ChunkIndex: c.Index, Text: c.Text, Embedding: vecs[i]}
				}
				if _, _, _, err := st.UpsertDoc(ctx, sd, sc); err != nil {
					log.Printf("upsert: %v", err)
					continue
				}
				atomic.AddInt64(&chunks, int64(len(cs)))
			}
		}()
	}
	go func() {
		defer close(jobs)
		for _, d := range docs {
			jobs <- d
		}
	}()
	wg.Wait()
	return atomic.LoadInt64(&chunks)
}

func embedBatched(ctx context.Context, e embed.Embedder, texts []string, batch int) ([][]float32, error) {
	if batch <= 0 {
		batch = 32
	}
	out := make([][]float32, 0, len(texts))
	for i := 0; i < len(texts); i += batch {
		end := i + batch
		if end > len(texts) {
			end = len(texts)
		}
		v, err := e.Embed(ctx, texts[i:end])
		if err != nil {
			return nil, err
		}
		out = append(out, v...)
	}
	return out, nil
}

func runQueries(ctx context.Context, eng *search.Engine, qs []string, mode search.Mode, smoke bool) modeStats {
	lat := make([]float64, 0, len(qs))
	for _, q := range qs {
		t0 := time.Now()
		_, err := eng.Query(ctx, q, 10, mode)
		ms := float64(time.Since(t0).Microseconds()) / 1000.0
		if err != nil {
			if !smoke {
				log.Printf("query (%s): %v", mode, err)
			}
			continue
		}
		lat = append(lat, ms)
	}
	stats := modeStats{Mode: string(mode), N: len(lat)}
	if len(lat) == 0 {
		return stats
	}
	sort.Float64s(lat)
	stats.P50Ms = pct(lat, 0.50)
	stats.P95Ms = pct(lat, 0.95)
	stats.P99Ms = pct(lat, 0.99)
	stats.P999 = pct(lat, 0.999)
	stats.MaxMs = lat[len(lat)-1]
	return stats
}

func pct(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	idx := int(math.Ceil(p*float64(len(sorted)))) - 1
	if idx < 0 {
		idx = 0
	}
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}

func printTable(r result) {
	fmt.Printf("# doc-index-service bench — %d docs, %d queries\n", r.Docs, r.Queries)
	fmt.Printf("# %s, %s/%s, embedder=%s\n", r.StartedAt.Format("2006-01-02 15:04:05Z"), r.GOOS, r.GOARCH, r.Embedder)
	fmt.Println("## index throughput")
	fmt.Printf("docs/sec   : %.1f\n", r.IndexDocsS)
	fmt.Printf("chunks/sec : %.1f\n", r.IndexChnksS)
	fmt.Println("## query latency (ms)")
	fmt.Printf("%-9s %-6s %-6s %-6s %-6s %-6s\n", "mode", "p50", "p95", "p99", "p999", "max")
	for _, m := range r.Modes {
		fmt.Printf("%-9s %-6.1f %-6.1f %-6.1f %-6.1f %-6.1f\n",
			m.Mode, m.P50Ms, m.P95Ms, m.P99Ms, m.P999, m.MaxMs)
	}
	fmt.Println("# Local-machine numbers. See bench/README.md for methodology.")
}

func envOr(k, fallback string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return fallback
}

func hostname() string {
	h, err := os.Hostname()
	if err != nil {
		return "unknown"
	}
	return h
}
