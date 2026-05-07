// Command indexer bulk-loads synthetic documents through the index path.
//
// It is the easiest way to populate Postgres for benches and demos: it
// uses the same UpsertDoc / embed-batch path as the HTTP server, so any
// behaviour difference between the server and the indexer is a bug.
package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/SAY-5/doc-index-service/internal/chunker"
	"github.com/SAY-5/doc-index-service/internal/seed"
	"github.com/SAY-5/doc-index-service/internal/store"
	"github.com/SAY-5/doc-index-service/pkg/embed"
)

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	var (
		count     = flag.Int("n", 1000, "number of synthetic docs to insert")
		seedV     = flag.Int64("seed", 42, "RNG seed")
		minBytes  = flag.Int("min", 200, "minimum body length in chars")
		maxBytes  = flag.Int("max", 2000, "maximum body length in chars")
		workers   = flag.Int("workers", 4, "concurrent indexers")
		batch     = flag.Int("batch", 32, "embedder batch size")
		dsnFlag   = flag.String("dsn", "", "Postgres DSN (defaults to $DATABASE_URL)")
		embedFlag = flag.String("embed", "", "embed sidecar base URL (defaults to $EMBED_URL)")
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

	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer cancel()

	st, err := store.Open(ctx, dsn)
	if err != nil {
		return fmt.Errorf("store: %w", err)
	}
	defer st.Close()

	emb := embed.New(embedURL)
	gen := seed.NewGenerator(*seedV)

	docs := make([]seed.Doc, *count)
	for i := 0; i < *count; i++ {
		docs[i] = gen.Next(i, *minBytes, *maxBytes)
	}

	jobs := make(chan seed.Doc, *workers*2)
	var (
		inserted int64
		failed   int64
		wg       sync.WaitGroup
	)
	start := time.Now()

	for w := 0; w < *workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for d := range jobs {
				if err := indexOne(ctx, st, emb, d, *batch); err != nil {
					if ctx.Err() != nil {
						return
					}
					atomic.AddInt64(&failed, 1)
					log.Printf("index failed: %v", err)
					continue
				}
				atomic.AddInt64(&inserted, 1)
			}
		}()
	}

	go func() {
		defer close(jobs)
		for _, d := range docs {
			select {
			case <-ctx.Done():
				return
			case jobs <- d:
			}
		}
	}()

	tick := time.NewTicker(2 * time.Second)
	defer tick.Stop()
	done := make(chan struct{})
	go func() {
		defer close(done)
		wg.Wait()
	}()

loop:
	for {
		select {
		case <-tick.C:
			n := atomic.LoadInt64(&inserted)
			elapsed := time.Since(start).Seconds()
			rate := float64(n) / elapsed
			log.Printf("progress: %d/%d (%.1f docs/s, %d failed)", n, *count, rate, atomic.LoadInt64(&failed))
		case <-done:
			break loop
		}
	}

	elapsed := time.Since(start)
	log.Printf("done: inserted=%d failed=%d in %s (%.1f docs/s)",
		inserted, failed, elapsed,
		float64(inserted)/elapsed.Seconds())
	return nil
}

func indexOne(ctx context.Context, st *store.Store, emb embed.Embedder, d seed.Doc, batch int) error {
	chunks := chunker.Split(d.Body, chunker.Options{})
	texts := make([]string, len(chunks))
	for i, c := range chunks {
		texts[i] = c.Text
	}
	embeds, err := embedBatched(ctx, emb, texts, batch)
	if err != nil {
		return err
	}
	hash := sha256.Sum256([]byte(d.Body))
	storeDoc := store.Doc{
		Source:      d.Source,
		Title:       d.Title,
		Body:        d.Body,
		ContentHash: hex.EncodeToString(hash[:]),
	}
	storeChunks := make([]store.Chunk, len(chunks))
	for i, c := range chunks {
		storeChunks[i] = store.Chunk{
			ChunkIndex: c.Index,
			Text:       c.Text,
			Embedding:  embeds[i],
		}
	}
	_, _, _, err = st.UpsertDoc(ctx, storeDoc, storeChunks)
	return err
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

func envOr(k, fallback string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return fallback
}
