// Command server runs the HTTP API for the doc-index-service.
package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/SAY-5/doc-index-service/internal/api"
	"github.com/SAY-5/doc-index-service/internal/rerank"
	"github.com/SAY-5/doc-index-service/internal/store"
	"github.com/SAY-5/doc-index-service/pkg/embed"
)

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	dsn := getenv("DATABASE_URL", "postgres://postgres:postgres@localhost:5432/docindex?sslmode=disable")
	embedURL := getenv("EMBED_URL", "http://localhost:8088")
	addr := getenv("HTTP_ADDR", ":8080")

	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer cancel()

	s, err := store.Open(ctx, dsn)
	if err != nil {
		return fmt.Errorf("store: %w", err)
	}
	defer s.Close()

	emb := embed.New(embedURL)

	// Reranker selection: RERANKER_KIND=cross-encoder routes /v1/query
	// rerank flags through the sidecar; "heuristic" (the default) keeps
	// everything in-process and is what CI exercises. "off" disables
	// rerank entirely so the server falls back to plain hybrid even when
	// clients ask for it.
	var reranker rerank.Reranker
	switch getenv("RERANKER_KIND", "heuristic") {
	case "off":
		reranker = nil
	case "cross-encoder":
		reranker = rerank.NewCrossEncoderReranker(embedURL)
	default:
		reranker = rerank.NewHeuristicReranker()
	}
	server := api.NewServer(s, emb, reranker)

	srv := &http.Server{
		Addr:              addr,
		Handler:           server.Routes(),
		ReadHeaderTimeout: 5 * time.Second,
	}

	errCh := make(chan error, 1)
	go func() {
		log.Printf("server listening on %s", addr)
		errCh <- srv.ListenAndServe()
	}()

	select {
	case <-ctx.Done():
		log.Println("shutting down")
		shutCtx, shutCancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer shutCancel()
		_ = srv.Shutdown(shutCtx)
		return nil
	case err := <-errCh:
		if errors.Is(err, http.ErrServerClosed) {
			return nil
		}
		return fmt.Errorf("listen: %w", err)
	}
}

func getenv(k, fallback string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return fallback
}
