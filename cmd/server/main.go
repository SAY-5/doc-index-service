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
	server := api.NewServer(s, emb)

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
