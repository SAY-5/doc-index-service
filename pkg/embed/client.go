// Package embed defines the contract between the Go services and the
// Python embedding sidecar. The HTTP client lives here so cmd/server,
// cmd/indexer, and bench/ can share one implementation.
package embed

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"
)

// Dim is fixed at 384 because we ship a single-model deployment. If the
// sidecar reports a different dimension at /healthz the server refuses to
// start, since the Postgres column is dimensionally typed.
const Dim = 384

// Embedder produces dense vectors for a batch of texts. Implementations
// must be safe for concurrent use.
type Embedder interface {
	Embed(ctx context.Context, texts []string) ([][]float32, error)
}

// Client is a small JSON-over-HTTP client for the sidecar. Construct via
// New. All RPCs honour ctx deadlines.
type Client struct {
	baseURL string
	http    *http.Client
}

// New returns a Client with a sensible default timeout. baseURL must not
// have a trailing slash.
func New(baseURL string) *Client {
	return &Client{
		baseURL: baseURL,
		http: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

type embedRequest struct {
	Texts []string `json:"texts"`
}

type embedResponse struct {
	Embeddings [][]float32 `json:"embeddings"`
	Model      string      `json:"model"`
	Dim        int         `json:"dim"`
}

// Embed calls POST /embed and returns the raw vectors. An empty input
// returns an empty slice without contacting the sidecar.
func (c *Client) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	body, err := json.Marshal(embedRequest{Texts: texts})
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/embed", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("embed: transport: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, fmt.Errorf("embed: status %d: %s", resp.StatusCode, string(raw))
	}
	var out embedResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("embed: decode: %w", err)
	}
	if len(out.Embeddings) != len(texts) {
		return nil, fmt.Errorf("embed: expected %d vectors, got %d", len(texts), len(out.Embeddings))
	}
	if out.Dim != Dim {
		return nil, fmt.Errorf("embed: dim mismatch (got %d, want %d)", out.Dim, Dim)
	}
	for i, v := range out.Embeddings {
		if len(v) != Dim {
			return nil, fmt.Errorf("embed: vector %d has length %d", i, len(v))
		}
	}
	return out.Embeddings, nil
}

// Health probes /healthz. ready=false signals the model is still loading.
type Health struct {
	Model string `json:"model"`
	Ready bool   `json:"ready"`
}

// Healthz pings the sidecar's health endpoint.
func (c *Client) Healthz(ctx context.Context) (Health, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/healthz", http.NoBody)
	if err != nil {
		return Health{}, err
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return Health{}, err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		return Health{}, errors.New("healthz: non-200")
	}
	var h Health
	if err := json.NewDecoder(resp.Body).Decode(&h); err != nil {
		return Health{}, err
	}
	return h, nil
}
