package rerank

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"time"
)

// CrossEncoderReranker shells out to the embed sidecar's /rerank endpoint
// and reorders candidates by the model's score. The HTTP boundary keeps
// the Go binary free of torch/transformers; the sidecar already owns
// model loading for embedding so adding rerank there is a small step.
type CrossEncoderReranker struct {
	BaseURL string
	HTTP    *http.Client
}

// NewCrossEncoderReranker returns a reranker pointed at the sidecar's
// base URL (e.g. "http://localhost:8088") with a 10 s timeout.
func NewCrossEncoderReranker(baseURL string) *CrossEncoderReranker {
	return &CrossEncoderReranker{
		BaseURL: baseURL,
		HTTP:    &http.Client{Timeout: 10 * time.Second},
	}
}

type rerankRequest struct {
	Query    string   `json:"query"`
	Passages []string `json:"passages"`
}

type rerankResponse struct {
	Scores []float64 `json:"scores"`
	Model  string    `json:"model"`
}

// Rerank implements Reranker.
func (r *CrossEncoderReranker) Rerank(ctx context.Context, query string, candidates []Candidate, topN int) ([]Scored, error) {
	if len(candidates) == 0 {
		return nil, nil
	}
	if topN <= 0 {
		topN = len(candidates)
	}

	passages := make([]string, len(candidates))
	for i, c := range candidates {
		passages[i] = c.Snippet
	}
	body, err := json.Marshal(rerankRequest{Query: query, Passages: passages})
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, r.BaseURL+"/rerank", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := r.HTTP.Do(req)
	if err != nil {
		return nil, fmt.Errorf("rerank: transport: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, fmt.Errorf("rerank: status %d: %s", resp.StatusCode, string(raw))
	}
	var out rerankResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("rerank: decode: %w", err)
	}
	if len(out.Scores) != len(candidates) {
		return nil, fmt.Errorf("rerank: expected %d scores, got %d", len(candidates), len(out.Scores))
	}

	scored := make([]Scored, len(candidates))
	for i, c := range candidates {
		scored[i] = Scored{Candidate: c, RerankScore: out.Scores[i]}
	}
	sort.SliceStable(scored, func(i, j int) bool {
		if scored[i].RerankScore != scored[j].RerankScore {
			return scored[i].RerankScore > scored[j].RerankScore
		}
		return scored[i].PriorRank < scored[j].PriorRank
	})
	if len(scored) > topN {
		scored = scored[:topN]
	}
	return scored, nil
}
