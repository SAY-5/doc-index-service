package rerank

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestCrossEncoder_HappyPath(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/rerank" {
			t.Errorf("path=%s", r.URL.Path)
		}
		var req rerankRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		// Score by passage length descending so the test can assert order.
		scores := make([]float64, len(req.Passages))
		for i, p := range req.Passages {
			scores[i] = float64(len(p))
		}
		_ = json.NewEncoder(w).Encode(rerankResponse{Scores: scores, Model: "stub"})
	}))
	defer srv.Close()

	r := NewCrossEncoderReranker(srv.URL)
	cands := []Candidate{
		{ChunkID: "a", Snippet: "x", PriorRank: 0},
		{ChunkID: "b", Snippet: "longer snippet", PriorRank: 1},
	}
	out, err := r.Rerank(context.Background(), "q", cands, 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 2 || out[0].ChunkID != "b" {
		t.Fatalf("expected b first by length, got %+v", out)
	}
}

func TestCrossEncoder_EmptyCandidates(t *testing.T) {
	r := NewCrossEncoderReranker("http://nowhere")
	out, err := r.Rerank(context.Background(), "q", nil, 10)
	if err != nil || len(out) != 0 {
		t.Fatalf("expected empty result, got %d err=%v", len(out), err)
	}
}

func TestCrossEncoder_TopNTruncates(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(rerankResponse{
			Scores: []float64{0.1, 0.5, 0.9},
			Model:  "stub",
		})
	}))
	defer srv.Close()

	cands := []Candidate{{ChunkID: "a"}, {ChunkID: "b"}, {ChunkID: "c"}}
	out, err := NewCrossEncoderReranker(srv.URL).Rerank(context.Background(), "q", cands, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 1 || out[0].ChunkID != "c" {
		t.Fatalf("expected only c, got %+v", out)
	}
}

func TestCrossEncoder_StatusError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte("model loading"))
	}))
	defer srv.Close()
	cands := []Candidate{{ChunkID: "a"}}
	_, err := NewCrossEncoderReranker(srv.URL).Rerank(context.Background(), "q", cands, 1)
	if err == nil || !strings.Contains(err.Error(), "503") {
		t.Fatalf("expected 503 error, got %v", err)
	}
}

func TestCrossEncoder_CountMismatch(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(rerankResponse{Scores: []float64{1.0}, Model: "stub"})
	}))
	defer srv.Close()
	cands := []Candidate{{ChunkID: "a"}, {ChunkID: "b"}}
	_, err := NewCrossEncoderReranker(srv.URL).Rerank(context.Background(), "q", cands, 1)
	if err == nil || !strings.Contains(err.Error(), "expected 2") {
		t.Fatalf("expected count-mismatch error, got %v", err)
	}
}

func TestCrossEncoder_TransportError(t *testing.T) {
	r := NewCrossEncoderReranker("http://127.0.0.1:1")
	_, err := r.Rerank(context.Background(), "q", []Candidate{{ChunkID: "a"}}, 1)
	if err == nil {
		t.Fatal("expected transport error")
	}
}

func TestCrossEncoder_BadJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("not json"))
	}))
	defer srv.Close()
	_, err := NewCrossEncoderReranker(srv.URL).Rerank(context.Background(), "q", []Candidate{{ChunkID: "a"}}, 1)
	if err == nil || !strings.Contains(err.Error(), "decode") {
		t.Fatalf("expected decode error, got %v", err)
	}
}
