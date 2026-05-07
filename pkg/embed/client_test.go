package embed

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func newServer(t *testing.T, h http.Handler) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)
	return srv
}

func TestClient_Embed_HappyPath(t *testing.T) {
	srv := newServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/embed" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		var req embedRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		out := embedResponse{
			Embeddings: make([][]float32, len(req.Texts)),
			Model:      "test",
			Dim:        Dim,
		}
		for i := range req.Texts {
			out.Embeddings[i] = make([]float32, Dim)
		}
		_ = json.NewEncoder(w).Encode(out)
	}))
	c := New(srv.URL)
	v, err := c.Embed(context.Background(), []string{"a", "b"})
	if err != nil {
		t.Fatal(err)
	}
	if len(v) != 2 || len(v[0]) != Dim {
		t.Fatalf("shape wrong: %d x %d", len(v), len(v[0]))
	}
}

func TestClient_Embed_EmptyShortCircuits(t *testing.T) {
	c := New("http://invalid.local:1") // would fail if reached
	v, err := c.Embed(context.Background(), nil)
	if err != nil || len(v) != 0 {
		t.Fatalf("empty input should not hit network: %v %d", err, len(v))
	}
}

func TestClient_Embed_DimMismatchRejected(t *testing.T) {
	srv := newServer(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(embedResponse{
			Embeddings: [][]float32{make([]float32, 128)},
			Model:      "x",
			Dim:        128,
		})
	}))
	c := New(srv.URL)
	_, err := c.Embed(context.Background(), []string{"x"})
	if err == nil || !strings.Contains(err.Error(), "dim mismatch") {
		t.Fatalf("expected dim mismatch error, got %v", err)
	}
}

func TestClient_Embed_NonOKBody(t *testing.T) {
	srv := newServer(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte("model loading"))
	}))
	c := New(srv.URL)
	_, err := c.Embed(context.Background(), []string{"x"})
	if err == nil || !strings.Contains(err.Error(), "503") {
		t.Fatalf("expected 503 error, got %v", err)
	}
}

func TestClient_Healthz(t *testing.T) {
	srv := newServer(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(Health{Model: "all-MiniLM-L6-v2", Ready: true})
	}))
	c := New(srv.URL)
	h, err := c.Healthz(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if !h.Ready || h.Model == "" {
		t.Fatalf("health: %+v", h)
	}
}

func TestClient_Embed_CountMismatch(t *testing.T) {
	srv := newServer(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(embedResponse{
			Embeddings: [][]float32{make([]float32, Dim)},
			Model:      "x",
			Dim:        Dim,
		})
	}))
	c := New(srv.URL)
	_, err := c.Embed(context.Background(), []string{"a", "b"})
	if err == nil || !strings.Contains(err.Error(), "expected 2") {
		t.Fatalf("expected count mismatch, got %v", err)
	}
}

func TestClient_Embed_PerVectorLengthMismatch(t *testing.T) {
	srv := newServer(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(embedResponse{
			Embeddings: [][]float32{make([]float32, Dim-1)},
			Model:      "x",
			Dim:        Dim,
		})
	}))
	c := New(srv.URL)
	_, err := c.Embed(context.Background(), []string{"x"})
	if err == nil || !strings.Contains(err.Error(), "vector 0") {
		t.Fatalf("expected per-vector length error, got %v", err)
	}
}

func TestClient_Embed_BadJSON(t *testing.T) {
	srv := newServer(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("not json"))
	}))
	c := New(srv.URL)
	_, err := c.Embed(context.Background(), []string{"x"})
	if err == nil || !strings.Contains(err.Error(), "decode") {
		t.Fatalf("expected decode error, got %v", err)
	}
}

func TestClient_Healthz_Non200(t *testing.T) {
	srv := newServer(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	c := New(srv.URL)
	if _, err := c.Healthz(context.Background()); err == nil {
		t.Fatal("expected non-200 error")
	}
}

func TestClient_Healthz_TransportError(t *testing.T) {
	c := New("http://127.0.0.1:1")
	if _, err := c.Healthz(context.Background()); err == nil {
		t.Fatal("expected transport error")
	}
}

func TestClient_Embed_TransportError(t *testing.T) {
	c := New("http://127.0.0.1:1")
	if _, err := c.Embed(context.Background(), []string{"x"}); err == nil {
		t.Fatal("expected transport error")
	}
}
