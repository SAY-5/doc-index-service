package api

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"

	"github.com/SAY-5/doc-index-service/internal/search"
	"github.com/SAY-5/doc-index-service/internal/store"
)

// fakeStore satisfies DocStore for the handler tests.
type fakeStore struct {
	upsertID        uuid.UUID
	upsertChunkN    int
	upsertExisted   bool
	upsertErr       error
	upsertCallCount int

	getDoc store.Doc
	getErr error

	list    []store.Doc
	listErr error
}

func (f *fakeStore) UpsertDoc(_ context.Context, _ store.Doc, _ []store.Chunk) (uuid.UUID, int, bool, error) {
	f.upsertCallCount++
	return f.upsertID, f.upsertChunkN, f.upsertExisted, f.upsertErr
}
func (f *fakeStore) GetDoc(_ context.Context, _ uuid.UUID) (store.Doc, error) {
	return f.getDoc, f.getErr
}
func (f *fakeStore) ListDocs(_ context.Context, _ time.Time, _ uuid.UUID, _ int) ([]store.Doc, error) {
	return f.list, f.listErr
}

// fakeEmbedder returns a fixed-length zero vector for each text.
type fakeEmbedder struct {
	calls int
	err   error
}

func (f *fakeEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	f.calls++
	if f.err != nil {
		return nil, f.err
	}
	out := make([][]float32, len(texts))
	for i := range texts {
		out[i] = make([]float32, 384)
	}
	return out, nil
}

// fakeSearch returns a canned result set.
type fakeSearch struct {
	results []search.Result
	err     error
}

func (f *fakeSearch) Query(_ context.Context, _ string, _ int, _ search.Mode) ([]search.Result, error) {
	return f.results, f.err
}

func newServer(s DocStore, e *fakeEmbedder, sr SearchEngine) *Server {
	return &Server{Store: s, Embedder: e, Search: sr, EmbedBatchSize: 32}
}

func do(t *testing.T, srv *Server, method, path string, body any) *httptest.ResponseRecorder {
	t.Helper()
	var rdr io.Reader
	if body != nil {
		raw, _ := json.Marshal(body)
		rdr = bytes.NewReader(raw)
	}
	req := httptest.NewRequest(method, path, rdr)
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	w := httptest.NewRecorder()
	srv.Routes().ServeHTTP(w, req)
	return w
}

func TestHandleHealth(t *testing.T) {
	srv := newServer(&fakeStore{}, &fakeEmbedder{}, &fakeSearch{})
	w := do(t, srv, "GET", "/healthz", nil)
	if w.Code != 200 {
		t.Fatalf("status %d", w.Code)
	}
}

func TestHandleIndex_HappyPath(t *testing.T) {
	id := uuid.New()
	st := &fakeStore{upsertID: id, upsertChunkN: 1}
	e := &fakeEmbedder{}
	srv := newServer(st, e, &fakeSearch{})
	w := do(t, srv, "POST", "/v1/index", map[string]string{
		"source": "https://example.com",
		"title":  "title",
		"body":   "this is a small body",
	})
	if w.Code != 200 {
		t.Fatalf("status %d body=%s", w.Code, w.Body.String())
	}
	var resp indexResp
	_ = json.Unmarshal(w.Body.Bytes(), &resp)
	if resp.DocID != id.String() {
		t.Fatalf("doc_id %q", resp.DocID)
	}
	if e.calls == 0 {
		t.Fatal("embedder not called")
	}
}

func TestHandleIndex_Validation(t *testing.T) {
	srv := newServer(&fakeStore{}, &fakeEmbedder{}, &fakeSearch{})
	cases := []struct {
		body any
		want int
	}{
		{nil, http.StatusBadRequest},
		{"not json object", http.StatusBadRequest},
		{map[string]string{"source": "", "title": "t", "body": "b"}, http.StatusBadRequest},
		{map[string]string{"source": "s", "title": "", "body": "b"}, http.StatusBadRequest},
		{map[string]string{"source": "s", "title": "t", "body": "  "}, http.StatusBadRequest},
	}
	for _, c := range cases {
		w := do(t, srv, "POST", "/v1/index", c.body)
		if w.Code != c.want {
			t.Fatalf("body=%v: status=%d want=%d", c.body, w.Code, c.want)
		}
	}
}

func TestHandleIndex_EmbedFailure(t *testing.T) {
	srv := newServer(&fakeStore{}, &fakeEmbedder{err: errors.New("down")}, &fakeSearch{})
	w := do(t, srv, "POST", "/v1/index", map[string]string{
		"source": "s", "title": "t", "body": "b",
	})
	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("status %d", w.Code)
	}
	var env ErrorEnvelope
	_ = json.Unmarshal(w.Body.Bytes(), &env)
	if !env.Retryable {
		t.Fatal("503 should be retryable")
	}
}

func TestHandleQuery_HappyPath(t *testing.T) {
	results := []search.Result{
		{DocID: "d1", ChunkID: "c1", Score: 0.5, Snippet: "s", Signals: map[string]float64{"bm25": 1, "vector": 0.9}},
	}
	srv := newServer(&fakeStore{}, &fakeEmbedder{}, &fakeSearch{results: results})
	w := do(t, srv, "POST", "/v1/query", map[string]any{"q": "hello", "k": 5, "mode": "hybrid"})
	if w.Code != 200 {
		t.Fatalf("status %d body=%s", w.Code, w.Body.String())
	}
	var resp queryResp
	_ = json.Unmarshal(w.Body.Bytes(), &resp)
	if len(resp.Results) != 1 || resp.Results[0].DocID != "d1" {
		t.Fatalf("body: %+v", resp)
	}
}

func TestHandleQuery_Validation(t *testing.T) {
	srv := newServer(&fakeStore{}, &fakeEmbedder{}, &fakeSearch{})
	cases := []struct {
		body any
		want int
	}{
		{nil, http.StatusBadRequest},
		{map[string]any{"q": ""}, http.StatusBadRequest},
		{map[string]any{"q": "x", "k": -1}, http.StatusBadRequest},
		{map[string]any{"q": "x", "mode": "wrong"}, http.StatusBadRequest},
	}
	for _, c := range cases {
		w := do(t, srv, "POST", "/v1/query", c.body)
		if w.Code != c.want {
			t.Fatalf("body=%v: status=%d want=%d", c.body, w.Code, c.want)
		}
	}
}

func TestHandleQuery_DownstreamError(t *testing.T) {
	srv := newServer(&fakeStore{}, &fakeEmbedder{}, &fakeSearch{err: errors.New("boom")})
	w := do(t, srv, "POST", "/v1/query", map[string]any{"q": "x"})
	if w.Code != 500 {
		t.Fatalf("status %d", w.Code)
	}
}

func TestHandleGetDoc(t *testing.T) {
	id := uuid.New()
	st := &fakeStore{getDoc: store.Doc{ID: id, Title: "t", Body: "b", Source: "s", CreatedAt: time.Now()}}
	srv := newServer(st, &fakeEmbedder{}, &fakeSearch{})
	w := do(t, srv, "GET", "/v1/docs/"+id.String(), nil)
	if w.Code != 200 {
		t.Fatalf("status %d", w.Code)
	}
}

func TestHandleGetDoc_NotFound(t *testing.T) {
	st := &fakeStore{getErr: store.ErrNotFound}
	srv := newServer(st, &fakeEmbedder{}, &fakeSearch{})
	w := do(t, srv, "GET", "/v1/docs/"+uuid.New().String(), nil)
	if w.Code != 404 {
		t.Fatalf("status %d", w.Code)
	}
}

func TestHandleGetDoc_BadID(t *testing.T) {
	srv := newServer(&fakeStore{}, &fakeEmbedder{}, &fakeSearch{})
	w := do(t, srv, "GET", "/v1/docs/not-a-uuid", nil)
	if w.Code != 400 {
		t.Fatalf("status %d", w.Code)
	}
}

func TestHandleListDocs_Empty(t *testing.T) {
	srv := newServer(&fakeStore{}, &fakeEmbedder{}, &fakeSearch{})
	w := do(t, srv, "GET", "/v1/docs", nil)
	if w.Code != 200 {
		t.Fatalf("status %d", w.Code)
	}
	var resp listResp
	_ = json.Unmarshal(w.Body.Bytes(), &resp)
	if len(resp.Items) != 0 {
		t.Fatalf("expected empty items: %+v", resp)
	}
}

func TestHandleListDocs_Pagination(t *testing.T) {
	now := time.Now().UTC()
	docs := make([]store.Doc, 50)
	for i := range docs {
		docs[i] = store.Doc{ID: uuid.New(), Title: "t", CreatedAt: now.Add(time.Duration(i) * time.Second)}
	}
	st := &fakeStore{list: docs}
	srv := newServer(st, &fakeEmbedder{}, &fakeSearch{})
	w := do(t, srv, "GET", "/v1/docs?limit=50", nil)
	if w.Code != 200 {
		t.Fatalf("status %d", w.Code)
	}
	var resp listResp
	_ = json.Unmarshal(w.Body.Bytes(), &resp)
	if resp.NextCursor == "" {
		t.Fatal("expected next_cursor when page is full")
	}
	if len(resp.Items) != 50 {
		t.Fatalf("items: %d", len(resp.Items))
	}
}

func TestHandleListDocs_BadLimit(t *testing.T) {
	srv := newServer(&fakeStore{}, &fakeEmbedder{}, &fakeSearch{})
	w := do(t, srv, "GET", "/v1/docs?limit=999", nil)
	if w.Code != 400 {
		t.Fatalf("status %d", w.Code)
	}
}

func TestHandleListDocs_BadCursor(t *testing.T) {
	srv := newServer(&fakeStore{}, &fakeEmbedder{}, &fakeSearch{})
	w := do(t, srv, "GET", "/v1/docs?cursor=!!!", nil)
	if w.Code != 400 {
		t.Fatalf("status %d", w.Code)
	}
}

func TestRequestIDMiddleware_EchoesHeader(t *testing.T) {
	srv := newServer(&fakeStore{}, &fakeEmbedder{}, &fakeSearch{})
	req := httptest.NewRequest("GET", "/healthz", nil)
	req.Header.Set("X-Request-ID", "client-supplied")
	w := httptest.NewRecorder()
	srv.Routes().ServeHTTP(w, req)
	if got := w.Header().Get("X-Request-ID"); got != "client-supplied" {
		t.Fatalf("expected echo, got %q", got)
	}
}

func TestRequestIDMiddleware_GeneratesWhenAbsent(t *testing.T) {
	srv := newServer(&fakeStore{}, &fakeEmbedder{}, &fakeSearch{})
	req := httptest.NewRequest("GET", "/healthz", nil)
	w := httptest.NewRecorder()
	srv.Routes().ServeHTTP(w, req)
	got := w.Header().Get("X-Request-ID")
	if got == "" || strings.Contains(got, " ") {
		t.Fatalf("unexpected request id %q", got)
	}
}

func TestHandleIndex_PersistFailure(t *testing.T) {
	st := &fakeStore{upsertErr: errors.New("db down")}
	srv := newServer(st, &fakeEmbedder{}, &fakeSearch{})
	w := do(t, srv, "POST", "/v1/index", map[string]string{
		"source": "s", "title": "t", "body": "b",
	})
	if w.Code != http.StatusInternalServerError {
		t.Fatalf("status %d", w.Code)
	}
}

func TestHandleGetDoc_LookupFailure(t *testing.T) {
	st := &fakeStore{getErr: errors.New("db blip")}
	srv := newServer(st, &fakeEmbedder{}, &fakeSearch{})
	w := do(t, srv, "GET", "/v1/docs/"+uuid.New().String(), nil)
	if w.Code != http.StatusInternalServerError {
		t.Fatalf("status %d", w.Code)
	}
}

func TestHandleListDocs_StoreError(t *testing.T) {
	st := &fakeStore{listErr: errors.New("list down")}
	srv := newServer(st, &fakeEmbedder{}, &fakeSearch{})
	w := do(t, srv, "GET", "/v1/docs", nil)
	if w.Code != http.StatusInternalServerError {
		t.Fatalf("status %d", w.Code)
	}
}

func TestHandleListDocs_WithCursor(t *testing.T) {
	enc, _ := EncodeCursor(Cursor{CreatedAt: time.Now(), ID: uuid.New().String()})
	srv := newServer(&fakeStore{}, &fakeEmbedder{}, &fakeSearch{})
	w := do(t, srv, "GET", "/v1/docs?cursor="+enc, nil)
	if w.Code != 200 {
		t.Fatalf("status %d", w.Code)
	}
}

func TestParseUUIDOrNil(t *testing.T) {
	if got := parseUUIDOrNil(""); got != uuid.Nil {
		t.Fatalf("empty should be nil, got %s", got)
	}
	if got := parseUUIDOrNil("not-a-uuid"); got != uuid.Nil {
		t.Fatalf("invalid should be nil, got %s", got)
	}
	want := uuid.New()
	if got := parseUUIDOrNil(want.String()); got != want {
		t.Fatalf("valid uuid mismatch: %s vs %s", got, want)
	}
}

func TestEmbedBatched_Empty(t *testing.T) {
	srv := newServer(&fakeStore{}, &fakeEmbedder{}, &fakeSearch{})
	out, err := srv.embedBatched(context.Background(), nil)
	if err != nil || out != nil {
		t.Fatalf("empty input should be (nil,nil): out=%v err=%v", out, err)
	}
}

func TestEmbedBatched_DefaultBatchSize(t *testing.T) {
	srv := &Server{Store: &fakeStore{}, Embedder: &fakeEmbedder{}, Search: &fakeSearch{}, EmbedBatchSize: 0}
	out, err := srv.embedBatched(context.Background(), []string{"a", "b", "c"})
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 3 {
		t.Fatalf("expected 3 vectors, got %d", len(out))
	}
}

func TestNewServer_Defaults(t *testing.T) {
	// NewServer accepts a *store.Store (concrete type). Pass a nil pool —
	// the constructor only reads the pointer, doesn't dial it; the
	// handlers never run in this test.
	srv := NewServer(nil, &fakeEmbedder{})
	if srv.EmbedBatchSize != 32 {
		t.Fatalf("default batch size %d", srv.EmbedBatchSize)
	}
	if srv.Search == nil {
		t.Fatal("Search should be wired")
	}
}

func TestRecoverer(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/boom", func(_ http.ResponseWriter, _ *http.Request) {
		panic("nope")
	})
	h := Recoverer(RequestID(mux))
	req := httptest.NewRequest("GET", "/boom", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	if w.Code != 500 {
		t.Fatalf("status %d", w.Code)
	}
}
