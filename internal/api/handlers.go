package api

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"net/http"
	"strconv"
	"strings"

	"github.com/google/uuid"

	"github.com/SAY-5/doc-index-service/internal/chunker"
	"github.com/SAY-5/doc-index-service/internal/rerank"
	"github.com/SAY-5/doc-index-service/internal/search"
	"github.com/SAY-5/doc-index-service/internal/store"
	"github.com/SAY-5/doc-index-service/pkg/embed"
)

// Server bundles the dependencies handlers need.
type Server struct {
	Store    DocStore
	Embedder embed.Embedder
	Search   SearchEngine
	// EmbedBatchSize controls the indexer-side batching when calling the
	// sidecar; 32 is small enough to fit in a single GPU step on cheap
	// hardware and large enough to amortise the HTTP round-trip.
	EmbedBatchSize int
}

// NewServer wires defaults against the concrete store and embedder.
// The reranker argument is optional; nil means rerank-flagged queries
// transparently fall back to plain hybrid.
func NewServer(s *store.Store, e embed.Embedder, r rerank.Reranker) *Server {
	eng := search.NewEngine(s, e)
	if r != nil {
		eng = eng.WithReranker(r)
	}
	return &Server{
		Store:          s,
		Embedder:       e,
		Search:         eng,
		EmbedBatchSize: 32,
	}
}

// Routes returns the configured http.Handler with middleware applied.
func (s *Server) Routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/index", s.handleIndex)
	mux.HandleFunc("POST /v1/query", s.handleQuery)
	mux.HandleFunc("GET /v1/docs/{id}", s.handleGetDoc)
	mux.HandleFunc("GET /v1/docs", s.handleListDocs)
	mux.HandleFunc("GET /healthz", s.handleHealth)
	return Recoverer(RequestID(mux))
}

type indexReq struct {
	Source string `json:"source"`
	Title  string `json:"title"`
	Body   string `json:"body"`
}

type indexResp struct {
	DocID         string `json:"doc_id"`
	ChunkCount    int    `json:"chunk_count"`
	AlreadyExists bool   `json:"already_exists"`
}

func (s *Server) handleIndex(w http.ResponseWriter, r *http.Request) {
	rid := RequestIDFrom(r.Context())
	var req indexReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		WriteError(w, rid, BadRequest("invalid_body", "expected JSON object"))
		return
	}
	req.Source = strings.TrimSpace(req.Source)
	req.Title = strings.TrimSpace(req.Title)
	if req.Source == "" || req.Title == "" || strings.TrimSpace(req.Body) == "" {
		WriteError(w, rid, BadRequest("missing_field", "source, title, and body are required"))
		return
	}

	chunks := chunker.Split(req.Body, chunker.Options{})
	texts := make([]string, len(chunks))
	for i, c := range chunks {
		texts[i] = c.Text
	}

	embeds, err := s.embedBatched(r.Context(), texts)
	if err != nil {
		WriteError(w, rid, Unavailable("embed_failed", err.Error()))
		return
	}

	hash := sha256.Sum256([]byte(req.Body))
	d := store.Doc{
		Source:      req.Source,
		Title:       req.Title,
		Body:        req.Body,
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

	id, n, existed, err := s.Store.UpsertDoc(r.Context(), d, storeChunks)
	if err != nil {
		WriteError(w, rid, &APIError{
			Status:  http.StatusInternalServerError,
			Code:    "persist_failed",
			Message: err.Error(),
		})
		return
	}
	WriteJSON(w, http.StatusOK, indexResp{
		DocID:         id.String(),
		ChunkCount:    n,
		AlreadyExists: existed,
	})
}

func (s *Server) embedBatched(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	bs := s.EmbedBatchSize
	if bs <= 0 {
		bs = 32
	}
	out := make([][]float32, 0, len(texts))
	for i := 0; i < len(texts); i += bs {
		end := i + bs
		if end > len(texts) {
			end = len(texts)
		}
		v, err := s.Embedder.Embed(ctx, texts[i:end])
		if err != nil {
			return nil, err
		}
		out = append(out, v...)
	}
	return out, nil
}

type queryReq struct {
	Q    string `json:"q"`
	K    int    `json:"k"`
	Mode string `json:"mode"`
}

type queryResp struct {
	Results []search.Result `json:"results"`
}

func (s *Server) handleQuery(w http.ResponseWriter, r *http.Request) {
	rid := RequestIDFrom(r.Context())
	var req queryReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		WriteError(w, rid, BadRequest("invalid_body", "expected JSON object"))
		return
	}
	q := strings.TrimSpace(req.Q)
	if q == "" {
		WriteError(w, rid, BadRequest("missing_field", "q is required"))
		return
	}
	if req.K < 0 || req.K > 100 {
		WriteError(w, rid, BadRequest("invalid_k", "k must be between 0 and 100"))
		return
	}
	mode, err := search.ParseMode(req.Mode)
	if err != nil {
		WriteError(w, rid, BadRequest("invalid_mode", err.Error()))
		return
	}

	results, err := s.Search.Query(r.Context(), q, req.K, mode)
	if err != nil {
		WriteError(w, rid, &APIError{
			Status:    http.StatusInternalServerError,
			Code:      "query_failed",
			Message:   err.Error(),
			Retryable: false,
		})
		return
	}
	WriteJSON(w, http.StatusOK, queryResp{Results: results})
}

func (s *Server) handleGetDoc(w http.ResponseWriter, r *http.Request) {
	rid := RequestIDFrom(r.Context())
	idStr := r.PathValue("id")
	id, err := uuid.Parse(idStr)
	if err != nil {
		WriteError(w, rid, BadRequest("invalid_id", "id is not a uuid"))
		return
	}
	d, err := s.Store.GetDoc(r.Context(), id)
	if err != nil {
		if errors.Is(err, store.ErrNotFound) {
			WriteError(w, rid, NotFound("not_found", "doc not found"))
			return
		}
		WriteError(w, rid, &APIError{
			Status:  http.StatusInternalServerError,
			Code:    "lookup_failed",
			Message: err.Error(),
		})
		return
	}
	WriteJSON(w, http.StatusOK, map[string]any{
		"id":           d.ID.String(),
		"source":       d.Source,
		"title":        d.Title,
		"body":         d.Body,
		"content_hash": d.ContentHash,
		"created_at":   d.CreatedAt,
	})
}

type listResp struct {
	Items      []listItem `json:"items"`
	NextCursor string     `json:"next_cursor,omitempty"`
}

type listItem struct {
	ID        string `json:"id"`
	Source    string `json:"source"`
	Title     string `json:"title"`
	CreatedAt string `json:"created_at"`
}

func (s *Server) handleListDocs(w http.ResponseWriter, r *http.Request) {
	rid := RequestIDFrom(r.Context())
	cur, err := DecodeCursor(r.URL.Query().Get("cursor"))
	if err != nil {
		WriteError(w, rid, BadRequest("invalid_cursor", err.Error()))
		return
	}
	limit := 50
	if l := r.URL.Query().Get("limit"); l != "" {
		n, err := strconv.Atoi(l)
		if err != nil || n <= 0 || n > 200 {
			WriteError(w, rid, BadRequest("invalid_limit", "limit must be 1..200"))
			return
		}
		limit = n
	}
	docs, err := s.Store.ListDocs(r.Context(), cur.CreatedAt, parseUUIDOrNil(cur.ID), limit)
	if err != nil {
		WriteError(w, rid, &APIError{
			Status:  http.StatusInternalServerError,
			Code:    "list_failed",
			Message: err.Error(),
		})
		return
	}
	resp := listResp{Items: make([]listItem, 0, len(docs))}
	for _, d := range docs {
		resp.Items = append(resp.Items, listItem{
			ID:        d.ID.String(),
			Source:    d.Source,
			Title:     d.Title,
			CreatedAt: d.CreatedAt.UTC().Format("2006-01-02T15:04:05.000Z"),
		})
	}
	if len(docs) == limit {
		next, _ := EncodeCursor(Cursor{
			CreatedAt: docs[len(docs)-1].CreatedAt,
			ID:        docs[len(docs)-1].ID.String(),
		})
		resp.NextCursor = next
	}
	WriteJSON(w, http.StatusOK, resp)
}

func parseUUIDOrNil(s string) uuid.UUID {
	if s == "" {
		return uuid.Nil
	}
	id, err := uuid.Parse(s)
	if err != nil {
		return uuid.Nil
	}
	return id
}

func (s *Server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	WriteJSON(w, http.StatusOK, map[string]any{"status": "ok"})
}
