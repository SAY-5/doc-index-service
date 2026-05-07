package api

import (
	"context"
	"time"

	"github.com/google/uuid"

	"github.com/SAY-5/doc-index-service/internal/search"
	"github.com/SAY-5/doc-index-service/internal/store"
)

// DocStore is the subset of *store.Store the API layer touches. The
// indirection lets us drop a fake into handler tests without standing up
// a real Postgres for every assertion.
type DocStore interface {
	UpsertDoc(ctx context.Context, d store.Doc, chunks []store.Chunk) (uuid.UUID, int, bool, error)
	GetDoc(ctx context.Context, id uuid.UUID) (store.Doc, error)
	ListDocs(ctx context.Context, after time.Time, afterID uuid.UUID, limit int) ([]store.Doc, error)
}

// SearchEngine abstracts the search.Engine behind a tiny method set.
type SearchEngine interface {
	Query(ctx context.Context, q string, k int, mode search.Mode) ([]search.Result, error)
}
