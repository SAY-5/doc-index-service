// Package api implements the HTTP layer of the doc-index-service.
//
// The package owns request/response types, middleware (request ID,
// recoverer), error envelope plumbing, opaque cursor encoding, and the
// HTTP handlers that glue together the chunker, embedder, and store.
package api

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"time"
)

// Cursor is the opaque token sent to clients for keyset pagination over
// the docs list. It carries the (created_at, id) pair of the last row
// returned, which lets the next query continue with WHERE
// (created_at, id) < ($createdAt, $id) — stable under concurrent inserts
// because the ordering is total.
type Cursor struct {
	CreatedAt time.Time `json:"t"`
	ID        string    `json:"i"`
}

// EncodeCursor serialises a Cursor to a URL-safe base64 string.
func EncodeCursor(c Cursor) (string, error) {
	raw, err := json.Marshal(c)
	if err != nil {
		return "", err
	}
	return base64.RawURLEncoding.EncodeToString(raw), nil
}

// DecodeCursor parses an encoded cursor. An empty input returns the zero
// value with no error so callers can use it as the "first page" sentinel.
func DecodeCursor(s string) (Cursor, error) {
	if s == "" {
		return Cursor{}, nil
	}
	raw, err := base64.RawURLEncoding.DecodeString(s)
	if err != nil {
		return Cursor{}, errors.New("cursor: invalid base64")
	}
	var c Cursor
	if err := json.Unmarshal(raw, &c); err != nil {
		return Cursor{}, errors.New("cursor: invalid payload")
	}
	if c.ID == "" {
		return Cursor{}, errors.New("cursor: empty id")
	}
	return c, nil
}
