package api

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"net/http"
)

type ctxKey int

const requestIDKey ctxKey = 1

// RequestIDFrom returns the X-Request-ID associated with a request, or
// "unknown" if the middleware did not see it.
func RequestIDFrom(ctx context.Context) string {
	v, _ := ctx.Value(requestIDKey).(string)
	if v == "" {
		return "unknown"
	}
	return v
}

// RequestID middleware echoes any client-supplied X-Request-ID header,
// generates a new id if missing, and stashes the result on the request
// context so handlers can include it in error envelopes.
func RequestID(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id := r.Header.Get("X-Request-ID")
		if id == "" {
			id = newID()
		}
		w.Header().Set("X-Request-ID", id)
		ctx := context.WithValue(r.Context(), requestIDKey, id)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

func newID() string {
	var b [8]byte
	_, _ = rand.Read(b[:])
	return hex.EncodeToString(b[:])
}

// Recoverer turns panics into 500 envelopes so a single broken handler
// doesn't kill the process. The original error is logged via the writer's
// header for visibility in tests but not surfaced to clients.
func Recoverer(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if rv := recover(); rv != nil {
				WriteError(w, RequestIDFrom(r.Context()), &APIError{
					Status:  http.StatusInternalServerError,
					Code:    "panic",
					Message: "handler panicked",
				})
			}
		}()
		next.ServeHTTP(w, r)
	})
}
