package api

import (
	"encoding/json"
	"errors"
	"net/http/httptest"
	"testing"
)

func TestWriteError_TypedAPIError(t *testing.T) {
	w := httptest.NewRecorder()
	WriteError(w, "req-123", BadRequest("invalid_body", "missing field"))
	if w.Code != 400 {
		t.Fatalf("status = %d", w.Code)
	}
	var env ErrorEnvelope
	if err := json.Unmarshal(w.Body.Bytes(), &env); err != nil {
		t.Fatal(err)
	}
	if env.Code != "invalid_body" || env.Message != "missing field" {
		t.Fatalf("envelope: %+v", env)
	}
	if env.RequestID != "req-123" {
		t.Fatalf("request id missing: %+v", env)
	}
	if env.Retryable {
		t.Fatalf("BadRequest should not be retryable by default")
	}
}

func TestWriteError_Unavailable_IsRetryable(t *testing.T) {
	w := httptest.NewRecorder()
	WriteError(w, "r", Unavailable("embed_down", "sidecar offline"))
	var env ErrorEnvelope
	_ = json.Unmarshal(w.Body.Bytes(), &env)
	if !env.Retryable {
		t.Fatalf("503 should be retryable, got %+v", env)
	}
}

func TestWriteError_PlainErrorBecomesInternal(t *testing.T) {
	w := httptest.NewRecorder()
	WriteError(w, "r", errors.New("boom"))
	if w.Code != 500 {
		t.Fatalf("plain error should map to 500, got %d", w.Code)
	}
	var env ErrorEnvelope
	_ = json.Unmarshal(w.Body.Bytes(), &env)
	if env.Code != "internal_error" {
		t.Fatalf("unexpected code: %v", env)
	}
}
