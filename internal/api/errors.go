package api

import (
	"encoding/json"
	"errors"
	"net/http"
)

// ErrorEnvelope is the canonical body of every non-2xx response. The
// Retryable flag tells clients whether re-issuing the same request without
// modification has any chance of succeeding.
type ErrorEnvelope struct {
	Code      string `json:"code"`
	Message   string `json:"message"`
	Retryable bool   `json:"retryable"`
	RequestID string `json:"request_id"`
}

// APIError is a typed error that handlers can return; the middleware turns
// it into a JSON ErrorEnvelope. Plain errors map to 500 / internal_error.
type APIError struct {
	Status    int
	Code      string
	Message   string
	Retryable bool
}

func (e *APIError) Error() string { return e.Code + ": " + e.Message }

// BadRequest builds a 400 envelope.
func BadRequest(code, msg string) *APIError {
	return &APIError{Status: http.StatusBadRequest, Code: code, Message: msg}
}

// NotFound builds a 404 envelope.
func NotFound(code, msg string) *APIError {
	return &APIError{Status: http.StatusNotFound, Code: code, Message: msg}
}

// Conflict builds a 409 envelope.
func Conflict(code, msg string) *APIError {
	return &APIError{Status: http.StatusConflict, Code: code, Message: msg}
}

// Unavailable builds a 503 envelope, marked retryable.
func Unavailable(code, msg string) *APIError {
	return &APIError{Status: http.StatusServiceUnavailable, Code: code, Message: msg, Retryable: true}
}

// WriteError renders err as the canonical envelope. requestID is pulled
// from the response writer's context-injected header by the middleware.
func WriteError(w http.ResponseWriter, requestID string, err error) {
	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		apiErr = &APIError{
			Status:  http.StatusInternalServerError,
			Code:    "internal_error",
			Message: "unexpected server error",
		}
	}
	env := ErrorEnvelope{
		Code:      apiErr.Code,
		Message:   apiErr.Message,
		Retryable: apiErr.Retryable,
		RequestID: requestID,
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(apiErr.Status)
	_ = json.NewEncoder(w).Encode(env)
}

// WriteJSON writes a successful payload with content-type set.
func WriteJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}
