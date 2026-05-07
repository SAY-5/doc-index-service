package api

import (
	"testing"
	"time"
)

func TestCursor_RoundTrip(t *testing.T) {
	want := Cursor{
		CreatedAt: time.Date(2026, 5, 6, 12, 34, 56, 0, time.UTC),
		ID:        "00000000-0000-0000-0000-000000000001",
	}
	enc, err := EncodeCursor(want)
	if err != nil {
		t.Fatal(err)
	}
	if enc == "" {
		t.Fatal("empty encoding")
	}
	got, err := DecodeCursor(enc)
	if err != nil {
		t.Fatal(err)
	}
	if !got.CreatedAt.Equal(want.CreatedAt) || got.ID != want.ID {
		t.Fatalf("round-trip mismatch: %+v vs %+v", got, want)
	}
}

func TestCursor_EmptyIsFirstPage(t *testing.T) {
	c, err := DecodeCursor("")
	if err != nil {
		t.Fatalf("empty cursor should not error: %v", err)
	}
	if !c.CreatedAt.IsZero() || c.ID != "" {
		t.Fatalf("empty cursor should be zero value")
	}
}

func TestCursor_RejectsInvalid(t *testing.T) {
	cases := []string{
		"not-base64-!!!",
		"////",
	}
	for _, c := range cases {
		if _, err := DecodeCursor(c); err == nil {
			t.Fatalf("expected error for %q", c)
		}
	}
}

func TestCursor_RejectsEmptyID(t *testing.T) {
	enc, err := EncodeCursor(Cursor{CreatedAt: time.Now(), ID: ""})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := DecodeCursor(enc); err == nil {
		t.Fatal("expected error for empty id")
	}
}
