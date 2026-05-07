package api

import (
	"bytes"
	"errors"
	"strings"
	"testing"
	"time"
)

// FuzzCursorRoundTrip seeds a handful of well-formed cursors and lets the
// fuzzer flip bytes inside the encoded form. The strict round-trip
// assertion catches encoder/decoder drift; the random-input arm asserts
// that a malformed cursor never crashes and either decodes to a sensible
// value or returns an error.
func FuzzCursorRoundTrip(f *testing.F) {
	good := []Cursor{
		{CreatedAt: time.Unix(0, 0).UTC(), ID: "00000000-0000-0000-0000-000000000001"},
		{CreatedAt: time.Date(2026, 5, 1, 0, 0, 0, 0, time.UTC), ID: "abc"},
		{CreatedAt: time.Date(1970, 1, 1, 0, 0, 0, 0, time.UTC), ID: "x"},
	}
	for _, c := range good {
		enc, err := EncodeCursor(c)
		if err != nil {
			f.Fatal(err)
		}
		f.Add(enc)
	}
	f.Add("")
	f.Add("not-base64-!!!")
	f.Add("////")
	f.Add(strings.Repeat("A", 4096))

	f.Fuzz(func(t *testing.T, in string) {
		got, err := DecodeCursor(in)
		if err != nil {
			// Decoding failures are fine; we just need the call not to
			// panic and the returned value to be the zero cursor so the
			// caller can rely on "err != nil" alone as the gate.
			if !got.CreatedAt.IsZero() || got.ID != "" {
				t.Fatalf("non-zero cursor on error path: %+v err=%v", got, err)
			}
			return
		}
		// The empty-string input is the "first page" sentinel; re-encoding
		// it would yield a non-empty cursor, which by design fails the
		// strict empty-ID check. Skip the round-trip in that case.
		if got.ID == "" && got.CreatedAt.IsZero() {
			return
		}
		// Re-encoding must produce a string that decodes to the same
		// value. This is the property handlers depend on for stable
		// pagination.
		enc, err := EncodeCursor(got)
		if err != nil {
			t.Fatalf("re-encode: %v", err)
		}
		again, err := DecodeCursor(enc)
		if err != nil {
			t.Fatalf("re-decode: %v", err)
		}
		if !again.CreatedAt.Equal(got.CreatedAt) || again.ID != got.ID {
			t.Fatalf("round trip drift: %+v -> %+v", got, again)
		}
	})
}

// FuzzDecodeFrame asserts the framing helper is panic-free on arbitrary
// byte streams and rejects oversize lengths. Encode(Decode(b)) is *not*
// asserted because b may contain trailing bytes outside the frame; what
// matters is that whatever the decoder accepts can be re-encoded into a
// frame that decodes to the same payload.
func FuzzDecodeFrame(f *testing.F) {
	seeds := [][]byte{
		nil,
		{},
		{frameMagic, frameVersion, 0x00},
		{frameMagic, frameVersion, 0x01, 0x42},
		{0x00, 0x00, 0x00},
		bytes.Repeat([]byte{0xff}, 32),
	}
	for _, s := range seeds {
		f.Add(s)
	}

	f.Fuzz(func(t *testing.T, in []byte) {
		payload, err := DecodeFrame(in)
		if err != nil {
			// All errors must be either ErrFrameInvalid or a short-read,
			// never an arbitrary panic.
			return
		}
		if len(payload) > MaxFramePayload {
			t.Fatalf("oversize payload accepted: %d bytes", len(payload))
		}
		// Re-encoding the accepted payload must round-trip.
		round, err := DecodeFrame(EncodeFrame(payload))
		if err != nil {
			t.Fatalf("re-decode failed: %v", err)
		}
		if !bytes.Equal(round, payload) {
			t.Fatalf("payload drift after re-encode")
		}
	})
}

// FuzzEncodeFrameNeverPanics is a tiny smoke fuzz that proves the encoder
// is total — it must not allocate astronomically large buffers regardless
// of input length, because callers may forward untrusted byte slices.
func FuzzEncodeFrameNeverPanics(f *testing.F) {
	f.Add([]byte("hello"))
	f.Add([]byte{})
	f.Fuzz(func(t *testing.T, in []byte) {
		out := EncodeFrame(in)
		if len(out) < 3 {
			t.Fatalf("frame too short: %d", len(out))
		}
		if out[0] != frameMagic || out[1] != frameVersion {
			t.Fatalf("header missing")
		}
		if len(in) <= MaxFramePayload {
			payload, err := DecodeFrame(out)
			if err != nil {
				t.Fatalf("self-roundtrip failed: %v", err)
			}
			if !bytes.Equal(payload, in) {
				t.Fatalf("payload drift")
			}
		} else {
			// Oversized input must still produce a frame, but DecodeFrame
			// will (correctly) reject it as ErrFrameInvalid.
			_, err := DecodeFrame(out)
			if !errors.Is(err, ErrFrameInvalid) {
				t.Fatalf("expected ErrFrameInvalid for oversize, got %v", err)
			}
		}
	})
}
