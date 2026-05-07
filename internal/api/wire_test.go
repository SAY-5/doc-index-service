package api

import (
	"bytes"
	"errors"
	"testing"
)

func TestEncodeDecodeFrame_RoundTrip(t *testing.T) {
	cases := [][]byte{
		nil,
		{},
		[]byte("hello"),
		bytes.Repeat([]byte{0x42}, 1024),
		bytes.Repeat([]byte{0xff}, MaxFramePayload),
	}
	for _, in := range cases {
		got, err := DecodeFrame(EncodeFrame(in))
		if err != nil {
			t.Fatalf("decode err: %v (len=%d)", err, len(in))
		}
		if !bytes.Equal(got, in) {
			t.Fatalf("payload mismatch (len=%d)", len(in))
		}
	}
}

func TestDecodeFrame_RejectsShort(t *testing.T) {
	for _, b := range [][]byte{nil, {0xD1}, {0xD1, 0x01}} {
		if _, err := DecodeFrame(b); !errors.Is(err, ErrFrameInvalid) {
			t.Fatalf("expected ErrFrameInvalid, got %v", err)
		}
	}
}

func TestDecodeFrame_RejectsBadMagic(t *testing.T) {
	if _, err := DecodeFrame([]byte{0x00, 0x01, 0x00}); !errors.Is(err, ErrFrameInvalid) {
		t.Fatalf("expected ErrFrameInvalid, got %v", err)
	}
}

func TestDecodeFrame_RejectsBadVersion(t *testing.T) {
	if _, err := DecodeFrame([]byte{frameMagic, 0xff, 0x00}); !errors.Is(err, ErrFrameInvalid) {
		t.Fatalf("expected ErrFrameInvalid, got %v", err)
	}
}

func TestDecodeFrame_RejectsOversizeLength(t *testing.T) {
	// Hand-craft a uvarint of 1 << 40 (way over MaxFramePayload).
	frame := []byte{frameMagic, frameVersion, 0x80, 0x80, 0x80, 0x80, 0x80, 0x10}
	if _, err := DecodeFrame(frame); !errors.Is(err, ErrFrameInvalid) {
		t.Fatalf("expected ErrFrameInvalid, got %v", err)
	}
}
