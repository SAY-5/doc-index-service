package api

import (
	"encoding/binary"
	"errors"
	"io"
)

// Length-prefixed framing helpers.
//
// The cursor and (eventually) other small opaque blobs we hand to clients
// benefit from a defensive frame: a magic byte, a version byte, and a
// uvarint length so we can rotate payload formats without breaking
// in-flight cursors. The encoding stays a few bytes long for the typical
// case while giving the decoder an easy way to reject anything that does
// not start with our magic prefix.
//
// MaxFramePayload bounds the payload at 64 KiB. That is deliberately tight
// — cursors are tiny and any unbounded "length" coming from a client is
// almost certainly a fuzzer or a confused caller; refusing it early is
// safer than allocating gigabytes from a uvarint.
const (
	frameMagic      byte = 0xD1
	frameVersion    byte = 0x01
	MaxFramePayload      = 64 * 1024
)

// ErrFrameInvalid is returned when a frame is malformed or its length
// exceeds MaxFramePayload. Callers translate it to 400 Bad Request.
var ErrFrameInvalid = errors.New("frame: invalid")

// EncodeFrame wraps payload with a magic byte, a version byte, and a
// uvarint length, in that order. It always succeeds.
func EncodeFrame(payload []byte) []byte {
	buf := make([]byte, 0, len(payload)+binary.MaxVarintLen64+2)
	buf = append(buf, frameMagic, frameVersion)
	var lenBuf [binary.MaxVarintLen64]byte
	n := binary.PutUvarint(lenBuf[:], uint64(len(payload)))
	buf = append(buf, lenBuf[:n]...)
	buf = append(buf, payload...)
	return buf
}

// DecodeFrame undoes EncodeFrame. It returns ErrFrameInvalid for any
// header mismatch, oversize length, or short read; otherwise the payload
// slice (which aliases into b — copy if you need to retain it).
func DecodeFrame(b []byte) ([]byte, error) {
	if len(b) < 3 {
		return nil, ErrFrameInvalid
	}
	if b[0] != frameMagic {
		return nil, ErrFrameInvalid
	}
	if b[1] != frameVersion {
		return nil, ErrFrameInvalid
	}
	rest := b[2:]
	n, consumed := binary.Uvarint(rest)
	if consumed <= 0 {
		return nil, ErrFrameInvalid
	}
	if n > MaxFramePayload {
		return nil, ErrFrameInvalid
	}
	rest = rest[consumed:]
	if uint64(len(rest)) < n {
		return nil, io.ErrUnexpectedEOF
	}
	return rest[:n], nil
}
