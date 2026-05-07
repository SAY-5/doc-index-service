package search

import (
	"strings"
	"testing"
	"unicode/utf8"
)

// FuzzSanitizeBM25Query asserts the three invariants the rest of the
// retrieval stack relies on, no matter what bytes a client throws at the
// /v1/query endpoint:
//
//   - the output is valid UTF-8 (so it can be passed to Postgres without a
//     decode error from the wire driver),
//   - the output never contains a NUL byte (Postgres rejects NULs in text),
//   - the output is bounded by MaxQueryRunes (so a megabyte of garbage
//     can't blow up the planner).
//
// The corpus is seeded with strings that historically tripped up
// hand-rolled sanitisers: BOMs, lone surrogates, replacement runes, mixed
// whitespace, and embedded controls.
func FuzzSanitizeBM25Query(f *testing.F) {
	seeds := []string{
		"",
		"hello world",
		"  hello\tworld  ",
		"\x00\x00\x00",
		"abc\x00def",
		"abc\x01\x02\x03def",
		"abc\xffxyz",
		"\xed\xa0\x80",
		"\ufeffbom-prefixed",
		"café résumé",
		"a" + strings.Repeat("b", MaxQueryRunes*2),
		"\"unbalanced quote",
		"and or not",
		"emoji 🚀 spans two surrogates",
	}
	for _, s := range seeds {
		f.Add(s)
	}

	f.Fuzz(func(t *testing.T, in string) {
		got := SanitizeBM25Query(in)
		if !utf8.ValidString(got) {
			t.Fatalf("invalid utf-8 output from %q: %q", in, got)
		}
		if strings.ContainsRune(got, 0) {
			t.Fatalf("NUL leaked into sanitised output from %q", in)
		}
		if n := utf8.RuneCountInString(got); n > MaxQueryRunes {
			t.Fatalf("output exceeds MaxQueryRunes (%d > %d)", n, MaxQueryRunes)
		}
		// Idempotence: sanitising twice must not change anything. This
		// guards against future edits that add a normalisation pass which
		// is not a fixed point.
		if again := SanitizeBM25Query(got); again != got {
			t.Fatalf("not idempotent: %q -> %q", got, again)
		}
	})
}
