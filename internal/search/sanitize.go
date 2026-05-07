package search

import (
	"strings"
	"unicode"
	"unicode/utf8"
)

// MaxQueryRunes caps the length of an incoming query before it is passed
// to Postgres. websearch_to_tsquery is happy to consume long strings, but
// allowing megabyte-sized queries lets a caller blow up the planner; 4 KiB
// of runes is generous for any human-formed query and still bounds the
// worst case.
const MaxQueryRunes = 4096

// SanitizeBM25Query normalises a free-form query string for the keyword
// retriever. The intent is *not* to escape every websearch operator —
// websearch_to_tsquery already swallows unbalanced quotes and stray
// punctuation gracefully — but to make the call deterministic and
// crash-safe in three ways:
//
//  1. drop NUL bytes and other C0/C1 control runes (Postgres rejects NULs
//     in text columns and we'd rather not surface that as a 500),
//  2. collapse runs of unicode whitespace to a single ASCII space so the
//     ts parser sees the same token boundaries we do, and
//  3. truncate to MaxQueryRunes runes (not bytes — splitting mid-rune
//     would corrupt UTF-8).
//
// The returned string is always valid UTF-8 and never contains a NUL.
// Empty / whitespace-only inputs return "".
func SanitizeBM25Query(q string) string {
	if q == "" {
		return ""
	}

	var b strings.Builder
	b.Grow(len(q))

	runes := 0
	prevSpace := true // suppress leading whitespace
	for i := 0; i < len(q); {
		r, size := utf8.DecodeRuneInString(q[i:])
		i += size

		// Replace the unicode replacement marker (RuneError on a 1-byte
		// step means invalid UTF-8) with a space so the token boundary is
		// preserved without leaking garbage downstream.
		if r == utf8.RuneError && size == 1 {
			if !prevSpace {
				b.WriteByte(' ')
				prevSpace = true
				runes++
			}
			continue
		}

		// Drop NULs and other control characters outright; they are
		// either rejected by Postgres or have no useful semantics for a
		// text query.
		if r == 0 || (unicode.IsControl(r) && !unicode.IsSpace(r)) {
			continue
		}

		// Drop format / zero-width runes (BOM, ZWNJ, ZWJ, ZWSP, …). These
		// are token-boundary noise — keeping them would mean repeated
		// sanitisation isn't a fixed point.
		if isZeroWidth(r) {
			continue
		}

		if unicode.IsSpace(r) {
			if !prevSpace {
				b.WriteByte(' ')
				prevSpace = true
				runes++
			}
			continue
		}

		b.WriteRune(r)
		prevSpace = false
		runes++

		if runes >= MaxQueryRunes {
			break
		}
	}

	return strings.TrimRight(b.String(), " ")
}

// isZeroWidth recognises the formatting / zero-width runes we want
// stripped: BOM, the zero-width joiner family, soft hyphen, and the
// directional formatting marks. They are dropped uniformly so the
// sanitiser is a fixed-point function.
func isZeroWidth(r rune) bool {
	switch r {
	case '\ufeff', // BOM
		'\u200b', // ZWSP
		'\u200c', // ZWNJ
		'\u200d', // ZWJ
		'\u00ad', // soft hyphen
		'\u2060', // word joiner
		'\u180e': // mongolian vowel separator
		return true
	}
	// LRM/RLM and other directional formatting characters.
	if r >= '\u200e' && r <= '\u200f' {
		return true
	}
	if r >= '\u202a' && r <= '\u202e' {
		return true
	}
	if r >= '\u2066' && r <= '\u2069' {
		return true
	}
	return false
}
