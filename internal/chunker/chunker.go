// Package chunker splits document bodies into bounded chunks for embedding.
//
// The chunker walks the input by Unicode code points so it never splits a
// rune in half, prefers paragraph and sentence boundaries when one exists
// within the configured window, and falls back to a hard cut when no
// boundary is found. Empty inputs return a single empty chunk so callers
// can rely on at-least-one-row semantics during persistence.
package chunker

import (
	"strings"
	"unicode/utf8"
)

// Default sizes are tuned to fit comfortably under the 256-token window of
// MiniLM-class embedders (~500 chars at English densities) while keeping
// some overlap so adjacent chunks share context.
const (
	DefaultTargetChars = 500
	DefaultMaxChars    = 700
	DefaultOverlap     = 50
)

// Options controls chunk sizing. Zero values fall back to defaults.
type Options struct {
	TargetChars int
	MaxChars    int
	Overlap     int
}

// Chunk is one unit of text plus its position in the source.
type Chunk struct {
	Index int
	Text  string
}

// Split partitions body into Chunk values. The returned slice always has
// at least one entry; an empty body yields a single empty chunk.
func Split(body string, opts Options) []Chunk {
	if opts.TargetChars <= 0 {
		opts.TargetChars = DefaultTargetChars
	}
	if opts.MaxChars <= 0 {
		opts.MaxChars = DefaultMaxChars
	}
	if opts.MaxChars < opts.TargetChars {
		opts.MaxChars = opts.TargetChars
	}
	if opts.Overlap < 0 {
		opts.Overlap = 0
	}
	if opts.Overlap >= opts.TargetChars {
		opts.Overlap = opts.TargetChars / 4
	}

	body = strings.TrimSpace(body)
	if body == "" {
		return []Chunk{{Index: 0, Text: ""}}
	}

	runes := []rune(body)
	if len(runes) <= opts.MaxChars {
		return []Chunk{{Index: 0, Text: string(runes)}}
	}

	var out []Chunk
	idx := 0
	cursor := 0
	for cursor < len(runes) {
		end := cursor + opts.TargetChars
		if end > len(runes) {
			end = len(runes)
		}
		// Try to extend to a clean boundary up to MaxChars.
		hardEnd := cursor + opts.MaxChars
		if hardEnd > len(runes) {
			hardEnd = len(runes)
		}
		end = preferBoundary(runes, cursor, end, hardEnd)

		piece := strings.TrimSpace(string(runes[cursor:end]))
		if piece != "" {
			out = append(out, Chunk{Index: idx, Text: piece})
			idx++
		}
		if end >= len(runes) {
			break
		}
		next := end - opts.Overlap
		if next <= cursor {
			next = end
		}
		cursor = next
	}

	if len(out) == 0 {
		out = append(out, Chunk{Index: 0, Text: string(runes)})
	}
	return out
}

// preferBoundary scans [target, hardEnd) for a sentence or whitespace break
// and returns the position at the end of that break, or target if none.
func preferBoundary(runes []rune, _, target, hardEnd int) int {
	if target >= len(runes) {
		return len(runes)
	}
	// Look forward up to hardEnd for the end of a sentence.
	for i := target; i < hardEnd; i++ {
		r := runes[i]
		if r == '.' || r == '!' || r == '?' || r == '\n' {
			// Include the punctuation, then any trailing whitespace.
			j := i + 1
			for j < hardEnd && isSpace(runes[j]) {
				j++
			}
			return j
		}
	}
	// Fall back: walk forward to the next whitespace boundary.
	for i := target; i < hardEnd; i++ {
		if isSpace(runes[i]) {
			return i
		}
	}
	return target
}

func isSpace(r rune) bool {
	return r == ' ' || r == '\t' || r == '\n' || r == '\r'
}

// CountRunes returns the rune length of s. Exported for callers that want
// to sanity-check chunk sizes without round-tripping through Split.
func CountRunes(s string) int { return utf8.RuneCountInString(s) }
