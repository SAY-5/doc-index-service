package chunker

import (
	"strings"
	"testing"
)

func TestSplit_Empty(t *testing.T) {
	got := Split("", Options{})
	if len(got) != 1 || got[0].Text != "" {
		t.Fatalf("empty input should yield one empty chunk, got %#v", got)
	}
}

func TestSplit_ShortFits(t *testing.T) {
	body := "one short paragraph that easily fits."
	got := Split(body, Options{TargetChars: 500, MaxChars: 700})
	if len(got) != 1 {
		t.Fatalf("short body should be one chunk, got %d", len(got))
	}
	if got[0].Text != body {
		t.Fatalf("unexpected text: %q", got[0].Text)
	}
}

func TestSplit_LongIsPartitioned(t *testing.T) {
	body := strings.Repeat("alpha beta gamma. ", 200) // ~3600 chars
	got := Split(body, Options{TargetChars: 500, MaxChars: 700, Overlap: 50})
	if len(got) < 5 {
		t.Fatalf("expected at least 5 chunks, got %d", len(got))
	}
	// Indices must be monotonically increasing from zero.
	for i, c := range got {
		if c.Index != i {
			t.Fatalf("index %d != position %d", c.Index, i)
		}
	}
	// Each chunk under MaxChars.
	for _, c := range got {
		if CountRunes(c.Text) > 700 {
			t.Fatalf("chunk too long: %d", CountRunes(c.Text))
		}
	}
}

func TestSplit_PrefersSentenceBoundary(t *testing.T) {
	// Build text where a sentence ends slightly past the target.
	prefix := strings.Repeat("a ", 240) // 480 chars
	body := prefix + "end. " + strings.Repeat("z ", 400)
	got := Split(body, Options{TargetChars: 480, MaxChars: 600, Overlap: 20})
	if len(got) < 2 {
		t.Fatalf("expected multi-chunk split")
	}
	if !strings.HasSuffix(strings.TrimSpace(got[0].Text), "end.") {
		t.Fatalf("first chunk should end at sentence boundary, got %q", got[0].Text)
	}
}

func TestSplit_UnicodeSafe(t *testing.T) {
	body := strings.Repeat("日本語のテスト。", 200)
	got := Split(body, Options{TargetChars: 200, MaxChars: 280, Overlap: 20})
	if len(got) < 2 {
		t.Fatalf("expected multi-chunk split")
	}
	for _, c := range got {
		// String must be valid UTF-8 and not produce replacement runes from broken sequences.
		if strings.ContainsRune(c.Text, '�') {
			t.Fatalf("chunk has replacement rune: %q", c.Text)
		}
	}
}

func TestSplit_OverlapSane(t *testing.T) {
	body := strings.Repeat("word ", 300)
	got := Split(body, Options{TargetChars: 200, MaxChars: 250, Overlap: 50})
	if len(got) < 3 {
		t.Fatalf("expected at least 3 chunks")
	}
	// Adjacent chunks should share at least one common token (overlap > 0).
	first := strings.Fields(got[0].Text)
	second := strings.Fields(got[1].Text)
	if len(first) == 0 || len(second) == 0 {
		t.Fatalf("empty fields")
	}
	// The trailing word of chunk 0 should appear in chunk 1.
	tail := first[len(first)-1]
	found := false
	for _, w := range second {
		if w == tail {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("expected overlap, last word of chunk0 %q not in chunk1", tail)
	}
}
