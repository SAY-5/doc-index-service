package search

import (
	"strings"
	"testing"
	"unicode/utf8"
)

func TestSanitizeBM25Query_Table(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want string
	}{
		{"empty", "", ""},
		{"plain", "hello world", "hello world"},
		{"trim_repeat_space", "  hello   world  ", "hello world"},
		{"strip_nul", "hel\x00lo", "hello"},
		{"strip_control", "hi\x01\x02there", "hithere"},
		{"unicode_kept", "café résumé", "café résumé"},
		{"newlines_collapse", "a\nb\tc", "a b c"},
		{"bom_stripped", "\ufeffhello", "hello"},
		{"only_spaces", "   \t\n  ", ""},
		{"replacement_rune_invalid_byte", "abc\xffxyz", "abc xyz"},
	}
	for _, c := range cases {
		got := SanitizeBM25Query(c.in)
		if got != c.want {
			t.Errorf("%s: got %q want %q", c.name, got, c.want)
		}
	}
}

func TestSanitizeBM25Query_NeverContainsNUL(t *testing.T) {
	in := "abc\x00def\x00\x00gh"
	got := SanitizeBM25Query(in)
	if strings.Contains(got, "\x00") {
		t.Fatalf("NUL leaked: %q", got)
	}
}

func TestSanitizeBM25Query_AlwaysValidUTF8(t *testing.T) {
	for _, in := range []string{
		"\xff\xfe\xfd",
		"abc\xc3\x28def",
		"\xed\xa0\x80",
		"plain ascii",
	} {
		got := SanitizeBM25Query(in)
		if !utf8.ValidString(got) {
			t.Fatalf("invalid utf8 from %q: %q", in, got)
		}
	}
}

func TestSanitizeBM25Query_TruncatesToBudget(t *testing.T) {
	in := strings.Repeat("a", MaxQueryRunes+200)
	got := SanitizeBM25Query(in)
	if utf8.RuneCountInString(got) > MaxQueryRunes {
		t.Fatalf("over budget: %d", utf8.RuneCountInString(got))
	}
}

func TestSanitizeBM25Query_ZeroWidthRunes(t *testing.T) {
	cases := []string{
		"a\u200bb", // ZWSP
		"a\u200cb", // ZWNJ
		"a\u200db", // ZWJ
		"a\u00adb", // soft hyphen
		"a\u2060b", // word joiner
		"a\u180eb", // mongolian vowel separator
		"a\ufeffb", // BOM mid-string
		"a\u200eb", // LRM
		"a\u200fb", // RLM
		"a\u202ab", // LRE
		"a\u202eb", // RLO
		"a\u2066b", // LRI
		"a\u2069b", // PDI
	}
	for _, in := range cases {
		got := SanitizeBM25Query(in)
		if got != "ab" {
			t.Errorf("input %q -> %q, want %q", in, got, "ab")
		}
	}
}
