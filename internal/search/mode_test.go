package search

import "testing"

func TestParseMode(t *testing.T) {
	cases := []struct {
		in      string
		want    Mode
		wantErr bool
	}{
		{"", ModeHybrid, false},
		{"hybrid", ModeHybrid, false},
		{"keyword", ModeKeyword, false},
		{"vector", ModeVector, false},
		{"nope", "", true},
	}
	for _, c := range cases {
		got, err := ParseMode(c.in)
		if (err != nil) != c.wantErr {
			t.Fatalf("%q: err=%v wantErr=%v", c.in, err, c.wantErr)
		}
		if !c.wantErr && got != c.want {
			t.Fatalf("%q: got %v want %v", c.in, got, c.want)
		}
	}
}
