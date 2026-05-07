// Package seed produces deterministic synthetic documents for tests and
// benchmarks.
//
// The generator is deliberately small: a fixed pool of topics, a fixed
// pool of templates, and a single seeded *rand.Rand. Two runs with the
// same seed produce byte-identical output, which keeps benchmarks
// reproducible across machines without needing to commit a corpus.
package seed

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math/rand"
	"strings"
)

// Doc is one synthetic document.
type Doc struct {
	Source string
	Title  string
	Body   string
	Hash   string
}

// Generator emits Doc values from a seeded source. Not safe for
// concurrent use — wrap multiple Generators if you need parallelism.
type Generator struct {
	r *rand.Rand
}

// NewGenerator returns a Generator with the given seed. Use 42 for the
// "official" benchmark corpus.
func NewGenerator(seed int64) *Generator {
	return &Generator{r: rand.New(rand.NewSource(seed))}
}

// Topic groups vocabulary so similar docs cluster in embedding space.
type topic struct {
	name     string
	subjects []string
	verbs    []string
	objects  []string
	tail     string
}

var topics = []topic{
	{
		name:     "databases",
		subjects: []string{"the query planner", "an index scan", "the optimizer", "a sequential scan", "the buffer cache"},
		verbs:    []string{"reorders", "uses", "evaluates", "rewrites", "estimates"},
		objects:  []string{"a hash join", "the predicate", "the cost model", "the row count", "the access path"},
		tail:     "before producing the final plan.",
	},
	{
		name:     "distributed",
		subjects: []string{"each replica", "the coordinator", "a follower", "the leader", "the gossip layer"},
		verbs:    []string{"applies", "fences", "replicates", "validates", "commits"},
		objects:  []string{"a log entry", "the heartbeat", "the snapshot", "the configuration change", "the term boundary"},
		tail:     "before acknowledging the write.",
	},
	{
		name:     "compilers",
		subjects: []string{"the front end", "the type checker", "the lowering pass", "the register allocator", "the backend"},
		verbs:    []string{"emits", "rewrites", "infers", "specialises", "spills"},
		objects:  []string{"the IR", "a phi node", "the call graph", "the control flow", "the live range"},
		tail:     "before machine code is selected.",
	},
	{
		name:     "networking",
		subjects: []string{"the connection", "an idle stream", "the receiver", "the sender", "the congestion controller"},
		verbs:    []string{"acknowledges", "retransmits", "shrinks", "probes", "smooths"},
		objects:  []string{"the window", "the round-trip estimate", "a lost segment", "the pacing rate", "the RTO"},
		tail:     "while the loss event is in flight.",
	},
	{
		name:     "machine-learning",
		subjects: []string{"the encoder", "a residual block", "the attention head", "the optimiser", "the scheduler"},
		verbs:    []string{"projects", "normalises", "scales", "blends", "warms"},
		objects:  []string{"the gradient", "the activation", "a learned bias", "the learning rate", "the dropout mask"},
		tail:     "during the forward pass.",
	},
	{
		name:     "operating-systems",
		subjects: []string{"the kernel", "a user thread", "the scheduler", "the page allocator", "the TLB"},
		verbs:    []string{"flushes", "preempts", "reclaims", "evicts", "promotes"},
		objects:  []string{"the working set", "a dirty page", "the run queue", "the inode cache", "the wait list"},
		tail:     "before returning to userspace.",
	},
}

// Next returns a single deterministic Doc. The Body has between minChars
// and maxChars runes (rough; we cap on whole sentences).
func (g *Generator) Next(i int, minChars, maxChars int) Doc {
	t := topics[g.r.Intn(len(topics))]
	span := maxChars - minChars + 1
	if span < 1 {
		span = 1
	}
	target := minChars + g.r.Intn(span)

	var b strings.Builder
	for b.Len() < target {
		b.WriteString(t.subjects[g.r.Intn(len(t.subjects))])
		b.WriteString(" ")
		b.WriteString(t.verbs[g.r.Intn(len(t.verbs))])
		b.WriteString(" ")
		b.WriteString(t.objects[g.r.Intn(len(t.objects))])
		b.WriteString(" ")
		b.WriteString(t.tail)
		b.WriteString(" ")
	}
	body := strings.TrimSpace(b.String())
	title := fmt.Sprintf("%s note %d", t.name, i)
	source := fmt.Sprintf("synthetic://%s/%d", t.name, i)
	sum := sha256.Sum256([]byte(body))
	return Doc{
		Source: source,
		Title:  title,
		Body:   body,
		Hash:   hex.EncodeToString(sum[:]),
	}
}

// QueryWorkload builds a mixed pool of queries against the topic vocab.
// kinds: keyword (single rare-ish term), multi (two terms ANDed),
// semantic (paraphrased subject+object so vector retrieval is favoured).
func (g *Generator) QueryWorkload(n int) []string {
	out := make([]string, 0, n)
	for i := 0; i < n; i++ {
		t := topics[g.r.Intn(len(topics))]
		switch g.r.Intn(3) {
		case 0:
			out = append(out, t.objects[g.r.Intn(len(t.objects))])
		case 1:
			a := t.subjects[g.r.Intn(len(t.subjects))]
			b := t.objects[g.r.Intn(len(t.objects))]
			out = append(out, a+" "+b)
		default:
			a := t.verbs[g.r.Intn(len(t.verbs))]
			b := t.objects[g.r.Intn(len(t.objects))]
			out = append(out, "how does "+a+" affect "+b)
		}
	}
	return out
}
