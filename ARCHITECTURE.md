# Architecture

This document explains the design choices behind the doc-index-service.
The README covers what the service does and how to run it; this file
covers *why* the boundaries are where they are.

## Why hybrid retrieval

A single retriever has a known failure mode:

* **BM25 alone** cannot find documents that don't share surface terms
  with the query. A search for "vector index resize" will miss a doc
  that talks only about "growing an HNSW graph at runtime".
* **Vector alone** scores by topical similarity, so a query for the
  exact phrase "ON CONFLICT DO NOTHING" will retrieve the top-K
  topically relevant chunks rather than the chunk that contains those
  five tokens. Pretrained embedders cannot reliably distinguish
  rare-term overlap from paraphrase.

Running both retrievers and fusing the rankings recovers the strengths
of each. There is no claim that this is novel — it is the standard
approach in IR literature (Cormack, Clarke, Büttcher 2009 and
descendants).

## Why RRF and not a weighted score sum

Reciprocal rank fusion assigns each document a score
`sum_over_lists(1 / (k + rank))` and sorts descending. The properties
that matter here:

1. **Scale invariance.** BM25 scores live on `[0, ~10]` and depend on
   document length and corpus statistics; cosine similarity lives on
   `[-1, 1]` (or `[0, 1]` for a normalised model). A weighted sum
   `α · bm25 + β · cosine` requires either a fixed normalisation
   (which breaks the moment a query is unusually short) or a
   per-query rescaling step (which is brittle). RRF only sees ranks.
2. **Graceful degradation.** When one retriever returns a noisy list,
   weighted score fusion can amplify the noise; RRF caps each list's
   per-document contribution at `1 / (k + 1)` for the top hit.
3. **No tunable knob.** The literature consensus is `k = 60`, applied
   here. If we needed per-corpus tuning we would have to track that
   parameter; we don't.

The cost of RRF is throwing away within-list score magnitude. For the
top-K reranking step the magnitude is largely redundant with the rank,
so we accept the loss. We *do* preserve the raw BM25 and cosine scores
in the response envelope as `signals.bm25` / `signals.vector` so a
caller who wants to second-guess the fusion can.

## The Go ↔ Python split

The embedding sidecar lives in Python because:

1. **Pretrained model availability.** The Hugging Face `transformers`
   and `sentence-transformers` ecosystem ships ready-to-use weights
   plus matching tokenizers behind a stable Python API.
2. **Tokenizer parity.** Reproducing a SentencePiece or BPE tokenizer
   in Go is a project of its own. ONNX Runtime in Go can execute the
   model graph, but you still need a tokenizer; mismatched tokenisation
   is an extremely subtle source of retrieval-quality bugs.
3. **Indexing is rate-limited by the embedder, not the HTTP boundary.**
   On the machine that produced the bench numbers in the README, the
   round-trip cost of `POST /embed` is negligible compared to the
   per-batch model forward pass with a real model.

The trade-off is that the deployment now has two language runtimes.
The interface between them is intentionally tiny — one `POST /embed`
call with a fixed JSON shape — so the Go side can be tested against a
fake (`fakeEmbedder` in the API tests) and the Python side can be
tested in isolation against `httpx.AsyncClient` over an in-process
ASGI app.

## Embedder choice and the swap path

The default embedder in CI and `docker-compose.yml` is `HashEmbedder`,
a deterministic bag-of-token-hashes function. It is *not* semantic;
its only job is to produce 384-dimensional vectors quickly enough that
unit tests don't need to download a 90 MB model. The unit tests assert
exactly what HashEmbedder is supposed to provide: dimension, shape,
determinism, and a weak signal that overlapping-token sentences score
above disjoint-token sentences.

The production path is `EMBEDDER_KIND=hf`, which loads
`sentence-transformers/all-MiniLM-L6-v2` (~22 MB, 384-d, normalised
cosine output). The `Embedder` Protocol in `app/embedder.py` makes the
swap one environment variable; everything downstream sees identically
shaped vectors.

We pin the embedder dimension at 384 because the Postgres column type
(`vector(384)`) is dimensionally typed and changing it requires a
migration. The sidecar enforces this on startup: if the model reports
a different dimension it refuses to serve.

## pgvector: HNSW vs IVFFlat

`pgvector` ships two index types:

| Property            | HNSW                              | IVFFlat                                          |
| ------------------- | --------------------------------- | ------------------------------------------------ |
| Build time          | Slow                              | Fast (centroids first)                           |
| Recall at speed     | Higher                            | Lower at the same `probes`                       |
| Memory              | Higher (graph stored)             | Lower (lists)                                    |
| Tuning              | `m`, `ef_construction`, `ef`      | `lists`, `probes`                                |
| Index updates       | Lazy, in-place                    | Lists do not split; rebuild as the corpus grows  |

For a 50 K – 1 M chunk corpus on a single node where queries are
latency-sensitive and writes are not bursty, HNSW is the right pick:
build cost is paid once at ingest, recall stays high, and there's no
"rebuild to compensate for centroid drift" maintenance task.

The migration uses `m = 16, ef_construction = 64`, which are the
pgvector defaults. The query-time `ef` parameter (a session GUC,
`hnsw.ef_search`) is left at its default of 40; raising it improves
recall at the cost of latency, and is the right knob to reach for if a
specific query seems to miss obvious neighbours.

## tsvector storage layout

`doc_chunks.tsv` is `GENERATED ALWAYS AS (to_tsvector('english',
text)) STORED`. The `STORED` is deliberate: query time is
latency-sensitive, and storing the parsed lexemes avoids a per-row
`to_tsvector` call inside the GIN scan. The cost is disk space and a
small per-write overhead; both are acceptable for the access pattern
(many reads per write).

The query path uses `websearch_to_tsquery('english', $1)` rather than
`plainto_tsquery` because it accepts the syntax users already know
from Google-style search (`foo OR bar`, `"exact phrase"`, `-excluded`)
without crashing on missing operators.

`ts_rank_cd` is used over `ts_rank` because it accounts for the
density of matches (cover density) rather than just the count, which
matches the long-tail corpus assumption better.

## What's deliberately not here

* **A learned cross-encoder reranker.** Top-K rerank with a
  cross-encoder typically lifts precision by several points, but it
  needs another model service, another GPU dependency, and another
  set of failure modes. The fusion step is hot-swappable so this can
  be added later without changing the API contract.
* **Per-tenant isolation.** All rows are pooled. Adding a `tenant_id`
  column and a row-level security policy is straightforward; the
  reason it isn't here is that the project scope is single-tenant.
* **Streaming responses.** The `/v1/query` response is a single JSON
  body. Switching to NDJSON or SSE would be a change to the handler
  layer; the search engine already returns results lazily-mergeable.
* **A score-cache.** Repeated queries pay the full search cost. The
  obvious next step is a Redis-backed LRU keyed on `(q, mode, k)` —
  it is mentioned in the project spec as deferred for v1.
* **A materialised `ts_rank_cd` projection.** The keyword-path latency
  in the bench numbers is dominated by `ts_rank_cd` recomputation
  inside the `ORDER BY`. A precomputed `chunk_score` column updated
  by a trigger would shave most of the cost, at the price of a
  per-update write. The trade-off is corpus-dependent; for a write-
  heavy ingest pattern the recompute is the cheaper choice.
