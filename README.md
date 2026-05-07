# doc-index-service

A small document index that combines keyword and dense-vector retrieval
behind a single `/v1/query` endpoint. The HTTP API and bulk indexer are
written in Go; embeddings are produced by a separate Python sidecar; the
storage layer is Postgres 16 with `pgvector` and `tsvector`.

## What this studies

* **Hybrid retrieval design.** Each query runs through two retrievers
  (`ts_rank_cd` over a `tsvector` GIN index, cosine distance over a 384-d
  HNSW index) and the two ranked lists are fused with reciprocal rank
  fusion. RRF is preferred to a weighted score sum because BM25 and
  cosine live on different scales — see `ARCHITECTURE.md`.
* **Go ↔ Python service split.** Embedding lives in Python because the
  ecosystem of pretrained sentence-transformer weights and tokenizers is
  not portable to Go without re-implementing tokenisation by hand. The
  HTTP boundary between the Go services and the embedder is fine for
  indexing throughput; query-path embedding is one round-trip per query.
* **Idempotent indexing.** `POST /v1/index` is keyed by
  `sha256(body)`, so retried writes do not duplicate chunks. The
  conflict path is a single-roundtrip `INSERT ... ON CONFLICT DO
  NOTHING RETURNING id`.
* **Cursor pagination.** `GET /v1/docs` uses opaque base64 cursors that
  carry the `(created_at, id)` pair of the last row returned. The order
  is total, so pagination is stable under concurrent inserts.

## Modules

| Path                         | Role                                                              |
| ---------------------------- | ----------------------------------------------------------------- |
| `cmd/server`                 | HTTP API: `/v1/index`, `/v1/query`, `/v1/docs/{id}`, `/v1/docs`   |
| `cmd/indexer`                | CLI bulk-loader; shares the upsert path with the HTTP server      |
| `internal/api`               | Handlers, error envelope, request-id middleware, opaque cursor    |
| `internal/store`             | `pgxpool` Postgres client and hand-rolled queries                 |
| `internal/search`            | Hybrid search: BM25 + vector candidates, reciprocal rank fusion   |
| `internal/rerank`            | Optional cross-encoder rerank stage; heuristic fallback in CI     |
| `internal/chunker`           | Unicode-safe text chunker with sentence-boundary preference       |
| `internal/seed`              | Deterministic synthetic-doc generator and query workload          |
| `pkg/embed`                  | Go HTTP client for the embedding sidecar                          |
| `services/embed-sidecar`     | FastAPI service: `POST /embed`, `GET /healthz`                    |
| `migrations`                 | `golang-migrate`-format SQL migrations                            |
| `bench`                      | Stand-alone benchmark harness; writes JSON artefacts              |

## Quickstart

```sh
# Prereqs: Docker (or any Postgres 16 with pgvector), Go 1.22, Python 3.12.

make up                # Postgres + embed sidecar + server in compose
make migrate DSN=...   # apply schema (or let CI's bench-smoke do it)
make seed              # 1k synthetic docs through the indexer
make bench             # 50k docs, 1k queries; writes bench/results/*.json
```

For a hermetic stack without Docker:

```sh
# Postgres+pgvector somewhere reachable, then:
EMBEDDER_KIND=hash make embed-up    # in one shell
make dev                            # in another
```

`EMBEDDER_KIND=hash` selects the deterministic bag-of-token-hashes
embedder used in CI; export `EMBEDDER_KIND=hf` to switch to
`sentence-transformers/all-MiniLM-L6-v2` (the path documented in
`ARCHITECTURE.md`).

## Architecture

```
       ┌────────────────────┐        ┌────────────────────┐
       │ cmd/server (Go)    │        │ cmd/indexer (Go)   │
       │ /v1/index /v1/query│        │ bulk loader        │
       └─────┬────────┬─────┘        └─────────┬──────────┘
             │        │                        │
             │        │     POST /embed        │
             │        └────────┐  ┌────────────┘
             │                 ▼  ▼
             │         ┌──────────────────┐
             │         │ services/        │
             │         │ embed-sidecar    │ (FastAPI, Python)
             │         │ HashEmbedder     │
             │         │ MiniLM-L6-v2     │
             │         └──────────────────┘
             ▼
   ┌──────────────────────────────┐
   │ Postgres 16 + pgvector       │
   │  docs (id, hash, body, …)    │
   │  doc_chunks (text,           │
   │    embedding vector(384),    │
   │    tsv tsvector GENERATED)   │
   │  GIN(tsv)  +  HNSW(embedding)│
   └──────────────────────────────┘
```

The hybrid search engine issues one BM25 query and one vector query in
parallel, fuses with RRF (`k = 60`), and returns the top-`k` chunks
with both signal scores attached.

## Tests

* **Go unit tests** — `internal/{api,chunker,search,seed}` and
  `pkg/embed`. Coverage on these packages is gated at 80% in CI; the
  current run sits at ~90%.
* **Python tests** — `services/embed-sidecar/tests`. Cover the
  `HashEmbedder` (determinism, cosine-similarity sanity, idempotency)
  and the FastAPI endpoints (`POST /embed`, `GET /healthz`).
* **bench-smoke** — A 1000-doc / 100-query smoke run against the
  real Postgres + sidecar in CI. Asserts the JSON artefact lands in
  `bench/results/`.
* **Integration tests** — `internal/store` is exercised end-to-end by
  `bench-smoke`. Extending unit-level integration tests via
  `testcontainers-go` is on the deferred list.

## Bench

100,000 synthetic docs, 1000 queries, 16 workers, hash embedder,
M-series Mac. Numbers from `bench/results/bench-20260507-220545.json`.

```
# doc-index-service bench — 100000 docs, 1000 queries
# 2026-05-07 22:05:45Z, darwin/arm64, embedder=hash-bag-384
## index throughput
docs/sec   : 134.4
chunks/sec : 333.0
## query latency (ms)
mode      p50    p95    p99    p999   max
keyword   157.0  287.4  454.2  501.8  521.5
vector    2.8    6.4    11.6   28.3   39.8
hybrid    171.1  311.4  463.9  569.8  570.6
```

The 50 k baseline is still committed at
`bench/results/bench-20260507-071619.json` for back-comparison.

### Why 100 k and not 200 k

`make bench` defaults to 200 k; this machine got bogged down by HNSW
index growth past ~30 k docs (insert rate dropped from ~150 docs/sec to
~40 docs/sec by 50 k and would not have completed 200 k in the run
budget). The numbers above are the highest scale that finished in a
single uninterrupted session. Two follow-ups would unlock the 200 k
ceiling on the same hardware:

* drop the HNSW index before bulk load and rebuild it after, the
  pgvector-recommended pattern for cold loads, and
* shard `doc_chunks` by hash so HNSW build cost is amortised across
  smaller partitions.

Both are deferred — they're real engineering, not knob-twiddling.

### Regression gate

`cmd/bench-regress` diffs two JSON artefacts and fails on >30 % drift
per metric. CI runs a fresh 5 k bench on every push and compares it
against the committed `bench/results/baseline-5k.json`; the gate is
per-metric so a single mode regressing isn't masked by a faster one.

```sh
make bench-regress \
    BASELINE=bench/results/baseline-5k.json \
    FRESH=bench/results/bench-YYYYMMDD-HHMMSS.json
```

A few notes:

* The hash embedder is *much* faster than MiniLM, so the vector path
  numbers above are dominated by Postgres + HNSW, not the embedder. Run
  with `EMBEDDER_KIND=hf` to see the model-bound figure.
* Keyword latency is high because `ts_rank_cd` is computed inside the
  ORDER BY without a precomputed score column, and it scales roughly
  linearly with corpus size — that's the ~2.4× growth from 50 k to
  100 k visible above.
* Concurrent indexing peaks around 16 workers on this machine; on real
  hardware the embedder is the bottleneck.

Re-run the benchmark with `make bench`. Pass `-docs` and `-queries` to
shrink it; the JSON schema lives in `bench/README.md`.

## Layout

```
.
├── bench/                      # bench harness + results/
├── cmd/{server,indexer}/       # Go binaries
├── internal/                   # api, chunker, search, seed, store
├── migrations/                 # golang-migrate SQL
├── pkg/embed/                  # Go HTTP client for the sidecar
├── services/embed-sidecar/     # FastAPI Python service
├── Dockerfile.server           # multi-target Go image (server / indexer)
├── docker-compose.yml          # local stack
├── Makefile                    # entry points
└── .github/workflows/ci.yml    # lint / typecheck / test / smoke / build
```

## Reranking

`/v1/query` accepts an optional `mode=hybrid+rerank` flag. The hybrid
fusion stage runs as before, but instead of returning the top-k fused
candidates directly, the engine forwards the top-20 to a Reranker:

* `HeuristicReranker` (the default) — pure-Go token-overlap with a
  length penalty. No model download, runs in CI, sub-millisecond cost.
* `CrossEncoderReranker` — calls the sidecar's `POST /rerank`, which
  loads `cross-encoder/ms-marco-MiniLM-L-6-v2` (≈80 MB) and scores
  `(query, passage)` pairs jointly.

Selection happens once at server start via `RERANKER_KIND`:

```sh
RERANKER_KIND=heuristic       # default; in-process
RERANKER_KIND=cross-encoder   # routes to sidecar /rerank
RERANKER_KIND=off             # rerank-flagged queries fall back to plain hybrid
```

Trade-off: the cross-encoder adds ~50-200 ms per query in exchange for
a measurable top-1 precision lift on hybrid queries where RRF is
"directionally right but ranks the wrong chunk first." Use the heuristic
when you can't afford that latency or you're in CI.

The rerank flag is back-compat: `mode=hybrid` (the default) still does
plain RRF and is unchanged.

## What this is *not*

* Not an LLM. There is no generation, no answer synthesis, no chat path.
* No live web crawl. Documents arrive only through `POST /v1/index`.
* No auth. Single-tenant, trusted clients only. Adding API keys is
  cheap; multi-tenancy at the schema level is not.
* No streaming results. Responses are JSON arrays.
* No GraphQL. REST only.
* No materialised view of `ts_rank_cd` results. See bench notes above.

## Licence

MIT — see `LICENSE`.
