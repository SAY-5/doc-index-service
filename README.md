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

50,000 synthetic docs, 1000 queries, hash embedder, M-series Mac.
Numbers from `bench/results/bench-20260507-071619.json`.

```
# doc-index-service bench — 50000 docs, 1000 queries
# 2026-05-07 07:16:19Z, darwin/arm64, embedder=hash-bag-384
## index throughput
docs/sec   : 133.3
chunks/sec : 330.6
## query latency (ms)
mode      p50    p95    p99    p999   max
keyword   64.7   119.3  205.9  342.0  373.0
vector    2.8    6.8    10.7   25.3   29.3
hybrid    72.0   166.7  254.4  527.4  1033.9
```

A few notes:

* The hash embedder is *much* faster than MiniLM, so the vector path
  numbers above are dominated by Postgres + HNSW, not the embedder. Run
  with `EMBEDDER_KIND=hf` to see the model-bound figure.
* Keyword latency is high because `ts_rank_cd` is computed inside the
  ORDER BY without a precomputed score column. A materialised
  `chunk_score` would close most of that gap; see
  `ARCHITECTURE.md` for why it isn't done here yet.
* Concurrent indexing peaks around 8 workers on this machine; the
  embedder is the bottleneck on real hardware (it isn't here, but the
  HTTP round-trip still costs).

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

## What this is *not*

* Not an LLM. There is no generation, no answer synthesis, no chat path.
* No learned reranker. The fusion is purely RRF; a cross-encoder rerank
  step is an obvious next move but is out of scope.
* No live web crawl. Documents arrive only through `POST /v1/index`.
* No auth. Single-tenant, trusted clients only. Adding API keys is
  cheap; multi-tenancy at the schema level is not.
* No streaming results. Responses are JSON arrays.
* No GraphQL. REST only.
* No materialised view of `ts_rank_cd` results. See bench notes above.

## Licence

MIT — see `LICENSE`.
