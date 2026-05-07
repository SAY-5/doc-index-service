# doc-index-service developer entry points.
#
# These targets stay deliberately thin: each one shells out to the tool
# that actually does the work so you can read the underlying command and
# reproduce it without make.

SHELL := /usr/bin/env bash
.SHELLFLAGS := -eu -o pipefail -c

GO        ?= go
PYTHON    ?= python3
SIDECAR   := services/embed-sidecar
VENV      := $(SIDECAR)/.venv
DSN       ?= postgres://postgres:postgres@localhost:5432/docindex?sslmode=disable
EMBED_URL ?= http://localhost:8088

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | awk -F':.*?##' '{printf "%-20s %s\n", $$1, $$2}'

.PHONY: tidy
tidy: ## Tidy go.mod
	$(GO) mod tidy

.PHONY: build
build: ## Build all Go binaries
	$(GO) build ./...

.PHONY: test
test: test-go test-py ## Run all tests

.PHONY: test-go
test-go: ## Run Go unit tests
	$(GO) test ./...

.PHONY: test-go-cover
test-go-cover: ## Run Go tests with coverage
	$(GO) test -race -covermode=atomic -coverprofile=coverage.out ./internal/... ./pkg/...
	$(GO) tool cover -func=coverage.out | tail -n 1

.PHONY: test-py
test-py: $(VENV)/.installed ## Run Python sidecar tests
	cd $(SIDECAR) && EMBEDDER_KIND=hash .venv/bin/python -m pytest tests/

.PHONY: lint
lint: lint-go lint-py ## Run all linters

.PHONY: lint-go
lint-go: ## Run golangci-lint
	@if command -v golangci-lint >/dev/null 2>&1; then \
		golangci-lint run ./...; \
	elif [ -x "$$HOME/go/bin/golangci-lint" ]; then \
		"$$HOME/go/bin/golangci-lint" run ./...; \
	else \
		echo "golangci-lint not installed; skipping"; \
	fi

.PHONY: lint-py
lint-py: $(VENV)/.installed ## Run ruff + black --check on the sidecar
	cd $(SIDECAR) && .venv/bin/ruff check .
	cd $(SIDECAR) && .venv/bin/black --check .

.PHONY: typecheck
typecheck: typecheck-py ## Run all type checks

.PHONY: typecheck-py
typecheck-py: $(VENV)/.installed ## Run mypy on the sidecar
	cd $(SIDECAR) && .venv/bin/mypy app

.PHONY: migrate
migrate: ## Apply SQL migrations against $(DSN)
	@if ! command -v psql >/dev/null 2>&1; then \
		echo "psql is required"; exit 1; \
	fi
	@for f in migrations/*.up.sql; do \
		echo ">> $$f"; \
		psql "$(DSN)" -v ON_ERROR_STOP=1 -f "$$f"; \
	done

.PHONY: seed
seed: ## Insert 1k synthetic docs through the indexer
	$(GO) run ./cmd/indexer -n 1000 -workers 4 -seed 42

.PHONY: bench
bench: ## Run the benchmark harness (200k docs, 1k queries by default)
	$(GO) run ./bench -docs 200000 -queries 1000

.PHONY: bench-smoke
bench-smoke: ## Tiny bench, used by CI
	$(GO) run ./bench -docs 1000 -queries 100 -smoke

.PHONY: bench-regress
bench-regress: ## Compare two bench JSONs; fails when any metric drifts > $(TOL)
	@if [ -z "$(BASELINE)" ] || [ -z "$(FRESH)" ]; then \
		echo "usage: make bench-regress BASELINE=path/to/baseline.json FRESH=path/to/fresh.json [TOL=0.30]"; exit 2; \
	fi
	$(GO) run ./cmd/bench-regress -baseline "$(BASELINE)" -fresh "$(FRESH)" -tol $${TOL:-0.30}

.PHONY: embed-up
embed-up: $(VENV)/.installed ## Start the embed sidecar locally on :8088
	cd $(SIDECAR) && EMBEDDER_KIND=hash .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8088

.PHONY: dev
dev: ## Run the Go server locally (requires migrate + embed-up first)
	DATABASE_URL="$(DSN)" EMBED_URL="$(EMBED_URL)" $(GO) run ./cmd/server

.PHONY: up
up: ## Bring the docker compose stack up
	docker compose up --build -d

.PHONY: down
down: ## Tear the compose stack down
	docker compose down -v

.PHONY: clean
clean: ## Remove build artefacts and caches
	$(GO) clean ./...
	rm -f coverage.out
	rm -rf $(SIDECAR)/.venv $(SIDECAR)/.pytest_cache $(SIDECAR)/**/__pycache__

$(VENV)/.installed: $(SIDECAR)/pyproject.toml
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	cd $(SIDECAR) && .venv/bin/pip install -e ".[test,dev]"
	touch $(VENV)/.installed
