# ─── Food Waste Redistribution Platform – Makefile ────────────────────────────
SHELL       := /bin/bash
GO          := go
BINARY      := bin/server
MIGRATE_DIR := migrations
DB_URL      ?= $(DATABASE_URL)

.PHONY: up down seed test lint gen gen-drift eval bundle demo reset report

# ── Docker Compose lifecycle ───────────────────────────────────────────────────
up:
	docker compose up -d
	@echo "Waiting for postgres..."
	@until docker compose exec postgres pg_isready -q; do sleep 1; done
	$(MAKE) _migrate-up

down:
	docker compose down -v --remove-orphans

# ── Database migrations ────────────────────────────────────────────────────────
_migrate-up:
	$(GO) run -tags 'postgres' \
		github.com/golang-migrate/migrate/v4/cmd/migrate \
		-path=$(MIGRATE_DIR) -database="$(DB_URL)" up

_migrate-down:
	$(GO) run -tags 'postgres' \
		github.com/golang-migrate/migrate/v4/cmd/migrate \
		-path=$(MIGRATE_DIR) -database="$(DB_URL)" down 1

seed:
	$(GO) run ./cmd/seed/...

# ── Tests ──────────────────────────────────────────────────────────────────────
test:
	$(GO) test -race -coverprofile=coverage.out -covermode=atomic ./...
	$(GO) tool cover -html=coverage.out -o coverage.html
	@echo "Coverage report: coverage.html"

# ── Lint ───────────────────────────────────────────────────────────────────────
lint:
	@which golangci-lint > /dev/null 2>&1 || \
		(curl -sSfL https://raw.githubusercontent.com/golangci/golangci-lint/master/install.sh | sh -s -- -b $$(go env GOPATH)/bin)
	golangci-lint run ./...

# ── Code generation ────────────────────────────────────────────────────────────
gen:
	$(GO) generate ./...

# Ensures go generate produces no diff (CI gate)
gen-drift:
	$(MAKE) gen
	@if ! git diff --exit-code; then \
		echo "ERROR: go generate produced uncommitted changes"; \
		exit 1; \
	fi

# ── ML model evaluation ───────────────────────────────────────────────────────
eval:
	$(GO) run ./cmd/eval/... --bundle=$(MODEL_BUNDLE_DIR)

# ── ML model bundle packaging ─────────────────────────────────────────────────
bundle:
	$(GO) run ./cmd/bundle/... --out=$(MODEL_BUNDLE_DIR)

# ── Demo data + full stack ────────────────────────────────────────────────────
demo: up seed
	@echo "Demo stack ready. API at http://localhost:$(PORT)"

# ── Full reset (nuke DB + redis, re-apply migrations, seed) ──────────────────
reset: down up seed
	@echo "Reset complete."

# ── Generate PDF / metrics report ─────────────────────────────────────────────
report:
	$(GO) run ./cmd/report/... --output=report.pdf
	@echo "Report written to report.pdf"
