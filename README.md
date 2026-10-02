# Annapurna — Food Surplus Redistribution Platform (SIH26234)

Backend for the AI-driven institutional food-surplus platform: surplus
capture, quality/safety verification, NGO matching, logistics routing and an
append-only QR audit chain. Go + PostgreSQL + Redis (+ optional Python ML
sidecar).

## Quick start

```bash
cp .env.example .env          # defaults match docker-compose
docker compose up -d          # postgres on :5433, redis on :6379
make migrate-up               # apply database/migrations
make seed                     # deterministic demo data (idempotent)
make run                      # API on http://localhost:8080
```

Without Docker (any PostgreSQL 14+ and Redis):

```bash
export DATABASE_URL="postgres://user:pass@localhost:5432/foodwaste?sslmode=disable"
export REDIS_URL="redis://localhost:6379/0"
go run ./cmd/migrate up && go run ./cmd/seed && go run ./cmd/server
```

Every command loads `.env` automatically (real environment variables win), so
`.env` is the only place you need to keep configuration.

Demo login: `kitchen@example.com` / `demo123` (also `ngo@`, `logistics@`,
`admin@example.com` — same password).

## Verifying a checkout

```bash
go test ./...            # unit tests (auth, qrchain, uploads)
make vet                 # go vet
./scripts/smoke.sh       # end-to-end HTTP test against a running server
```

Current smoke-test coverage (55 checks): health/readiness, login, rotating
refresh tokens with reuse detection, logout, surplus create/validate, quality
verification (valid image, non-image → 415, missing batch → 422), approval,
the safety-rejected guard, QR timeline/verify/append with idempotent replay,
diversion and its state guards, the SSE stream scoping, and offline-sync
dedupe.

`scripts/smoke.sh` drives the real API: login → create surplus → upload a
photo for the quality check → approve → inspect/verify the QR chain → divert →
offline-sync replay. It prints a PASS/FAIL per check and exits non-zero on
failure. Point it elsewhere with `BASE_URL=http://host:port ./scripts/smoke.sh`.

## Layout

| Path | Purpose |
| --- | --- |
| `cmd/server` | HTTP API (chi router) |
| `cmd/worker` | Asynq worker (background jobs) |
| `cmd/migrate` | Dependency-free migration runner (`up`, `down`, `version`, `force`, `drop`) |
| `cmd/seed` | Deterministic demo dataset (idempotent, rebuilds QR chains) |
| `database/migrations` | SQL migrations (`*.up.sql` / `*.down.sql`) |
| `internal/services` | Business logic (surplus, quality, approve, matching, sync, QR) |
| `internal/qrchain` | Canonical SHA-256 hash chain for QR events |
| `internal/httpapi` | Handlers, middleware, error envelope |
| `internal/auth` | JWT access tokens, rotating refresh tokens |
| `internal/files` | Multipart upload validation + EXIF-stripping re-encode |
| `scripts/smoke.sh` | End-to-end smoke test |

## Key API surface

`POST /api/v1/auth/login|refresh|logout|logout-all`, `GET /auth/me`,
`POST|GET /api/v1/surplus`, `GET /surplus/{id}`,
`POST /surplus/{id}/approve|divert`, `POST /quality/check` (multipart:
`batch_id` + `image`), `POST /sync/batch` (offline outbox replay, ≤50 items),
`GET /qr/{batch_id}`, `GET /qr/{batch_id}/verify`,
`POST /qr/{batch_id}/event`, `GET /events/stream` (SSE), plus `/health`,
`/ready`, `/metrics`.

The frozen contract (enum vocabularies, error codes, per-endpoint roles) lives
in `Context/annapurna01-main/annapurna01-main/08_API_CONTRACT.md`.

## Notes

- `qr_events` is append-only: a database trigger rejects UPDATE/DELETE, and
  each row hashes the previous row (`prev_hash` → `event_hash`), so tampering
  is detectable with `GET /qr/{batch_id}/verify`.
- Offline replays are idempotent twice over: a Redis dedupe fast path and a
  unique index on `client_event_id` (the durable guard). `client_ts` /
  `client_event_id` are provenance only and are never hashed into the chain.
- Feature flags `FEATURE_MOCK_ML`, `FEATURE_MOCK_CV`, `FEATURE_MOCK_ROUTING`
  let the API run without the Python inference sidecar.
