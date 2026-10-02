# 03 — BACKEND STRUCTURE (P1, Go 1.23+)

```
backend/
├── go.mod  go.sum  Makefile  .env.example
├── Dockerfile               # multi-stage: golang:1.23-builder → gcr.io/distroless/static
├── cmd/
│   ├── server/main.go       # chi router, middleware, graceful shutdown, /api/v1
│   ├── worker/main.go       # Asynq server + periodic tasks (cron) | redis DB1
│   └── seedcheck/main.go    # generates/validates seed QR chains via internal/qrchain (D30)
├── internal/
│   ├── config/              # env loading (fail fast), all D27 pins
│   ├── httpapi/
│   │   ├── gen/              # oapi-codegen STRICT server stubs — GENERATED, do not edit (CI drift gate)
│   │   ├── handlers/         # one file per resource: auth kitchen prediction planning feedback waste surplus
│   │   │                     # quality matching routing qr sensors alerts processing analytics reports events(sse)
│   │   │                     # admin health sync devices app
│   │   └── middleware/      # request_id, recoverer, slog request logging, auth(jwt), rbac, rate_limit,
│   │                         # idempotency, app_version(426 below min, skips /app/config & /auth/*)
│   ├── auth/                # golang-jwt/v5 mint/verify (jti), bcrypt, refresh rotation + family revoke
│   ├── rbac/                # requireRole(...), QR event→role map, row-scope helpers (kitchen_id/recipient_id)
│   ├── store/               # pgx v5 pool + sqlc-generated queries (store/queries/*.go, sqlc.yaml)
│   │                        # tx helper (retry serialization failures), pagination helper
│   ├── services/            # business logic — one file per domain, mirrors v2 rules 1:1:
│   │                        # surplus(state machine) quality approve matching routing qr sensor alert
│   │                        # processing analytics diversion report(sync+async) push appconfig sync admin audit
│   ├── qrchain/             # canonical hash: sha256(prev||batch_id||event||actor_id||server_ts||evidence_hash) (D30)
│   ├── mlclient/            # mlserving HTTP client: per-endpoint timeout, sony/gobreaker circuit breaker,
│   │                        # FEATURE_MOCK_* fallbacks (contract-shaped fixtures), error → AppError mapping
│   ├── pushx/               # Firebase Admin SDK (FCM v1): send, token registry (device_tokens), prune on
│   │                        # invalid-credential errors; PUSH disabled → no-op
│   ├── redisx/              # go-redis v9: cache-aside, sliding-window limiter (Lua), idempotency store,
│   │                        # SET NX locks, pub/sub (SSE fan-out + app-config refresh), sync dedupe,
│   │                        # danger-zone clock (Lua pipeline), graceful degrade (in-memory fallbacks)
│   ├── asynqx/              # task type names + handlers: reports, push, expiry_sweep, sensor_bulk,
│   │                        # offer_expiry, drift_check(calls sidecar) | enqueuer helper
│   ├── processing/          # canonical KPI formulas (D9 v3) + parity fixture test vs integration/processing
│   ├── files/               # upload validation (magic bytes JPG/PNG), size, stdlib re-encode (drops EXIF),
│   │                        # UUID names, storage/uploads confinement; bundle static serving (sha256 manifest)
│   ├── metrics/             # prometheus: http duration, cv/ml/sidecar latency, safety_decisions_total,
│   │                        # qr_chain_verify_failures, sync_* , push_*, redis_degraded
│   └── sse/                 # event stream writer (chi), Last-Event-ID resume via events:recent:{kitchen}
├── storage/uploads/         # gitignored, UUID filenames
├── static/bundles/cv/        # versioned device bundles written by cv/export/bundle.py (`make bundle`)
└── tests embedded as *_test.go:
    unit/  (auth, rbac, qrchain, state machine, matching weights, sync replay, processing parity)
    api/   one file per handler (401/403/404/409/413/415/422/426/429)
    contract/ test_openapi_live.py equivalent: live responses ↔ openapi.json
    e2e/    full flow incl. offline-sync replay + push stub + mlserving stub
```
Layer rule: handlers → services → store; services call mlclient/pushx/redisx/asynqx, never raw SQL outside store, never other domains directly. `internal/httpapi/gen` is generated — contract-first, CI fails if `go generate ./...` produces a diff.