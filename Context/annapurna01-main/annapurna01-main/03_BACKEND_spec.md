# 03 — BACKEND SPEC (v3 FINAL, Go 1.23+)

Read `01_MASTER_spec.md` (D1–D30), `08_API_CONTRACT.md`, `04_DATABASE_spec.md`, `05_REDIS_spec.md` first. Implement exactly those; never rename endpoints or fields. All business rules are identical to v2 — only the language changed.

## 1. Stack
Go 1.23+, **chi** router, **oapi-codegen strict server** generated from `docs/api/openapi.json` (handlers implement the generated interface — compile-time contract), pgx v5 pool + **sqlc** typed queries, **golang-migrate** (SQL lives in `database/migrations`, D14), golang-jwt/v5, golang.org/x/crypto/bcrypt, go-playground/validator/v10, **slog** (JSON, request_id), prometheus/client_golang, **maroto/v2** (ESG PDF), **Asynq** (jobs/cron), **sony/gobreaker** (sidecar), Firebase Admin SDK for Go (FCM v1), go-redis/v9, google/uuid, testcontainers-go + testify + gomock. Verify versions when pinning.

## 2. Application skeleton
`cmd/server`: chi router; middleware order: request_id → recoverer(slog + 500 INTERNAL) → auth(JWT) → rbac → app_version → rate_limit → idempotency → handler. CORS only from `ALLOWED_ORIGINS` (docs; native apps need none). Lifespan: ping Postgres + Redis, warm mlserving (dummy predict), log versions. `/health` (process), `/ready` (DB required; redis, mlserving, push reported), `/metrics` (internal only).
Errors: `AppError{Code, Message, Status}` → `{"detail","code","message"}`. Validation → 422 `VALIDATION_ERROR`; panics → 500 `INTERNAL` (no stack to client). Codes: `SURPLUS_EXPIRED, INVALID_STATE_TRANSITION, SAFETY_REJECTED_CANNOT_APPROVE, IMAGE_TOO_LARGE, IMAGE_UNSUPPORTED, IMAGE_UNREADABLE, CV_UNAVAILABLE, ML_UNAVAILABLE, NO_ELIGIBLE_RECIPIENT, MATCH_NOT_FOUND, QR_CHAIN_BROKEN, IDEMPOTENCY_KEY_REUSED, FORBIDDEN_ROLE, RATE_LIMITED, REFRESH_TOKEN_REUSED, APP_VERSION_UNSUPPORTED, SYNC_ITEM_REJECTED, SYNC_OUT_OF_ORDER, ROUTE_INFEASIBLE`.

## 3. Security
Access JWT HS256 (configurable), 60-min expiry, `jti`; **refresh tokens (D22):** 256-bit random, sha256-hashed in `refresh_tokens` with family_id, 30-day expiry, rotate on use; reuse of a rotated token → revoke the family, 401 `REFRESH_TOKEN_REUSED`; logout denylists jti (Redis) + deletes token row; `/auth/logout-all` revokes all families. bcrypt cost 12. `requireRole(...)` middleware; row scoping: KITCHEN only its `kitchen_id`, NGO only its matches, SYSTEM limited to `/sensors/readings`. `X-App-Version` < `APP_MIN_VERSION` → 426 `APP_VERSION_UNSUPPORTED` with store-link payload (skips `/app/config`, `/auth/*`). Upload hardening: magic-byte whitelist (JPG/PNG), ≤ MAX_UPLOAD_MB, stdlib decode→re-encode (drops EXIF/scripts), UUID filename, path-confined to `storage/uploads`. No secrets in logs. Audit-log every approve/override/admin action.

## 4. Services (business logic — ported 1:1 from v2)
**prediction**: validate; if history missing fetch last 28 same-meal consumptions/surpluses; `mlclient.PredictDemand`; on failure statistical fallback (same-weekday median ×1.03, `model_version="fallback-stat"`); log to `prediction_log`; cache 60 s.
**planning**: same with overrides; never persists.
**feedback**: store actuals → `production.consumed_qty`, `outcome_feedback`.
**surplus**: table-driven state machine: allowed = {PENDING_SAFETY:{AVAILABLE,HOLD,DIVERTED,EXPIRED}, HOLD:{AVAILABLE,DIVERTED,EXPIRED}, AVAILABLE:{MATCHED,DIVERTED,EXPIRED}, MATCHED:{IN_TRANSIT,AVAILABLE(if all declined),EXPIRED}, IN_TRANSIT:{DELIVERED}}; else 409. Create writes QR `CREATED` (actor=user). Expiry sweep = Asynq cron 1 min under `lock:expiry_sweep`.
**quality**: validate upload → sanitize/save (internal/files) → `mlclient.CvPredict` (timeout 8 s; on failure `visual.status=RISK, reason="CV unavailable, manual inspection"`, never GOOD) → sensor summary (danger minutes from Redis else recompute from DB) → `mlclient.AssessSafety` → store `quality_checks` → `requires_human_approval=true` always. `device_preview` in request stored for telemetry ONLY (D20). Does NOT change surplus status.
**approve**: human decision; APPROVE only if latest fusion ELIGIBLE, else 409 unless ADMIN `override_reason` (audit-logged). Writes QR `APPROVED` (actor=user, `evidence_hash` of latest quality_check).
**matching**: candidates = active recipients, category accepted, free capacity ≥ min(quantity, 5 kg), window reachable before expiry. Score (weights sum 100): need 20, capacity fit 20, distance 25 (exp decay), window overlap 15, shelf-life slack 10, fairness 10 (penalty if received in last 48 h). Reasons from top-3 components + `score_breakdown`. Persist OFFERED; **on offer → asynqx push task for that NGO's devices**; `respond` → ACCEPTED/DECLINED; first ACCEPTED → MATCHED + QR `MATCHED`.
**routing**: stops from accepted matches → `mlclient.BuildRoute` (sidecar OR-Tools VRPTW: capacity, service 10 min, recipient windows, hard deadline = expiry − 30 min); infeasible → 409 `ROUTE_INFEASIBLE`; persist route+stops; fallback nearest-neighbour haversine (label `solver`).
**qr**: role→event map; sequence validator; hash via `internal/qrchain` (D30, server order); `client_ts` stored, never hashed; `verify` recomputes the chain, returns first broken index; per-batch `SELECT … FOR UPDATE` (pgx tx) to serialize concurrent events.
**sync (D24)**: `POST /sync/batch` — items ordered; per item: dedupe (`sync:dedupe:*` then DB UNIQUE `client_event_id`) → replay the item's normal service path with actor from JWT → `{client_event_id, status ACCEPTED|DUPLICATE|REJECTED, code?, body?}`. QR items replay per-batch FIFO: out-of-sequence → REJECTED `SYNC_OUT_OF_ORDER`. Replaying a whole batch changes nothing. 120 batches/min/user.
**push (D23 v3)**: `pushx.Send(userIDs, event_type, title, body, data)` → asynqx task → FCM v1 (chunks ≤500; honors 429 Retry-After) → error codes (UNREGISTERED/INVALID_CREDENTIALS) prune `device_tokens`. Never blocks requests: failures → metric + log only. Push disabled → no-op.
**appconfig (D26)**: `GET /app/config` → {min_supported_version, force_update, feature_flags, model_bundle{server{onnx url,sha256}, device{tflite url,sha256,size_bytes}, thresholds, preprocess}, display{carbon_factor_source, danger_zone_band_c}}; from env + `MODEL_BUNDLE_DIR/manifest.json`; changes pub/sub `chan:system` refresh workers.
**sensor**: validate ranges (temp −30..120, humidity 0..100, energy ≥ 0), reject stale > 24 h; ingest per 05_REDIS (XADD + latest + fan-out); threshold rules → alerts (dedupe 10 min); batch attach in cold store.
**alert**: create, publish SSE + push (WARN/CRITICAL), list/ack.
**processing**: `internal/processing` canonical formulas (D9 v3), `efficiency_score` D3, UPSERT kitchen/date, `mlclient.DetectAnomalies` → anomalies + alerts.
**analytics**: impact from SQL views; waste_avoided_kg = Σ(delivered) + Σ(diverted) (documented in `docs/evaluation/metrics_definitions.md`); carbon = kg × factor with source; meals_equivalent = kg / 0.4 (configurable). Cache 30 s.
**diversion**: records stream/quantity; QR terminal event optional.
**report**: ESG summary PDF (maroto) / CSV; > 2 s → Asynq task → 202 `{job_id}`; job status in Redis.

## 5. mlserving client (D28)
Per-endpoint timeouts: predict 2 s, cv 8 s, safety 2 s, route 10 s, anomalies 5 s. One gobreaker per endpoint (fail fast after 5 consecutive failures, 30 s half-open). `FEATURE_MOCK_ML` / `FEATURE_MOCK_CV` / `FEATURE_MOCK_ROUTING` → deterministic contract-shaped fixtures without calling the sidecar (keeps the app unblocked from day 1). Sidecar auth: internal network + `MLSERVING_TOKEN` header. Map sidecar errors → `ML_UNAVAILABLE` / `CV_UNAVAILABLE` (never leak stack traces). See 07 §15 for the sidecar contract.

## 6. Workers (Asynq, Redis DB1)
`expiry_sweep` (cron 1 min, Redis lock), `offer_expiry` (30 min unanswered → EXPIRED + re-match), `sensor_bulk` (XREADGROUP → pgx COPY every 1 s/500 rows), `jobs_reports` (ESG PDF), `jobs_push` (FCM sends + receipt pruning), `drift_check` (daily, calls sidecar /v1/drift), `stale_sensor_check` (5 min). Graceful shutdown: drain in-flight tasks ≤ 10 s.

## 7. Observability
Metrics: `http_request_duration_seconds`, `sidecar_request_duration_seconds{endpoint}`, `cv_inference_seconds`, `ml_predict_seconds`, `safety_decisions_total{status}`, `qr_chain_verify_failures_total`, `sync_batch_size`, `sync_rejected_total`, `push_sent_total`, `push_failures_total`, `redis_degraded`. slog JSON with request_id, user_id, batch_id, app_version. pprof disabled in prod.

## 8. Testing (targets)
Coverage ≥ 80 % on services. Must-have: invalid transition → 409; NGO cannot read other NGO's matches; `.exe` renamed `.jpg` → 415; 9 MB → 413; sidecar returning 500 → contract-valid RISK response; approve blocked on fusion REJECTED; QR tamper → verify false; processing zero/negative → 422; idempotent POST replay; refresh rotation + reuse-revocation family; sync batch replayed twice → zero duplicates; QR sync out-of-order → SYNC_OUT_OF_ORDER; `/app/config` below-min version → 426; processing parity fixture vs `integration/processing` (Python) — identical outputs; Redis down suite (in-memory fallbacks); mlserving down suite (mock fallbacks); e2e with mocks < 10 s.

## 9. Build order
skeleton+config+openapi gen → auth(+refresh)/rbac/errors → app/config+bundle serving → kitchen+prediction(mock) → waste+surplus+state machine → quality(mock CV) → approve+QR chain → matching+push(FCM) → sync → routing(haversine mock) → sensors+alerts+redis → processing → analytics+reports → Asynq crons → swap mocks for real mlserving → hardening+tests → load test (k6, 50 users, p95 < 500 ms non-CV).

## 10. Definition of done
All 42 rows in 08_API_CONTRACT live and equal to openapi.json (generated stubs — zero drift); contract test green; no service imports `ml/`, `cv/`, `integration/` (only `mlclient`); runs with Redis stopped AND mlserving stopped AND push disabled; `make e2e` passes on seed data including the offline-sync replay path.