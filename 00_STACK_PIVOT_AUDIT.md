# 00 — STACK PIVOT AUDIT (v3 FINAL — Go backend + Flutter mobile)

**Question answered:** "The backend doesn't work on Go and the frontend doesn't work on Flutter."

**Root cause:** the v2 pack specified **FastAPI (Python)** for the backend and **Expo React Native (TypeScript)** for the app. The API contracts, database, Redis semantics, QR chain, safety design and patent claims are language-agnostic and were never the problem — the *stack decisions* were. v3 replaces them with **Go 1.23+ backend** and **Flutter (Dart 3) mobile app**, keeping every frozen contract (D1–D18, D22, D24–D26) byte-identical wherever possible. This is the last pivot the pack needs: everything below is stack reality, not preference.

## A. What survives untouched
| Area | Status |
|---|---|
| API contract (08) — all 42 endpoints, payloads, enums, RBAC matrix | unchanged (token type + bundle manifest updated, endpoints not) |
| Database schema (04) — all 21 tables, hash chain, indexes, seeds | unchanged (migration *tooling* changes: Alembic → golang-migrate SQL) |
| Redis design (05) — key catalogue, TTLs, Lua patterns | unchanged semantics (client library: redis-py → go-redis; jobs: arq → Asynq) |
| ML pipelines (07) — LightGBM/CatBoost demand, conformal, newsvendor, fusion, anomalies | unchanged, all still Python — now served via the sidecar (below) |
| QR chain, fairness matching, VRPTW (OR-Tools), ESG formulas, state machines | unchanged |
| Offline-first protocol (D24), sync semantics, capture pipeline rules (D25), app-config control (D26), refresh-token auth (D22) | unchanged semantics |
| Patent claims (master §11) — all five, incl. edge-gated evidence-bound chain | stack-agnostic, unchanged |

## B. What changes and why (locked as decisions)
| # | v2 assumption | v3 decision |
|---|---|---|
| 1 | Backend = FastAPI/Python | **D27 Backend = Go 1.23+**: chi router, **oapi-codegen contract-first** (server stubs generated from the frozen `openapi.json` — CI fails on drift), pgx v5 + sqlc, golang-migrate, golang-jwt/v5, go-playground/validator, slog, prometheus/client_golang |
| 2 | Go can import Python ML/CV | It can't. **D28: one Python inference sidecar `mlserving/`** (FastAPI, internal :8001, shared-secret): wraps demand, fusion, CV, OR-Tools routing, anomaly detection. Go calls it with timeout + circuit breaker (sony/gobreaker) + `FEATURE_MOCK_*` fallbacks. Principle 1 amended: *one repo, one Go API, one Python inference service, one DB* — not microservice sprawl |
| 3 | Workers = arq (Python) | **D29 jobs = Asynq** (Redis DB1, cron scheduler): reports (PDF via maroto v2), push, expiry sweep, sensor bulk-insert, drift check (calls sidecar), offer expiry |
| 4 | App = Expo React Native | **D19(v3) App = Flutter, Dart 3**, single codebase, all 4 roles. Riverpod, go_router, dio, Drift (SQLite), flutter_secure_storage, flutter_map (OSM — no map API key needed for the demo), mobile_scanner, camera, fl_chart, intl en/hi |
| 5 | Device CV = same int8 ONNX via onnxruntime-react-native | Flutter has no first-class ONNX runtime; **`tflite_flutter` is officially unmaintained** (verified live). **D20(v3): device runs TFLite int8 via `flutter_litert` (maintained successor, classic `Interpreter` API, bundled native runtimes).** Server keeps ONNX int8. **One Keras checkpoint → two exports (TFLite int8 device, ONNX server) with a cross-runtime parity gate (D21 v3)** |
| 6 | Classifier trained in PyTorch (timm) | **v3 primary: train in Keras/TF** (EfficientNet-Lite0 — designed for TFLite; MobileNetV3-Large alt) because native int8 TFLite quantization is mature there. Reason: **ai_edge_torch has an open MobileNetV3 int8 bug** (google-ai-edge/litert-torch #499, still open Jan 2026). PyTorch path stays as a documented alternative with that caveat, not the default |
| 7 | Push = Expo Push Service | **D23(v3): FCM HTTP v1 via the official Firebase Admin SDK for Go**; client uses firebase_messaging. No Expo dependency at all; still best-effort, D23 semantics unchanged |
| 8 | Formulas canonical in `integration/processing` (D9) | **D9(v3): canonical KPI formulas in Go `internal/processing`;** `integration/processing` (Python) mirrors them for ML features; a committed parity fixture (inputs→expected outputs) is asserted in both languages in CI |
| 9 | Alembic in `database/migrations` (D14) | **D14(v3): still one location, now plain SQL migrations** `0001..0008_{up,down}.sql` run by golang-migrate. Same versions, same content |
| 10 | QR hash seed generation in Python | **D30: Go owns the canonical hash-chain implementation** (`internal/qrchain`) — it is byte-concatenation + SHA-256, language-neutral by construction; `cmd/seedcheck` generates/validates seed chains; golden vectors shared with ML tests |

## C. Ripple effects applied across the pack
- **01_MASTER** — topology `phones → nginx → Go API → {Postgres, Redis, mlserving → ml/cv/routing}`, ports +8001 sidecar; env matrix (Go + Flutter dart-define + sidecar); phases: sidecar skeleton lands Phase 1 (mocked in Go until then); risk register gains sidecar-latency and dual-artifact-drift entries with their gates.
- **02_MOBILE** — full Flutter rewrite (structure + spec): dio interceptors (refresh single-flight, Idempotency-Key), Drift outbox with the same states, flutter_litert preview + gate reading quantization constants (scale/zero-point) from the bundle manifest, FCM deep links, patrol/integration_test E2E.
- **03_BACKEND** — full Go rewrite: layout `cmd/{server,worker,seedcheck}`, `internal/{httpapi,auth,rbac,store,services,qrchain,mlclient,pushx,redisx,asynqx,processing,files}`; every v2 business rule ported 1:1 (state machine table, matching weights, sync replay, approve/override audit).
- **04_DATABASE** — golang-migrate; tables/indexes/seeds identical; `seed_check.py` → `cmd/seedcheck`.
- **05_REDIS** — go-redis + Lua unchanged; Asynq owns `asynq:*` keys in DB1; sensor stream consumer is now Go (XADD/XREADGROUP + pgx COPY — faster than arq, same semantics).
- **06_CV** — training recipe in Keras; `export/to_tflite_int8.py` + `export/to_onnx.py` + `bundle.py` → bundle carries BOTH artifacts + manifest (two sha256s + quantization constants + preprocess constants); parity gate: TFLite vs ONNX argmax agreement ≥ 99% and mean |Δunsafe_prob| ≤ 0.03 on golden fixtures + test set, else the bundle is not shipped.
- **07_ML_TRAINING** — pipeline unchanged; new §15 "Serving (mlserving/)" defines the internal contract (endpoints, auth, timeouts, health, mock flags); fusion stays server-side (sidecar), never on device.
- **08_API_CONTRACT** — endpoints unchanged; device token = FCM token; `/app/model-bundle` manifest now lists `server{onnx}` + `device{tflite}` + thresholds; push envelope = FCM data message with the same `{type,id,deep_link}` shape.

## D. Verified by live research this turn (re-verify at kickoff)
- `tflite_flutter`: **"no longer maintained"** per its own successor; `flutter_litert` ^3.8.0 (Feb 2026) is the maintained drop-in (Interpreter API source-compatible, auto-bundled native runtimes, classic `Interpreter` supports int8/quantized I/O — `CompiledModel`/LiteRT-Next is float32-only, so the spec pins the classic Interpreter).
- `litert-torch` (ai_edge_torch) PyTorch→TFLite converter is Beta; **MobileNetV3 int8 quantization bug open since Feb 2025** (#499) → hence Keras as the primary training path.
- Go stack pins (chi, oapi-codegen, pgx v5, sqlc, golang-migrate, golang-jwt/v5, Asynq, maroto v2, Firebase Admin SDK for Go, go-redis v9, sony/gobreaker): all actively maintained — verify versions at kickoff.

## E. Honest tradeoffs (know them for judge Q&A)
1. **Two model artifacts now exist (ONNX server + TFLite device) instead of one.** Mitigation: single Keras checkpoint, versioned together in one bundle, CI parity gate before any release. The RN design's "one file" elegance is gone — the Flutter constraint costs exactly this, and the parity gate is the answer.
2. **Two backend languages (Go + Python sidecar).** This is the standard production pattern (Go for IO/transactions, Python for models) — defensible and common at FAANG scale; the alternative (all-Python backend) contradicts your team's stack. The sidecar is internal-only, one service, mocked in Go until ready.
3. **Flutter gains:** no map API key (flutter_map/OSM), best-in-class QR scanner (mobile_scanner), typed SQLite (Drift), FCM without any Expo account, and polished UI motion — all demo wins.
4. Nothing about the safety story, human-approval rule, evidence chain, offline demo moment, or patent claims changed.