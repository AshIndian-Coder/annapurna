# 01 — MASTER SPEC (v3 FINAL — Go backend + Flutter app — give this to any AI first)

You are building **SIH26234**: a mobile-first platform that predicts institutional-kitchen demand, verifies surplus food safety, redistributes it to NGOs with optimized routes, tracks custody tamper-evidently, monitors processing units, and reports ESG impact. The user-facing product is **one Flutter app** (kitchen, NGO, driver, admin); the API is a **Go service**; Python lives only in a thin inference sidecar and the training pipeline. Problem statement (MoFPI): predict demand/surplus, detect expiry/quality via sensors + images, connect surplus to NGOs, optimize logistics, monitor processing efficiency, detect inefficiencies, produce sustainability analytics, support production planning.

Coverage map (PS requirement → feature → module):
| PS requirement | Feature | Module |
|---|---|---|
| Predict demand/surplus | quantile forecast + newsvendor | ml/demand via mlserving |
| Expiry/quality via sensors + images | on-device preview → CV cascade + time-temperature fusion | cv (Keras; TFLite device + ONNX server), ml/quality, mobile/cv, integration/iot |
| Connect to NGOs/buyers | fair explainable matching + FCM push offers | backend services, mobile NGO role |
| Optimize routes | OR-Tools VRPTW (Python, via sidecar) | integration/routing |
| Monitor processing | KPI + anomaly detection | internal/processing (Go), ml/analytics |
| Detect inefficiencies | EWMA/CUSUM + IsolationForest alerts | ml/analytics via sidecar |
| Sustainability/ESG | impact + PDF/CSV report | Go report service (maroto) |
| Production planning | what-if planner | backend, mobile WhatIf |
| Secondary buyers / unsafe-for-humans | recovery hierarchy (feed/compost/biogas) | backend diversion |
| Field operation on phones | offline-first Flutter app, camera QR + food capture, push | mobile/ (D19–D26) |

## 1. Non-negotiable principles
1. **One repo, one Go API, one Python inference service (mlserving), one DB.** Redis is an accelerator only — the system runs with Redis down.
2. Humans approve every redistribution. AI recommends, never certifies food safety.
3. Every number on a slide is reproducible via `make eval`.
4. Synthetic data is always labelled. Estimates are labelled.
5. Contracts are frozen: `docs/api/openapi.json` is the source of truth; the Go server is **generated from it** (oapi-codegen, CI drift gate); changes go through CONTRACT CHANGE REQUIRED.
6. Mock-first: the Flutter app and Go API run with fakes until real modules arrive (`FEATURE_MOCK_*` flags live in Go, `USE_MOCKS` in the app).
7. Mobile-first, offline-first: the core field flows (QR scans, waste log, surplus create, food photo) work with no network; sync later (D24).

## 2. Locked decisions — D1–D18 (v1, unchanged except noted)
D1 UUID ids + `batch_code` (e.g. B-10291). D2 `downtime_pct = downtime_min/runtime_min*100`. D3 `efficiency_score = clip(100 − 1.2·material_loss_pct − 1.5·downtime_pct, 0, 100)` (example 500/460/420/35/125 → material_loss 8.0, downtime 8.33, energy/kg 0.272, score **77.9**). D4 quality response = `visual` + `safety_decision` blocks. D5 approval is a human endpoint; QR actor from JWT. D6 ISO weekday 0=Mon. D7 functions `ml.predict.predict_demand`, `cv.inference.cv_predict`, alias `process_image_cv` (both invoked **via mlserving**, D28). D8 surplus statuses `PENDING_SAFETY, AVAILABLE, HOLD, MATCHED, IN_TRANSIT, DELIVERED, EXPIRED, DIVERTED`. **D9 (v3): canonical KPI formulas live in Go `internal/processing`; `integration/processing` (Python) mirrors them for ML features; a committed parity fixture (inputs→expected) is asserted in both languages in CI.** D10 carbon factor configurable, cited, labelled estimate. D11 schema additions via proposal (approved: migrations 0004–0008). D12 Redis roles: rate limit, cache, pub/sub+streams, locks, jobs, idempotency. D13 image models live only in `cv/`. **D14 (v3): migrations = plain SQL, one location (`database/migrations`), run by golang-migrate.** D15 `ml/models/demand_model.pkl` is a symlink to the promoted registry version. D16 roles: KITCHEN, NGO, LOGISTICS, ADMIN + non-human `SYSTEM` service token. D17 timestamps stored UTC, shown IST. D18 money/quantities: kg, °C, kWh, km, minutes.

## 3. Locked decisions — mobile (v2, D19–D26; v3 amendments marked)
- **D19 (v3): Form factor:** ONE **Flutter (Dart 3)** app for all 4 roles; folder `mobile/`. Riverpod + go_router + dio + Drift + flutter_map. Tablet layout for KITCHEN/ADMIN; phone for NGO/LOGISTICS. Native Android (and iOS if time) builds via standard `flutter build` / CI — no Expo.
- **D20 (v3): CV authority split:** the phone runs a **TFLite int8** classifier via `flutter_litert` (classic Interpreter API) ONLY for (a) the image-quality gate (blur/dark → retake before upload) and (b) an instant preview chip. The authoritative decision is ALWAYS `POST /quality/check` → mlserving `cv_predict` (ONNX int8) → fusion. The device can never issue a final REJECTED/ELIGIBLE; heatmap + model_version + `evidence_hash` live server-side.
- **D21 (v3): Model bundle:** one Keras checkpoint → exports `model_int8.tflite` (device) AND `model_int8.onnx` (server), versioned together in `cv/models/cv-vN/bundle/` with `manifest.json` (version, two sha256s, size, **TFLite input quantization scale/zero-point + preprocess constants**). App fetches `GET /app/model-bundle`, verifies sha256 before activating, falls back to the bundled copy. **Parity gate: TFLite vs ONNX argmax agreement ≥ 99% and mean |Δunsafe_prob| ≤ 0.03 on golden fixtures + test set, or the bundle does not ship.**
- **D22 Auth:** access JWT 60 min + refresh token 30 days, rotating; refresh-reuse detection revokes the family; tokens in flutter_secure_storage; `X-App-Version` header on every request.
- **D23 (v3): Push: FCM HTTP v1** via the official Firebase Admin SDK for Go; client firebase_messaging. Triggers: `alert.created` (WARN/CRITICAL), `match.offered`, `qr.event` RECEIVED, `surplus.status_changed`. SSE stays for foreground live tiles only. Push is best-effort: no data lives only in a push.
- **D24 Offline-first:** local Drift outbox; every mutation carries `client_event_id` (UUIDv7) + optional `client_ts`; `POST /sync/batch` replays in order with per-item idempotency (ACCEPTED / DUPLICATE / REJECTED). QR chain order = server-recorded order; `client_ts` is provenance, never hashed. Offline-allowed: QR events, waste, attendance, surplus create, photo capture (queued upload). Online-only: predict, what-if, match, route, reports.
- **D25 Capture:** camera package, 4:3, resize to long edge 1600 px, JPEG q80, target ≤ 1 MB, EXIF/GPS stripped client-side (re-encode); `MAX_UPLOAD_MB=8` stays as the server bound.
- **D26 App config:** `GET /app/config` returns `min_supported_version`, `force_update`, feature flags, active model-bundle pointer, display copy. The app hard-blocks below min version with a store/deep link.

## 4. Locked decisions — Go stack (v3, D27–D30)
- **D27 Backend = Go 1.23+**: chi router; **oapi-codegen strict server** generated from the frozen openapi.json; pgx v5 pool + sqlc queries; golang-migrate (D14); golang-jwt/v5; bcrypt; go-playground/validator; slog (JSON, request_id); prometheus/client_golang; maroto v2 for PDF; sony/gobreaker for the sidecar client. Multi-stage Dockerfile → distroless.
- **D28 Python inference sidecar `mlserving/`**: internal FastAPI on :8001 (compose network only) with shared-secret header; endpoints `/v1/predict-demand`, `/v1/assess-safety`, `/v1/cv-predict`, `/v1/build-route`, `/v1/detect-anomalies`, `/health|/ready`. It imports only `ml`, `cv`, `integration.routing`. Go calls it with per-call timeouts (demand 2 s, CV 8 s, route 10 s), a circuit breaker per endpoint, and mock fallbacks — a sidecar outage degrades the demo, never kills it.
- **D29 Jobs = Asynq** (Redis DB1): reports, push sends, expiry sweep (cron 1 min, lock), sensor bulk-insert (XREADGROUP + pgx COPY), offer expiry (30 min), drift check (daily → calls sidecar). No Python worker queue.
- **D30 Canonical hash chain lives in Go** (`internal/qrchain`): `hash = H(prev_hash || batch_id || event_type || actor_id || server_ts_iso || evidence_hash)`, ASCII byte concatenation + SHA-256 — language-neutral by construction. `cmd/seedcheck` generates/validates seed chains; golden vectors committed for cross-language tests.

## 5. Core workflows (state machines)
Surplus: `PENDING_SAFETY → (approve) AVAILABLE | (hold) HOLD | (reject) DIVERTED/EXPIRED`; `AVAILABLE → MATCHED → IN_TRANSIT → DELIVERED`; any → `EXPIRED` by sweep.
QR: `CREATED → APPROVED → MATCHED → PICKED_UP → HANDED_OFF → RECEIVED`, 409 on skip/repeat, D30 chain; events queued offline sync per-batch FIFO; `client_ts` displayed in the timeline as captured time.
Safety decision (fusion, sidecar): hard rules first (expired → REJECTED; danger-zone budget exceeded → REJECTED), then monotone LightGBM; thresholds tuned for unsafe-recall ≥ 0.98; low confidence → HOLD (abstain). Device preview cannot override any of this.

## 6. End-to-end data flow (mobile)
Attendance+menu → `/predict-demand` (Redis cache 60 s → Go → mlserving → ml lib) → quantile forecast → newsvendor production → surplus logged in app (offline-capable) → capture photo → on-device gate+preview (D20) → upload (≤1 MB) → mlserving `cv_predict` → fusion with sensor danger-minutes → human approve in app → match (fairness + reasons) → NGO FCM push + accept → VRPTW route (sidecar OR-Tools) → driver QR scan (offline-capable) → sync → verify chain → outcome feedback → drift monitor → ESG report (PDF share from phone).

## 7. Environment matrix (`.env.example` union)
Go API: PORT=8000, DATABASE_URL, REDIS_URL, JWT_SECRET, JWT_EXPIRE_MIN=60, REFRESH_SECRET, REFRESH_EXPIRE_DAYS=30, ALLOWED_ORIGINS (docs only), MLSERVING_URL=http://mlserving:8001, MLSERVING_TOKEN, MODEL_BUNDLE_DIR, MAX_UPLOAD_MB=8, CARBON_FACTOR_KG_CO2E_PER_KG, DANGER_ZONE_MIN_C=5, DANGER_ZONE_MAX_C=60 (verify against FSSAI/WHO guidance before citing), FEATURE_MOCK_ML, FEATURE_MOCK_CV, APP_MIN_VERSION, GOOGLE_APPLICATION_CREDENTIALS (FCM service account JSON), FEATURE_MOCK_ROUTING. Sidecar: MLFLOW/registry path, CV_MODEL_PATH, CV_DEVICE=cpu, ROUTING_PROVIDER=haversine|osrm|ors, ROUTING_API_KEY. Mobile (dart-define): API_URL, USE_MOCKS, FEATURE_ON_DEVICE_CV.

## 8. Delivery plan (adapt days to your SIH timeline)
Phase 0 (day 1): freeze contract + openapi.json → generate Go stubs; repo skeleton; CI (lint + gen-drift + test); Flutter project + FCM config on two phones. Phase 1: DB+auth(+refresh)+seed (Go); app shell on mocks; mlserving skeleton with mock responses; ML baseline + synthetic data; CV baseline in Keras + first bundle (both exports + parity). Phase 2: real predict via sidecar; quality pipeline (device gate + server fusion); surplus state machine; matching + FCM offers. Phase 3: routing via sidecar; QR chain + offline sync; sensors+alerts; processing anomaly. Phase 4: impact/ESG, what-if, NGO/driver flows, offline polish. Phase 5: evaluation docs, load test, demo rehearsal ×3 (two phones + scrcpy mirror), backup video, patent disclosure. Integrate every 1–2 days; `main` always demoable.

## 9. Quality gates (CI must enforce)
Go lint/vet + gen-drift (openapi ↔ generated stubs); unit (Go + Dart); API contract test (live responses equal openapi.json); mock fixtures (Go + Dart) validate against openapi; migration from empty DB; seed minimums; no-leakage test; conformal coverage ≥ 0.85 on holdout; CV unsafe-recall ≥ 0.98 at operating point on test set; **TFLite/ONNX parity gate (D21)**; **device preprocess parity (golden vectors incl. quantization scale/zero-point)**; Go↔Python processing-formula parity fixture; sync replay idempotent (same batch twice = zero duplicates); Flutter integration_test demo flow; Docker `up` healthy in < 90 s; app cold start < 2.5 s, release APK < 40 MB.

## 10. Demo (5 min, phone-first) and judge Q&A
Script in `docs/demo/DEMO_SCRIPT.md` (two phones mirrored with scrcpy): kitchen phone predict (p10–p90 band) → what-if → surplus → capture photo (on-device preview instantly) → upload → server fusion HOLD while simulator causes temp excursion push → approve a good batch → NGO phone gets FCM push → accepts → route on flutter_map → driver phone scans QR in **airplane mode** → sync → verify chain green → ESG PDF shared from phone. Prepare answers: real vs synthetic data; how unsafe donations are prevented; offline behaviour (the airplane-mode scan IS the answer); per-kitchen cost; NGO onboarding; FSSAI alignment (verify the Food Safety and Standards (Food Recovery and Distribution) Regulations, 2019 before citing); carbon factor source; why Go + sidecar (IO vs models — the standard production split); why two model artifacts (parity gate); why metrics are trustworthy.

## 11. Risk register
| Risk | Mitigation |
|---|---|
| CV fails on cooked Indian food (domain shift) | collect own ~1.5k photos, fine-tune, abstain/HOLD, show cross-domain numbers honestly |
| Synthetic data criticised | benchmark on a public real dataset, calibrate simulator, label clearly |
| Live demo breaks | `make demo` deterministic seed, offline mode, backup video, release APK installed on both phones before travel |
| Push silently fails (emulators, no Play services) | real device with Play services for push; local notifications fallback; demo moment is offline-first (D24) so nothing depends on push |
| TFLite/ONNX divergence | one Keras checkpoint, one bundle, D21 parity gate blocks shipping |
| Sidecar down/slow at demo | Go mock fallbacks + circuit breaker; `/ready` shows mlserving status; restart in compose |
| Team bandwidth (two backend languages) | mlserving is ~150 lines of glue; all business logic in Go; mocks unblock everyone day 1 |
| Scope creep | P1 features frozen at Phase 4; web admin out of scope |
| Contract drift | openapi is generated-into-code; CI fails on drift |
| AGPL exposure (Ultralytics) | Apache/MIT models only (section 12) |

## 12. Model choices (v3)
Demand: **LightGBM** (quantile) + **CatBoost** ensemble, conformal P10–P90, newsvendor decision; optional Chronos-Bolt-tiny zero-shot member (all Apache-2.0). Safety fusion: **monotone LightGBM** + isotonic calibration, sidecar-side only. Processing anomalies: EWMA/CUSUM + IsolationForest. Images: **train EfficientNet-Lite0 (primary) or MobileNetV3-Large in Keras** from ImageNet weights → export **TFLite int8 (device, flutter_litert)** + **ONNX int8 (server, onnxruntime)** from the same checkpoint, D21 parity gate. PyTorch/timm path is a documented alternative (ONNX mature; ai_edge_torch→TFLite has an open MobileNetV3 int8 bug — verify before choosing it). Detection (optional, later, server-side only): YOLOX-Tiny or RTMDet-Tiny (Apache-2.0); avoid Ultralytics (AGPL) on the patent path.

## 13. Patent-readiness (not legal advice)
**Process:** consult a registered Indian patent agent *before* any public release of the repo, pitch deck or demo video; novelty can be lost by public disclosure (India's grace periods are narrow — verify with the agent). File an Indian **provisional** application first (then complete within 12 months). Check SIH/your institution's IP terms for ownership before filing. Software "per se" is excluded (Sec 3(k)); claims must show technical effect (sensor data + image processing + control decision), so frame as a *system and method*.
**Claim-worthy inventive concepts (draft ideas to test against prior art):**
1. *Exposure-aware eligibility gate*: redistribution eligibility computed from a monotonic fusion of image-derived unsafe probability, cumulative time-above-safe-temperature, remaining shelf-life and food-category prior, with calibrated abstention that forces human review.
2. *Forecast-to-redistribution pre-allocation*: production chosen at a cost-ratio quantile of a conformally calibrated demand distribution, and the predicted surplus distribution pre-notifies candidate recipients before the surplus physically exists.
3. *Edge-gated evidence-bound custody chain*: an on-device quality gate that blocks unusable captures before upload, while every lifecycle event binds to the server-side safety-decision snapshot (hash of CV result + sensor summary) — including offline-captured events whose provenance timestamps survive sync reordering.
4. *Shelf-life-deadline VRPTW with fairness penalty* mixing recipient need, recency fairness and batch expiry as hard time windows.
5. *Recovery hierarchy routing*: automatic diversion of human-unsafe food to feed/compost/biogas with per-stream impact accounting.
All five are stack-agnostic — the Go/Flutter pivot changes nothing. **Repo artefacts:** `docs/patent/INVENTION_DISCLOSURE.md` (problem, prior art, embodiments, flowcharts, example numbers), `CLAIMS_DRAFT.md`, `PRIOR_ART_LOG.md` (Google Patents / InPASS / Espacenet queries), `DRAWINGS/` (block diagram incl. edge/server split, flowcharts, sequence diagram, data-structure diagram). Keep a dated lab notebook of design decisions (commit history helps as evidence).

## 14. Honest-claims policy
Allowed: "measured on held-out rolling-origin split, WAPE x%", "unsafe recall y% at threshold t on dataset Z", "device/server agreement z% across runtimes (TFLite vs ONNX, same checkpoint)". Forbidden: unmeasured accuracy, "certified", "guaranteed safe", invented carbon factors, claiming the phone model "approves" food.