# 08 — API CONTRACT (v3 FINAL — single source; serves Go backend, Flutter app, mocks)

Base `http://localhost:8000/api/v1` · Swagger `/docs` · Auth `Authorization: Bearer <JWT>` · IDs UUID · time ISO-8601 (UTC stored) · units kg/°C/kWh/km/min · weekday 0=Mon.
**Contract-first (D27):** `docs/api/openapi.json` is the source of truth; the Go server implements oapi-codegen strict stubs generated from it (CI drift gate); Dart mock fixtures validate against it in the same CI job.
Headers: `X-Request-ID` (returned), `X-App-Version` (all mobile clients — enforced vs `min_supported_version`, D26), `Idempotency-Key` (optional on all POST that create; on mobile it equals the `client_event_id`). Pagination: `?limit=50&cursor=` → `{items, next_cursor}`.
Error: `{"detail":"...","code":"SURPLUS_EXPIRED","message":"human text"}`.
Status: 400,401,403,404,409 state conflict,413 file too large,415 bad image type,422,426 app version unsupported,429 (with `Retry-After`),500,503 dependency down.
Codes: SURPLUS_EXPIRED, INVALID_STATE_TRANSITION, SAFETY_REJECTED_CANNOT_APPROVE, IMAGE_TOO_LARGE, IMAGE_UNSUPPORTED, IMAGE_UNREADABLE, CV_UNAVAILABLE, ML_UNAVAILABLE, NO_ELIGIBLE_RECIPIENT, MATCH_NOT_FOUND, QR_CHAIN_BROKEN, IDEMPOTENCY_KEY_REUSED, FORBIDDEN_ROLE, RATE_LIMITED, REFRESH_TOKEN_REUSED, APP_VERSION_UNSUPPORTED, SYNC_ITEM_REJECTED, SYNC_OUT_OF_ORDER, ROUTE_INFEASIBLE.

## Enums
role: KITCHEN|NGO|LOGISTICS|ADMIN|SYSTEM · meal_type: BREAKFAST|LUNCH|SNACK|DINNER · waste cause: OVERPRODUCTION|LOW_ATTENDANCE|SPOILAGE|EXPIRY|STORAGE_ISSUE|PREPARATION_ERROR|OTHER · surplus status: PENDING_SAFETY|AVAILABLE|HOLD|MATCHED|IN_TRANSIT|DELIVERED|EXPIRED|DIVERTED · visual.status: GOOD|RISK|REJECTED · safety decision: ELIGIBLE|HOLD|REJECTED · risk: LOW|MEDIUM|HIGH · qr event: CREATED|APPROVED|MATCHED|PICKED_UP|HANDED_OFF|RECEIVED · diversion: ANIMAL_FEED|COMPOST|BIOGAS · alert type: TEMP_EXCURSION|HUMIDITY_HIGH|ENERGY_SPIKE|DOWNTIME_HIGH|MATERIAL_LOSS_HIGH|EXPIRY_SOON|SENSOR_STALE · severity: INFO|WARN|CRITICAL · sync item status: ACCEPTED|DUPLICATE|REJECTED · device platform: ANDROID|IOS.

## Endpoints
| # | Method Path | Roles | Notes |
|---|---|---|---|
| 1 | POST /auth/login | public | rate-limited 10/min per IP AND device id; returns access + refresh |
| 2 | GET /auth/me | any | |
| 3 | POST /auth/logout | any | denylists jti in Redis |
| 4 | GET /kitchen/overview | KITCHEN,ADMIN | dashboard cards |
| 5 | POST /kitchen/attendance | KITCHEN | accepts client_event_id (offline dedupe) |
| 6 | POST /kitchen/menu | KITCHEN | |
| 7 | POST /predict-demand | KITCHEN,ADMIN | cached 60 s (Go → mlserving) |
| 8 | POST /planning/what-if | KITCHEN | |
| 9 | POST /feedback/outcome | KITCHEN | actuals |
| 10 | POST /waste | KITCHEN | accepts client_event_id |
| 11 | GET /waste/analytics | KITCHEN,ADMIN | |
| 12 | POST /surplus | KITCHEN | → PENDING_SAFETY |
| 13 | GET /surplus?status=&cursor= | all human | |
| 14 | GET /surplus/{id} | all human | |
| 15 | POST /quality/check | KITCHEN | multipart; optional device_preview block stored as telemetry ONLY (D20) |
| 16 | POST /surplus/{id}/approve | KITCHEN,ADMIN | human decision |
| 17 | POST /surplus/{id}/divert | KITCHEN,ADMIN | |
| 18 | POST /match | KITCHEN | triggers FCM push to offered NGOs (D23) |
| 19 | POST /matches/{id}/respond | NGO | accept/decline |
| 20 | GET /recipients | all human | |
| 21 | POST /route | KITCHEN,LOGISTICS | |
| 22 | POST /qr/{batch_id}/event | per event | actor from token; accepts client_event_id + client_ts (provenance only, never hashed — D24) |
| 23 | GET /qr/{batch_id} | all human | timeline shows client_ts as "captured at" |
| 24 | GET /qr/{batch_id}/verify | all human | |
| 25 | POST /sensors/readings | SYSTEM,ADMIN | batch array allowed |
| 26 | GET /sensors/latest, /sensors/history | KITCHEN,ADMIN | latest from Redis |
| 27 | GET /alerts, POST /alerts/{id}/ack | KITCHEN,ADMIN | |
| 28 | POST /processing/metrics | KITCHEN,ADMIN | |
| 29 | GET /processing/analytics | KITCHEN,ADMIN | |
| 30 | GET /analytics/impact | all human | cached 30 s |
| 31 | GET /reports/esg?from=&to=&format=pdf|csv | KITCHEN,ADMIN | async job → 202 `{job_id}`; GET /reports/jobs/{id} |
| 32 | GET /events/stream | all human | SSE — foreground-only on mobile; polling + FCM is primary (D23) |
| 33 | /admin/users (GET,POST,PATCH), /admin/models, /admin/audit-log, POST /admin/reseed | ADMIN | reseed dev only |
| 34 | GET /health, /ready, /metrics | public/internal | |
| 35 | POST /auth/refresh | public | rotating refresh (D22); reuse → 401 REFRESH_TOKEN_REUSED + family revoked |
| 36 | POST /auth/logout-all | any | revokes all refresh families of the user |
| 37 | POST /devices, DELETE /devices/{id} | any | register/prune push token (FCM registration token) |
| 38 | POST /sync/batch | all human | offline outbox replay (D24); ≤50 items; 120/min |
| 39 | GET /app/config | any (auth optional) | min_supported_version, flags, bundle pointer, display copy (D26) |
| 40 | GET /app/model-bundle | all human | manifest (server ONNX + device TFLite, both sha256s — D21); binaries are static files |
| 41 | POST /notifications/test | ADMIN | sends a stub push to own devices (demo/dev) |
| 42 | GET /qr/{batch_id}/label?format=svg|png | KITCHEN,LOGISTICS,ADMIN | printable batch label with QR + batch_code |

## Key payloads
**Login** → `{"access_token","refresh_token","token_type":"bearer","expires_in":3600,"user":{"id","name","role"}}`
**Refresh** req `{"refresh_token"}` → same shape as login (new pair; old refresh invalidated; reuse of an old one revokes the family).
**Devices** req `{"platform":"ANDROID","token":"<FCM registration token>","app_version":"1.0.0"}` → `{"device_id","active":true}`.
**Sync batch** req `{"items":[{"client_event_id":"018f...","kind":"qr_event|waste|attendance|surplus|quality_upload","payload":{...same shape as the direct endpoint...},"client_ts":"..."}]}` → `{"results":[{"client_event_id","status":"ACCEPTED|DUPLICATE|REJECTED","code"?,"body"?}],"applied":n,"duplicates":n}`. QR items replay per-batch FIFO; out-of-sequence → REJECTED `SYNC_OUT_OF_ORDER`.
**App config** → `{"min_supported_version":"1.0.0","force_update":false,"feature_flags":{...},"model_bundle":{"version":"cv-v1","server":{"url":"/static/bundles/cv/cv-v1/model_int8.onnx","sha256":"..."},"device":{"url":"/static/bundles/cv/cv-v1/model_int8.tflite","sha256":"...","size_bytes":5200000},"thresholds":{...},"preprocess":{...}},"display":{"carbon_factor_source":"...","danger_zone_band_c":[5,60]}}`

**Predict** req: `{"attendance":450,"meal_type":"LUNCH","menu":["rice","dal","paneer"],"day_of_week":4,"date":"2026-10-02","historical_consumption":[..]?,"historical_surplus":[..]?}` (history optional; backend fills from DB).
res:
```json
{"predicted_consumption":420,"recommended_production":435,"expected_surplus":15,"surplus_risk":"LOW",
 "prediction_interval":{"p10":401,"p50":420,"p90":441,"coverage_target":0.8},
 "recommended_quantile":0.7,
 "top_drivers":[{"feature":"attendance","effect_kg":+38.2},{"feature":"weekday","effect_kg":-6.1}],
 "confidence":0.87,"model_version":"demand-v1","data_source":"SYNTHETIC|REAL"}
```
`confidence` is deprecated, derived from interval width; the app shows the interval.

**Surplus create** req `{"food_name","quantity_kg","prepared_at","expiry_at","meal_id?","food_category?","client_event_id"?}` → `{"batch_id","batch_code","status":"PENDING_SAFETY","safety_status":"PENDING"}`.

**Quality check** multipart: `image` (jpg/png ≤8 MB; app sends ≤1 MB per D25), `batch_id`, `temperature_c?`, `device_preview?` (JSON string, §06/5 shape). res:
```json
{"batch_id":"uuid","visual":{"status":"RISK","risk_level":"MEDIUM","confidence":0.84,"reason":"Visible discoloration","detections":[{"label":"quality_risk","confidence":0.84,"bbox":[x,y,w,h]}],"heatmap_url":"/files/..","image_quality":"OK","ood":false,"model_version":"cv-v1"},
 "safety_decision":{"status":"HOLD","reasons":["CV_RISK","DANGER_MINUTES_42"],"danger_zone_minutes":42,"hours_to_expiry":2.5,"model_version":"fusion-v1"},
 "requires_human_approval":true,
 "quality_status":"RISK","risk_level":"MEDIUM","confidence":0.84,"reason":"...","detections":[],"model_version":"cv-v1"}
```
(bottom flat fields = backward compat.)

**Approve** req `{"decision":"APPROVE|HOLD|REJECT","note":"..."}` → `{"batch_id","status","qr_event":{...}}`; 409 `SAFETY_REJECTED_CANNOT_APPROVE` if fusion says REJECTED (admin override requires `override_reason`, audit-logged).

**Match** req `{"batch_id","max_results":5}` → matches with `recipient_id,name,type,capacity_kg,distance_km,pickup_window,score,score_breakdown{need,capacity,distance,window,shelf_life,fairness},reasons[]`.

**Route** req `{"batch_id","recipient_ids":[...],"vehicle_id"?}` → `{"route_id","distance_km","eta_minutes","solver":"ortools-vrptw|haversine","stops":[{"sequence","recipient_id","name","lat","lng","eta","deadline_ok"}]}`.

**QR event** req `{"event_type":"PICKED_UP","location?":{lat,lng},"client_event_id"?,"client_ts"?}` → `{"batch_id","event_type","timestamp","hash","prev_hash"}` ; verify → `{"valid":true,"length":6,"broken_at":null}`.

**Sensor reading** req `{"readings":[{"sensor_id","location_id","timestamp","temperature_c","humidity_pct","energy_kwh","batch_id"?}]}` → `{"accepted":n,"alerts":[...]}`.

**Processing** req `{"date","kitchen_id?","raw_material_kg","output_kg","waste_kg","runtime_minutes","downtime_minutes","energy_kwh"}` → `{"material_loss_pct":8.0,"production_efficiency":0.92,"downtime_pct":8.33,"energy_per_kg":0.272,"efficiency_score":77.9,"anomalies":[]}`.

**Impact** → `{"waste_avoided_kg","redistributed_kg","diverted_kg","estimated_carbon_saved_kg","carbon_factor_used","carbon_factor_source","successful_redistributions","meals_equivalent","trend":[{date,...}]}`.

**Push envelope (server→FCM v1, D23)**: notification {title, body} + data = `{"type":"match.offered|alert.created|qr.event|surplus.status_changed","id":"...","deep_link":"surplus/{id}|match/{id}|route/{id}"}`. Push is best-effort; state lives in the API.

**SSE events**: `alert.created`, `surplus.status_changed`, `match.offered`, `qr.event`, `sensor.reading`, `job.done`, `app.config_changed` → each `{type,data,ts}`.

## RBAC matrix
KITCHEN: own kitchen data all ops. NGO: recipients, own matches (respond), QR RECEIVED. LOGISTICS: routes, QR PICKED_UP/HANDED_OFF. ADMIN: everything + override. SYSTEM: /sensors/readings only.
QR event permission: CREATED/APPROVED/MATCHED by kitchen/admin; PICKED_UP/HANDED_OFF by logistics; RECEIVED by NGO.

## Contract tests
CI generates/validates `openapi.json` against committed file; Go stubs are generated from it (`go generate` diff must be empty); **Go and Dart mock fixtures are validated against it in the same CI job**; sync-batch replay determinism test runs in both.