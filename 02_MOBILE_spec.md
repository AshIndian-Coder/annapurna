# 02 — MOBILE APP SPEC (v3 FINAL — Flutter, Dart 3)

Read `01_MASTER_spec.md` (D19–D26) and `08_API_CONTRACT.md` first. The app tells the judge one story on a phone: **Predict → Capture → Verify → Approve → Match → Route → Scan (offline) → Prove → Report.** It is a field tool, not a dashboard port.

## 1. Stack (pinned at kickoff; verified Oct 2026 — re-verify versions)
Flutter stable (3.x, Dart 3, Material 3), **Riverpod 2** (code-gen providers; stateNotifier for flows), **go_router** (typed routes, deep links), **dio** (+ interceptors), **Drift** (typed SQLite, the offline DB), flutter_secure_storage (tokens), shared_preferences (config kv), **camera** (capture), **mobile_scanner** (QR — best-in-class, no platform channel work), **flutter_litert** (classic `Interpreter`, int8 — the maintained TFLite runtime; `tflite_flutter` is dead, do not use), firebase_messaging + flutter_local_notifications (FCM push + local fallback), **flutter_map** (OSM tiles — **no API key needed**; markers via flutter_map_marker_popup; list fallback offline), fl_chart (charts), intl + flutter_localizations (en/hi), freezed + json_serializable (DTOs), flutter_image_compress (resize/EXIF strip), path_provider, share_plus (ESG PDF share), connectivity_plus (network watcher), uuid (UUIDv7 client_event_id), flutter_lints strict. E2E: patrol (native interactions incl. permissions) — flutter integration_test acceptable. Set your own applicationId/bundle id BEFORE the first FCM registration.

## 2. Service layer (golden rule)
`core/api/client.dart`: base `API_URL` (dart-define), injects Bearer + `X-App-Version`; on 401 → single-flight refresh (queue concurrent calls; on refresh failure clear tokens → Login); `Idempotency-Key` = client_event_id on creates; multipart for images; errors normalized to `AppError{status,code,message}`; GET retry 2× on network error; 20 s timeout. Every service method has `if (USE_MOCKS) return mocks.<fn>(payload)`; mock fixtures are the exact JSON fixtures from 08_API_CONTRACT, contract-tested in CI against openapi.json (same fixtures the Go suite uses). Live updates: **foreground polling via Riverpod timers (15 s) + SSE is optional** — polling keeps the app simple and judges never see the difference; FCM covers the background. (If SSE is kept, use flutter_client_sse or raw HttpClient stream; it is foreground-only per D23.)

## 3. Roles and navigation
| Role | Tabs | Key screens |
|---|---|---|
| KITCHEN (phone/tablet) | Home · Plan · Surplus · More | Dashboard, Prediction, WhatIf, WasteLog, SurplusCreate/List, QualityCapture, QualityResult, Recipients, RouteMap, QrLabel, SensorMonitor, Alerts, Processing, Impact, EsgReport |
| NGO (phone) | Offers · History · Scan | Offers, OfferDetail, ReceiveScan |
| LOGISTICS (phone) | Today · Scan | RouteList, StopDetail, ScanFlow |
| ADMIN (tablet) | Ops · Models · Admin | Impact, Alerts, Processing, Users, Models, AuditLog, SystemHealth, AppConfig |
Role comes from `/auth/me` (server-driven); go_router redirects per role; navigation is UX only — the Go API enforces RBAC.

## 4. The capture flow (centerpiece — rehearse this)
1. SurplusCreate → status PENDING_SAFETY (offline-capable: outbox).
2. QualityCapture: live camera (camera package), on-screen frame guide, 4:3. On shutter: flutter_image_compress → long edge 1600 px, JPEG q80 (≤1 MB target), EXIF/GPS stripped by re-encode (D25).
3. **On-device quality gate** (`cv/quality_gate.dart`): Laplacian variance < τ_blur or luminance out of band → full-screen "Retake: too dark / blurry" — never upload a dead image.
4. **On-device preview** (`cv/preview.dart`): flutter_litert Interpreter on `model_int8.tflite` (bundle D21) → chip `Looks OK` / `Possible risk` / `Not sure` + ms latency. Label: "preview — final decision after upload". NEVER blocks or approves (D20).
5. Upload (multipart, queued if offline; `pending_uploads` holds the file). Optimistic UI: batch moves to QualityResult with "awaiting server decision" state.
6. Server result renders `SafetyDecisionCard`: Visual status + confidence, heatmap image, danger-minutes bar vs budget, hours-to-expiry, final ELIGIBLE/HOLD/REJECTED chip, reason chips. Buttons: Approve (disabled on REJECTED; ADMIN override → reason dialog), Hold, Divert. Persistent disclaimer: "Decision support, not food-safety certification."
7. Latency budget: shutter→preview < 1 s; upload+server decision p95 < 4 s on 4G (device int8 inference is typically tens of ms on mid-range Android).

## 5. Offline-first protocol (D24 — a judged feature)
- Every mutation goes through `mutate()` → online: direct call + record; offline: Drift outbox enqueue.
- Outbox row: {client_event_id (UUIDv7), kind, payload, dependsOn?, createdAt, state}. States: QUEUED → SENDING → DONE | FAILED(retryable ×5) | DEAD (red banner + "contact admin").
- Sync triggers: app start, app resume (AppLifecycleState.resumed), connectivity regain, 30 s while online with pending items.
- `POST /sync/batch` {items:[{client_event_id, kind, payload, client_ts}]} (max 50/batch) → per-item [{client_event_id, status: ACCEPTED|DUPLICATE|REJECTED, code?, body?}]. DUPLICATE = server already applied (idempotent replay, safe).
- Ordering: QR events replay per-batch FIFO (outbox `dependsOn` chains scans of the same batch). Chain hashes server order; the timeline shows `client_ts` as "captured at" (D24).
- Conflict policy: QR events append-only (no conflict); waste/attendance last-write-wins (documented); surplus create is server-authoritative (201 returns canonical batch, app adopts server ids); quality capture never re-uploads on DUPLICATE (server returns original decision).
- Offline-allowed: waste, attendance, surplus create, QR scan, photo capture (queued). Online-only (cloud icon): predict, what-if, match, route, ESG report.
- Airplane-mode demo moment: driver scans PICKED_UP offline → sync on reconnect → chain green. Rehearse it.

## 6. Push notifications (D23 v3 — FCM)
`firebase_messaging` obtains the FCM token after login + permission → `POST /devices`. Triggers from Go: `match.offered` (NGO), `alert.created` WARN/CRITICAL (KITCHEN/ADMIN), `qr.event` RECEIVED (KITCHEN), `surplus.status_changed` (KITCHEN). Foreground: in-app banner + haptic; background: system notification; tap → go_router deep link (`surplus/{id}`, `match/{id}`, `route/{id}`). Push is ALWAYS best-effort — the app refetches on resume (Riverpod invalidate); no state exists only in a push. Fallback if push is unavailable (emulator without Play services): flutter_local_notifications polling. Requires `google-services.json`/`GoogleService-Info.plist` + a real device with Play services for the push demo; note in runbook.

## 7. Screen specs (key ones)
**Login**: email/password, "Demo as Kitchen/NGO/Driver/Admin" (hidden when USE_MOCKS=false), language toggle en/hi.
**Dashboard (KITCHEN)**: KPI cards (expected diners, forecast p50 band, recommended production, surplus risk badge, waste today, available kg, redistributed today), live alert strip, next-action FAB, mini trend. Pull-to-refresh; skeletons everywhere.
**Prediction**: form (date, meal, attendance, menu chips) → IntervalBand p10–p50–p90 + recommended production marker + top drivers + model_version + data_source badge (SYNTHETIC shown). States INPUT→LOADING→RESULT→ACTION.
**WhatIf**: attendance slider, menu toggles, day chips; debounce 400 ms; delta vs baseline.
**WasteLog**: big-number keypad entry (kg), cause chips, meal; works offline ("queued" badge); analytics tab: by-cause donut, trend, top recurring cause insight.
**SurplusList**: status chips + expiry countdown; per-item stepper (Created → Quality → Approve → Match → Route → Track); actions enabled per status.
**Recipients**: ranked cards with score breakdown bar, reason chips, capacity/distance/window; multi-select → Send offers.
**RouteMap**: flutter_map with numbered markers + polyline, ETA, per-stop deadline_ok; list fallback offline (no tiles).
**QrLabel**: full-screen QR (batch_code) for the driver phone to scan + share/print link (`GET /qr/{batch_id}/label`).
**SensorMonitor**: live tiles temp/humidity/energy with bands + sparkline; per-batch danger-minutes gauge.
**NGO Offers**: card with food, kg, distance, pickup window, expiry countdown; accept/decline; ReceiveScan scans the driver's QR → RECEIVED event (offline-capable).
**Driver RouteList**: today's stops, big type, one-hand; StopDetail: call button, map deep link; ScanFlow: full-screen mobile_scanner, confirmation toast with captured time.
**Admin Models**: version list with metrics, drift status, promote/rollback (ADMIN API); Users CRUD; AuditLog with filters; SystemHealth (DB/Redis/mlserving/CV/sensors last-seen); AppConfig (flags, min version).
**EsgReport**: date range, generate (202 job polling), open/share PDF via share_plus.

## 8. UX standards
Thumb-reach: primary actions bottom, ≥ 44 dp targets; text ≥ 16 sp; contrast 4.5:1; status = icon + text (never colour alone); skeletons + empty + error states everywhere (error shows `code` + retry); snackbars for mutations; confirmation dialogs for approve/divert/override; Hindi strings for every label (intl ARB files, no hard-coded text); dark mode; demo font scale; haptics subtle; no fake precision — round sensibly, never render fields the API didn't return; offline banner with queued-count.

## 9. Performance budgets
Cold start < 2.5 s; release APK < 40 MB; capture→preview < 1 s mid-range Android (Redmi-class test device); lists 60 fps (ListView.builder + const widgets); device inference < 300 ms; lazy-load heavy screens; image cache with size caps; Riverpod autoDispose on detail screens.

## 10. Security
Tokens in flutter_secure_storage (Keychain/Keystore) only; no tokens in logs/shared_preferences; EXIF+GPS stripped at capture; no PII in outbox beyond user id; TLS only; certificate pinning optional post-SIH; logout-all button (revokes family); app never stores passwords; debug banners off in release builds.

## 11. Testing
Unit: outbox state machine, sync merge (ACCEPTED/DUPLICATE/REJECTED), refresh single-flight, **preprocess parity (golden vectors from cv/ CI — exact tensor equality incl. int8 quantization)**, formatters, DTO round-trip vs openapi. Contract: mocks vs openapi.json (same CI job as Go). E2E (patrol, real device): login→predict→surplus→capture(gate+preview)→decision→approve→offer→accept→route→offline scan→sync→verify→impact. Device matrix: mid Android (with Play services), one iPhone if built, one tablet; airplane-mode suite; push suite on real device.

## 12. Build order
Flutter project + FCM config on both demo phones → core/api + auth(+refresh) + Login → app shell + mock services → Dashboard + Prediction/WhatIf → Surplus + Capture flow (gate+preview+upload) → decision + approve → outbox + sync → Recipients + NGO offers + FCM → Route + Driver + offline scan → QR label + verify → Sensors/Alerts → Processing/Impact/EsgReport → Admin → hi localization + polish → release build + smoke test on both phones → patrol green.

## 13. Definition of done
Full judge flow on two real phones in mock and live modes with zero screen-level code changes; airplane-mode scan syncs and verifies green; FCM push arrives on a real device; mocks pass contract CI; preprocess parity test green; APK < 40 MB, cold start < 2.5 s; no dio calls outside `data/services/`; every screen has empty/error/offline states; Hindi complete.