# 02 — MOBILE APP STRUCTURE (P3, Flutter / Dart 3)

```
mobile/
├── pubspec.yaml             # pinned deps; analysis_options.yaml strict lint
├── l10n.yaml                 # flutter_localizations + intl, locales en/hi
├── build.yaml                # freezed / json_serializable codegen
├── android/  ios/            # app id, FCM: google-services.json / GoogleService-Info.plist, permissions
├── lib/
│   ├── main.dart  app.dart   # ProviderScope, bootstrap: config → auth → router
│   ├── router.dart           # go_router, role-aware guarded routes, deep links (surplus/{id}, match/{id}, route/{id})
│   ├── core/
│   │   ├── api/              # dio + interceptors: auth header, refresh-on-401 single-flight queue,
│   │   │                     #   Idempotency-Key (=client_event_id), X-App-Version, multipart, 20 s timeout, GET retry ×2
│   │   ├── errors/           # AppError{status,code,message}; typed handling
│   │   ├── theme/  constants/  validators/  (mirrors of 08 enums + zod-equivalent checks)
│   │   └── result/           # sealed Result<T> for service calls
│   ├── data/
│   │   ├── dtos/             # json_serializable models = API payloads (contract-tested)
│   │   ├── services/         # auth kitchen prediction planning waste surplus quality matching route qr
│   │   │                     # sensor alert processing analytics report admin sync app
│   │   │                     # ONLY network layer (golden rule); mock switch: USE_MOCKS → fixtures identical to 08
│   │   └── mocks/            # fixtures from docs/api; contract test vs openapi.json
│   ├── features/             # Riverpod providers + screens + widgets per feature
│   │   ├── auth/             # Login (demo-role buttons), biometrics post-SIH
│   │   ├── kitchen/         # dashboard prediction what_if waste surplus_{list,create} quality_{capture,result}
│   │   │                     # recipients route_map qr_label sensors alerts processing impact esg_report
│   │   ├── ngo/              # offers offer_detail history receive_scan
│   │   ├── driver/           # route_list stop_detail scan_flow
│   │   └── admin/            # users models(promote/rollback) audit_log system_health app_config
│   ├── offline/
│   │   ├── db/               # Drift: outbox, kv_cache, pending_uploads
│   │   ├── outbox.dart       # enqueue(client_event_id, kind, payload, dependsOn) → QUEUED→SENDING→
│   │   │                     #   DONE | FAILED(retry×5) | DEAD
│   │   └── sync.dart         # POST /sync/batch ordered replay (QR per-batch FIFO), merge results
│   ├── cv/
│   │   ├── litert.dart       # flutter_litert classic Interpreter (int8) — loaded from bundle (D21)
│   │   ├── preprocess.dart   # resize/normalize/quantize with scale+zero_point FROM manifest.json
│   │   ├── quality_gate.dart # Laplacian variance + luminance (same τ as thresholds.yaml)
│   │   └── preview.dart      # chip GOOD|RISK|UNKNOWN only — never final (D20)
│   ├── notifications/
│   │   ├── fcm.dart          # firebase_messaging: token, permission, foreground/local bridges
│   │   └── channels.dart     # flutter_local_notifications; deep-link routing per event type
│   └── shared/widgets/       # StatCard IntervalBand(p10-p50-p90) TrendChart(fl_chart) Sparkline RiskGauge
│                            #   Timeline SafetyDecisionCard SurplusCard RecipientCard StopCard QrView
│                            #   Skeleton EmptyState ErrorState StatusBadge OfflineBanner Toast Sheet
├── assets/cv-bundle/         # fallback copy: model_int8.tflite + manifest.json (D21)
├── test/                     # unit: outbox reducer, sync merge, preprocess parity (golden vectors), dtos, formatters
├── integration_test/         # demo_flow_test.dart (patrol or flutter integration_test) + offline_scan_test.dart
└── screenshots/ (README)     # for the SIH submission deck
```

Rules: only `data/services/` performs HTTP; screens never call dio. `cv/` may only produce preview + gate output. Offline queue lives in `offline/`, used via a `mutate()` wrapper — screens call the same API online or offline.