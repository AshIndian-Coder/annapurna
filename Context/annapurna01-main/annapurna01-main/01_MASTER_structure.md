# 01 — MASTER FILE STRUCTURE (v3 FINAL — Go + Flutter)

Legend: `# does / needs`

```
SIH26234-Food-Waste/
├── README.md                 # pitch, diagram, 3-command quickstart
├── LICENSE                   # Apache-2.0 (keeps CV + app licensing clean)
├── Makefile                  # up seed test train eval bundle demo lint report reset
├── docker-compose.yml        # dev: postgres, redis, backend(Go), worker(Go), mlserving(Python), iot-simulator
├── docker-compose.prod.yml   # nginx TLS, healthchecks, resource limits
├── .env.example  .gitignore  .pre-commit-config.yaml
├── .github/workflows/        # ci.yml (lint, gen-drift, tests, build)  deploy.yml
│
├── database/   (P2)          # 04_DATABASE_*
│   ├── schema.md  erd.mmd
│   ├── migrations/           # golang-migrate SQL, ONE location (D14 v3)
│   │   └── 0001..0008_{up,down}.sql
│   ├── seeds/01..11_*.sql    # idempotent, FK order
│   ├── views/  scripts/      # reset_db.sh backup.sh export_training_data.py
│   └── tests/                # constraint + seed-minimum checks (run via backend test suite)
├── redis/      (P1+P2)       # 05_REDIS_* — conf + KEYS.md (catalogue + TTLs + Asynq keys)
├── backend/    (P1, Go)      # 03_BACKEND_*
│   ├── go.mod  Makefile  Dockerfile(multi-stage→distroless)  .env.example
│   ├── cmd/server/           # chi router, middleware, graceful shutdown
│   ├── cmd/worker/           # Asynq worker + cron scheduler
│   ├── cmd/seedcheck/        # QR seed-chain generation/validation (D30)
│   ├── internal/httpapi/     # oapi-codegen STRICT stubs from docs/api/openapi.json (contract-first)
│   ├── internal/{auth,rbac,store(pgx+sqlc),services,qrchain,mlclient,pushx,redisx,asynqx,processing,files,metrics}
│   ├── storage/uploads/  static/bundles/cv/
│   └── tests embedded (unit + api + contract + e2e)
├── mlserving/  (P1.5, Python) # 07 §15 — internal inference sidecar (D28)
│   ├── main.py (FastAPI, :8001, shared-secret)  routers/  auth.py
│   ├── adapters → imports ml/, cv/, integration/routing ONLY
│   └── requirements.txt  Dockerfile  tests/
├── ml/         (P4)          # 07_ML_TRAINING_* — LightGBM/CatBoost/fusion/anomaly (Python, unchanged)
├── cv/         (P5)          # 06_CV_* — Keras training; dual export (TFLite int8 device + ONNX server)
│   ├── training/ models/cv-vN/{model.keras, model_int8.tflite, model_int8.onnx, thresholds.yaml, bundle/}
│   ├── export/{to_tflite_int8.py, to_onnx.py, bundle.py}  parity/  eval/  fixtures/  tests/
├── mobile/     (P3, Flutter) # 02_MOBILE_* — Dart 3 app, all 4 roles (D19 v3)
│   ├── lib/{core,data,features,offline,cv,notifications,shared}  assets/cv-bundle/
│   ├── android/ ios/ (FCM config)  integration_test/  test/
│   └── pubspec.yaml  l10n.yaml  build.yaml(freezed/json_serializable)
├── integration/ (P5)          # iot-simulator/ processing/ routing/(OR-Tools, via sidecar) camera/ seed-data/
├── ops/                      # nginx, prometheus, grafana, runbook, demo scripts, scrcpy mirror
└── docs/
    ├── architecture/*.mmd  api/{API_CONTRACT.md,openapi.json,postman.json}
    ├── CONTRACT_DECISIONS.md  demo/  ppt/  evaluation/  model_cards/
    ├── compliance/{food_safety,privacy,limitations}.md
    └── patent/{INVENTION_DISCLOSURE.md,CLAIMS_DRAFT.md,PRIOR_ART_LOG.md,DRAWINGS/}
```

## Ownership and import rules (no gaps, no overlaps)
| Folder | Owner | May import / call | Must NOT |
|---|---|---|---|
| backend (Go) | P1 | HTTP→`mlserving` only; Postgres, Redis | ml, cv, integration directly |
| mlserving (Python) | P1.5 | `ml`, `cv`, `integration.routing` | backend, mobile, DB writes |
| ml | P4 | `integration.processing` (mirror) | backend, cv, mobile |
| cv | P5 | nothing project-internal | ml, backend, mobile |
| mobile (Flutter) | P3 | HTTP only via `data/services`; device CV only from `cv` bundle artifact | all Python/Go |
| database | P2 | — | — |

## Runtime topology (v3)
`phones (Flutter, 2+ roles) → nginx → backend (Go API) → {Postgres, Redis, mlserving (Python) → {ml lib, cv lib, OR-Tools}}`; `worker (Go + Asynq)` shares backend image, sends FCM push, calls mlserving for drift; `iot-simulator` posts to backend; bundle downloads: `app → GET /app/model-bundle → static file (sha256-verified)`; demo mirror: `scrcpy` (Android → projector).

## Ports
backend 8000 · mlserving 8001 (internal only) · postgres 5432 · redis 6379 · prometheus 9090 · grafana 3000

## Superseded from v2
`frontend/` web app (v1) and the Expo React Native plan (v2) are both gone — replaced by the Flutter app per D19(v3). The Python backend is reduced to `mlserving/` (inference only) per D28. No other deletions.