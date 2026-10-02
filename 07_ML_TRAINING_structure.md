# 07 — ML / MODEL TRAINING STRUCTURE (P4, `ml/` + `mlserving/`)

```
ml/
├── README.md  MODEL_CARD.md  DATA_CARD.md  pyproject.toml  requirements.txt  Makefile
│   # make data | features | train | tune | calibrate | eval | backtest | explain | register | drift | test
├── configs/   demand.yaml  fusion.yaml  anomaly.yaml  costs.yaml(Cu,Co,carbon factor)
├── data_gen/  kitchen_simulator.py  profiles.yaml(hostel,corporate,hospital,caterer)  calibrate_to_real.py
├── data/      raw/  external/(public benchmark)  interim/  processed/(parquet)  splits.json
├── demand/
│   ├── preprocess.py  calendar.py  weather.py(optional)  features.py
│   ├── baselines.py   # seasonal-naive, moving avg, same-weekday median
│   ├── train.py       # LightGBM/CatBoost/XGBoost, rolling-origin CV
│   ├── tune.py        # Optuna
│   ├── quantile.py    # P10..P90
│   ├── conformal.py   # CQR calibration
│   ├── ensemble.py    # stacking
│   ├── newsvendor.py  # production quantile from costs
│   ├── evaluate.py    # WAPE MAE RMSE bias pinball coverage per-slice
│   ├── backtest_policy.py  # kg waste avoided vs history
│   └── explain.py     # SHAP top_drivers
├── quality/           # tabular only — images live in cv/ (D13); fusion runs SERVER-SIDE only, via mlserving (D20)
│   ├── time_temp.py  food_categories.py  safety_rules.py  fusion_features.py
│   ├── train_fusion.py  calibrate.py  inference.py(assess_safety)
├── analytics/
│   ├── waste_analysis.py  processing_analysis.py(imports integration.processing mirror — D9 v3 parity fixture)
│   ├── processing_anomaly.py  impact.py
├── monitoring/  drift.py  retrain_trigger.py  performance_tracker.py
├── registry/models/{demand,fusion,anomaly}/vN/   # model.joblib feature_schema.json metrics.json conformal.json card.json
├── predict.py   # predict_demand(payload)  (stable — consumed by mlserving, NOT by Go)
├── train.py     # orchestrates all stages
├── models/demand_model.pkl   # symlink → promoted registry version (legacy contract)
├── notebooks/   # exploration only
└── tests/  test_predict_contract test_no_leakage test_newsvendor test_conformal_coverage
            test_time_temp test_fusion_monotonic test_processing_formulas test_registry_load

mlserving/                     # P1.5 — the ONLY Python service (D28). Internal, shared-secret, :8001.
├── main.py                    # FastAPI app: routers, X-Internal-Token check, /health /ready
├── routers/  demand.py  quality.py  cv.py  routing.py  anomalies.py  drift.py  health.py
│                              # POST /v1/predict-demand  /v1/assess-safety  /v1/cv-predict
│                              # POST /v1/build-route      /v1/detect-anomalies  /v1/drift
├── adapters.py                # imports ONLY: ml.predict, ml.quality.inference, cv.inference,
│                              #   integration.routing.route_adapter (never DB, never backend)
├── error_codes.py             # typed errors → JSON {code,message} for the Go client to map
├── requirements.txt  Dockerfile(python-slim, mounts ml/ cv/ integration/ + storage/uploads)
└── tests/                     # contract tests mirroring 07 §15 (timeouts, auth, shapes)
```
v3 note: nothing here moves to the phone. The only on-device model is the CV TFLite bundle owned by `cv/` (06). Demand, fusion, routing and anomaly inference stay on the server — the phone is a thin client plus an advisory CV preview (D20). Go never imports Python; it speaks HTTP to `mlserving/` only.