# 07 — ML MODEL-TRAINING SPEC (v3 FINAL — best pipeline for this problem)

Read `01_MASTER_spec.md` first. Goal: calibrated demand forecasts → cost-optimal production → honest uncertainty → server-side safety fusion → anomaly detection, all reproducible with one command. **Nothing here runs on the phone (D20); the only on-device model is the CV bundle owned by `cv/` (06). Go consumes everything in this file through `mlserving/` (§15).**

## 1. Which models (pretrained / lightweight)
| Task | Pick | Why |
|---|---|---|
| Demand forecasting (primary) | **LightGBM** (quantile + L1/Tweedie objectives), ensemble with **CatBoost** (+ XGBoost) | Best accuracy per effort on tabular+calendar data, trains in seconds, no GPU, small artifacts, SHAP-friendly |
| Optional pretrained time-series | **Chronos-Bolt (tiny/mini, Apache-2.0)** zero-shot ensemble member/baseline per kitchen | Pretrained, light; useful when a kitchen has little history. Verify current model names/licences. N-HiTS/TFT not worth the complexity |
| Uncertainty | **Conformalized quantile regression (MAPIE or own)** | Distribution-free coverage guarantee |
| Production decision | **Newsvendor** on predicted quantile | Converts forecast → action; minimises expected cost |
| Safety fusion (tabular) | **LightGBM with monotone constraints** (+ isotonic calibration) | Risk must not decrease with more exposure; < 50 ms CPU in mlserving |
| Processing anomalies | **EWMA/CUSUM + IsolationForest** | Explainable, no labels needed |
| Image model | lives in `cv/` (Keras EfficientNet-Lite0 / MobileNetV3-Large); **one checkpoint → TFLite int8 (device) + ONNX int8 (server)** with the D21 parity gate | see 06 |
No deep nets for demand unless benchmark proves gain. Say so in the pitch: "simplest model that wins, measured".

## 2. Data strategy (critical for credibility)
1. **Public real benchmark** (e.g. a restaurant-visitors forecasting dataset; verify licence) proves the pipeline beats baselines on real noisy data.
2. **Calibrated simulator** `kitchen_simulator.py`: 2–3 years × 4 profiles. Signals: weekday/weekend, holidays/festivals (`holidays` lib + manual Indian calendar), exam weeks (hostel), month-end, menu popularity (per-item latent effect), weather, random attendance absence (beta noise), special-event shocks, drift (new menu season). Output column `data_source=SYNTHETIC`. `calibrate_to_real.py` matches noise/seasonality stats to the public benchmark.
3. **Real data** from pilots/DB export as it appears (`v_training_set`). Final model trained on real+synthetic with source flag; report metrics per source.
Minimum training columns: date, kitchen_id, day_of_week(ISO), meal_type, menu list, attendance (expected & actual), prepared_qty, consumed_qty, surplus_qty, waste_qty.
Target: `consumed_qty` (demand actually eaten). Note: censored demand (sold out) — if shortage flagged, model `consumed` as lower bound; include `ran_out` flag feature only for training-time analysis.

## 3. Features (only those available at inference)
Calendar: dow, is_weekend, month, week-of-year, holiday, festival, days-to-holiday, exam flag, month-end. Attendance: expected, ratio to 28-day mean. Lags of consumption/surplus (1,7,14,28) *same meal type*; rolling mean/median/std (7,14,28); same-weekday mean (4 weeks); trend slope; per-kitchen target encoding (out-of-fold!); menu multi-hot of top-N items + menu category counts + menu popularity score (OOF target encoding) + "new item" flag; weather (temp, rain) if available with offline fallback; prior-day waste. Guard: `test_no_leakage.py` recomputes features with data truncated at t and asserts equality.

## 4. Validation design
Rolling-origin (expanding window) with ≥5 folds, horizon = 1 day ahead (and 7-day report), gap = 0. Separate **calibration fold** (last 15 % before test) for conformal. Final hold-out = last 8 weeks, touched once. Stratified reporting by meal_type, weekday, kitchen profile, holiday vs normal.

## 5. Training stages (`make train`)
1. `preprocess` → cleaning report (rows in/out/reasons; clip negatives; dedupe; fill small gaps; flag outliers via robust z).
2. `features` → parquet + `feature_schema.json` (names, dtypes, order, defaults).
3. `baselines` → seasonal-naive(7), same-weekday median, 7-day moving avg: record WAPE.
4. `train` → LightGBM (objective `regression_l1` and `tweedie`), CatBoost, XGBoost with early stopping on fold.
5. `tune` → Optuna TPE, 100 trials, metric = fold-mean WAPE, search: num_leaves 15–127, learning_rate 0.01–0.1, min_data_in_leaf 10–100, feature_fraction 0.5–1, bagging 0.5–1, lambda_l1/l2 0–10; seed fixed; pruner Median.
6. `quantile` → LightGBM quantile models at 0.1, 0.5, 0.9 (+ 0.7/0.8 used by newsvendor).
7. `ensemble` → ridge stacking on OOF predictions (non-negative weights).
8. `conformal` → CQR: scores on calibration fold, adjust quantile bands for target coverage 80 % (P10–P90); store `conformal.json`.
9. `newsvendor` → critical ratio `q* = Cu/(Cu+Co)`; `Cu` cost of under-production per kg (default 1.0), `Co` cost of surplus per kg (default 1.0, configurable per kitchen; ratio typically chosen so q*≈0.6–0.8). `recommended_production = Q(q*)`; `expected_surplus = E[max(Q−D,0)]`; risk LOW/MED/HIGH by `expected_surplus/Q` thresholds (5 %, 12 %) and historical surplus frequency — thresholds documented, not "validated".
10. `evaluate` → WAPE, MAE, RMSE, bias, pinball loss, interval coverage, width; per-slice; vs baselines; write `metrics.json`.
11. `backtest_policy` → simulate on hold-out: policy production = newsvendor; compare with the kitchen's actual prepared quantity: kg waste avoided, shortage days, service level. **This is the headline metric** ("x kg/week saved, shortage rate ≤ y %"). Include sensitivity to Cu/Co.
12. `explain` → SHAP; global importance + per-request top-5 drivers via TreeSHAP (cached in mlserving).
13. `register` → `registry/models/demand/vN/`; update symlink `models/demand_model.pkl` only after promotion gate passes.

## 6. Promotion gate
Candidate promoted only if: holdout WAPE ≤ best baseline × 0.85 (relative improvement ≥ 15 %); coverage within 80 ±5 %; no slice worse than baseline by > 5 %; inference < 50 ms; tests green; human (ADMIN/P4) confirms. Rollback = flip pointer (pub/sub `model:active:*` refresh).

## 7. Public inference contract (consumed by mlserving, unchanged shape)
`predict_demand(payload) -> dict`: predicted_consumption (p50 int), recommended_production, expected_surplus, surplus_risk, prediction_interval{p10,p50,p90,coverage_target}, recommended_quantile, top_drivers, confidence (deprecated proxy = clip(1 − (p90−p10)/(2·p50),0,1)), model_version, data_source. Validates input (raises `ValueError` with code: NEGATIVE_ATTENDANCE, UNKNOWN_MEAL_TYPE, BAD_DOW, EMPTY_HISTORY_OK_IF_DB). Cold start: unseen kitchen → global model + profile prior. Loads artifact once; thread-safe. Deterministic.

## 8. Safety fusion model (server-side only — D20)
Inputs: `cv_unsafe_prob` (**server** ONNX value — the device preview must never be a feature), `danger_zone_minutes` (from `time_temp.py`: minutes with temp in 5–60 °C band *unless food is served hot/cold-chain compliant* — thresholds in `fusion.yaml`, cite FSSAI/WHO/USDA guidance after verification), `hours_to_expiry`, `hours_since_prepared`, `food_category_risk`, `humidity_mean`, `cv_image_quality`. Hard rules first: expired → REJECTED; danger minutes > budget (default 240 min cumulative, configurable per category) → REJECTED. Model: LightGBM with `monotone_constraints` (+ for unsafe_prob, danger minutes, hours_since_prepared; − for hours_to_expiry), trained on simulated+labelled scenarios (labels from a transparent risk function + expert rules, then noise) — **state plainly that fusion labels are rule-derived, not microbiological ground truth**. Calibrate (isotonic), set threshold for unsafe recall ≥ 0.98; HOLD band between thresholds; output reason codes (`CV_RISK`, `DANGER_MINUTES_n`, `EXPIRY_NEAR`, `CATEGORY_HIGH_RISK`). `assess_safety(batch, cv_result, sensor_summary) -> {status, reasons, danger_zone_minutes, hours_to_expiry, model_version}`. Target latency < 50 ms CPU.

## 9. Processing analytics
Canonical formulas live in Go `internal/processing` (D9 v3); `integration/processing` mirrors them for ML features, and a **committed parity fixture (inputs→expected outputs) is asserted in both Go and Python CI**. `efficiency_score` D3. Anomalies: per-kitchen EWMA (λ=0.3, 3σ) on loss %, downtime %, energy/kg + CUSUM for drift + IsolationForest (contamination 0.05) on the 3-vector; output `{metric, value, expected, z, method, severity}`. Minimum 14 days before enabling IF; before that z-score vs global priors.

## 10. Monitoring and feedback
`feedback` rows (actual vs predicted) → nightly `performance_tracker` (Asynq `drift_check` cron → sidecar `/v1/drift`) computes rolling WAPE and coverage; `drift.py` PSI (>0.2 warn, >0.3 alert) on key features and target residual; `retrain_trigger` sets a flag (human approves). Model version visible in app UI and admin screens. **Telemetry loop:** compare `quality_checks.device_preview` vs the server decision over time to publish the device/server agreement metric (eval doc, master §14); device preview never feeds any model.

## 11. Metrics you may claim (only after measuring)
WAPE/MAE/RMSE vs baselines on real benchmark and hold-out; interval coverage; policy backtest kg saved; fusion unsafe-recall & precision; processing anomaly precision on injected anomalies (synthetic, labelled as such); device/server agreement % across runtimes. Targets: WAPE 6–10 % on realistic noise (≈ "90–94 % accuracy" in layman terms, never say 98 % unless on low-noise synthetic and labelled), coverage 78–82 % for 80 % band, device/server agreement ≥ 99 % (same checkpoint, both int8).

## 12. Reproducibility
Pinned requirements; global seed; `make train` from clean clone produces identical `metrics.json` ±1e-6; artifacts carry git SHA and data hash; `MODEL_CARD.md` filled (intended use, data, metrics, limitations, fairness across kitchen types).

## 13. Tests
Contract keys/types/ranges; leakage; newsvendor monotonic in Cu/Co; conformal coverage on synthetic iid; fusion monotonicity (grid test) + **fusion rejects any request that tries to pass device_preview as input**; processing parity fixture vs Go; registry load from clean process; latency; exceptions carry codes.

## 14. Definition of done
`make train eval backtest` prints the headline table; promoted artifact loads in mlserving; `predict_demand` sample from API contract returns valid JSON through the full chain (Go → sidecar → ml); docs/evaluation files generated; no unlabelled synthetic claims; parity/telemetry hooks for the device bundle in place.

## 15. Serving (mlserving/ — the D28 sidecar contract)
Internal FastAPI on `:8001`, compose-network only, `X-Internal-Token: $MLSERVING_TOKEN` required on every route. It imports ONLY `ml.predict`, `ml.quality.inference`, `cv.inference`, `integration.routing` — no DB, no Redis writes. Routes (request/response shapes are EXACTLY the domain functions they wrap):
| Route | Wraps | Timeout | Notes |
|---|---|---|---|
| POST /v1/predict-demand | predict_demand | 2 s | contract §7 |
| POST /v1/assess-safety | assess_safety | 2 s | contract §8 |
| POST /v1/cv-predict | cv_predict {image_path, batch_id} | 8 s | reads shared `storage/uploads` volume; semaphore 2, threadpool |
| POST /v1/build-route | route_adapter.build_route | 10 s | OR-Tools VRPTW; haversine fallback |
| POST /v1/detect-anomalies | processing_anomaly | 5 s | |
| POST /v1/drift | monitoring.drift | 30 s | called by Asynq cron daily |
| GET /health, /ready | — | — | ready = models loaded (demand, fusion, cv ONNX warm) |
Errors: JSON `{code,message}` typed (e.g. `ML_INVALID_INPUT`, `CV_ERROR`, `ROUTE_INFEASIBLE`) — the Go client maps to AppError codes, never leaks stack traces. `FEATURE_MOCK_*` on the Go side bypasses the sidecar entirely (fixtures ship in Go tests). Lifespan: load registry artifacts + CV model once, warm with dummy inputs. `/metrics` (prometheus) internal only. The sidecar is a thin (~150-line) wrapper — all logic stays in `ml/` and `cv/`, unchanged by this pivot.