# ml_pipeline — the models behind SIH26234

Two halves. The **CPU half** (this file) needs no GPU, no PyTorch and no TensorFlow.
The **image half** needs an NVIDIA card and its own requirements file.

```
demand      forecast consumption and pick a production quantity      ml_demand_*
quality     is this batch safe to redistribute?                      ml_quality_*, ml_cv_*
processing  where is the factory losing money?                        ml_analytics_processing_*
logistics   who gets the food, and in what order                     ml_logistics_*
sustainability  what did it all achieve?                            ml_analytics_sustainability
```

## Setup

```
cd annapurna/ml_pipeline
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/WSL/macOS: source .venv/bin/activate
pip install -r ml_requirements.txt
```

For the image half on an RTX 5050 (Blackwell, sm_120), install torch **first** from the
cu128 index — see [ml_requirements-cv.txt](ml_requirements-cv.txt).

## Run

```
python ml_run_all.py --list          # the whole plan, including the GPU stages
python ml_run_all.py                 # every CPU stage
python ml_run_all.py --with-cv       # ...and the image stages
python ml_run_all.py --from demand   # resume after a failure
```

Each stage is its own process, so a failure names the stage that died instead of leaving
a half-written artefact. `tests` runs first: a broken invariant stops everything before
an hour of training is spent.

## What each stage produces, and where it lands

| Stage | Artefact | Consumer |
|---|---|---|
| `demand` | `registry/models/demand/vN/` + `card.json` | `mlserving` → `POST /predict-demand` |
| `shelf-life`, `quality` | `registry/models/{shelf_life,fusion}/vN/` | `/quality/check`, `/sensors/*` |
| `processing` | anomaly model + per-day KPIs | `/processing/analytics` |
| `sustainability` | impact + ESG report | `/analytics/impact`, `/reports/esg` |
| `logistics-*` | solver timings, match and route results | `/match`, `/route` |
| `cv-train` | `runs/<name>/{model.pt,calib.joblib,metrics.json,preds_*.json}` | `/quality/check` |
| `cv-export` | `model_fp32.onnx`, `model_int8.onnx`, `manifest.json` | `/app/model-bundle` |
| `cv-report` | `runs/<name>/eval_report.json` | judges, model cards |

## The demand half in one paragraph

Features are built so that **every value for a row at date `t` comes from data before
`t`** — lags on a reindexed daily grid (so a lag-7 really is 7 calendar days, not 7
observations), expanding target encoding, and past-only gap filling. That is not a
convention, it is enforced: `ml_demand_features.assert_no_leakage` rebuilds the frame
with the future removed and requires every column to match, and it runs in the `tests`
stage. It has already caught four real leaks during development.

Then: two naive baselines, five expanding-window folds for model selection, a LightGBM
quantile ensemble at P10/P50/P75/P90/P95, CQR conformalisation on a **separate
calibration fold**, one untouched holdout, and a policy backtest that replays the
newsvendor production plan against what the kitchen actually cooked.

### Measured, on 540 days of SYNTHETIC data

| | WAPE | coverage (target 0.80) | bias |
|---|---|---|---|
| **LightGBM + CQR** | **0.180** | 0.837 | −1.7 kg |
| same-weekday median | 0.334 | — | — |
| seasonal naive (7 d) | 0.433 | — | — |

Policy backtest: **77.6 % of surplus avoided** versus the historical plan, at a 12 %
shortage rate.

**Where the remaining 18 % goes, and why you should say so before a judge asks.**
The simulator makes consumption an exact function of *realised* attendance. Handed that,
an oracle scores WAPE 0.000. A kitchen decides production the night before, when only
*expected* attendance exists. So essentially all of the residual error is turnout
uncertainty, which no night-before forecast can remove. Saying this unprompted is worth
more than the number.

### The operating point is a business decision, and the code shows its work

`Cu/(Cu+Co)` on the newsvendor cost gives q\*=0.75 at Cu=3 — and still runs out of food on
~15 % of days, because the newsvendor optimum only knows about *linear* costs and a
kitchen running out is not a linear cost. So the shipped rule is
`q* = max(Cu/(Cu+Co), service_level)`, the report says which constraint bound, and
`metrics.json → policy_backtest.frontier` prints the whole waste-versus-shortage curve.
That table, not a single number, is the answer to "how do you know this trade-off is
right".

## The image half in one paragraph

Photos of the same dish, taken ten times in one session, are one observation. So splits
are made by **whole source** (a kitchen the model has never seen) and by **dish group**,
and perceptual-hash near-duplicates are removed before the split is drawn — a random
photo split overstates cross-domain accuracy by 5–15 points on food imagery.

Training is staged: a frozen-trunk warm-up on the head, then a full fine-tune with
cosine decay, warm-up, EMA weights, mixup/cutmix, label smoothing, AMP and gradient
clipping. It writes a **full-state checkpoint every epoch** — model, EMA, optimizer,
scheduler, scaler, epoch, and the RNG states for Python/NumPy/Torch — so
`--resume` continues from the exact step, not from a warm start.

### The gate that decides whether a model ships

```
unsafe recall >= 0.98 on the unseen-source test split
```

Not accuracy. A model can have 97 % accuracy and release 3 % of unsafe batches, and those
are not the same claim. `ml_cv_eval_report.py` prints the confusion matrix, the
hold-threshold sweep, the recommended operating point, and the cost-weighted loss against
two alternatives — **always HOLD** and **always ELIGIBLE**. A model that does not beat
"always hold everything" has no business shipping, and the report says so.

### Two artifacts, one checkpoint (D21)

The server runs ONNX int8. The phone runs TFLite int8 (`flutter_litert`). Both are
exports of the same weights, so `ml_cv_export.py` gates them against each other:
**≥99 % argmax agreement and mean |Δ unsafe probability| ≤ 0.03**, measured on the test
split, before a `manifest.json` is written.

`--require-device` makes a missing TFLite artifact a hard failure. A bundle whose
`device` entry is absent is not a bundle — the phone would silently run something other
than what the parity report describes.

## Honest-claims policy

Allowed: *"WAPE 0.180 on a 56-day holdout, 18× better than the same-weekday median"*,
*"unsafe recall 0.98 at τ on the kitchen_b test split of N photos"*, *"device/server
agreement 99.2 % across runtimes from one checkpoint"*.

Forbidden: *"certified safe"*, *"guaranteed"* , any accuracy from a random split, any
number whose test split shares a source with train, any synthetic number presented as a
real-kitchen result.

Two claims the code refuses to make for you, on purpose:

* A camera cannot detect bacteria. The CV model estimates **visible deterioration** only.
  It is fused with time-temperature exposure and always ends with a human. The fusion,
  the abstention and the human step are load-bearing, not decoration.
* `data_source: SYNTHETIC` is attached to every metric produced from a simulator. The
  pipeline is measured; the kitchen is not.

## Layout

| File | Role |
|---|---|
| `ml_utils.py` | paths, atomic writes, deterministic seeding, device/dtype, timers, provenance |
| `ml_cv_common.py` | preprocessing contract, loaders, EMA, mixup/cutmix, metrics, checkpoints |
| `ml_cv_train_cls.py` | the classifier trainer (staged, resumable, VRAM-fitting) |
| `ml_cv_prepare_data.py` | pHash dedupe, group-stratified splits, the leakage audit |
| `ml_cv_export.py` | ONNX fp32/int8, D21 gate, `manifest.json` |
| `ml_cv_eval_report.py` | the report a judge reads |
| `ml_cv_quality_gate.py` | blur/exposure/glare gate (phone + server) |
| `ml_cv_ood.py` | Mahalanobis out-of-distribution detection |
| `ml_cv_gradcam.py` | evidence heatmaps and `evidence_hash` for the custody chain |
| `ml_demand_features.py` | leakage-safe features + the executable leakage assertion |
| `ml_demand_model.py` | quantiles, CQR, newsvendor, `predict_demand` |
| `ml_demand_train.py` | baselines, rolling CV, evaluation, backtest, promotion gate |
| `ml_tests_test_demand.py` | 22 executable checks on the demand half |

## Known limits

* The simulators are the only data. Every number above is a **pipeline check**, not a
  result about a real kitchen.
* The TFLite path depends on community converters with open bugs. The parity gate is
  load-bearing, not ceremonial.
* `portion_kg` and the Cu/Co ratio are assumptions, not measurements. Replace them with
  the kitchen's real figures before quoting a waste-reduction percentage.