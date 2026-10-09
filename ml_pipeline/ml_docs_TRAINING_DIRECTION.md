# TRAINING DIRECTION — what is built, what is measured, what is not

PS 26234. This file records the state of the models honestly, including the parts that
do not work yet. Read section 8 before quoting any number to a judge.

## 1. The one-paragraph version

Demand forecasting, safety fusion, processing-efficiency anomaly detection, matching,
routing and impact accounting are implemented, tested and runnable on CPU. The image
classifier is implemented with a modern recipe and a resumable trainer, but it has been
developed against a **synthetic photo fixture**, not a photo library, so no accuracy
claim is attached to it. The single biggest missing input is data; everything else is
engineering that is ready for it.

## 2. What the problem statement asks, and where it is answered

| PS requirement | Module | Status |
|---|---|---|
| Predict demand and surplus in real time | `ml_demand_*` | done, measured on synthetic data |
| Identify items nearing expiry via sensors + images | `ml_quality_time_temp`, `ml_quality_train_fusion`, `ml_cv_*` | models done; CV accuracy unmeasured |
| Connect surplus to NGOs and secondary buyers | `ml_logistics_match` | done |
| Optimise transportation routes | `ml_logistics_route` | done (VRPTW + haversine fallback) |
| Monitor processing efficiency | `ml_analytics_processing_anomaly` | done |
| Detect inefficiencies: overproduction, loss, downtime, energy | `ml_analytics_processing_anomaly` + `ml_demand_*` | done |
| Sustainability analytics and ESG reports | `ml_analytics_sustainability` | done |
| Support smart production planning | `ml_demand_train` (what-if quantile) | done |

## 3. The demand model, and the decision it feeds

Recipe: leakage-safe features → two naive baselines → five expanding-window folds for
model selection → LightGBM quantile ensemble (P10/P50/P75/P90/P95) → CQR conformalisation
on a dedicated calibration fold → one untouched 56-day holdout → newsvendor policy
backtest → promotion gate.

Measured on 540 synthetic days: **WAPE 0.180**, versus 0.334 for the same-weekday median
and 0.433 for seasonal naive. 80 % band covers 83.7 % of outcomes.

Three decisions worth defending, because each was a deliberate choice against the obvious
one:

1. **Run-out days are dropped from training.** When the kitchen runs out, recorded
   consumption is a *lower bound* on demand. Training on it teaches the model to
   under-predict precisely when the kitchen is busiest.
2. **The conformal fold is separate from the calibration-of-threshold fold.** A model
   whose quantile levels were fitted on the same data that set its interval correction
   produces intervals that look calibrated and are not.
3. **The service floor overrides the newsvendor ratio.** See below.

### Why the operating point is P90 and not the cost optimum

The frontier in `metrics.json → policy_backtest.frontier` is the honest version of this
decision:

| quantile | surplus (kg) | shortage days | expected cost |
|---|---|---|---|
| 0.50 | 5 035 | 214 | 22 310 |
| 0.75 | 10 280 | 108 | 17 230 |
| 0.85 | 13 770 | 65 | 18 040 |
| **0.90** | **15 670** | **52** | 19 030 |

At Cu/Co = 3 the cost optimum is q\*=0.75, which still runs short on ~15 % of days. A
kitchen running out is a service failure, not a linear cost, so the shipped rule is
`q* = max(Cu/(Cu+Co), service_level)`. The report states which constraint bound and the
implied Cu/Co, rather than quietly entering Cu=9 and calling it an economic result.

## 4. The safety fusion, and why it is monotone

The fusion model is a **monotone** LightGBM over `{CV unsafe probability, cumulative
danger-zone minutes, hours to expiry, food-category prior}`:

* more visual risk → never lowers the safety verdict,
* more time above the safe band → never lowers it,
* less time to expiry → never lowers it.

Hard rules run first (expired → REJECTED; danger-zone budget exceeded → REJECTED), and
low confidence abstains to HOLD. `ml_tests_test_fusion_monotonic.py` asserts monotonicity
by perturbing one input at a time and requiring the output never to move the wrong way.
That test is not decoration — monotonicity is a claim a judge will test by hand.

Note: LightGBM 4.7 rejects monotone constraints on a *quantile* objective, so the
monotone stage is an `L2` regression on the logit and the quantile levels come from the
calibration wrapper. Both halves are asserted in the test.

## 5. The image model

Recipe: EfficientNet-Lite0 (ImageNet) → frozen-trunk head warm-up → full fine-tune with
warm-up + cosine decay, EMA weights, mixup/cutmix, label smoothing, AMP and gradient
clipping. Batch size is fitted to the card and halved on OOM, which is what keeps 8 GB
of VRAM comfortable.

**Checkpointing.** Every epoch writes `last.pt` with model, EMA, optimizer, scheduler,
scaler, epoch, global step, best score, and the RNG states for Python, NumPy and Torch.
`--resume` continues from the exact step. Verified: a 2-epoch run resumed to 4 epochs
kept its best epoch and its selection. A run killed mid-epoch loses that epoch, not the
whole run — the honest trade for checkpoint-on-epoch rather than on-batch.

**Data discipline.** The test split is whole *sources*. `ml_cv_prepare_data.py` removes
perceptual-hash near-duplicates before drawing splits, and refuses to build a report that
claims a headline number when a group or a duplicate cluster straddles a boundary.

## 6. The two-artifact problem (D21)

Flutter needs TFLite; PyTorch exports ONNX. So the same checkpoint produces two artifacts
and they are gated against each other before a bundle is written:

* ONNX fp32 vs PyTorch: max |Δp| < 1e-3 — this catches a wrong normalisation constant.
* ONNX int8 vs fp32: ≥ 98 % argmax agreement — this catches a bad calibration set.
* TFLite vs ONNX: ≥ 99 % argmax agreement and mean |Δ unsafe prob| ≤ 0.03 — this is the
  claim the app's manifest makes, so it is the one that gets measured.

`--require-device` fails the export if no TFLite artifact was produced. The converters
involved have open bugs (the MobileNetV3 int8 issue in `litert-torch`), which is exactly
why the gate exists and why the check is on agreement rather than on "the file exists".

## 7. Bugs this review actually found

Kept here because "what went wrong" is the part of a training document worth reading.

| Found | Effect | Fix |
|---|---|---|
| `actual_diners`, `waste_qty` were model inputs | WAPE 0.028 — oracle-level, because they are post-hoc outcomes | excluded via `POST_HOC_COLUMNS`; the honest number is 0.180 |
| global-median gap filling | 4 features leaked the future for early rows | past-only expanding median, then lag, then 0 |
| shrinkage prior from "earliest 30 rows" | unstable tie-break, changed with the frame length | fixed constant prior; the leakage guard caught it |
| `seasonal_naive(history, dates)` returned one column per kitchen in the *history* | silently predicted the wrong kitchen | query frame with keys, positionally aligned |
| `expected_surplus_quantiles` indexed a dict of arrays by `0.01` | `KeyError` on any batch | row-wise grid interpolation |
| ONNX export spilled weights to `model_fp32.onnx.data` | 272 kB `.onnx` pointing at a 13.5 MB sidecar; the manifest hashed a file with no weights in it | `external_data=False` + a self-containment assertion |
| opset 17 | static quantisation died on "no adapter to version 17 for Pad" | opset 13 |
| Grad-CAM hooked `getattr(net, "forward_features")` | `AttributeError` — it is a bound method | wrap the bound method, restore in `finally` |
| Grad-CAM hooked the classifier | captured `(B, 1280)` — timm already pooled, nothing to localise | explained at `forward_features` output |
| `blur_min = 0.055` on a 0–1 scale | 36× too high; rejected every photograph | 0.0018 (100/255²), verified with a blur ladder |
| eval-report threshold sweep computed the complement of the spec rule | reported `unsafe_recall = 0.0` for a perfect model | the sweep now calls the same `evaluate()` the trainer uses |
| `python a.py b.py` as one pipeline stage | ran `a.py`, put `b.py` in `sys.argv`, reported success | one stage per script |
| UTF-8 crash in a dependency's printer | `UnicodeEncodeError` on a checkmark killed a completed export | streams forced to UTF-8 with replacement |

## 8. What is NOT established — read before quoting anything

1. **No real photo library has been used.** The CV numbers in this repository come from
   a synthetic fixture and mean nothing about food photographs. The first honest CV
   number requires `data/cv/raw/<kitchen>/<CLASS>/` from at least two real kitchens.
2. **No 95 % image accuracy is claimed.** The safety-relevant metric is unsafe recall
   at an operating point, and a claim of "95 % accuracy" for a food-safety classifier is
   the wrong claim even if it were true — it hides how the errors are distributed.
3. **All numbers are synthetic.** Every artefact carries `data_source: SYNTHETIC`.
4. **The waste-reduction percentage is a simulation result.** It compares the model's
   plan with a simulated manager who already uses same-weekday history and attendance.
5. **The device artifact may not exist yet.** With no TFLite converter installed, the
   export produces server artifacts and refuses to write a complete bundle.

## 9. The ordered path from here

1. **Collect photos.** 1 500–2 000 across 4 classes from ≥ 2 kitchens, several sessions
   each, and hours-since-cooking as a deliberate axis. This is the bottleneck; nothing
   else matters until it exists.
2. `ml_cv_prepare_data.py --test-sources kitchen_b --require-unseen-test-source`, then
   `ml_cv_train_cls.py` for a real run, then `ml_cv_eval_report.py`. If unsafe recall is
   below 0.98, collect more photos — do not lower the target.
3. `ml_cv_export.py --tflite --bundle --require-device` to close D21.
4. `ml_demand_train.py --tune` on real kitchen history once any exists.
5. Benchmark the demand model against a public real dataset (licence permitting) so the
   baseline-versus-LightGBM gap is measured on real noise rather than simulator noise.
6. `pip freeze > ml_requirements.lock` after a run whose numbers you intend to quote.

## 10. Questions to have ready

* **"How do you prevent unsafe donations?"** A camera cannot detect bacteria. Three
  independent signals must agree — visible deterioration, cumulative time above the safe
  temperature band, and time to expiry — fused by a model constrained to be monotone, and
  any disagreement abstains to a human. The phone can only demand a retake; it can never
  issue a safety verdict.
* **"What happens with no network?"** The scan, the capture and the waste log are
  offline-first. The batch syncs later, and the custody chain is recorded in
  server-arrival order, so a scan captured offline is provably the same scan.
* **"How do you know the accuracy is real?"** The test split is whole kitchens, not
  random photos; near-duplicates are removed by perceptual hash before the split is
  drawn; and the report refuses to print a headline number when any of that is not true.
* **"Why these two model artifacts?"** One checkpoint has to satisfy an ONNX server and a
  TFLite phone. They are gated against each other at ≥ 99 % argmax agreement, and a
  bundle without both does not ship.