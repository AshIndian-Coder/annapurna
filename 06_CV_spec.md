# 06 — COMPUTER VISION SPEC (v3 FINAL — Keras training, dual export, server-authoritative with on-device gate)

## 1. Honest scope
A camera cannot detect bacteria or certify safety. The module estimates **visible deterioration** (mould, discolouration, sliminess, drying, separation, foreign objects). It outputs decision support that is fused with time-temperature exposure and expiry (mlserving → ml/quality) and always ends with a human. The phone's preview is advisory only. State this on the slide; judges reward it.

## 2. Model recommendation (v3 — Flutter-driven)
| Need | Recommended | Why | Alternatives / notes |
|---|---|---|---|
| **Main classifier (GOOD / RISK / REJECTED / NOT_FOOD)** | **EfficientNet-Lite0 (Keras, ImageNet weights), 224 px** | designed for TFLite (SE blocks removed), clean native int8 TFLite quantization, Apache-2.0 (TF), ~4–5 M params | **MobileNetV3-Large (tf.keras.applications)** as the accuracy-hungry alternative — also TFLite-native. Pick by your own benchmark |
| **Localisation of defects (optional, server-side only)** | **YOLOX-Tiny/Nano** (Apache-2.0) or **RTMDet-Tiny** (Apache-2.0) | permissive licence, small; runs via mlserving, never on device in v1 | **Avoid Ultralytics YOLOv5/v8/11** (AGPL-3.0) for a patent-path product. Verify licences before use |
| **OOD / unknown food guard** | Mahalanobis distance on the penultimate embedding (server) | free, calibrated with val set | Optional DINOv2-small (Apache-2.0) embeddings |
| **Explainability** | Grad-CAM on the conv head (server) | shows "where" without box labels | heatmap stays server-side (D20) |
| **Image-quality gate** | Classical OpenCV (Laplacian variance, histogram) | deterministic, 2 ms; SAME constants run on the phone from `thresholds_device.yaml` | |
**Why Keras and not PyTorch (v3):** the phone runs TFLite, and native TF→TFLite int8 quantization is mature. The PyTorch path (timm → ONNX server + `litert_torch`/ai_edge_torch → TFLite device) is documented as an alternative, **but ai_edge_torch has an open MobileNetV3 int8 bug (google-ai-edge/litert-torch #499, open as of Jan 2026) — verify it is fixed before choosing that path.**

## 3. One checkpoint, two runtimes, one parity gate (v3 core, D20/D21)
- **Source of truth:** `model.keras` checkpoint. Export pipeline: `to_tflite_int8.py` (full-integer PTQ, ~200 representative images; records the input tensor **scale/zero_point** into `preprocess.json`) and `to_onnx.py` (tf2onnx) — both from the SAME checkpoint, versioned together in `cv/models/cv-vN/bundle/`.
- **Server (authoritative):** int8 ONNX via onnxruntime (mlserving, threadpool) — full cascade (§4), heatmap, OOD, fusion evidence, storage in `quality_checks`, `evidence_hash` into the QR chain.
- **Device (advisory, from the bundle):** int8 TFLite via **`flutter_litert` classic Interpreter** (the maintained TFLite runtime; `tflite_flutter` is unmaintained — do not use). Quality gate (blur/exposure) + preview probabilities only. The device CANNOT emit final REJECTED/ELIGIBLE; its UI vocabulary is "Looks OK / Possible risk / Not sure".
- **Preprocessing parity:** `bundle/preprocess.json` ships resize policy, normalization constants AND the TFLite int8 quantization scale/zero-point. `mobile/lib/cv/preprocess.dart` reads constants from the manifest — never hardcodes. The Dart preprocess quantizes to int8 with those constants; the Go/Python preprocess uses the fp32 ONNX path. CI parity tests feed the committed golden fixtures through BOTH runtimes.
- **D21 parity gate (ships or it doesn't):** TFLite vs ONNX argmax agreement ≥ 99% AND mean |Δunsafe_prob| ≤ 0.03 on golden fixtures + test set. `test_parity_gate` runs in CI; `make bundle` refuses to update the active pointer otherwise.

## 4. Server pipeline (cascade with abstention)
1. `validate`: extension + magic bytes JPG/PNG, ≤ MAX_UPLOAD_MB, decodable, dims ≥ 224; else typed error (IMAGE_UNSUPPORTED / UNREADABLE / TOO_LARGE).
2. `quality_gate`: variance of Laplacian < τ_blur → `image_quality=BLURRY`; mean luminance <40 or >220 or >30 % clipped → DARK/OVEREXPOSED. Return `visual.status=RISK, reason="Image unusable—retake"`, confidence 0 (never GOOD on a bad image).
3. `preprocess`: BGR→RGB, resize (shorter side 256, centre crop 224), ImageNet normalise; identical constants in train/inference (fp32 ONNX path) and mirrored (quantized) on device.
4. `classifier` (int8 ONNX): softmax → temperature-scaled p(GOOD), p(RISK), p(REJECTED), p(NOT_FOOD).
5. `ood`: Mahalanobis distance d; d > τ_ood or p(NOT_FOOD) high → `ood=true`, status RISK, reason "Not recognised as food / unusual", requires human.
6. Decision: `unsafe_prob = p(RISK)+p(REJECTED)`. If `unsafe_prob ≥ τ_unsafe` (operating point for unsafe recall ≥ 0.98 on validation) → status = argmax{RISK,REJECTED}; elif `max_prob < τ_conf` → abstain → RISK ("uncertain"); else GOOD. REJECTED only if p(REJECTED) ≥ τ_reject.
7. `heatmap`: Grad-CAM overlay PNG (saved under uploads, URL returned); cue text from top CAM region colour stats.
8. Return contract (§6).

## 5. Device-side contract (bundle vN — what the phone may do)
```json
{"device_preview": {"quality": "OK|BLURRY|DARK|OVEREXPOSED",
  "preview_status": "GOOD|RISK|UNKNOWN", "unsafe_prob": 0.31,
  "device_model_version": "cv-v1", "device_latency_ms": 74}}
```
Rules: `thresholds_device.yaml` ⊆ `thresholds.yaml` (gate τ's + preview bands only); no OOD stats on device (UNKNOWN covers it); `preview_status` never reaches the fusion model or any decision field — stored in `quality_checks.device_preview` for telemetry/eval ONLY (D20). The eval report includes "device/server agreement %" measured **across runtimes** (TFLite int8 device vs ONNX int8 server, same checkpoint) — report the measured number, not the expectation.

## 6. Output contract (mlserving wraps as `visual`)
```python
def cv_predict(image_path: str, batch_id: str) -> dict:
    return {
     "batch_id": batch_id, "quality_status": "RISK", "risk_level": "MEDIUM", "confidence": 0.84,
     "reason": "Visible discoloration detected", "detections": [{"label":"quality_risk","confidence":0.84,"bbox":None}],
     "requires_human_approval": True, "model_version": "cv-v1",
     # additive, optional:
     "probabilities": {"GOOD":0.1,"RISK":0.84,"REJECTED":0.05,"NOT_FOOD":0.01},
     "unsafe_prob": 0.89, "image_quality":"OK", "ood": False, "heatmap_path": "...", "latency_ms": 112 }
```
`risk_level`: LOW if GOOD and unsafe_prob<0.1; MEDIUM for RISK; HIGH for REJECTED or unsafe_prob ≥0.9. `requires_human_approval` always true. Never raise to callers for model errors: raise typed `CVError(code)`; mlserving → Go maps to `CV_UNAVAILABLE`.

## 7. Data plan (biggest lever on real accuracy)
**Public starters (verify licences and current availability):** Kaggle fresh-vs-rotten fruit & vegetable sets; Fruits-360 (clean fruit, good for pretraining features); Food-101 and UEC-Food256/Indian food sets (fresh appearance, food recognition; NOT_FOOD negatives from COCO/Places). Treat public rotten-fruit sets as auxiliary—their easy backgrounds inflate accuracy.
**Own data (mandatory for credibility):** 1,500–3,000 photos of Indian institutional food. Protocol: for each dish (rice, dal, sabzi, roti, curd/paneer dishes, fruit) photograph under 3 lightings and 2 phones at t=0, 4, 8, 12, 24 h at room temperature and refrigerated; label by visible-change rubric (not microbiology) by two annotators; keep `session_id`. Dispose of spoiled food safely; do not taste-test. Add augmentation of real canteen trays/steel plates/banana leaf. **Capture with the app itself** (D25 pipeline: 1600 px, q80) so train-time images match inference-time images.
**Classes:** GOOD (no visible issue), RISK (early change: sheen loss, dryness, slight discolouration, odd separation), REJECTED (mould, slime, strong discolouration, foreign matter), NOT_FOOD (empty tray, hands, random).
**Splits:** group by `session_id`+`source`; test set contains a source never seen in training (cross-domain test). Dedupe with perceptual hash across splits.

## 8. Training recipe (Keras)
- Init ImageNet weights (EfficientNet-Lite0). Stage 1: freeze base, train head 5 epochs lr 1e-3. Stage 2: unfreeze, lr 1e-4 base / 1e-3 head, AdamW wd 0.02 (or Adam), cosine decay, 25 epochs, batch 64, EarlyStopping on val macro-F1 **and** unsafe-recall (restore best).
- Loss: categorical cross-entropy, label smoothing 0.05 + class weights (or focal γ=2); optional asymmetric cost (penalise GOOD-predicted-for-REJECTED 5×) via sample weights.
- Augmentation (tf.data): RandomResizedCrop(0.6–1), flip, small rotation, brightness/contrast ±25 %, **hue jitter ≤ 0.03 (colour is the signal)**, Gaussian blur/noise, JPEG quality 40–95 (mirrors the app's q80), light cutmix (0.1). Eval: deterministic centre crop (no TTA — latency).
- Seeds fixed, 3 runs, report mean±std. Commit `metrics.json` + `card.json` per version.
- Calibrate: temperature scaling on val; pick τ_unsafe on val so unsafe recall ≥ 0.98; verify on test (report both). Fit OOD stats on train embeddings; τ_ood at 95 % in-dist retention.
- Export: `to_tflite_int8.py` (full-integer PTQ, representative set ~200) → `model_int8.tflite`; `to_onnx.py` (tf2onnx) → `model_int8.onnx`; compare Keras fp32 vs TFLite vs ONNX accuracy drop (< 1 pt each) and latency; `make bundle` assembles bundle + manifest + golden vectors + **runs the D21 parity gate** before updating the active pointer.

## 9. Evaluation (publish in docs/evaluation/cv_results.md)
Macro-F1, per-class recall, **unsafe recall & precision at operating point**, confusion matrix, PR curves, ECE, abstention rate, OOD AUROC, cross-domain test, robustness (brightness ±, blur, JPEG 30, 1/2 res), latency p50/p95 (server ONNX on a modest CPU ≤ 300 ms incl. preprocess; device TFLite < 300 ms mid-range Android), model size, **device/server agreement % across runtimes (D21 gate numbers)**. Realistic expectations: 97–99 % on same-source public data; 85–93 % macro-F1 on unseen cooked-food sources. Report the lower numbers—they are more believable.

## 10. Fixtures and tests
`fixtures/` 10 images + `outcomes.json`: fresh (GOOD), early-change (RISK), mouldy (REJECTED), blurry, dark, non-food, txt-renamed-jpg, corrupt, oversize, tiny. Tests: contract keys/types exact; batch_id preserved; `requires_human_approval` true; model_version present; blurry → not GOOD; corrupt → CVError; deterministic across 3 runs; latency with generous CI bound; clean process import; **bundle manifest sha256s match files; thresholds_device ⊆ thresholds; golden vectors regenerated deterministically; D21 parity gate green.**

## 11. Integration
Go backend → mlserving → `cv.inference.cv_predict` (timeout 8 s, threadpool, semaphore 2). Upload saving happens in Go (`internal/files`); the sidecar reads the shared `storage/uploads` volume path. The Flutter app sees the Go API's JSON (`visual` block, heatmap URL) + its own advisory preview. P4 consumes `unsafe_prob` (server value, never the device preview) as a fusion feature. Model version shown in app UI and stored in `quality_checks`. Bundle served from `backend/static/bundles/cv/` with the version pointer in `/app/config` (D21).

## 12. Patent/novelty notes
Novel = the *gated cascade with calibrated abstention tied to an unsafe-recall operating point* + fusion with time-temperature exposure + **the edge-gated evidence-bound capture (master claim 3): on-device rejection of unusable images before upload, while the authoritative, hash-chained decision stays server-side** — not "a CNN classifies rotten food", which is prior art. The multi-runtime parity gate is an engineering control, also document it as an embodiment detail. Record the operating-point procedure and ablations (CV only vs CV+exposure; with/without device gate) in evaluation docs as evidence of technical effect.

## 13. Definition of done
`make train eval export bundle` reproduces committed metrics; `cv_predict` works in a clean process; TFLite int8 < 15 MB (device bundle), ONNX int8 < 25 MB; unsafe recall ≥ 0.98 on test at operating point with reported precision; abstain + OOD + quality gate tested; heatmap works; fixtures pass; **D21 parity gate green in CI**; Dart preprocess parity test green; CV README lists exact dataset licences.