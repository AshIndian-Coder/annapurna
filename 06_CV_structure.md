# 06 — COMPUTER VISION STRUCTURE (P5, `cv/` — Keras training, dual export)

```
cv/
├── README.md  MODEL_CARD.md  requirements.txt  Makefile(data|train|calibrate|export|bundle|eval|bench|test)
├── configs/
│   ├── classifier.yaml      # backbone (efficientnet_lite0 | mobilenet_v3_large), img size, lr, epochs, aug, classes
│   ├── detector.yaml        # optional YOLOX-tiny / RTMDet-tiny (server-side only, via mlserving)
│   ├── thresholds.yaml      # blur, exposure, abstain band, OOD, operating point (written by calibrate.py)
│   └── classes.yaml         # GOOD, RISK, REJECTED, NOT_FOOD + spoilage cue taxonomy
├── data/                    # gitignored; DVC optional
│   ├── raw/{public,own_collected}/   interim/   processed/{train,val,test}/   splits.csv
│   ├── DATA_CARD.md         # sources, licences, labelling protocol, session ids
│   └── collection_protocol.md  # timed room-temp photos of rice, dal, sabzi, roti, paneer, fruit
├── training/
│   ├── build_dataset.py     # merge sources, dedupe (phash), group split by session/source
│   ├── dataset.py           # tf.data pipeline (train aug / eval deterministic)
│   ├── train_classifier.py  # Keras transfer learning: head then fine-tune, EarlyStopping on
│   │                        #   val macro-F1 AND unsafe-recall, cosine schedule, 3 seeds
│   ├── calibrate.py         # temperature scaling, abstain band, unsafe-recall operating point → thresholds.yaml
│   ├── ood_fit.py           # penultimate-embedding class-conditional Gaussians (Mahalanobis) → ood.npz
│   └── explain.py           # Grad-CAM (tf-keras-vis or manual) overlay PNG
├── models/                  # versioned artifacts per cv-vN: model.keras (checkpoint = source of truth),
│   │                        #   model_int8.tflite (device), model_int8.onnx (server), ood.npz, thresholds.yaml,
│   │                        #   metrics.json, card.json, bundle/
│   └── cv-vN/bundle/         # DEVICE bundle (D21 v3): model_int8.tflite, thresholds_device.yaml, labels.json,
│                            #   preprocess.json (resize policy + normalize + INT8 scale/zero_point from TFLite
│                            #   converter), manifest.json {version, sha256(tflite), sha256(thresholds), size_bytes}
├── inference/               # SERVER-side runtime (Python, called via mlserving)
│   ├── __init__.py           # exports cv_predict
│   ├── predict.py           # cv_predict(image_path,batch_id)->dict (contract, §6)
│   ├── validate.py quality_gate.py preprocess.py   # preprocessing constants == bundle/preprocess.json (fp32 path)
│   ├── classifier.py        # onnxruntime session (int8 ONNX), singleton, thread-safe
│   ├── ood.py calibrate.py heatmap.py postprocess.py
├── export/
│   ├── to_tflite_int8.py    # TFLiteConverter, full-integer PTQ, ~200 representative images,
│   │                        #   dumps input quantization scale/zero_point into bundle/preprocess.json
│   ├── to_onnx.py           # tf2onnx from the SAME checkpoint → model_int8.onnx
│   ├── bundle.py            # assembles cv-vN/bundle/ + copies bundle + onnx into backend/static/bundles/cv/
│   └── parity/  golden_vectors.py   # 10 fixtures → {input, tflite_out, onnx_out, torch/keras fp32 ref} JSON
├── eval/    evaluate.py  robustness.py  crossdomain.py  latency_bench.py  parity_report.py  report.md(generated)
├── fixtures/                # 10 known demo images + expected outcomes.json (committed, small)
└── tests/   test_validate test_quality_gate test_contract test_known_images test_abstain test_latency
            test_bundle_manifest test_parity_gate   # D21 gate: tflite vs onnx argmax ≥99%, mean |Δp| ≤ 0.03
```
Imports nothing from backend/ml/mobile. mlserving reaches it only via its adapter. The Flutter app consumes ONLY the compiled `bundle/` artifact (D21) — never Python. The optional detector runs server-side only; the phone runs the int8 classifier for gate+preview only (D20).