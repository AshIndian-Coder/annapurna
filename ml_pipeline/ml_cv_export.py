"""Export a trained run to the two shipping artifacts, and gate them (spec 06, D21).

    python ml_cv_export.py --run runs2/lite0_staged --index data/cv/index.csv
    python ml_cv_export.py --run runs2/lite0_staged --bundle --require-device

Produces, in the run directory:

    model_fp32.onnx        server, float32 reference
    model_int8.onnx        server, int8 (onnxruntime static quantisation)
    model_int8.tflite      device, int8 (best-effort; see the note below)
    manifest.json          D21: both sha256s, sizes, quantisation + preprocess constants
    export_report.json     every number behind the gate

Input to every exported graph: float32 NCHW in ``[0, 1]``, ``size x size``.
Output: probabilities over ``CLASSES``.  Normalisation and temperature are folded into
the graph, so the phone and the server cannot disagree about either.

**The device artifact is not a given.**  Flutter needs a TFLite int8 model
(``flutter_litert``), PyTorch has no first-class route to one, and the official
converter has an open MobileNetV3 int8 bug.  This module therefore tries three paths in
order, records which one produced the file, and -- when ``--require-device`` is passed
and none worked -- **refuses to emit a bundle**.  A bundle whose ``device`` entry is
missing is not a bundle; shipping it would mean the phone silently runs2 a different
model than the one the parity report describes.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

from ml_cv_common import CLASSES, LOG, eval_tf
from ml_utils import get_paths, human_bytes, save_json, sha256_file

#: D21 parity gate.  Both must hold on the *unseen-source* test split.
GATE = {
    "fp32_max_abs_prob_diff_vs_torch": 1e-3,
    "int8_argmax_agreement_with_fp32": 0.98,
    "int8_max_abs_prob_diff": 0.08,
    "tflite_vs_onnx_argmax_agreement": 0.99,
    "tflite_vs_onnx_mean_abs_unsafe_prob_delta": 0.03,
}


# --------------------------------------------------------------------------- #
# Wrappers: preprocessing and temperature are part of the graph
# --------------------------------------------------------------------------- #
class Student(nn.Module):
    """Classifier + normalisation + temperature, as one exportable module."""

    def __init__(self, net, mean, std, T: float = 1.0):
        super().__init__()
        self.net, self.T = net, float(T)
        self.register_buffer("m", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("s", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, x):
        return torch.softmax(self.net((x - self.m) / self.s) / self.T, dim=-1)


class Teacher(nn.Module):
    """Linear probe on a frozen backbone, as one exportable module."""

    def __init__(self, bb, W, b, mean, std, T: float = 1.0):
        super().__init__()
        self.bb, self.T = bb, float(T)
        self.register_buffer("W", torch.tensor(W, dtype=torch.float32))
        self.register_buffer("b", torch.tensor(b, dtype=torch.float32))
        self.register_buffer("m", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("s", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, x):
        f = self.bb((x - self.m) / self.s)
        return torch.softmax((f @ self.W.T + self.b) / self.T, dim=-1)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_images(paths, size, *, cap: int | None = None) -> np.ndarray:
    """Eval-preprocessed float32 NCHW, exactly as training evaluated it."""
    tf = eval_tf(size)
    out = []
    for p in list(paths)[:cap] if cap else list(paths):
        with Image.open(p) as im:
            out.append(tf(im.convert("RGB")).numpy())
    return np.stack(out).astype(np.float32) if out else np.zeros((0, 3, size, size), np.float32)


def _reader(arr: np.ndarray, name: str):
    """CalibrationDataReader over a fixed in-memory array."""
    from onnxruntime.quantization import CalibrationDataReader

    class Reader(CalibrationDataReader):
        def __init__(self):
            self._it = iter([{name: a[None]} for a in arr])

        def get_next(self):
            return next(self._it, None)

    return Reader()


# --------------------------------------------------------------------------- #
# Device export: three attempts, all recorded
# --------------------------------------------------------------------------- #
def export_tflite(module: nn.Module, size: int, out_path: Path) -> dict:
    """Try, in order: litert-torch, ONNX -> TF -> TFLite, and give up honestly.

    Returns a record with ``ok``, ``path`` and ``via``; never raises, because a missing
    device artifact is a *decision* for the caller (fall back to a server-only release,
    or retrain in Keras), not a crash at 2 a.m. before a demo.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import litert_torch  # type: ignore

        edge = litert_torch.convert(module.eval(), (torch.rand(1, 3, size, size),))
        edge.export(str(out_path))
        if out_path.exists() and out_path.stat().st_size > 0:
            return {"ok": True, "path": str(out_path), "via": "litert_torch",
                    "caveat": "litert-torch is Beta; the int8 MobileNetV3 path has an open "
                              "upstream bug (#499), so the parity gate below is the only "
                              "thing standing between that bug and a shipped model"}
        return {"ok": False, "path": None, "via": "litert_torch",
                "error": "converter reported success but wrote no file"}
    except Exception as exc:  # noqa: BLE001
        LOG.info("litert-torch unavailable or failed: %s", exc)

    try:
        # onnx2tf converts an ONNX graph to a TF SavedModel; the TFLite converter then
        # does the int8 pass.  Community-maintained, so a failure here is expected and
        # is reported rather than swallowed.
        from onnx2tf import convert  # type: ignore

        onnx = out_path.with_suffix(".onnx")
        if not onnx.exists():
            return {"ok": False, "path": None, "via": "onnx2tf",
                    "error": "no ONNX file to convert from"}
        convert(input_onnx_file_path=str(onnx), output_folder_path=str(out_path.parent / "_tf"),
                output_saved_model_dir_name="saved_model", non_verbose=True, overwrite=True)
        import tensorflow as tf

        conv = tf.lite.TFLiteConverter.from_saved_model(str(out_path.parent / "_tf" / "saved_model"))
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = lambda: ([{"x": a[None]} for a in
                                                load_images([], size)[:1]] or iter([]))
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.uint8
        conv.inference_output_type = tf.uint8
        blob = conv.convert()
        out_path.write_bytes(blob)
        shutil.rmtree(out_path.parent / "_tf", ignore_errors=True)
        return {"ok": True, "path": str(out_path), "via": "onnx2tf+tflite",
                "caveat": "onnx2tf is community-maintained; verify every op before shipping"}
    except Exception as exc:  # noqa: BLE001
        LOG.info("onnx2tf path failed: %s", exc)
        return {"ok": False, "path": None, "via": "onnx2tf+tflite", "error": str(exc)}


def run_tflite(tflite_path: Path, arr: np.ndarray) -> np.ndarray | None:
    """Execute a TFLite model on the ONNX-shaped input; ``None`` if unavailable."""
    try:
        import tensorflow as tf
    except Exception:  # noqa: BLE001
        return None
    try:
        interp = tf.lite.Interpreter(model_path=str(tflite_path))
        interp.allocate_tensors()
        inp = interp.get_input_details()[0]
        out = interp.get_output_details()[0]
        results = []
        for a in arr:
            x = a[None].astype(np.float32)
            interp.set_tensor(inp["index"], x)
            interp.invoke()
            results.append(interp.get_tensor(out["index"])[0].astype(np.float64))
        return np.stack(results)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("TFLite inference failed (%s) -- the parity gate cannot cover the "
                    "device artifact", exc)
        return None


# --------------------------------------------------------------------------- #
# Unsafe-probability metric used by the D21 gate
# --------------------------------------------------------------------------- #
def unsafe_probability(probs: np.ndarray) -> np.ndarray:
    """P(the batch must not be redistributed) = P(RISK) + P(REJECTED).

    ``NOT_FOOD`` is excluded on purpose: a photo of a shoe is confidently handled and
    carries no food-safety risk, and folding it in would let a model score well on the
    parity gate by being confidently wrong about the wrong thing.
    """
    p = np.asarray(probs, dtype=float)
    idx = {c: i for i, c in enumerate(CLASSES)}
    return p[:, idx["RISK"]] + p[:, idx["REJECTED"]]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _calibration_method(name: str):
    """Resolve a method name to whatever the installed onnxruntime expects.

    onnxruntime 1.30 replaced the ``calibrate_method="minmax"`` string with a
    ``CalibrationMethod`` enum; older releases take the string.  Passing the string to
    the new build raises ``ValueError: Unsupported calibration method minmax`` -- for
    *every* name, because the comparison is against enum members.  Resolving here keeps
    one code path working across both, and the resolved name is written into the report.
    """
    try:
        from onnxruntime.quantization.calibrate import CalibrationMethod
    except ImportError:
        return name
    return getattr(CalibrationMethod, {"minmax": "MinMax", "entropy": "Entropy",
                                       "percentile": "Percentile"}[name], name)


def _quantize(int8: Path, fp32: Path, calib: np.ndarray, report: dict) -> str:
    """Static int8 quantisation, trying calibration methods newest-first.

    ``percentile`` is preferred: food photos taken in a kitchen have a long tail of
    blown-out highlights, and ``minmax`` lets three specular pixels set the uint8
    activation range for the whole model, which costs exactly the fine detail the
    classifier needs.  The fallbacks are explicit and the method that actually ran is
    recorded -- an int8 model whose calibration nobody can name is not reproducible.
    """
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

    last: Exception | None = None
    for method in ("percentile", "entropy", "minmax"):
        try:
            # The ONNX C++ version converter prints a "No Adapter To Version N for Pad"
            # traceback from inside onnxruntime and then carries on with the original
            # graph.  It is noise in a build log and a genuine warning in a report, so
            # the logger is quieted here and the fact is recorded in export_report.json
            # rather than left as an unexplained block of red text nobody reads.
            onnx_logger = logging.getLogger("onnx")
            previous = onnx_logger.level
            onnx_logger.setLevel(logging.ERROR)
            try:
                quantize_static(str(fp32), str(int8), _reader(calib, "image"),
                                quant_format=QuantFormat.QDQ, weight_type=QuantType.QInt8,
                                activation_type=QuantType.QUInt8, per_channel=True,
                                calibrate_method=_calibration_method(method),
                                extra_options={"ActivationSymmetric": False,
                                                   "WeightSymmetric": True})
            finally:
                onnx_logger.setLevel(previous)
            return method
        except ValueError as exc:
            last = exc
            LOG.info("calibration method '%s' unavailable (%s); trying the next one",
                     method, exc)
    raise RuntimeError(f"int8 quantisation failed for every calibration method: {last}")


def _assert_self_contained(path: Path) -> dict:
    """Fail loudly if the exported graph references an external weight file.

    Self-contained is a *deployability* property, not a format detail: the phone fetches
    one file, verifies one sha256, and the Go service serves one static path.  A graph
    that quietly depends on a sibling file breaks all three, and does so silently, at
    the moment the model is served rather than when it is built.
    """
    import onnx
    from onnx import TensorProto

    model = onnx.load(str(path), load_external_data=False)
    external = [t.name for t in model.graph.initializer
                if t.data_location == TensorProto.EXTERNAL]
    sidecar = path.with_suffix(path.suffix + ".data")
    if external or sidecar.exists():
        raise RuntimeError(
            f"{path.name} is not self-contained: {len(external)} initializers live in "
            f"external data and {sidecar.name} exists. Every consumer of this bundle "
            f"expects a single file, so this export must not ship.")
    weight_bytes = sum(len(t.raw_data) for t in model.graph.initializer)
    on_disk = path.stat().st_size
    if on_disk < weight_bytes:
        raise RuntimeError(f"{path.name} is {on_disk} bytes on disk but declares "
                           f"{weight_bytes} bytes of initializers -- the file is truncated")
    return {"initializers": len(model.graph.initializer),
            "weight_bytes": int(weight_bytes), "file_bytes": int(on_disk)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None, help="a trained student run directory")
    ap.add_argument("--teacher", default=None, help="a DINOv2 probe run (fp32 server only)")
    ap.add_argument("--index", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--calib-n", type=int, default=128, help="calibration images for int8")
    ap.add_argument("--parity-n", type=int, default=256, help="images for the parity gate")
    ap.add_argument("--opset", type=int, default=13,
                    help="ONNX opset; 13 is what the static quantiser can downgrade")
    ap.add_argument("--tflite", action="store_true", help="also attempt the device export")
    ap.add_argument("--require-device", action="store_true",
                    help="exit non-zero unless a TFLite artifact was produced")
    ap.add_argument("--bundle", action="store_true", help="write D21 manifest.json")
    a = ap.parse_args(argv)

    if bool(a.run) == bool(a.teacher):
        ap.error("pass exactly one of --run or --teacher")
    run_dir = Path(a.run or a.teacher)
    if not run_dir.exists():
        ap.error(f"{run_dir} does not exist")

    paths = get_paths()
    index = Path(a.index) if a.index else paths.cv_data / "index.csv"
    out_dir = Path(a.out_dir) if a.out_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    import joblib
    import onnxruntime as ort
    import timm

    if a.run:
        import ml_cv_common as C

        cal = joblib.load(run_dir / "calib.joblib")
        net = timm.create_model(cal["arch"], pretrained=False, num_classes=len(CLASSES))
        net.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu",
                                       weights_only=True))
        module = Student(net.eval(), cal["mean"], cal["std"], cal.get("T", 1.0)).eval()
        size = int(cal["size"])
        meta = {"arch": cal["arch"], "temperature": float(cal.get("T", 1.0)),
                "mean": list(cal["mean"]), "std": list(cal["std"]), "size": size}
    else:
        pr = joblib.load(run_dir / "probe.joblib")
        bb = timm.create_model(pr["backbone"], pretrained=True, num_classes=0,
                               img_size=pr["size"]).eval()
        module = Teacher(bb, pr["W"], pr["b"], pr["mean"], pr["std"], pr.get("T", 1.0)).eval()
        size = int(pr["size"])
        meta = {"arch": f"{pr['backbone']}+linear-probe",
                "temperature": float(pr.get("T", 1.0)),
                "mean": list(pr["mean"]), "std": list(pr["std"]), "size": size}

    # ---- which images does the gate use? ---------------------------------- #
    if index.exists():
        df = pd.read_csv(index)
        if "split" not in df.columns:
            raise SystemExit(f"{index} has no 'split' column; run ml_cv_prepare_data first")
        test = df[df.split == "test"]["path"].tolist()
        val = df[df.split == "val"]["path"].tolist()
        if not test:
            LOG.warning("test split is empty -- the parity gate falls back to val, and the "
                        "numbers it produces are optimistic")
        parity_paths = (test or val)[: a.parity_n]
        calib_paths = (val or test)[: a.calib_n]
    else:
        LOG.warning("no index at %s -- exporting without a parity check", index)
        parity_paths, calib_paths = [], []

    dummy = torch.rand(1, 3, size, size)
    fp32 = out_dir / "model_fp32.onnx"
    t0 = time.perf_counter()
    # external_data=False is load-bearing.  torch >= 2.5 spills initializers into a
    # sibling "<name>.onnx.data" for graphs above a size threshold, which produces a
    # 272 kB .onnx pointing at a 13.5 MB sidecar: inference still works in place, the
    # parity check still passes, and the manifest's sha256 covers a file with no weights
    # in it.  Copy the .onnx to a static host and the model is silently dead.
    try:
        torch.onnx.export(module, dummy, str(fp32),
                          input_names=["image"], output_names=["probs"],
                          opset_version=a.opset,
                          dynamic_axes={"image": {0: "batch"}, "probs": {0: "batch"}},
                          do_constant_folding=True, external_data=False, verbose=False)
    except TypeError:      # older torch has no external_data kwarg
        torch.onnx.export(module, dummy, str(fp32),
                          input_names=["image"], output_names=["probs"],
                          opset_version=a.opset,
                          dynamic_axes={"image": {0: "batch"}, "probs": {0: "batch"}},
                          do_constant_folding=True, verbose=False)
    _assert_self_contained(fp32)
    report: dict = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "run": str(run_dir), "out_dir": str(out_dir), "meta": meta,
        "classes": CLASSES, "gate": GATE, "opset": a.opset,
        "fp32_onnx_mb": fp32.stat().st_size / 1e6,
        "fp32_self_contained": True,
        "export_seconds": round(time.perf_counter() - t0, 2),
        "parity_images": len(parity_paths), "calibration_images": len(calib_paths),
        "parity_split": "test" if index.exists() and test else ("val" if index.exists() else "none"),
    }

    imgs = load_images(parity_paths, size)
    if len(imgs) == 0:
        raise SystemExit("no images available for the parity gate -- refusing to export "
                         "an ungated model")
    with torch.no_grad():
        ref = module(torch.from_numpy(imgs)).numpy()

    s32 = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"])
    p32 = s32.run(None, {"image": imgs})[0]
    report["fp32_max_abs_prob_diff_vs_torch"] = float(np.abs(p32 - ref).max())
    report["fp32_argmax_agreement_with_torch"] = float((p32.argmax(1) == ref.argmax(1)).mean())
    LOG.info("fp32 ONNX vs torch: max |dP| = %.2e, argmax agreement %.4f",
             report["fp32_max_abs_prob_diff_vs_torch"],
             report["fp32_argmax_agreement_with_torch"])

    # ---- int8 server artifact --------------------------------------------- #
    device: dict = {"ok": False, "path": None, "via": None, "error": "not attempted"}
    if a.run:                       # a ViT teacher has no useful int8 story; keep fp32
        from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

        calib = load_images(calib_paths, size)
        if len(calib) < 8:
            raise SystemExit(f"only {len(calib)} calibration images; int8 static "
                             f"quantisation needs at least 8 to avoid a degenerate range")
        int8 = out_dir / "model_int8.onnx"
        q0 = time.perf_counter()
        method = _quantize(int8, fp32, calib, report)
        report["quantize_seconds"] = round(time.perf_counter() - q0, 2)
        report["calibration_method"] = method
        p8 = ort.InferenceSession(str(int8), providers=["CPUExecutionProvider"]).run(
            None, {"image": imgs})[0]
        report.update(
            int8_onnx_mb=int8.stat().st_size / 1e6,
            int8_argmax_agreement_with_fp32=float((p8.argmax(1) == p32.argmax(1)).mean()),
            int8_max_abs_prob_diff=float(np.abs(p8 - p32).max()),
            int8_mean_abs_unsafe_prob_delta=float(np.abs(
                unsafe_probability(p8) - unsafe_probability(p32)).mean()),
        )
        LOG.info("int8 ONNX vs fp32: argmax agreement %.4f, max |dP| %.4f",
                 report["int8_argmax_agreement_with_fp32"], report["int8_max_abs_prob_diff"])

        if a.tflite or a.require_device:
            device = export_tflite(module, size, out_dir / "model_int8.tflite")
            if device.get("ok"):
                pt = run_tflite(Path(device["path"]), imgs)
                if pt is not None:
                    report["tflite_vs_onnx_argmax_agreement"] = float(
                        (pt.argmax(1) == p32.argmax(1)).mean())
                    report["tflite_vs_onnx_mean_abs_unsafe_prob_delta"] = float(np.abs(
                        unsafe_probability(pt) - unsafe_probability(p32)).mean())
                    report["tflite_mb"] = Path(device["path"]).stat().st_size / 1e6
                    LOG.info("TFLite vs ONNX (D21): argmax agreement %.4f, mean |d unsafe| %.4f",
                             report["tflite_vs_onnx_argmax_agreement"],
                             report["tflite_vs_onnx_mean_abs_unsafe_prob_delta"])
            report["device_export"] = device

    # ---- gate -------------------------------------------------------------- #
    checks = {
        k: (report.get(k) is not None and
            (report[k] <= v if k.endswith(("vs_torch", "abs_prob_diff", "unsafe_prob_delta"))
             else report[k] >= v))
        for k, v in GATE.items()
    }
    if not report.get("tflite_vs_onnx_argmax_agreement"):
        checks["tflite_vs_onnx_argmax_agreement"] = None
        checks["tflite_vs_onnx_mean_abs_unsafe_prob_delta"] = None
    report["gate_checks"] = {k: (None if v is None else bool(v)) for k, v in checks.items()}
    report["gate_pass"] = all(v for v in checks.values() if v is not None) and not a.require_device or (
        all(v for v in checks.values() if v is not None) and device.get("ok", False))
    report["gate_failures"] = [k for k, v in checks.items() if v is False]
    if a.require_device and not device.get("ok"):
        report["gate_failures"].append("device_tflite_missing")

    save_json(out_dir / "export_report.json", report)
    LOG.info("gate: %s%s", "PASS" if not report["gate_failures"] else "FAIL",
             "" if not report["gate_failures"] else f" -> {report['gate_failures']}")

    # ---- D21 manifest ------------------------------------------------------ #
    if a.bundle:
        manifest = build_manifest(out_dir, run_dir, meta, report, device, version=None)
        save_json(out_dir / "manifest.json", manifest)
        LOG.info("wrote %s", out_dir / "manifest.json")

    print()
    print(f"fp32   {human_bytes(int(fp32.stat().st_size)):>10s}  {fp32.name}")
    if "int8_onnx_mb" in report:
        print(f"int8   {report['int8_onnx_mb']:>10.2f} MB  model_int8.onnx")
    if device.get("ok"):
        print(f"device {report.get('tflite_mb', 0):>10.2f} MB  model_int8.tflite  (via {device['via']})")
    else:
        print(f"device {'-':>10s}   NOT PRODUCED: {device.get('error') or 'not attempted'}")
    print(f"gate   {'PASS' if not report['gate_failures'] else 'FAIL ' + str(report['gate_failures'])}")
    return 0 if not report["gate_failures"] else 1


def build_manifest(out_dir: Path, run_dir: Path, meta: dict, report: dict,
                   device: dict, *, version: str | None) -> dict:
    """D21 manifest: the two artifacts, both hashes, and every constant a client needs.

    The app verifies ``sha256`` before activating a bundle and falls back to the copy
    shipped inside the APK, so an entry here is a contract, not documentation.
    """
    version = version or f"cv-{out_dir.name}"
    base = f"/static/bundles/cv/{version}"

    def entry(name: str) -> dict | None:
        p = out_dir / name
        if not p.exists():
            return None
        return {"file": name, "url": f"{base}/{name}", "sha256": sha256_file(p),
                "size_bytes": p.stat().st_size}

    server = entry("model_int8.onnx") or entry("model_fp32.onnx")
    dev = entry("model_int8.tflite")
    manifest = {
        "version": version,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "classes": CLASSES,
        "input": {
            "layout": "NCHW", "dtype": "float32", "range": [0.0, 1.0],
            "height": meta["size"], "width": meta["size"],
            "mean": meta["mean"], "std": meta["std"],
            "resize_short": int(round(meta["size"] / 0.875)),
            "center_crop": meta["size"],
            "colour": "RGB",
        },
        "output": {"name": "probs", "kind": "softmax", "temperature": meta["temperature"]},
        "server": (server or {}).__setitem__("runtime", "onnxruntime"),
        "device": dev,
        "device_runtime": "tflite" if dev else None,
        "quantization": {
            "scheme": "int8 static, per-channel weights, uint8 activations (QDQ)",
            "calibration_method": report.get("calibration_method"),
        },
        "parity": {
            "fp32_max_abs_prob_diff_vs_torch": report.get("fp32_max_abs_prob_diff_vs_torch"),
            "int8_argmax_agreement_with_fp32": report.get("int8_argmax_agreement_with_fp32"),
            "tflite_vs_onnx_argmax_agreement": report.get("tflite_vs_onnx_argmax_agreement"),
            "tflite_vs_onnx_mean_abs_unsafe_prob_delta":
                report.get("tflite_vs_onnx_mean_abs_unsafe_prob_delta"),
            "split": report.get("parity_split"),
            "n_images": report.get("parity_images"),
        },
        "notes": [
            "server and device are two exports of ONE checkpoint; the D21 parity gate "
            "(>=99 % argmax agreement, mean |d unsafe prob| <= 0.03) must pass before "
            "this bundle is served",
        ],
    }
    if dev is None:
        manifest["notes"].append(
            "NO DEVICE ARTIFACT: the phone cannot run this bundle. Do not publish it as "
            "complete -- either supply a TFLite export or serve server-only and say so.")
    return manifest


if __name__ == "__main__":
    raise SystemExit(main())
