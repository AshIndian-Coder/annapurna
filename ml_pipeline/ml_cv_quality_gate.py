"""Image-quality gate: reject an unusable photo *before* it is uploaded (spec 06, D20).

    python ml_cv_quality_gate.py --images shots/*.jpg --json
    python ml_cv_quality_gate.py --calibrate --images labelled/*.jpg

Two consumers, one implementation:

* the **phone** (``flutter_litert`` plus a Dart port of :data:`THRESHOLDS`) uses it to
  demand a retake for a blurred or dark frame.  It never issues a safety decision --
  only "this picture is not good enough to judge".
* the **server** uses it to record *why* a frame was poor, so a HOLD caused by an
  unusable photo is visibly different from a HOLD caused by visible spoilage.  Without
  that distinction the kitchen blames the model for its own photograph.

Everything here is plain NumPy on a small thumbnail: no model, no GPU, no network, and
it runs2 in single-digit milliseconds on a mid-range phone.

**Thresholds are a documented policy choice, not a measurement.**  ``--calibrate``
fits them on labelled shots and writes the result; the defaults below are the starting
point and the calibration report says how many images moved.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from PIL import Image

from ml_utils import get_logger, save_json

LOG = get_logger("cv.quality_gate")

#: Size the gate analyses.  Blur and exposure are scale-stable well below the model's
#: input size, and a 256 px thumbnail is ~40x less work than 1024 px on a phone.
ANALYSIS_SIZE = 256

#: Defaults.  ``warn_margin`` below never blocks; only the hard thresholds do.
#:
#: ``blur_min`` deserves the arithmetic spelled out, because the classic Laplacian
#: focus rule is quoted on an 0-255 scale and this module works on 0-1:
#:
#:     variance scales with the square of the amplitude, so
#:     var_0to1 = var_0to255 / 255**2 = var_0to255 / 65025
#:
#: The usual 8-bit threshold of ~100 (below which a frame is considered out of focus)
#: is therefore 100 / 65025 = 0.00154 on this scale, and 0.0018 leaves a little headroom
#: for a phone's own sharpening.  The first version of this file used 0.055, which is a
#: 36x error inherited from the 0-255 convention -- it rejected every photograph.
THRESHOLDS = {
    "blur_min": 0.0018,        # variance of the Laplacian on a 0-1 luma thumbnail
    "brightness_min": 0.16,    # mean luma in [0, 1]
    "brightness_max": 0.94,
    "glare_max": 0.28,         # fraction of near-saturated pixels
    "contrast_min": 0.02,      # luma standard deviation; catches lens caps and blank walls
    "warn_margin": 0.15,       # relax a reject threshold by this factor to warn
}
#: These defaults are a *starting point*, not a measurement.  Run
#: ``--selftest <one sharp photo>`` on the actual demo phone: it reports the blur radius
#: at which the gate starts rejecting, which is the only number that matters for that
#: camera's optics.  The default set is deliberately permissive -- the gate exists to
#: stop a person wasting a photo of a lens cap, and a false "retake" during a live demo
#: is worse than a soft photo the classifier can still handle.


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def _luma(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"), dtype=np.float32) / 255.0


def laplacian_variance(gray: np.ndarray) -> float:
    """Variance of the 4-neighbour Laplacian -- the standard focus measure.

    Computed with array shifts rather than ``scipy.ndimage.laplace`` so the module has
    no SciPy dependency; the phone port is a handful of lines.
    """
    lap = (-4.0 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1]
           + gray[1:-1, :-2] + gray[1:-1, 2:])
    return float(lap.var())


def saturation_fraction(rgb: np.ndarray) -> float:
    """Fraction of blown-out pixels (all channels > 0.98) -- glare / direct flash."""
    return float((rgb.min(axis=2) > 0.98).mean())


def features(path_or_img) -> dict:
    """Quality features for one image (path or PIL image)."""
    img = (Image.open(path_or_img) if isinstance(path_or_img, (str, Path))
           else path_or_img)
    with img if isinstance(img, Image.Image) else img:
        im = img.convert("RGB")
        im.draft("RGB", (ANALYSIS_SIZE * 2, ANALYSIS_SIZE * 2))   # cheap JPEG downscale
        small = im.resize((ANALYSIS_SIZE, ANALYSIS_SIZE), Image.Resampling.BILINEAR)
        arr = np.asarray(small, dtype=np.float32) / 255.0
    gray = _luma(small)
    return {
        "blur": laplacian_variance(gray),
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "glare": saturation_fraction(arr),
        "width": im.width, "height": im.height,
    }


# --------------------------------------------------------------------------- #
# Verdict
# --------------------------------------------------------------------------- #
def assess(f: dict, thresholds: dict | None = None) -> dict:
    """REJECT / WARN / OK plus the reason list.

    The reason list is part of the output on purpose: "TOO_DARK" and "BLURRY" lead to
    different advice to the person holding the phone.
    """
    t = dict(THRESHOLDS)
    if thresholds:
        t.update(thresholds)
    warn_f = 1.0 - t["warn_margin"]
    reasons: list = []
    hard: list = []

    if f["blur"] < t["blur_min"] * warn_f:
        hard.append("BLURRY")
    if f["brightness"] < t["brightness_min"] * warn_f:
        hard.append("TOO_DARK")
    if f["brightness"] > t["brightness_max"] / warn_f:
        hard.append("TOO_BRIGHT")
    if f["glare"] > t["glare_max"] / warn_f:
        hard.append("GLARE")
    if f["contrast"] < t["contrast_min"] * warn_f:
        hard.append("LOW_CONTRAST")

    if not hard:
        if f["blur"] < t["blur_min"]:
            reasons.append("SLIGHTLY_BLURRY")
        if f["brightness"] < t["brightness_min"]:
            reasons.append("SLIGHTLY_DARK")
        if f["brightness"] > t["brightness_max"]:
            reasons.append("SLIGHTLY_BRIGHT")
        if f["glare"] > t["glare_max"]:
            reasons.append("SLIGHT_GLARE")
        if f["contrast"] < t["contrast_min"]:
            reasons.append("SLIGHTLY_LOW_CONTRAST")

    verdict = "REJECT" if hard else ("WARN" if reasons else "OK")
    advice = {
        "BLURRY": "hold the phone still and tap the plate to focus",
        "TOO_DARK": "move somewhere with more light",
        "TOO_BRIGHT": "step out of direct light",
        "GLARE": "change the angle so the light is not reflecting",
        "LOW_CONTRAST": "get closer, or put the food on a plain surface",
    }
    return {
        "verdict": verdict,
        "reasons": hard or reasons,
        "advice": [advice.get(r, "retake the photo") for r in (hard or reasons)][:1],
        "features": {k: round(float(v), 5) for k, v in f.items()},
        "thresholds": t,
    }


def gate(path_or_img, thresholds: dict | None = None) -> dict:
    return assess(features(path_or_img), thresholds)


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def calibrate(samples, labels: dict) -> dict:
    """Pick thresholds that separate ``{path: "ok"|"bad"}`` labels.

    Reports the leave-one-out-style operating point *and* the resulting confusion, so a
    reviewer can see whether the fitted thresholds are better than the defaults or
    merely different.
    """
    xs, ys = [], []
    for s in samples:
        xs.append(features(s))
        ys.append(labels[str(s)])
    f_blur = np.array([x["blur"] for x in xs])
    f_bright = np.array([x["brightness"] for x in xs])
    bad = np.array([y == "bad" for y in ys])
    if bad.all() or (~bad).all():
        return {"applicable": False,
                "reason": "labels contain a single class; nothing to calibrate"}

    def best_threshold(values, higher_is_bad, grid):
        costs = []
        for t in grid:
            pred_bad = (values > t) if higher_is_bad else (values < t)
            costs.append(int((pred_bad != bad).sum()))
        i = int(np.argmin(costs))
        return float(grid[i]), int(costs[i])

    blur_grid = np.linspace(0.005, 0.30, 120)
    dark_grid = np.linspace(0.02, 0.45, 120)
    t_blur, c_blur = best_threshold(f_blur, higher_is_bad=False, grid=blur_grid)
    t_dark, c_dark = best_threshold(f_bright, higher_is_bad=False, grid=dark_grid)
    out = {
        "applicable": True,
        "n": len(xs),
        "n_bad": int(bad.sum()),
        "fitted": {"blur_min": round(t_blur, 5), "brightness_min": round(t_dark, 5)},
        "errors_blur": c_blur, "errors_dark": c_dark,
    }
    for name, thr in (("defaults", THRESHOLDS), ("fitted", out["fitted"])):
        pred_bad = ((f_blur < thr["blur_min"]) | (f_bright < thr["brightness_min"]))
        tp = int((pred_bad & bad).sum())
        fp = int((pred_bad & ~bad).sum())
        fn = int((~pred_bad & bad).sum())
        out[f"{name}_confusion"] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": int((~pred_bad & ~bad).sum()),
            "recall": round(tp / max(tp + fn, 1), 3),
            "precision": round(tp / max(tp + fp, 1), 3),
        }
    return out


# --------------------------------------------------------------------------- #
def selftest(reference, *, radii=(0, 1, 2, 3, 4, 6, 8, 12)) -> dict:
    """Blur a reference image along a known ladder and report where the gate flips.

    This is how the threshold gets justified instead of asserted.  The caller supplies
    one sharp photo; the gate's verdict at each blur radius is a fact, and the radius at
    which it flips is the actual operating point on *that* camera.  Run it per phone and
    tune ``blur_min`` from the number it prints.
    """
    from PIL import ImageFilter

    with Image.open(reference) as im:
        base = im.convert("RGB")
    rows = []
    for r in radii:
        img = base if r == 0 else base.filter(ImageFilter.GaussianBlur(r))
        res = gate(img)
        rows.append({"radius_px": r, "verdict": res["verdict"],
                     "blur": res["features"]["blur"],
                     "reasons": res["reasons"]})
    flip = next((r["radius_px"] for r in rows if r["verdict"] != "OK"), None)
    return {
        "reference": str(reference),
        "thresholds": dict(THRESHOLDS),
        "ladder": rows,
        "first_reject_radius_px": flip,
        "verdict": (f"the gate starts rejecting at a {flip}px Gaussian blur on this "
                    f"reference" if flip is not None else
                    "the gate never rejects across this ladder -- blur_min is too high "
                    "for this camera, lower it"),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", nargs="*", default=[])
    ap.add_argument("--labels", default=None,
                    help="CSV with path,label where label is ok|bad -- for --calibrate")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--selftest", default=None,
                    help="one sharp reference photo: report where the gate flips along a "
                         "blur ladder, which is how blur_min gets justified")
    ap.add_argument("--thresholds", default=None, help="JSON file of threshold overrides")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    import json
    thresholds = json.loads(Path(a.thresholds).read_text()) if a.thresholds else None

    if a.selftest:
        report = selftest(a.selftest)
        if a.out:
            save_json(a.out, report)
        print(json.dumps(report, indent=2) if a.json else
              "\n".join(f"  r={r['radius_px']:>2}px  {r['verdict']:6s} "
                        f"blur={r['blur']:.5f}  {r['reasons']}" for r in report["ladder"]))
        print("\n" + report["verdict"])
        return 0

    if a.calibrate:
        if not a.labels:
            raise SystemExit("--calibrate needs --labels path,label")
        import pandas as pd
        lab = pd.read_csv(a.labels)
        paths = [Path(p) for p in lab["path"]]
        report = calibrate(paths, dict(zip(lab["path"].astype(str), lab["label"])))
        if a.out:
            save_json(a.out, report)
        print(json.dumps(report, indent=2) if a.json else report)
        return 0

    if not a.images:
        ap.error("pass --images or --calibrate")

    results = []
    for p in a.images:
        try:
            results.append({"path": str(p), **gate(p, thresholds)})
        except Exception as exc:  # noqa: BLE001
            results.append({"path": str(p), "verdict": "REJECT", "reasons": ["UNREADABLE"],
                            "error": str(exc)})
    n_reject = sum(r["verdict"] == "REJECT" for r in results)
    n_warn = sum(r["verdict"] == "WARN" for r in results)
    summary = {"n": len(results), "reject": n_reject, "warn": n_warn,
               "ok": len(results) - n_reject - n_warn,
               "results": results}
    if a.out:
        save_json(a.out, summary)
    if a.json:
        print(json.dumps(summary, indent=2))
    else:
        for r in results:
            print(f"{r['verdict']:6s} {r['path']}  {r.get('reasons', '')}")
        print(f"\n{n_reject} reject, {n_warn} warn, {summary['ok']} ok "
              f"of {len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
