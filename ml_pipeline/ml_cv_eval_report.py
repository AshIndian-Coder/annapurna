from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from ml_cv_common import (
    C2I,
    CLASSES,
    LOG,
    decide,
    evaluate,
    expected_calibration_error,
    pick_reject_threshold,
)
from ml_utils import get_paths, save_json


TARGET_UNSAFE_RECALL = 0.98

COST_FALSE_ACCEPT = 10.0
COST_FALSE_HOLD = 1.0

def _load_metrics(run: Path) -> dict | None:
    f = run / "metrics.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        LOG.error("%s is not valid JSON (%s) -- treating the run as unevaluated", f, exc)
        return None


def _block(split: str, m: dict | None) -> dict:
    if not m or split not in m:
        return {"available": False, "reason": f"run has no '{split}' metrics"}
    v = m[split]
    out = {
        "available": True,
        "n": v.get("n"),
        "macro_f1": v.get("macro_f1"),
        "accuracy": v.get("accuracy"),
        "unsafe_recall": v.get("unsafe_recall"),
        "false_accepts": v.get("false_accepts"),
        "good_auto_accept_rate": v.get("good_auto_accept_rate"),
        "ece": v.get("ece"),
        "tau": v.get("tau"),
        "tau_reject": v.get("tau_reject"),
        "actions": v.get("actions"),
        "risk_hold_rate": v.get("risk_hold_rate"),
        "risk_zone_recall": v.get("risk_zone_recall"),
        "reject_recall": v.get("reject_recall"),
        "confusion": v.get("confusion"),
        "per_class": v.get("per_class"),
    }
    if v.get("probs") and v.get("y_true"):
        p = np.asarray(v["probs"], dtype=float)
        yy = np.asarray(v["y_true"], dtype=int)
        fa, fb = int(v.get("false_accepts") or 0), int(
            (yy == C2I["GOOD"]).sum() - round((v.get("good_auto_accept_rate") or 0)
                                              * max(int((yy == C2I["GOOD"]).sum()), 1)))
        out["cost_weighted_loss"] = float(
            (COST_FALSE_ACCEPT * fa + COST_FALSE_HOLD * max(fb, 0)) / max(len(yy), 1))
    return out


def _flagged(probs: np.ndarray, tau: float) -> np.ndarray:
    """The serving rule: ``1 - P(GOOD) >= tau`` means "hold, do not redistribute".

    Defined once, here and in :func:`ml_cv_common.evaluate`.  The report calls
    ``evaluate`` for every threshold rather than re-deriving the numbers, so the sweep,
    the metrics block and the model the Go service serves are provably the same rule --
    an earlier version of this file computed the *complement* and reported an unsafe
    recall of 0.0 for a perfect model.
    """
    return (1.0 - np.asarray(probs, dtype=float)[:, C2I["GOOD"]]) >= tau


def _risk_zone_block(probs: np.ndarray, y: np.ndarray, metrics_block: dict) -> dict:
    """The RISK middle zone, recomputed from the saved predictions.

    ``RISK`` here is a **decision zone**, not a learned visual class: rows whose
    release score ``s = 1 - P(GOOD)`` is neither confidently releasable (``s < tau``)
    nor confidently rejectable (``s >= tau_reject``) are held for a human.  ``tau``
    comes from the run's metrics; ``tau_reject`` from the metrics when the run
    recorded it, otherwise calibrated here on the truly REJECTED rows of this split.
    """
    tau = metrics_block.get("tau")
    if tau is None:
        return {"available": False, "reason": "no tau in metrics"}
    tau = float(tau)
    tau_reject = metrics_block.get("tau_reject")
    calibrated_here = False
    if tau_reject is None:
        try:
            tau_reject = pick_reject_threshold(probs, y)
            calibrated_here = True
        except ValueError as exc:
            return {"available": False,
                    "reason": f"no REJECT zone on this split ({exc})",
                    "tau": tau,
                    "note": "every HOLD is risk_hold; RISK = human review only"}
    tau_reject = float(tau_reject)
    action = decide(probs, tau, tau_reject)
    risk_true = y == C2I["RISK"]
    rej_true = y == C2I["REJECTED"]
    return {
        "available": True,
        "tau": tau,
        "tau_reject": tau_reject,
        "tau_reject_source": ("from run metrics" if not calibrated_here
                              else "calibrated on this split (run did not record one)"),
        "definition": ("auto_accept: 1-P(GOOD) < tau | reject: >= tau_reject | "
                       "risk_hold: in between -- hold for human review, not a "
                       "learned visual class"),
        "actions": {a: int((action == a).sum())
                    for a in ("auto_accept", "risk_hold", "reject")},
        "risk_hold_rate": round(float((action == "risk_hold").mean()), 4),
        "risk_zone_recall": (round(float((action[risk_true] == "risk_hold").mean()), 4)
                             if risk_true.any() else None),
        "reject_recall": (round(float((action[rej_true] == "reject").mean()), 4)
                          if rej_true.any() else None),
        "note": ("RISK means the model is unsure, so a person decides. It is not "
                 "evidence of early spoilage unless a human confirms it."),
    }


def policy_comparison(probs: np.ndarray, y: np.ndarray) -> dict:
    """What the three shippable policies cost, on the same rows.

    A model that does not beat ``always HOLD`` has no business shipping; this block
    makes that check mechanical instead of a matter of opinion.
    """
    from ml_cv_common import GOOD

    n = max(len(y), 1)
    probs = np.asarray(probs, dtype=float)
    unsafe = np.isin(y, [C2I["RISK"], C2I["REJECTED"]])
    good = y == C2I["GOOD"]
    # argmax == GOOD, i.e. the most confident "release it" the model can make
    decided = probs.argmax(axis=1) == GOOD

    def cost(accept: np.ndarray) -> float:
        # NOT_FOOD rows are neither error: identifying a shoe as a shoe is the correct
        # answer and blocks redistribution for the right reason.  Counting them as
        # "good food held back" would make a perfect model look like it wastes 30 % of
        # the food it sees.
        fa = int((accept & unsafe).sum())
        fb = int(((~accept) & good).sum())
        return (COST_FALSE_ACCEPT * fa + COST_FALSE_HOLD * fb) / n

    return {
        "always_hold": {"false_accepts": 0, "good_blocked": int(good.sum()),
                        "cost_weighted_loss": cost(np.zeros(len(y), bool)),
                        "note": "zero risk, maximum waste -- the safe default"},
        "always_eligible": {"false_accepts": int(unsafe.sum()),
                            "good_blocked": 0,
                            "cost_weighted_loss": cost(np.ones(len(y), bool)),
                            "note": "maximum risk, zero waste -- never acceptable"},
        "model_argmax": {"false_accepts": int((decided & unsafe).sum()),
                         "good_blocked": int(((~decided) & good).sum()),
                         "cost_weighted_loss": cost(decided),
                         "note": "argmax == GOOD; see the sweep for the thresholded rule"},
    }


def threshold_sweep(probs: np.ndarray, y: np.ndarray, *,
                    taus: np.ndarray | None = None) -> list:
    """Unsafe recall and good-auto-accept across the hold threshold.

    ``tau`` rises -> more photos are held -> unsafe recall rises and good-food waste
    rises with it.  The operating point is a *business* decision, and this table is
    what makes it one.  Every row is computed by ``evaluate`` so the sweep cannot drift
    from the reported metrics.
    """
    if taus is None:
        taus = np.linspace(0.05, 0.95, 91)
    rows = []
    for t in taus:
        m = evaluate(probs, y, float(t))
        rows.append({
            "tau": round(float(t), 3),
            "unsafe_recall": m["unsafe_recall"],
            "false_accepts": m["false_accepts"],
            "good_auto_accept_rate": m["good_auto_accept_rate"],
            "macro_f1": round(m["macro_f1"], 4),
            "meets_target": m["meets_recall_target"],
        })
    return rows


def reliability_curve(probs: np.ndarray, y: np.ndarray, *, bins: int = 10) -> list:
    """Confidence vs empirical accuracy, per bin, for the predicted class."""
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y).astype(float)
    edges = np.linspace(conf.min(), conf.max() + 1e-9, bins + 1)
    out = []
    for i in range(bins):
        sel = (conf >= edges[i]) & (conf < edges[i + 1] if i < bins - 1 else conf <= edges[i + 1])
        if not sel.any():
            continue
        out.append({"bin": i, "n": int(sel.sum()),
                    "mean_confidence": round(float(conf[sel].mean()), 4),
                    "accuracy": round(float(correct[sel].mean()), 4),
                    "gap": round(float(conf[sel].mean() - correct[sel].mean()), 4)})
    return out


def build_report(run: Path, index_csv: Path) -> dict:
    metrics = _load_metrics(run)
    rep: dict = {
        "run": run.name, "path": str(run),
        "arch": (metrics or {}).get("arch"),
        "classes": CLASSES,
        "targets": {"unsafe_recall": TARGET_UNSAFE_RECALL,
                    "cost_false_accept": COST_FALSE_ACCEPT,
                    "cost_false_hold": COST_FALSE_HOLD},
    }
    # -- 1. split integrity ------------------------------------------------- #
    split_info: dict = {"available": False}
    if index_csv.exists():
        df = pd.read_csv(index_csv)
        for name, part in df.groupby("split"):
            split_info[name] = {
                "images": int(len(part)),
                "groups": int(part["group"].nunique()) if "group" in part else None,
                "sources": sorted(part["source"].unique()) if "source" in part else None,
                "labels": part["label"].value_counts().to_dict() if "label" in part else None,
            }
        test_src = set(df[df.split == "test"]["source"]) if "source" in df else set()
        other = set(df[df.split != "test"]["source"]) if "source" in df else set()
        split_info.update({
            "available": True,
            "test_unseen_source": not (test_src & other),
            "test_sources": sorted(test_src),
            "index_fingerprint": (df.get("phash") is not None) and bool(df["phash"].nunique()),
        })
        split_info["caveat"] = (
            "" if split_info["test_unseen_source"] else
            "the test split shares sources with train/val -- these numbers are optimistic")
    else:
        split_info["reason"] = f"no index at {index_csv}"
    rep["split_integrity"] = split_info
    rep["headline_claim_valid"] = bool(split_info.get("test_unseen_source"))
    rep["val"] = _block("val", metrics)
    rep["test"] = _block("test_unseen_sources", metrics) or _block("test", metrics)
    probs = y = None
    rep["evaluated_on"] = None
    rep["evaluated_split_name"] = None
    for name, split_name in (("preds_test.json", "test"), ("preds_val.json", "val")):
        pf = run / name
        if pf.exists():
            try:
                blob = json.loads(pf.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                LOG.error("%s is not valid JSON (%s) -- skipping it", pf, exc)
                continue
            probs = np.asarray(blob["probs"], dtype=float)
            y = np.asarray(blob["y_true"], dtype=int)
            rep["evaluated_on"] = name
            rep["evaluated_split_name"] = split_name
            rep["predictions_file"] = str(pf)
            rep["predictions_n"] = int(blob.get("n", len(y)))
            if blob.get("classes") and list(blob["classes"]) != CLASSES:
                LOG.error("%s was written for classes %s but this build uses %s -- the "
                          "probabilities do not line up and are ignored",
                          pf, blob["classes"], CLASSES)
                probs = y = None
                rep["evaluated_on"] = rep["evaluated_split_name"] = None
            break

    if probs is not None:
        on = rep["evaluated_split_name"]
        rep["safety"] = {
            "unsafe_recall_at_tau": rep[on].get("unsafe_recall"),
            "meets_target": bool((rep[on].get("unsafe_recall") or 0) >= TARGET_UNSAFE_RECALL),
            "target": TARGET_UNSAFE_RECALL,
        }
        rep["calibration"] = {
            "ece": round(expected_calibration_error(probs, y), 5),
            "reliability": reliability_curve(probs, y),
        }
        rep["threshold_sweep"] = threshold_sweep(probs, y)
        feasible = [r for r in rep["threshold_sweep"] if r["meets_target"]]
        rep["recommended_operating_point"] = (
            max(feasible, key=lambda r: r["good_auto_accept_rate"]) if feasible else None)
        if not feasible:
            rep["recommended_operating_point_note"] = (
                "NO threshold reaches the 0.98 unsafe-recall target on this split. "
                "Collect more photos; do not lower the target to make the slide fit.")
        rep["policies"] = policy_comparison(probs, y)
        rep["risk_zone"] = _risk_zone_block(probs, y, rep[on])
    else:
        rep["safety"] = {"available": False,
                         "reason": "no preds_test.json / preds_val.json beside metrics.json, "
                                   "so the threshold sweep and policy comparison cannot be "
                                   "recomputed; re-run ml_cv_train_cls.py to regenerate them"}

    # -- 5. device parity --------------------------------------------------- #
    exp = run / "export_report.json"
    rep["device_parity"] = (json.loads(exp.read_text(encoding="utf-8"))
                            if exp.exists()
                            else {"available": False,
                                  "reason": "no export_report.json; run ml_cv_export.py"})
    rep["device_parity"]["available"] = exp.exists()
    manifest = run / "manifest.json"
    rep["bundle"] = (json.loads(manifest.read_text(encoding="utf-8"))
                     if manifest.exists() else {"available": False,
                                                "reason": "no manifest.json (D21 bundle)"})

    # -- honest-claims block ------------------------------------------------ #
    rep["claims"] = {
        "may_claim": [
            f"macro-F1 {_n(rep['test'].get('macro_f1'), 4)} and unsafe recall "
            f"{_n(rep['test'].get('unsafe_recall'), 4)} on the "
            f"{'unseen-source' if rep['headline_claim_valid'] else 'held-out (source-shared)'} "
            f"test split of {_n(rep['test'].get('n'))} photos"
        ] if rep["test"].get("available") else [],
        "must_not_claim": [
            "certified safe -- the model estimates visible deterioration only",
            "any accuracy number computed with random splits or without near-duplicate removal",
        ],
        "caveats": [
            "A camera cannot detect bacteria; the fusion with time-temperature exposure "
            "and a human approval step are load-bearing, not decoration.",
            "Synthetic or simulator data, if any, is labelled and is not evidence about a "
            "real kitchen.",
        ],
    }
    return rep


def _n(v, nd: int = 4) -> str:
    """Format a metric for a report meant to be read aloud.

    ``0.26111111111111107`` in a slide is noise; ``0.2611`` is a number.  ``None`` prints
    as ``-`` rather than ``None`` so a missing metric does not read like a bug.
    """
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return f"{f:,.{nd}f}"


def render(report: dict) -> str:
    L: list = []
    add = L.append
    add(f"CV EVALUATION REPORT -- {report['run']}  (arch: {report.get('arch')})")
    add("=" * 78)
    si = report["split_integrity"]
    add(f"1. SPLIT INTEGRITY      {'OK' if si.get('test_unseen_source') else 'NOT PROVEN'}"
        f"   [headline claim {'valid' if report['headline_claim_valid'] else 'NOT supported'}]")
    if si.get("available"):
        for k in ("train", "val", "test"):
            if k in si:
                v = si[k]
                add(f"   {k:5s} {_n(v['images']):>6s} images  "
                    f"{_n(v['groups']) if v['groups'] is not None else '-':>5s} dishes  "
                    f"sources={v['sources']}")
        add(f"   test uses unseen sources: {_n(si['test_unseen_source'])}")
        if si.get("caveat"):
            add(f"   ! {si['caveat']}")
    else:
        add(f"   {si.get('reason')}")
    add("")
    add("2/3. ACCURACY AND SAFETY")
    for split in ("val", "test"):
        b = report[split]
        if not b.get("available"):
            add(f"   {split:4s}: {b.get('reason')}")
            continue
        add(f"   {split:4s}: n={_n(b['n'])}  macro_f1={_n(b['macro_f1'])}  "
            f"acc={_n(b['accuracy'])}  unsafe_recall={_n(b['unsafe_recall'])}  "
            f"false_accepts={_n(b['false_accepts'])}  ECE={_n(b['ece'])}")
        cm = b.get("confusion") or {}
        if cm:
            add("         confusion (rows = true):")
            labels = list(cm.keys())
            add("           " + "".join(f"{c[:9]:>11s}" for c in labels))
            for t in labels:
                row = cm[t]
                add(f"   {t[:9]:>9s}" + "".join(f"{row.get(c, 0):>11,d}" for c in labels))
    s = report.get("safety", {})
    if s.get("meets_target") is not None:
        add(f"   unsafe-recall target {_n(s['target'], 2)}: "
            f"{'MET' if s['meets_target'] else 'NOT MET'}")
    add("")
    add("4. CALIBRATION")
    cal = report.get("calibration") or {}
    if "ece" in cal:
        add(f"   ECE {_n(cal['ece'])}")
        for r in cal.get("reliability", []):
            add(f"     bin {r['bin']}  n={r['n']:5d}  conf={r['mean_confidence']:.3f}  "
                f"acc={r['accuracy']:.3f}  gap={r['gap']:+.3f}")
    if report.get("threshold_sweep"):
        add("")
        add("   hold-threshold sweep (tau = 1 - P(GOOD) needed to hold a batch):")
        for r in report["threshold_sweep"][::10]:
            flag = " <- meets target" if r["meets_target"] else ""
            add(f"     tau={r['tau']:.2f}  unsafe_recall={_n(r['unsafe_recall'], 3)}  "
                f"good_auto_accept={_n(r['good_auto_accept_rate'], 3)}  "
                f"false_accepts={_n(r['false_accepts'])}{flag}")
        op = report.get("recommended_operating_point")
        if op:
            add(f"   recommended operating point: tau={op['tau']} "
                f"(unsafe_recall {_n(op['unsafe_recall'], 3)}, "
                f"good auto-accept {_n(op['good_auto_accept_rate'], 3)})")
        elif report.get("recommended_operating_point_note"):
            add(f"   ! {report['recommended_operating_point_note']}")
    if report.get("policies"):
        add("")
        add("6. COST-WEIGHTED LOSS vs ALTERNATIVE POLICIES")
        for name, p in report["policies"].items():
            add(f"     {name:16s} loss={p['cost_weighted_loss']:8.3f}  "
                f"false_accepts={p['false_accepts']:4d}  good_blocked={p['good_blocked']:4d}")
    add("")
    add("5. DEVICE PARITY (D21)")
    dp = report.get("device_parity") or {}
    if not dp.get("available"):
        add(f"   {dp.get('reason')}")
    else:
        add(f"   fp32 vs torch  max|dP| = {dp.get('fp32_max_abs_prob_diff_vs_torch')}")
        add(f"   int8 vs fp32   argmax agreement = {dp.get('int8_argmax_agreement_with_fp32')}")
        add(f"   tflite vs onnx argmax agreement = "
            f"{dp.get('tflite_vs_onnx_argmax_agreement')}")
        add(f"   tflite vs onnx mean|d unsafe prob| = "
            f"{dp.get('tflite_vs_onnx_mean_abs_unsafe_prob_delta')}")
    b = report.get("bundle") or {}
    if b.get("device") is None and b.get("available") is not False:
        add("   ! manifest has NO device artifact -- the phone cannot run this bundle")
    add("")
    add("CLAIMS")
    for c in report["claims"]["may_claim"]:
        add(f"   may: {c}")
    for c in report["claims"]["must_not_claim"]:
        add(f"   NOT: {c}")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--index", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    paths = get_paths()
    index = Path(a.index) if a.index else paths.cv_data / "index.csv"

    runs = ([a.run] if a.run else
            [str(p.parent) for p in sorted(paths.runs.glob("*/metrics.json"))] if a.all else [])
    if not runs:
        raise SystemExit("pass --run <dir> or --all")
    for r in runs:
        rep = build_report(Path(r), index)
        print(render(rep))
        print()
        save_json(Path(r) / "eval_report.json", rep)
        LOG.info("wrote %s", Path(r) / "eval_report.json")


if __name__ == "__main__":
    raise SystemExit(main())
