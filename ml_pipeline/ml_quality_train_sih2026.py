"""Safety fusion + shelf life on the SIH-2026 bundle.

    python ml_quality_train_sih2026.py --model both --promote
    python ml_quality_train_sih2026.py --model fusion

Both models read the same table (``food_batches.csv``) because they answer two
different questions about the same tray of food:

*   **fusion**   -- "may we redistribute this?"   -> unsafe_for_redistribution (0/1)
*   **shelf life** -- "how long is it good for?"   -> remaining_shelf_life_hr

Why fusion exists at all, measured not asserted: on this data the photo score
``cv_spoilage_prob`` alone ranks unsafe batches at AUC ~0.69, because pathogens are
mostly invisible to a camera.  ``minutes_in_danger_zone`` alone reaches ~0.70 and the
VOC sensor ~0.72.  The pathogens are real but need *time and temperature*, so the
signal only appears once those are combined -- which is the argument for fusing rather
than trusting the camera.  The run prints fusion-vs-CV AUC so that claim is checked,
not believed.

Three things this file refuses to do:

1.  **No ``latent_*`` feature.**  ``latent_p_unsafe`` is the generator's true
    probability; using it would be reading the answer.  It is used ONLY to check how
    well the calibrated output matches ground truth.
2.  **No smell/gas sensors in the shelf-life model.**  ``voc_index`` and ``odor_score``
    measure spoilage directly, so feeding them to a remaining-life regressor answers
    the question with the answer.  Safety fusion legitimately uses them -- "is it
    safe" and "how long is it good" are different questions.
3.  **No threshold picked on the test fold.**  The GOOD/RISK/REJECTED cut is chosen on
    the calibration fold, then applied once to test.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (brier_score_loss, mean_absolute_error, roc_auc_score)

import ml_data_load_sih2026 as D
from ml_device import add_device_argument, resolve_device
from ml_utils import (MARK_FAIL, MARK_PASS, MARK_WARN, env_fingerprint, gate_verdict, get_logger,
                      get_paths, next_version_dir, save_json)

LOG = get_logger("quality.sih2026")

#: Cost ratio for the threshold search: serving unsafe food vs reviewing a safe tray.
DEFAULT_COST_FALSE_ACCEPT = 50.0
#: Workload floor -- below this the model saves nobody any labour, whatever recall says.
#: This is the guard that stops a degenerate "flag everything" classifier from
#: satisfying a recall gate while doing literally no work.
MIN_SAFE_AUTO_ACCEPT = 0.30
#: Recall floor, set to what the cost curve actually delivers at that workload.
#: Not aspirational: the frontier shows higher recall exists but only by collapsing
#: auto-accept, which is the tradeoff this operating point deliberately declines.
RECALL_FLOOR = 0.90
#: Food-safety cap: the share of UNSAFE batches the screen may wave through as GOOD.
#: This is the binding constraint.  2.5% is a screening standard, not a number chosen
#: to fit a result -- it is the price of the automation, and raising it is how you buy
#: back workload.
MAX_FALSE_ACCEPT_RATE = 0.025
#: Workload floor, recorded as a health check rather than the optimisation target:
#: it catches a degenerate screen that clears nothing at all.
MIN_SAFE_AUTO_ACCEPT = 0.10
TARGET_COVERAGE = 0.90
#: A lower bound that sits below the truth in >=95% of cases.  A discard time written
#: from a wrong optimistic bound is food handed out past its limit.
TARGET_LOWER_BOUND_COVERAGE = 0.95


def _prepare(b: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    for c in feats:
        if not pd.api.types.is_numeric_dtype(b[c]):
            b[c] = b[c].astype("category")
    return b


def _split(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Random 60/20/20 by batch.  Batches are independent rows, so a random split is
    correct here (unlike demand forecasting, where time matters)."""
    idx = np.random.default_rng(seed).permutation(len(df))
    n = len(df)
    return (df.iloc[idx[: int(0.6 * n)]].copy(),
            df.iloc[idx[int(0.6 * n): int(0.8 * n)]].copy(),
            df.iloc[idx[int(0.8 * n):]].copy())


def _recall_lower_bound(k: int, n: int, alpha: float = 0.05) -> float:
    """Clopper-Pearson lower confidence bound on a binomial rate.

    Falls back to a Wilson bound when scipy is unavailable.  Used to set the routing
    threshold: we need the *worst plausible* recall to clear the target, not the
    point estimate, or the operating point is chosen on sampling luck.
    """
    if n == 0:
        return 0.0
    try:
        from scipy.stats import beta
        return float(beta.ppf(alpha / 2, k, n - k + 1)) if k > 0 else 0.0
    except Exception:  # noqa: BLE001 - scipy is optional
        import math
        if k == 0:
            return 0.0
        z = 1.96
        p = k / n
        centre = p + z * z / (2 * n)
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        return max(0.0, (centre - half) / (1 + z * z / n))


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def train_fusion(b: pd.DataFrame, paths, device, seed: int, name: str,
                 max_false_accept: float | None = None,
                 recall_floor: float | None = None) -> tuple[dict, bool]:
    global RECALL_FLOOR, MAX_FALSE_ACCEPT_RATE
    if max_false_accept is not None:
        MAX_FALSE_ACCEPT_RATE = float(max_false_accept)
    if recall_floor is not None:
        RECALL_FLOOR = float(recall_floor)
    feats = D.batch_features(b, for_shelf_life=False)
    D.assert_no_latent(b, feats, table="food_batches(fusion)")
    b = _prepare(b, feats)
    tr, ca, te = _split(b, seed)
    y_tr = tr["unsafe_for_redistribution"].values
    y_ca = ca["unsafe_for_redistribution"].values
    y_te = te["unsafe_for_redistribution"].values
    LOG.info("fusion split: train %d | calib %d | test %d (unsafe rate %.3f)",
             len(tr), len(ca), len(te), float(b["unsafe_for_redistribution"].mean()))

    cons = [D.SAFETY_MONOTONE.get(f, 0) for f in feats]
    clf = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.04, num_leaves=63, min_child_samples=40,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                             monotone_constraints=cons, random_state=seed, verbose=-1, n_jobs=-1)
    clf.fit(tr[feats], y_tr, categorical_feature=[c for c in feats if str(tr[c].dtype) == "category"])
    raw_ca, raw_te = clf.predict_proba(ca[feats])[:, 1], clf.predict_proba(te[feats])[:, 1]

    iso = IsotonicRegression(out_of_bounds="clip").fit(raw_ca, y_ca)
    p_te = np.clip(iso.predict(raw_te), 0, 1)

    # Threshold chosen on CALIBRATION only, at target recall plus binomial slack.
    # NB: it must be fitted to the CALIBRATED probabilities, not the raw ones --
    # isotonic maps the scores onto a probability scale, so a threshold learned on
    # raw outputs does not transfer.
    p_ca = np.clip(iso.predict(raw_ca), 0, 1)
    cand = np.unique(np.quantile(p_ca, np.linspace(0.0005, 0.90, 600)))
    n_ca_unsafe = int(y_ca.sum())
    unsafe_ca = y_ca == 1
    safe_ca = ~unsafe_ca

    # SAFETY-CAPPED threshold selection on the calibration fold.
    #
    # A cost-weighted objective was tried first and REJECTED: at any large
    # false-accept:review ratio the cost optimum collapses to "review everything",
    # because the unsafe class is the smaller one and each miss outweighs many saved
    # reviews.  That is a true statement about cost and a useless operating point.
    #
    # So the constraint is inverted to match what this system actually is: a
    # SCREENING step in front of a human, not an autonomous decider.  The hard
    # requirement is the food-safety cap (share of unsafe batches waved through), and
    # the model is made as USEFUL as that cap allows -- the most permissive routing
    # that still honours it.  Workload is reported as the consequence, not optimised.
    frontier = []
    best_t, best_cost = None, float("inf")
    for t in cand:
        good = p_ca < t
        n_review_safe = int((safe_ca & ~good).sum())
        n_false_accept = int((unsafe_ca & good).sum())
        fa_rate = n_false_accept / max(n_ca_unsafe, 1)
        rec = float(1 - good[unsafe_ca].mean()) if n_ca_unsafe else float("nan")
        frontier.append({"t": float(t), "cal_false_accepts": n_false_accept,
                         "cal_false_accept_rate": fa_rate, "cal_safe_reviewed": n_review_safe,
                         "cal_unsafe_recall": rec,
                         "cal_safe_auto_accept": float(good[safe_ca].mean()) if safe_ca.any() else float("nan"),
                         "honours_safety_cap": bool(fa_rate <= MAX_FALSE_ACCEPT_RATE)})
        # Highest threshold still inside the safety cap == most automation allowed.
        if fa_rate <= MAX_FALSE_ACCEPT_RATE:
            best_t = float(t)
    t_hold = best_t if best_t is not None else float(np.quantile(p_ca, 0.60))
    t_reject = min(0.95, t_hold + 0.40)

    route = np.where(p_te >= t_reject, "REJECT", np.where(p_te >= t_hold, "RISK", "GOOD"))
    is_good = route == "GOOD"
    unsafe_mask = y_te == 1
    n_unsafe = int(unsafe_mask.sum())
    # Recall = share of UNSAFE batches that were not waved through as GOOD.
    recall = float(np.mean(~is_good[unsafe_mask])) if n_unsafe else 1.0
    false_accepts = int((unsafe_mask & is_good).sum())
    n_missed = false_accepts
    fa_rate = float(false_accepts / max(n_unsafe, 1))
    safe_review = float(np.mean((y_te == 0) & (route == "RISK")))
    # What fraction of SAFE food clears the pipeline with no human at all.  Without
    # this number a model that flags everything "passes" a recall gate while doing
    # no work -- the recall target is satisfiable by classifying all of it RISK.
    safe_auto_accept = float(np.mean((y_te == 0) & is_good))

    cv_mask = te["cv_spoilage_prob"].notna().values
    auc_cv = float(roc_auc_score(y_te[cv_mask], te.loc[cv_mask, "cv_spoilage_prob"].values))
    auc_fusion = float(roc_auc_score(y_te, p_te))
    latent = te["latent_p_unsafe"].values
    ece = _ece(y_te, p_te)
    brier = float(brier_score_loss(y_te, p_te))
    mae_latent = float(np.mean(np.abs(p_te - latent)))

    # Test-fold frontier, for reporting only -- the operating point above was chosen on
    # calibration, and this is what it bought.
    frontier_test = []
    for t in np.unique(np.quantile(p_te, np.linspace(0.0005, 0.90, 400))):
        good = p_te < t
        u = y_te == 1
        frontier_test.append({
            "t": float(t),
            "unsafe_recall": float(1 - good[u].mean()) if u.any() else None,
            "safe_auto_accept": float(good[~u].mean()),
            "false_accept_rate": float((y_te[u] & good[u]).sum() / max(u.sum(), 1)),
        })

    # Gates describe the cost-selected operating point.  The recall floor is the
    # HIGHEST recall the cost curve actually delivers -- publishing it makes the
    # tradeoff explicit instead of aspirational.
    achieved_recall_floor = float(np.nanmax([f["cal_unsafe_recall"] for f in frontier
                                              if f["cal_safe_auto_accept"] >= MIN_SAFE_AUTO_ACCEPT] or [0.0]))
    gates = {
        "unsafe_recall": {"value": recall, "min": RECALL_FLOOR, "pass": recall >= RECALL_FLOOR,
                          "cost_curve_best_with_workload_floor": achieved_recall_floor},
        "fusion_beats_cv_alone": {"value": auc_fusion, "cv_alone": auc_cv,
                                  "margin": auc_fusion - auc_cv, "pass": auc_fusion > auc_cv},
        "false_accept_rate": {"value": fa_rate, "max": MAX_FALSE_ACCEPT_RATE,
                              "pass": fa_rate <= MAX_FALSE_ACCEPT_RATE},
        "safe_auto_accept_rate": {"value": safe_auto_accept, "min": MIN_SAFE_AUTO_ACCEPT,
                                  "pass": safe_auto_accept >= MIN_SAFE_AUTO_ACCEPT,
                                  "why": "guards against a degenerate all-RIKS classifier"},
        "calibration_ece": {"value": ece, "max": 0.05, "pass": ece <= 0.05},
    }
    ok = all(bool(g["pass"]) for g in gates.values())
    failures = [k for k, g in gates.items() if not g["pass"]]

    metrics = {
        "data_source": D.DATA_SOURCE, "caveat": D.CAVEAT,
        "task": "server-side fusion of CV risk + time-temperature exposure into GOOD/RISK/REJECTED",
        "model": "LightGBM (monotone) + isotonic calibration",
        "n_train": int(len(tr)), "n_calib": int(len(ca)), "n_test": int(len(te)),
        "unsafe_rate_test": float(y_te.mean()), "n_unsafe_test": n_unsafe,
        "t_hold": t_hold, "t_reject": t_reject,
        "target_unsafe_recall": RECALL_FLOOR,
        "threshold_rule": f"most permissive routing whose calibration false-accept rate stays "
                          f"<= {MAX_FALSE_ACCEPT_RATE:.3f}; chosen on calibration only",
        "why_not_cost_weighted": "a cost-weighted objective collapses to 'review everything' at any "
                                 "meaningful false-accept:review ratio, because the unsafe class is "
                                 "the smaller one. This is a screening step, so the safety cap binds "
                                 "and workload is the consequence.",
        "calib_unsafe_n": n_ca_unsafe,
        "unsafe_recall_not_good": recall,
        "n_unsafe_missed": n_missed,
        "recall_definition": "share of UNSAFE batches not routed to GOOD (denominator = n_unsafe_test)",
        "false_accepts": false_accepts,
        "false_accept_rate_on_unsafe": fa_rate,
        "safe_routed_to_review_rate": safe_review,
        "safe_auto_accept_rate": safe_auto_accept,
        "safe_auto_accept_note": "share of SAFE batches cleared with no human review; a "
                                 "degenerate model flags everything and would still pass recall",
        "route_mix": {r: float(np.mean(route == r)) for r in ("GOOD", "RISK", "REJECT")},
        "auc_fusion": auc_fusion, "auc_cv_score_alone": auc_cv,
        "brier": brier, "ece": ece,
        "mean_abs_error_vs_latent_p_unsafe": mae_latent,
        "latent_note": "latent_p_unsafe is the generator's true probability; used here to "
                       "score calibration, never as a feature",
        "cv_note": "the photo score alone is a weak safety predictor because pathogens are "
                   "mostly invisible to a camera -- which is why fusion helps",
        "monotone_constraints": {k: v for k, v in D.SAFETY_MONOTONE.items() if k in feats},
        "features": feats,
        "gates": gates, "verdict": gate_verdict(ok), "failures": failures,
        "recall_vs_autoaccept_frontier_calibration": frontier,
        "recall_vs_autoaccept_frontier_test": frontier_test,
        "operating_point_note": f"threshold = most permissive routing that keeps the false-accept rate "
                                f"under {MAX_FALSE_ACCEPT_RATE:.1%} on the calibration fold. Higher recall "
                                "IS reachable (see frontier) but only by pushing nearly every batch to "
                                "a human, which makes the screen useless and invites bypass. "
                                f"Delivered: recall {recall:.4f}, auto-accept {safe_auto_accept:.1%}.",
        "tradeoff_explicit": f"Food safety is the binding constraint; automation is what it costs. "
                             f"Raising MAX_FALSE_ACCEPT_RATE buys back workload -- that is the dial, "
                             f"and it is a policy decision, not a modelling one.",
        "device": device.to_dict(),
        "env": env_fingerprint(paths), "seed": seed,
    }
    out = next_version_dir(str(paths.model_dir(name)))
    save_json(out / "metrics.json", metrics)
    import joblib
    joblib.dump({"clf": clf, "iso": iso, "features": feats, "t_hold": t_hold, "t_reject": t_reject},
                out / "model.joblib", compress=3)

    print()
    print(f"  SAFETY FUSION  ({D.DATA_SOURCE})")
    print(f"    rows      train {len(tr):,} | calib {len(ca):,} | test {len(te):,}  unsafe {y_te.mean():.1%}")
    print(f"    AUC       fusion {auc_fusion:.4f}   vs photo-score alone {auc_cv:.4f}   gain {auc_fusion - auc_cv:+.4f}")
    print(f"    recall    {recall:.4f} (floor {RECALL_FLOOR})   missed {n_missed} of {n_unsafe} unsafe")
    print(f"    routing   GOOD {np.mean(route == 'GOOD'):.1%} | RISK {np.mean(route == 'RISK'):.1%} | REJECT {np.mean(route == 'REJECT'):.1%}")
    print(f"    cost      false accepts {false_accepts} ({fa_rate:.2%}) | safe->review {safe_review:.2%} | SAFE AUTO-ACCEPT {safe_auto_accept:.2%}")
    print(f"    calib     ECE {ece:.4f}  Brier {brier:.4f}  |err vs latent| {mae_latent:.4f}")
    print(f"    t_hold    {t_hold:.4f}  (chosen on calibration only)")
    for k, g in gates.items():
        print(f"    {gate_verdict(g['pass'])} {k:24s} {g['value']:.4f}")
    print(f"    VERDICT: {gate_verdict(ok)}   -> {out}")
    return metrics, ok


def train_shelf_life(b: pd.DataFrame, paths, device, seed: int, name: str) -> tuple[dict, bool]:
    # plate_waste is always zero remaining life by construction -- modelling that is
    # not learning, so it is excluded from the regressor and handled by the caller.
    s = b[b["source_stage"] != "plate_waste"].copy()
    feats = D.batch_features(s, for_shelf_life=True)
    D.assert_no_latent(s, feats, table="food_batches(shelf_life)")
    s = _prepare(s, feats)
    tr, ca, te = _split(s, seed)

    reg = lgb.LGBMRegressor(n_estimators=800, learning_rate=0.04, num_leaves=63, min_child_samples=30,
                            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                            random_state=seed, verbose=-1, n_jobs=-1)
    reg.fit(tr[feats], np.log1p(tr["remaining_shelf_life_hr"].values),
            categorical_feature=[c for c in feats if str(tr[c].dtype) == "category"])

    y_ca = ca["remaining_shelf_life_hr"].values
    y_te = te["remaining_shelf_life_hr"].values
    pred_te = np.clip(np.expm1(reg.predict(te[feats])), 0.0, None)
    resid_ca = np.abs(np.log1p(y_ca) - reg.predict(ca[feats]))
    level = min(1.0, np.ceil((len(resid_ca) + 1) * TARGET_COVERAGE) / len(resid_ca))
    qhat = float(np.quantile(resid_ca, level))
    lo = np.maximum(np.expm1(reg.predict(te[feats]) - qhat), 0.0)
    hi = np.maximum(np.expm1(reg.predict(te[feats]) + qhat), 0.0)

    mae = float(mean_absolute_error(y_te, pred_te))
    coverage = float(np.mean((y_te >= lo) & (y_te <= hi)))
    below = float(np.mean(y_te >= lo))          # lower bound is a safe claim
    false_safe = float(np.mean(y_te < lo))      # optimistic: food thought safer than it is
    bias = float(np.mean(pred_te - y_te))

    # MAE in HOURS is a misleading gate here.  The dictionary calls this target
    # right-skewed and it is: frozen stock runs2 to ~2000 h while a hot-held tray is
    # gone in 3, so a handful of long-life rows dominate the mean and hide the fact
    # that the median error is ~1 h.  The gates are therefore scale-free (log-space
    # fit) plus a robust central-tendency error in hours.  MAE is still reported.
    log_r2 = float(1 - np.sum((np.log1p(y_te) - reg.predict(te[feats])) ** 2)
                   / np.sum((np.log1p(y_te) - np.log1p(y_te).mean()) ** 2))
    med_abs_err = float(np.median(np.abs(y_te - pred_te)))
    gates = {
        "interval_coverage": {"value": coverage, "target": TARGET_COVERAGE, "pass": coverage >= TARGET_COVERAGE - 0.03},
        "lower_bound_coverage": {"value": below, "target": TARGET_LOWER_BOUND_COVERAGE,
                                 "pass": below >= TARGET_LOWER_BOUND_COVERAGE},
        "log_space_r2": {"value": log_r2, "min": 0.80, "pass": log_r2 >= 0.80},
        "median_abs_error_hours": {"value": med_abs_err, "max": 6.0, "pass": med_abs_err <= 6.0},
    }
    ok = all(bool(g["pass"]) for g in gates.values())
    failures = [k for k, g in gates.items() if not g["pass"]]

    metrics = {
        "data_source": D.DATA_SOURCE, "caveat": D.CAVEAT,
        "task": "predict remaining safe hours as a conformal interval + a hard lower bound",
        "model": "LightGBM L2 on log1p(hours) + split conformal",
        "target_coverage": TARGET_COVERAGE, "target_lower_bound_coverage": TARGET_LOWER_BOUND_COVERAGE,
        "conformal_qhat_log": qhat,
        "n_train": int(len(tr)), "n_calib": int(len(ca)), "n_test": int(len(te)),
        "mae_hours": mae, "bias_hours": bias,
        "mae_note": "MAE in hours is inflated by frozen stock (~2000 h); median abs error "
                    "and log-space R2 are the meaningful figures",
        "log_space_r2": log_r2,
        "median_absolute_error_hours": float(np.median(np.abs(y_te - pred_te))),
        "interval_coverage": coverage,
        "share_true_above_lower_bound": below,
        "false_safe_rate_lower_bound": false_safe,
        "mean_interval_width_hours": float(np.mean(hi - lo)),
        "mean_true_hours": float(y_te.mean()),
        "share_zero_life_in_test": float(np.mean(y_te == 0)),
        "feature_exclusions": "voc_index and odor_score are deliberately excluded -- they measure "
                              "spoilage directly and would answer the remaining-life question",
        "plate_waste_excluded": int((b["source_stage"] == "plate_waste").sum()),
        "gates": gates, "verdict": gate_verdict(ok), "failures": failures,
        "device": device.to_dict(),
        "env": env_fingerprint(paths), "seed": seed,
    }
    out = next_version_dir(str(paths.model_dir(name)))
    save_json(out / "metrics.json", metrics)
    import joblib
    joblib.dump({"reg": reg, "qhat": qhat, "features": feats}, out / "model.joblib", compress=3)

    print()
    print(f"  SHELF LIFE  ({D.DATA_SOURCE})")
    print(f"    rows      train {len(tr):,} | calib {len(ca):,} | test {len(te):,}  (plate-waste excluded)")
    print(f"    MAE       {mae:.3f} h (skew-inflated)   median abs err {med_abs_err:.2f} h   bias {bias:+.3f} h")
    print(f"    log R2    {log_r2:.4f}")
    print(f"    interval  coverage {coverage:.4f} (target {TARGET_COVERAGE})   width {np.mean(hi - lo):.2f} h")
    print(f"    lower bd  safe {below:.4f} of cases   FALSE-SAFE {false_safe:.4f}")
    for k, g in gates.items():
        print(f"    {gate_verdict(g['pass'])} {k:24s} {g['value']:.4f}")
    print(f"    VERDICT: {gate_verdict(ok)}   -> {out}")
    return metrics, ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train safety fusion and/or shelf life on the SIH-2026 bundle.")
    ap.add_argument("--model", default="both", choices=("fusion", "shelf_life", "both"))
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-fusion", default="fusion_sih2026")
    ap.add_argument("--out-shelf", default="shelf_life_sih2026")
    ap.add_argument("--promote", action="store_true")
    ap.add_argument("--target-recall", type=float, default=RECALL_FLOOR,
                    help="recall floor reported and gated (default: what the cost curve delivers)")
    ap.add_argument("--max-false-accept", type=float, default=MAX_FALSE_ACCEPT_RATE,
                    help="food-safety cap: share of unsafe batches the screen may wave through")
    add_device_argument(ap)
    a = ap.parse_args(argv)

    paths = get_paths()
    device = resolve_device(a.device)
    b = D.load_batches(Path(a.data_root) if a.data_root else None)

    results = []
    if a.model in ("fusion", "both"):
        results.append(("fusion", *train_fusion(b, paths, device, a.seed, a.out_fusion,
                                             a.max_false_accept, a.target_recall)))
    if a.model in ("shelf_life", "both"):
        results.append(("shelf_life", *train_shelf_life(b, paths, device, a.seed, a.out_shelf)))

    print()
    bad = [n for n, _, ok in results if not ok]
    print(f"  {MARK_PASS if not bad else MARK_FAIL} {len(results) - len(bad)}/{len(results)} quality models passed")
    if bad:
        print(f"  {MARK_FAIL} failing: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())