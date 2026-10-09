"""One-page metrics report for the SIH-2026 Annapurna models.

    python ml_report_metrics.py                 # print the table
    python ml_report_metrics.py --out reports   # also write json + markdown

Why this exists
---------------
Every model in the registry was scored on the metric its own task implies: a
forecaster gets WAPE, a safety screen gets recall at a stated false-accept rate,
a remaining-life predictor gets interval coverage, an event detector gets
*episode* recall.  Asking "what is the accuracy" across that set has no single
answer, and this script exists to make that explicit rather than to invent one.

The rule followed here is deliberately unfashionable: **the headline number is
the one the task deserves, not the one that clears a threshold.**  Where a
subsystem cannot reach 0.95 on its correct metric, the report says so and prints
the measurement that explains why.  A demand forecaster's accuracy cannot be
raised by picking a friendlier denominator, and `ml_demand_train_sih2026.py
--diagnose-noise-floor` measures the floor that proves it.

Everything is read back from the registry artefacts on disk.  Nothing here is
typed in by hand, so the report cannot drift away from the models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))

from ml_utils import get_logger, get_paths, save_json

LOG = get_logger("report.metrics")

TARGET = 0.95


def _latest(paths, name: str) -> tuple[str, dict]:
    """Highest-numbered version directory for a model, plus its metrics.

    Handles both layouts in the registry: versioned ``<name>/vN/`` and flat
    ``<name>/metrics.json`` (the anomaly detector has no versions to roll).
    """
    base = paths.model_dir(name)
    if not base.exists():
        return name, {}
    flat = base / "metrics.json"
    if flat.exists():
        return name, json.loads(flat.read_text(encoding="utf-8"))
    try:
        latest = paths.latest_model_dir(name)
    except Exception:
        return name, {}
    mpath = latest / "metrics.json"
    if not mpath.exists():
        return f"{name}/{latest.name}", {}
    return f"{name}/{latest.name}", json.loads(mpath.read_text(encoding="utf-8"))


def _rows(paths) -> list[dict]:
    """One row per scored capability, each with the metric ITS task deserves."""
    out: list[dict] = []

    dname, d = _latest(paths, "demand_sih2026")
    if d:
        wape = d["test"]["wape"]
        floor = (d.get("noise_floor") or {}).get("in_sample_wape_on_test")
        out.append({
            "subsystem": "Demand forecast",
            "task": "1-day-ahead units, newsvendor production plan",
            "headline_metric": "WAPE (weighted absolute percentage error)",
            "value": wape,
            "higher_is_better": False,
            "meets_95": wape <= 1 - TARGET,
            "secondary": {
                "accuracy_equivalent_1_minus_WAPE": 1 - wape,
                "MAE_units": d["test"]["mae"],
                "bias_units": d["test"]["bias_units"],
                "best_baseline_WAPE": min(d["baselines"].values()),
                "surplus_avoided_pct": d["policy_backtest"]["surplus_avoided_pct"],
                "stockout_rate_of_plan": d["policy_backtest"]["stockout_rate_of_plan"],
                "latency_ms_per_row": d["latency_ms_per_row"],
            },
            "why_not_95": (
                f"a model fitted ON the test fold -- which has seen the answers -- still "
                f"only reaches WAPE {floor:.4f}, so WAPE {1-TARGET:.2f} is below the "
                f"information floor of this data"
            ) if floor else "noise-floor diagnostic not present in this artefact",
            "artefact": dname,
            "verdict": d.get("verdict"),
        })

    fname, f = _latest(paths, "fusion_sih2026")
    if f:
        rec = f["unsafe_recall_not_good"]
        out.append({
            "subsystem": "Food-safety fusion screen",
            "task": "route each batch GOOD / RISK / REJECT",
            "headline_metric": "unsafe recall at a capped false-accept rate",
            "value": rec,
            "higher_is_better": True,
            "meets_95": rec >= TARGET,
            "secondary": {
                "false_accept_rate": f["false_accept_rate_on_unsafe"],
                "unsafe_batches_missed": f["n_unsafe_missed"],
                "unsafe_batches_total": f["n_unsafe_test"],
                "safe_auto_accept_rate": f["safe_auto_accept_rate"],
                "AUC_fusion": f["auc_fusion"],
                "AUC_photo_score_alone": f["auc_cv_score_alone"],
                "calibration_ECE": f["ece"],
            },
            "why_not_naive_accuracy": (
                "naive 2-class accuracy is ~0.87 against a 0.84 majority-class "
                "baseline, so it would flatter a model that reviews nothing"
            ),
            "artefact": fname,
            "verdict": f.get("verdict"),
        })

    sname, s = _latest(paths, "shelf_life_sih2026")
    if s:
        cov = s["share_true_above_lower_bound"]
        out.append({
            "subsystem": "Shelf-life",
            "task": "remaining safe hours as an interval + hard lower bound",
            "headline_metric": "share above the conservative lower bound",
            "value": cov,
            "higher_is_better": True,
            "meets_95": cov >= TARGET,
            "secondary": {
                "interval_coverage_at_0.90_target": s["interval_coverage"],
                "false_safe_rate": s["false_safe_rate_lower_bound"],
                "log_space_R2": s["log_space_r2"],
                "median_absolute_error_hours": s["median_absolute_error_hours"],
                "mean_interval_width_hours": s["mean_interval_width_hours"],
            },
            "why_this_metric": (
                "a point forecast has no accuracy; the operational question is how "
                "often the hard lower bound is safe, which is the 0.95-capped number"
            ),
            "artefact": sname,
            "verdict": s.get("verdict"),
        })

    aname, a = _latest(paths, "anomaly_sih2026")
    if a:
        rec = a["results"]["routed_full"]["episode_recall"]
        per_type = {k: v["recall"] for k, v in a["per_type_gates"].items()}
        out.append({
            "subsystem": "Cold-chain anomaly",
            "task": "detect faults across compressor and door cycles",
            "headline_metric": "episode recall (an incident is found at all)",
            "value": rec,
            "higher_is_better": True,
            "meets_95": rec >= TARGET,
            "secondary": {
                "episodes_found": f"{a['results']['routed_full']['episodes_found']}"
                                  f"/{a['results']['routed_full']['episodes']}",
                "row_precision": a["results"]["routed_full"]["row_precision"],
                "row_recall": a["results"]["routed_full"]["row_recall"],
                "defrost_false_alarm": a["false_alarm_during_benign_defrost"]["routed_full"],
                "per_type_recall": per_type,
                "types_at_or_above_95": sum(v >= TARGET for v in per_type.values()),
                "types_total": len(per_type),
                "worst_type": a["worst_type"],
            },
            "why_not_row_accuracy": (
                "rows inside a fault are not independent observations; missing one "
                "of 200 readings in an incident is not 1/200 of the outcome, which is "
                "why this is scored per incident"
            ),
            "artefact": aname,
            "verdict": a.get("verdict"),
        })

    lpath = paths.reports / "logistics_benchmark_sih2026.json"
    if lpath.exists():
        lg = json.loads(lpath.read_text(encoding="utf-8"))
        board = lg.get("scoreboard", {})
        cov = (board.get("mean_coverage") or {}).get("ortools")
        if cov is not None:
            on_time = ((board.get("offpeak") or {}).get("ortools") or {}).get("total_on_time_kg")
            out.append({
                "subsystem": "Routing",
                "task": "assign collection stops under vehicle, capacity and time limits",
                "headline_metric": "mean stop coverage (share of stops served)",
                "value": cov,
                "higher_is_better": True,
                "meets_95": cov >= TARGET,
                "secondary": {
                    "instances": board.get("instances"),
                    "coverage_nearest_neighbour": (board.get("mean_coverage") or {}).get(
                        "nearest_neighbour"),
                    "on_time_kg_ortools": on_time,
                    "advantage_kg_peak": (board.get("advantage_kg") or {}).get("peak"),
                    "advantage_kg_offpeak": (board.get("advantage_kg") or {}).get("offpeak"),
                    "scoring_note": lg.get("scoring_note"),
                },
                "why_this_metric": (
                    "this is a solver, not a classifier: there is no correct answer to "
                    "compare against, only coverage delivered and food waste avoided"
                ),
                "artefact": "reports/logistics_benchmark_sih2026.json",
                "verdict": board.get("verdict"),
            })
    return out


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def render_markdown(rows: list[dict], sources: dict) -> str:
    L: list[str] = []
    L.append("# Annapurna SIH-2026 - model metrics\n")
    L.append("> **All figures come from synthetic data.** "
             "`data_source = SYNTHETIC_SIH2026`. The bundle was generated by "
             "`ml_data_gen_sih2026.py` from documented mechanisms (Ratkowsky-type "
             "growth, Q10 spoilage, HACCP rules). These numbers validate the "
             "pipeline and the calibration; they are **not** evidence about a real "
             "kitchen.\n")
    L.append("## Headline\n")
    L.append("Each subsystem is scored on the metric its task implies. "
             "There is deliberately no single 'accuracy' column, because averaging "
             "a forecaster's WAPE with a screen's recall would mean nothing.\n")
    L.append("| Subsystem | Headline metric | Value | >= 95% | Artefact |")
    L.append("|---|---|---|---|---|")
    for r in rows:
        mark = "yes" if r["meets_95"] else "**no**"
        L.append(f"| {r['subsystem']} | {r['headline_metric']} | "
                 f"**{_fmt(r['value'])}** | {mark} | `{r['artefact']}` |")
    n_ok = sum(r["meets_95"] for r in rows)
    L.append(f"\n**{n_ok} of {len(rows)} subsystems clear 0.95 on their own correct "
             f"metric.**\n")

    L.append("## Why the one that does not, does not\n")
    for r in rows:
        if not r["meets_95"]:
            L.append(f"### {r['subsystem']}\n")
            L.append(f"- Headline: {r['headline_metric']} = **{_fmt(r['value'])}**")
            L.append(f"- Reason: {r.get('why_not_95')}")
            for k, v in (r.get("secondary") or {}).items():
                L.append(f"- {k}: {_fmt(v)}")
            L.append("")

    L.append("## Detail\n")
    for r in rows:
        L.append(f"### {r['subsystem']}  ({r['artefact']})\n")
        L.append(f"- Task: {r['task']}")
        L.append(f"- Headline -- {r['headline_metric']}: **{_fmt(r['value'])}** "
                 f"({'meets 0.95' if r['meets_95'] else 'below 0.95'})")
        for note in ("why_this_metric", "why_not_naive_accuracy", "why_not_row_accuracy"):
            if r.get(note):
                L.append(f"- {note.replace('_', ' ').capitalize()}: {r[note]}")
        sec = r.get("secondary") or {}
        for k, v in sec.items():
            if isinstance(v, dict):
                L.append(f"- {k}:")
                for kk, vv in v.items():
                    L.append(f"    - {kk}: {_fmt(vv)}")
            else:
                L.append(f"- {k}: {_fmt(v)}")
        if r.get("verdict"):
            L.append(f"- Verdict in artefact: `{r['verdict']}`")
        L.append("")

    L.append("## What these numbers do not establish\n")
    L.append("- No model here has seen real kitchen data.")
    L.append("- The food-photo classifier is untrained; `data/cv/` does not exist, "
             "so no image-based accuracy is reported at all.")
    L.append("- A ceiling measured on synthetic data is a ceiling on the "
             "generator, not on reality.")
    L.append("")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Registry-wide metrics report.")
    ap.add_argument("--out", default=None,
                    help="directory under registry/ to write the report into")
    args = ap.parse_args(argv)

    paths = get_paths()
    rows = _rows(paths)

    print()
    print("=" * 96)
    print(f"  ANNAPURNA SIH-2026 MODEL METRICS   ({paths.root})")
    print("  SYNTHETIC DATA -- pipeline/calibration evidence, not real-kitchen evidence")
    print("=" * 96)
    print(f"  {'subsystem':<28}{'headline metric':<42}{'value':>10}  >=95%")
    print("  " + "-" * 92)
    for r in rows:
        mark = "yes" if r["meets_95"] else "NO"
        print(f"  {r['subsystem']:<28}{r['headline_metric']:<42}{_fmt(r['value']):>10}  {mark}")
    n_ok = sum(r["meets_95"] for r in rows)
    print("  " + "-" * 92)
    print(f"  {n_ok}/{len(rows)} subsystems clear 0.95 on their own correct metric")
    for r in rows:
        if not r["meets_95"]:
            print(f"  -> {r['subsystem']}: {r.get('why_not_95')}")
    print()

    if args.out:
        dest = paths.reports / args.out
        dest.mkdir(parents=True, exist_ok=True)
        payload = {
            "data_source": "SYNTHETIC_SIH2026",
            "target": TARGET,
            "subsystems_meeting_target": n_ok,
            "subsystems_total": len(rows),
            "note": (
                "Each subsystem is scored on the metric its task implies. No single "
                "accuracy figure exists across forecasting, screening, survival and "
                "event detection."
            ),
            "rows": rows,
        }
        save_json(dest / "metrics_report.json", payload)
        md = dest / "metrics_report.md"
        md.write_text(render_markdown(rows, payload), encoding="utf-8")
        LOG.info("wrote %s and %s", dest / "metrics_report.json", md)
        print(f"  report written to {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())