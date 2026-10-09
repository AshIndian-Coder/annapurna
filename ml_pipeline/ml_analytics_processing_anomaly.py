"""Processing-unit inefficiency detection (PS 26234, requirements #5 and #6).

    python ml_analytics_processing_anomaly.py --data data/raw/sim_processing.parquet

Three complementary detectors, because no single one covers the failure modes:

* **EWMA** for *spikes* -- a single bad day (yield drop, machine breakdown).
* **CUSUM** for *drifts* -- the slow 25 %-over-20-days energy creep that a level-based
  detector never fires on.
* **IsolationForest** for *joint* oddities -- a day where each metric alone looks fine
  but the combination is impossible (low output with normal downtime and normal energy).

Parity with the Go backend (D9)
-------------------------------
``downtime_pct`` here is ``downtime_min / runtime_min * 100`` -- the *same* formula as
``internal/processing.DowntimePct`` in the Go service, with ``runtime`` meaning
productive runtime as the API contract defines it.  The previous version computed
``100 * downtime / (downtime + runtime)``, which is the share-of-clock-time
definition.  For the spec's worked example (35 min down, 420 min runtime) Go returns
8.33 and the Python side returned 7.69: the dashboard, the anomaly detector and the
ESG report disagreed by 8 % on the same day.  ``ml_tests_test_processing_parity.py``
now asserts the two agree.

Evaluation is against **injected** episodes, so precision/recall here is a pipeline
check on a simulator, never evidence about a real plant.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from ml_utils import format_table, get_logger, get_paths, save_json, set_determinism

LOG = get_logger("analytics.anomaly")

#: KPI vector fed to the unsupervised detectors.
METRICS = ["loss_pct", "downtime_pct", "energy_per_kg", "overprod_pct"]
LABEL = {"loss_pct": "raw-material loss", "downtime_pct": "machine downtime",
         "energy_per_kg": "energy per kg", "overprod_pct": "overproduction"}
#: Guard rail: with fewer than this many days, IsolationForest is not fitted (spec 07 §9).
MIN_DAYS_FOR_IF = 14

CAVEAT = ("episodes are injected into SYNTHETIC data; the precision/recall below is a "
          "pipeline check, not evidence about a real plant")


# --------------------------------------------------------------------------- #
# KPIs
# --------------------------------------------------------------------------- #
def derived(df: pd.DataFrame) -> pd.DataFrame:
    """Per-day KPIs.  These four numbers are the entire input to the detectors.

    ``downtime_pct`` uses ``runtime`` (productive minutes) as the denominator, matching
    ``internal/processing.DowntimePct`` in the Go backend.
    """
    d = df.sort_values(["unit_id", "date"]).reset_index(drop=True).copy()
    raw = d["raw_in_kg"].astype(float)
    out = d["output_kg"].astype(float)
    downtime = d["downtime_h"].astype(float) * 60.0   # hours -> minutes (API unit is minutes)
    runtime = d["runtime_h"].astype(float) * 60.0
    ordered = d["ordered_kg"].astype(float).clip(lower=1e-9)
    d["loss_pct"] = 100.0 * (raw - out) / raw.clip(lower=1e-9)
    d["downtime_pct"] = np.where(runtime > 0, 100.0 * downtime / runtime.clip(lower=1e-9), 0.0)
    d["energy_per_kg"] = d["energy_kwh"].astype(float) / out.clip(lower=1e-9)
    d["overprod_pct"] = 100.0 * (out - ordered) / ordered
    return d


# --------------------------------------------------------------------------- #
# Detectors
# --------------------------------------------------------------------------- #
def ewma_upper(x: np.ndarray, warm: int = 14, lam: float = 0.3, k: float = 3.0):
    """Flag ``x[t]`` when it is ``k`` sigma above the EWMA level.

    Flagged points do **not** update the level or the variance.  That is the whole
    point: a real equipment failure must not be absorbed into "normal" after three bad
    days, which is what an unguarded EWMA does.
    """
    x = np.asarray(x, dtype=float)
    if x.size <= warm:
        return np.zeros(x.size, bool), np.zeros(x.size), np.full(x.size, float("nan"))
    m, v = float(np.mean(x[:warm])), float(np.var(x[:warm])) + 1e-9
    flags, z, exp_ = np.zeros(x.size, bool), np.zeros(x.size), np.full(x.size, m)
    for t in range(warm, x.size):
        z[t], exp_[t] = (x[t] - m) / np.sqrt(v), m
        if z[t] > k:
            flags[t] = True
            continue
        v = lam * (x[t] - m) ** 2 + (1 - lam) * v
        m = lam * x[t] + (1 - lam) * m
    return flags, z, exp_


def cusum_upper(x: np.ndarray, warm: int = 14, k: float = 0.5, h: float = 5.0):
    """Tabular CUSUM for persistent upward drift.  Resets on alarm."""
    x = np.asarray(x, dtype=float)
    if x.size <= warm:
        return np.zeros(x.size, bool), float("nan")
    mu, sd = float(np.mean(x[:warm])), float(np.std(x[:warm])) + 1e-9
    s, flags = 0.0, np.zeros(x.size, bool)
    for t in range(warm, x.size):
        s = max(0.0, s + (x[t] - mu) / sd - k)
        if s > h:
            flags[t], s = True, 0.0
    return flags, mu


def _standardise(g: pd.DataFrame, warm: int) -> np.ndarray:
    """Z-score against the *first* ``warm`` days only, so the detectors never see
    post-incident statistics and quietly normalise a persistent failure away."""
    X = g[METRICS].to_numpy(float)
    mu = X[:warm].mean(axis=0)
    sd = X[:warm].std(axis=0) + 1e-9
    return (X - mu) / sd


def detect(df: pd.DataFrame, *, warm: int = 14, lam: float = 0.3, contamination: float = 0.05,
           seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all three detectors per unit.  Returns ``(per_day_flags, alerts)``."""
    d = derived(df)
    alerts: list[dict] = []
    frames = []
    for unit, g in d.groupby("unit_id", sort=False):
        g = g.reset_index(drop=True)
        n = len(g)
        ew = np.zeros(n, bool)
        cu = np.zeros(n, bool)
        for m in METRICS:
            x = g[m].to_numpy(float)
            f, z, exp_ = ewma_upper(x, warm, lam)
            c, mu = cusum_upper(x, warm)
            ew |= f
            cu |= c
            base_sd = float(np.std(x[:warm])) + 1e-9
            for t in np.flatnonzero(f):
                alerts.append(dict(date=g.date[t], unit_id=unit, metric=m, value=float(x[t]),
                                   expected=float(exp_[t]), z=float(z[t]), method="EWMA"))
            for t in np.flatnonzero(c):
                alerts.append(dict(date=g.date[t], unit_id=unit, metric=m, value=float(x[t]),
                                   expected=float(mu), z=float((x[t] - mu) / base_sd),
                                   method="CUSUM"))

        # IsolationForest is fitted on the first 60 % of *post-warmup* days only and then
        # applied to everything, so it does not score its own training window.
        isf = np.zeros(n, bool)
        if n > warm * 2:
            Z = _standardise(g, warm)
            fit_end = max(warm + 20, int(0.6 * n))
            iso = IsolationForest(n_estimators=200, contamination=contamination,
                                  random_state=seed, n_jobs=-1).fit(Z[warm:fit_end])
            isf[fit_end:] = iso.predict(Z[fit_end:]) == -1
            isf[:warm] = False
        frames.append(g.assign(f_ewma=ew, f_cusum=cu, f_if=isf))

    res = pd.concat(frames, ignore_index=True)
    res["n_methods"] = res[["f_ewma", "f_cusum", "f_if"]].sum(axis=1).astype(int)
    al = pd.DataFrame(alerts)
    if not al.empty:
        al["severity"] = np.where(al.z > 5.0, "HIGH", "MEDIUM")
        al["label"] = al.metric.map(LABEL)
    return res, al


# --------------------------------------------------------------------------- #
# Evaluation against injected episodes
# --------------------------------------------------------------------------- #
def evaluate(res: pd.DataFrame, warm: int = 14) -> dict:
    out: dict = {}
    r = res.copy()
    r["is_ep"] = r.episode_id > 0
    # one day of grace *after* an episode: a detection that lands the day it ends is a
    # useful alert, not a false alarm.
    r["near_ep"] = r.is_ep | r.groupby("unit_id")["is_ep"].shift(1, fill_value=False)
    r = r[r.groupby("unit_id").cumcount() >= warm].copy()

    normal_days = int((~r.near_ep).sum())
    eps = r[r.is_ep].groupby("episode_id").agg(
        start=("date", "min"), end=("date", "max"), typ=("episode_type", "first"),
        unit=("unit_id", "first"))

    methods = {
        "EWMA": r.f_ewma,
        "CUSUM": r.f_cusum,
        "IsolationForest": r.f_if,
        "EWMA_or_CUSUM": r.f_ewma | r.f_cusum,
        "any_method": r.n_methods >= 1,
        "at_least_2_methods": r.n_methods >= 2,
    }
    for name, flag in methods.items():
        flag = flag.astype(bool)
        tp, fp = int((flag & r.near_ep).sum()), int((flag & ~r.near_ep).sum())
        detected, delays = 0, []
        for _, e in eps.iterrows():
            w = r[(r.unit_id == e.unit) & (r.date >= e.start) & (r.date <= e.end + pd.Timedelta(days=1))]
            hit = w[flag.loc[w.index]]
            if len(hit):
                detected += 1
                delays.append((hit.date.min() - e.start).days)
        out[name] = {
            "episode_recall": detected / max(len(eps), 1),
            "median_delay_days": float(np.median(delays)) if delays else None,
            "day_precision": tp / max(tp + fp, 1),
            "false_alarms_per_unit_month": fp / max(normal_days, 1) * 30,
        }
    out["ewma_or_cusum_by_type"] = {
        typ: {"episodes": len(g),
              "detected": int(sum(bool((r[(r.unit_id == e.unit) & (r.date >= e.start)
                                        & (r.date <= e.end + pd.Timedelta(days=1))]
                                        .f_ewma | r[(r.unit_id == e.unit) & (r.date >= e.start)
                                                    & (r.date <= e.end + pd.Timedelta(days=1))]
                                        .f_cusum).any())
                                    for _, e in g.iterrows()))}
        for typ, g in eps.groupby("typ")
    }
    out["n_episodes"] = int(len(eps))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Detect processing inefficiencies.")
    ap.add_argument("--data", default=None)
    ap.add_argument("--contamination", type=float, default=0.05)
    ap.add_argument("--warm", type=int, default=MIN_DAYS_FOR_IF)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    set_determinism(a.seed)

    paths = get_paths()
    data = Path(a.data) if a.data else paths.data_raw / "sim_processing.parquet"
    if not data.exists():
        LOG.error("missing %s -- run ml_data_gen_processing_simulator first", data)
        return 2
    df = pd.read_parquet(data)

    with __import__("ml_utils").Timer("detect"):
        res, alerts = detect(df, warm=a.warm, contamination=a.contamination, seed=a.seed)
    metrics = evaluate(res, warm=a.warm)
    metrics["data_source"] = sorted(df.get("data_source", pd.Series(["UNKNOWN"])).unique())
    metrics["caveat"] = CAVEAT
    metrics["downtime_pct_definition"] = ("downtime_min / runtime_min * 100 -- identical to "
                                          "Go internal/processing.DowntimePct (D9 parity)")

    out_dir = paths.model_dir("anomaly")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "metrics.json", metrics)
    save_json(out_dir / "config.json", {"warm": a.warm, "lam": 0.3, "contamination": a.contamination,
                                        "metrics": list(METRICS), "seed": a.seed})
    if len(alerts):
        alerts.to_csv(out_dir / "alerts.csv", index=False)
    # per-day flags feed the dashboard's "why is this day flagged" drill-down
    res.to_parquet(out_dir / "daily_flags.parquet", index=False)

    print(f"{metrics['n_episodes']} injected episodes (SYNTHETIC)")
    rows = [{"method": k, "episode_recall": round(v["episode_recall"], 4),
             "median_delay_days": v["median_delay_days"],
             "day_precision": round(v["day_precision"], 4),
             "false_alarms_per_unit_month": round(v["false_alarms_per_unit_month"], 3)}
            for k, v in metrics.items() if isinstance(v, dict) and "episode_recall" in v]
    print(format_table(rows))
    print("  by episode type (EWMA or CUSUM):", metrics["ewma_or_cusum_by_type"])
    print(f"  artifacts -> {out_dir}")
    print(f"  ({CAVEAT})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())