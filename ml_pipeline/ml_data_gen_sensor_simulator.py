"""Synthetic batch-level food-safety data.  ALL OUTPUT IS SYNTHETIC and RULE-DERIVED.

    python ml_data_gen_sensor_simulator.py --out data/raw/sim_safety.parquet

Each row is one cooked batch: a category, an elapsed time, a temperature trace with
**realistic irregular sampling** (the IoT gateway batches and occasionally drops),
features summarising it, humidity, and a simulated image "unsafe" probability.

Labels come from the transparent budget rule in ``ml_quality_time_temp.py`` +
``ml_configs_fusion.yaml``, with hidden per-batch variation.  They are NOT
microbiological ground truth and every artefact this writes says so.

Two things this deliberately models that a naive simulator misses:

* **Sampling gaps.**  Real deployments lose the gateway for minutes at a time.  We
  insert outage gaps, and the label is computed from the *gap-aware* integration, so a
  model cannot learn "10 minutes per sample" as a shortcut.
* **Sensor dropout.**  A few samples are NaN.  The feature extractor reports
  ``unobserved_minutes`` instead of inventing exposure.

Performance note: the per-batch Python loop is the bottleneck at 20k rows, so the
simulation is vectorised per scenario and only the trace *sampling* stays in a loop.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from ml_quality_time_temp import MAX_GAP_MIN, budget_used_hours, humidity_mult, load_cfg, summarize, temp_rate
from ml_utils import get_logger, get_paths, save_json

LOG = get_logger("data_gen.sensor")

STEP_MIN = 10
SCENARIOS = ["ambient", "hot_hold", "chilled", "mixed"]
SCEN_P = [0.30, 0.30, 0.25, 0.15]

#: Probability that a given sample is missing (gateway outage / battery).
DROPOUT_P = 0.03
#: Probability that a batch contains one extended outage of 1-5 samples.
OUTAGE_P = 0.22


def _cool(start, target, t, tau):
    return target + (start - target) * np.exp(-t / tau)


def _sample_gaps(rng, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Irregular sample times (minutes from t=0), their durations, and a dropout mask.

    Durations are *wall-clock* gaps; :func:`ml_quality_time_temp.summarize` caps them,
    which is exactly the behaviour we want the model to learn.
    """
    base = np.arange(n) * STEP_MIN + rng.normal(0, STEP_MIN * 0.15, n)
    base = np.maximum.accumulate(np.clip(base, 0.0, None))
    times = base + rng.choice([0.0, 0.0, 0.0, STEP_MIN * 0.5], size=n)  # sub-step jitter
    dt = np.diff(np.concatenate([[0.0], times]))
    if n > 1 and rng.random() < OUTAGE_P:
        k = int(rng.integers(1, min(6, n)))
        start = int(rng.integers(0, max(1, n - k)))
        times[start:start + k] += rng.uniform(3, 12) * STEP_MIN  # 30-120 min outage
        dt = np.diff(np.concatenate([[0.0], times]))
    dropout = rng.random(n) < DROPOUT_P
    return times, np.clip(dt, 0.0, None), dropout


def _trace(rng, scenario: str, n: int, amb: float) -> np.ndarray:
    t = np.arange(n) * STEP_MIN / 60.0
    start = rng.uniform(68, 85)
    if scenario == "ambient":
        T = _cool(start, amb, t, rng.uniform(0.7, 1.5))
    elif scenario == "hot_hold":
        T = np.full(n, rng.uniform(61, 70)) + rng.normal(0, 1.2, n)
        if rng.random() < 0.35:  # equipment failure: the holding oven gives up
            k = int(rng.integers(0, n))
            T[k:] = _cool(T[k], amb, t[: n - k], rng.uniform(0.7, 1.5))
    elif scenario == "chilled":
        T = _cool(start, rng.uniform(2, 6), t, rng.uniform(0.5, 1.2))
    else:  # ambient first, then into the fridge
        k = int(min(n - 1, max(1, rng.uniform(0.5, 4.0) * 60 / STEP_MIN)))
        T = _cool(start, amb, t, rng.uniform(0.7, 1.5))
        T[k:] = _cool(T[k], rng.uniform(2, 6), t[: n - k], rng.uniform(0.5, 1.2))
    return T + rng.normal(0, 0.5, n)


def simulate(n_batches: int = 20000, seed: int = 42, cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_cfg()
    rng = np.random.default_rng(seed)
    cats = list(cfg["categories"])
    cap = float(cfg["targets"]["max_hours_left"])
    lo, hi = cfg["danger_zone_c"]

    rows: list[dict] = []
    for _ in range(n_batches):
        cat = cats[int(rng.integers(len(cats)))]
        budget = float(cfg["categories"][cat]["budget_hours"])
        hours = min(float(rng.exponential(5.0) + 0.25), 30.0)
        n = max(3, int(hours * 60 / STEP_MIN) + 1)
        scen = str(rng.choice(SCENARIOS, p=SCEN_P))
        amb = float(rng.uniform(22, 38))
        hum = float(rng.uniform(40, 92))

        T = _trace(rng, scen, n, amb)
        _, dt, dropout = _sample_gaps(rng, n)
        T_obs = T.copy()
        T_obs[dropout] = np.nan

        used = budget_used_hours(T_obs, STEP_MIN, hum, cfg, dt_min=dt)
        budget_eff = budget * float(rng.lognormal(0, 0.15))  # hidden initial-load variation
        remaining = budget_eff - used

        last = T_obs[~np.isnan(T_obs)]
        rate_now = (float(temp_rate(last[-1:], cfg)[0]) * humidity_mult(hum, cfg)
                    if last.size else float(cfg["rate_multiplier"]["other_danger"]) * humidity_mult(hum, cfg))
        hours_left = float(np.clip(remaining / max(rate_now, 1e-6), 0, cap)) if remaining > 0 else 0.0
        cv_unsafe = float(1 / (1 + np.exp(-(6 * (used / budget_eff - 1) + rng.normal(0, 1.0)))))

        feats = summarize(T_obs, STEP_MIN, cfg, dt_min=dt, max_gap_min=MAX_GAP_MIN)
        rows.append({
            "category": cat,
            "hours_since_prepared": float(dt.sum()),
            "humidity_mean": hum,
            "cv_unsafe_prob": cv_unsafe,
            **feats,
            "hours_left_true": hours_left,
            "unsafe": int(remaining <= 0),
            "scenario": scen,
            "has_outage": bool(np.any(dt > MAX_GAP_MIN)),
            "budget_hours": budget,
            "data_source": "SYNTHETIC",
        })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate SYNTHETIC safety/shelf-life training data.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--group-col", default=None,
                    help="split by this column (batch id / kitchen / day) instead of by row")
    a = ap.parse_args(argv)

    paths = get_paths()
    out = Path(a.out) if a.out else paths.data_raw / "sim_safety.parquet"
    df = simulate(a.n, a.seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    # A quick self-check that the labels actually follow the documented rule.
    rule_ok = bool((df.loc[df.unsafe == 1, "hours_left_true"] == 0).all())
    info = {
        "rows": int(len(df)),
        "unsafe_share": float(df.unsafe.mean()),
        "median_hours_left_true": float(df.hours_left_true.median()),
        "batches_with_sensor_outage": int(df.has_outage.sum()),
        "median_unobserved_minutes": float(df.unobserved_minutes.median()),
        "labels_follow_budget_rule": rule_ok,
        "caveat": "SYNTHETIC and RULE-DERIVED. Proves the pipeline runs2; it is not "
                  "evidence about real food safety or real microbiology.",
    }
    save_json(out.with_suffix(".meta.json"), info)
    print(f"wrote {len(df):,} batches -> {out}")
    print(f"  unsafe share {df.unsafe.mean():.1%}   median hours_left_true "
          f"{df.hours_left_true.median():.1f}   outage batches {int(df.has_outage.sum()):,}")
    print(f"  labels follow the budget rule: {rule_ok}")
    print("  (SYNTHETIC, rule-derived labels: a pipeline check, not evidence about real food)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())