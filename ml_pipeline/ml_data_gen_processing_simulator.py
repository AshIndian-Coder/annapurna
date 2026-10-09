"""Synthetic processing-unit data with LABELLED injected inefficiencies. ALL OUTPUT IS SYNTHETIC.

Run from ml/:  python -m data_gen.processing_simulator --out data/raw/sim_processing.parquet
Episode types: yield_drop (raw-material loss), downtime_spike, energy_spike, overproduction, energy_drift.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from ml_utils import get_paths, save_json

import numpy as np
import pandas as pd

WEEK = [1.0, 1.0, 1.0, 1.0, 0.95, 0.6, 0.3]
TYPES = ["yield_drop", "downtime_spike", "energy_spike", "overproduction", "energy_drift"]


def simulate(days: int = 540, units: int = 4, seed: int = 7, start: str = "2024-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days, freq="D")
    frames = []
    for u in range(units):
        base_raw = rng.uniform(4000, 9000)
        base_loss, base_down = rng.uniform(0.03, 0.08), rng.uniform(0.03, 0.07)
        base_epk = rng.uniform(0.18, 0.45)
        loss = base_loss * rng.lognormal(0, 0.10, days)
        down = base_down * rng.lognormal(0, 0.20, days)
        epk = base_epk * rng.lognormal(0, 0.05, days)
        op = np.clip(rng.normal(0.03, 0.015, days), 0, 0.10)
        ep_id, ep_type = np.zeros(days, int), np.array([""] * days, dtype=object)

        t, k = 45, 0
        while t < days - 35:
            typ = TYPES[int(rng.integers(len(TYPES)))]
            n = int(rng.integers(20, 31)) if typ == "energy_drift" else int(rng.integers(1, 4))
            k += 1
            sl = slice(t, t + n)
            if typ == "yield_drop":
                loss[sl] *= rng.uniform(2.2, 3.5)
            elif typ == "downtime_spike":
                down[sl] *= rng.uniform(3, 5)
            elif typ == "energy_spike":
                epk[sl] *= rng.uniform(1.4, 1.9)
            elif typ == "overproduction":
                op[sl] += rng.uniform(0.12, 0.25)
            else:
                epk[sl] *= 1 + 0.25 * np.arange(1, n + 1) / n
            ep_id[sl], ep_type[sl] = k + 1000 * u, typ
            t += n + int(rng.integers(20, 40))

        raw = base_raw * np.array([WEEK[d.dayofweek] for d in dates]) * rng.lognormal(0, 0.05, days)
        loss, down = np.clip(loss, 0, 0.5), np.clip(down, 0, 0.9)
        out = raw * (1 - loss)
        frames.append(pd.DataFrame({
            "date": dates, "unit_id": f"U{u+1}", "raw_in_kg": raw, "output_kg": out,
            "ordered_kg": out / (1 + op), "runtime_h": 16 * (1 - down), "downtime_h": 16 * down,
            "energy_kwh": epk * out, "episode_id": ep_id, "episode_type": ep_type, "data_source": "SYNTHETIC",
        }))
    return pd.concat(frames, ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate SYNTHETIC processing-unit data.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--days", type=int, default=540)
    ap.add_argument("--units", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    paths = get_paths()
    out = Path(a.out) if a.out else paths.data_raw / "sim_processing.parquet"
    df = simulate(a.days, a.units, a.seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    save_json(out.with_suffix(".meta.json"), {
        "unit_days": int(len(df)),
        "injected_episodes": int(df.loc[df.episode_id > 0, "episode_id"].nunique()),
        "episode_types": sorted(set(df.loc[df.episode_id > 0, "episode_type"])),
        "caveat": "SYNTHETIC with LABELLED injected inefficiencies; detection precision/recall "
                  "on it is a pipeline check, not evidence about a real plant",
    })
    print(f"wrote {len(df):,} unit-days, {df.loc[df.episode_id>0,'episode_id'].nunique()} "
          f"injected episodes -> {out} (SYNTHETIC)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
