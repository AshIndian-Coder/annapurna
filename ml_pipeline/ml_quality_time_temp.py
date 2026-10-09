"""Time-temperature helpers for the safety and shelf-life models.

The one thing this module exists to get right is **exposure accounting on an unevenly
sampled sensor trace**.  The previous implementation assumed a perfectly regular trace::

    danger_zone_minutes = ((t > lo) & (t < hi)).sum() * step_min

That silently under-counts whenever the trace has a gap.  A batch that sat at 32 C for
four hours while the IoT gateway buffered offline produced *one* reading, so it
reported 10 minutes of danger-zone time instead of 240 and the batch was released to an
NGO.  The Go side already handles this correctly -- ``internal/redisx/dangerzone.go``
increments by real ``dt`` and caps any single step at 15 minutes -- so the Python model
features and the API disagreed.

Here every sample contributes its own duration:

* ``dt_min[i]`` is the time the sample at index ``i`` represents (defaults to a uniform
  step for callers that really do have a regular trace).
* A gap longer than ``max_gap_min`` is **capped**, not extrapolated, and the uncounted
  remainder is reported as ``unobserved_minutes``.  We refuse to invent exposure we
  did not measure; we surface it instead.
* NaN samples (sensor dropout) contribute nothing and are counted separately.

Threshold values and their FSSAI provenance live in ``ml_configs_fusion.yaml``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from ml_utils import get_logger

LOG = get_logger("quality.time_temp")

_CFG_PATH = Path(__file__).resolve().parent / "ml_configs_fusion.yaml"

#: Default cap on a single exposure increment.  Matches
#: ``internal/redisx/dangerzone.go`` (D9 parity) -- both sides must agree or the
#: dashboard and the model will report different danger minutes for the same batch.
MAX_GAP_MIN = 15.0


def load_cfg(path: str | Path | None = None) -> dict:
    """Load the fusion configuration.  Cached per path so hot inference loops stay cheap."""
    p = Path(path or _CFG_PATH)
    if not p.exists():
        raise FileNotFoundError(
            f"fusion config not found at {p}. It ships with the pipeline; do not generate it."
        )
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    validate_cfg(cfg)
    return cfg


def validate_cfg(cfg: dict) -> dict:
    """Fail fast on a config that cannot produce a monotone risk model."""
    required = ("danger_zone_c", "fastest_growth_c", "rate_multiplier", "humidity",
                "categories", "targets")
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"fusion config missing keys: {missing}")
    lo, hi = cfg["danger_zone_c"]
    f0, f1 = cfg["fastest_growth_c"]
    if not lo < f0 <= f1 < hi:
        raise ValueError(
            f"band nesting is wrong: danger_zone_c={cfg['danger_zone_c']} must contain "
            f"fastest_growth_c={cfg['fastest_growth_c']}"
        )
    for name, spec in cfg["categories"].items():
        if spec["budget_hours"] <= 0:
            raise ValueError(f"category {name}: budget_hours must be > 0")
        if not 0.0 <= spec["risk"] <= 1.0:
            raise ValueError(f"category {name}: risk must be in [0,1]")
    if not 0.0 < cfg["targets"]["unsafe_recall"] <= 1.0:
        raise ValueError("targets.unsafe_recall must be in (0,1]")
    return cfg


# --------------------------------------------------------------------------- #
# Rate model
# --------------------------------------------------------------------------- #
def temp_rate(temps: Sequence[float] | np.ndarray, cfg: dict) -> np.ndarray:
    """Relative speed at which the safe-time budget is consumed, per sample.

    Bands are applied cold-hot-last so the ordering is unambiguous: a sample at 65 C is
    ``hot_hold`` even though it is also above ``fastest_growth_c``.
    """
    t = np.asarray(temps, dtype=float)
    lo, hi = cfg["danger_zone_c"]
    f0, f1 = cfg["fastest_growth_c"]
    m = cfg["rate_multiplier"]
    r = np.full(t.shape, float(m["other_danger"]))
    r = np.where((t >= f0) & (t <= f1), float(m["fast_growth"]), r)
    r = np.where(t >= hi, float(m["hot_hold"]), r)
    r = np.where(t <= lo, float(m["chilled"]), r)
    return r


def humidity_mult(h: float, cfg: dict) -> float:
    hc = cfg["humidity"]
    return 1.0 + max(0.0, float(h) - float(hc["threshold_pct"])) * float(hc["slope_per_pct"])


# --------------------------------------------------------------------------- #
# Duration handling
# --------------------------------------------------------------------------- #
def _durations(n: int, step_min: float,
               dt_min: Sequence[float] | np.ndarray | None) -> np.ndarray:
    """Per-sample exposure durations, with gap capping and dropout accounting."""
    if dt_min is None:
        return np.full(n, float(step_min))
    dt = np.asarray(dt_min, dtype=float)
    if dt.shape != (n,):
        raise ValueError(f"dt_min has shape {dt.shape}, expected {(n,)}")
    return dt


def _cap_gaps(dt: np.ndarray, max_gap_min: float) -> tuple[np.ndarray, float]:
    """Cap each increment and return the minutes we deliberately did not count."""
    capped = np.minimum(dt, max_gap_min)
    return capped, float(np.maximum(dt - capped, 0.0).sum())


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def budget_used_hours(temps, step_min: float, humidity: float, cfg: dict, *,
                      dt_min=None, max_gap_min: float = MAX_GAP_MIN) -> float:
    """Hours of safe-time budget consumed by the trace."""
    t = np.asarray(temps, dtype=float)
    if t.size == 0:
        return 0.0
    dt, _ = _cap_gaps(_durations(t.size, step_min, dt_min), max_gap_min)
    r = temp_rate(t, cfg)
    ok = ~np.isnan(t)
    return float((r * dt)[ok].sum() / 60.0 * humidity_mult(humidity, cfg))


def summarize(temps, step_min: float, cfg: dict, *, dt_min=None,
              max_gap_min: float = MAX_GAP_MIN) -> dict[str, float]:
    """Sensor trace -> model features.

    ``danger_zone_minutes`` is the strict interval ``(lo, hi)``; ``hot_hold_minutes``
    and ``chilled_minutes`` are the two safe bands.  ``unobserved_minutes`` is the part
    of the trace that fell outside the gap cap -- reported, never silently absorbed.
    """
    t = np.asarray(temps, dtype=float)
    if t.size == 0:
        raise ValueError("empty temperature trace")
    dt, unobserved = _cap_gaps(_durations(t.size, step_min, dt_min), max_gap_min)
    ok = ~np.isnan(t)
    dt_ok = np.where(ok, dt, 0.0)

    lo, hi = cfg["danger_zone_c"]
    in_danger = ok & (t > lo) & (t < hi)
    hot = ok & (t >= hi)
    cold = ok & (t <= lo)
    last = t[ok][-1] if ok.any() else float("nan")
    return {
        "danger_zone_minutes": float(dt_ok[in_danger].sum()),
        "hot_hold_minutes": float(dt_ok[hot].sum()),
        "chilled_minutes": float(dt_ok[cold].sum()),
        "current_temp_c": float(last),
        "mean_temp_c": float(np.average(t[ok], weights=dt_ok[ok])) if ok.any() else float("nan"),
        "unobserved_minutes": float(unobserved),
        "n_dropout_samples": int((~ok).sum()),
    }


def trace_from_timestamps(temps, timestamps, cfg: dict, *,
                          max_gap_min: float = MAX_GAP_MIN,
                          assume_step_before_first: float = 10.0) -> tuple[np.ndarray, dict[str, float]]:
    """Build ``dt_min`` from an ISO timestamp series and summarise in one call.

    ``timestamps`` may be ``datetime`` objects or ISO-8601 strings.  The first sample is
    assumed to represent ``assume_step_before_first`` minutes, which matches how the
    simulator builds traces.
    """
    import pandas as pd

    ts = pd.to_datetime(pd.Index(timestamps), utc=True)
    if len(ts) != len(temps):
        raise ValueError(f"{len(temps)} temperatures vs {len(ts)} timestamps")
    deltas = pd.Series(ts).diff().dt.total_seconds().to_numpy() / 60.0
    deltas[0] = assume_step_before_first
    deltas = np.clip(np.nan_to_num(deltas, nan=assume_step_before_first), 0.0, None)
    return deltas, summarize(temps, float(np.median(deltas[deltas > 0]) or assume_step_before_first),
                             cfg, dt_min=deltas, max_gap_min=max_gap_min)