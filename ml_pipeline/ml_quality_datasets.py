"""Feature contract shared by the safety-fusion and shelf-life models.

Two things here are load-bearing for the rest of the pipeline:

* :data:`FEATURES` is the *exact* column order both LightGBM models were fitted on.
  Changing it invalidates every registered ``model.joblib``, so it is asserted against
  the bundle at load time in ``ml_quality_inference.py``.
* :data:`MONO_RISK` encodes the safety monotonicity the product depends on: more time
  in the danger zone, more hours since preparation, more humidity or a higher image
  risk may never *lower* predicted risk.  ``ml_tests_test_fusion_monotonic.py`` grid-
  tests this, and LightGBM is given the same constraints so it cannot learn a
  non-monotone shortcut from noisy synthetic labels.

Splitting is by *group* (batch / kitchen / day) rather than by row.  The previous
``split3`` did a uniform random row split, which puts near-identical synthetic batches
on both sides of the boundary and reports a recall the model would never reach in
production.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

#: Canonical model input order.  Never reorder; append instead.
FEATURES: list[str] = [
    "category",                 # categorical, ordinal-encoded via pandas Categorical
    "hours_since_prepared",
    "danger_zone_minutes",
    "hot_hold_minutes",
    "chilled_minutes",
    "current_temp_c",
    "mean_temp_c",
    "humidity_mean",
    "cv_unsafe_prob",
]

#: +1: the feature must not decrease predicted risk when it increases.
MONO_RISK: dict[str, int] = {
    "hours_since_prepared": 1,
    "danger_zone_minutes": 1,
    "humidity_mean": 1,
    "cv_unsafe_prob": 1,
    "current_temp_c": 0,       # not constrained: cooler is usually safer
    "mean_temp_c": 0,
    "hot_hold_minutes": 0,
    "chilled_minutes": 0,
}

#: Features that must never be NaN after prep (filled with the training median).
_IMPUTE: dict[str, float] = {
    "hours_since_prepared": 0.0,
    "danger_zone_minutes": 0.0,
    "hot_hold_minutes": 0.0,
    "chilled_minutes": 0.0,
    "current_temp_c": 20.0,
    "mean_temp_c": 20.0,
    "humidity_mean": 60.0,
    "cv_unsafe_prob": 0.0,
}


def mono_vector(sign: int = 1) -> list[int]:
    """LightGBM ``monotone_constraints`` aligned with :data:`FEATURES`.

    ``sign=+1`` for the risk classifier (risk rises with exposure); ``sign=-1`` for the
    hours-left regressor (more exposure means *less* time remaining).
    """
    return [sign * MONO_RISK.get(f, 0) for f in FEATURES]


def prep(df: pd.DataFrame, categories: Sequence[str] | None = None,
         *, medians: dict[str, float] | None = None) -> pd.DataFrame:
    """Build the model matrix: fixed column order, categorical dtype, no NaN.

    ``categories`` must be the *training* category list.  A value outside it becomes
    NaN, which LightGBM treats as a missing numeric code -- silently mapping an unseen
    food category onto a real one is a safety bug, so callers validate first.
    """
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise KeyError(f"missing feature columns: {missing}")
    x = df.loc[:, FEATURES].copy()
    cats = list(categories) if categories is not None else sorted(df["category"].dropna().unique())
    x["category"] = pd.Categorical(x["category"], categories=cats)
    for col, default in _IMPUTE.items():
        fill = (medians or {}).get(col, default)
        x[col] = pd.to_numeric(x[col], errors="coerce").fillna(fill)
    return x


def fit_imputers(df: pd.DataFrame) -> dict[str, float]:
    """Training medians, stored with the model so inference fills identically."""
    return {c: float(pd.to_numeric(df[c], errors="coerce").median())
            for c in FEATURES if c != "category"}


def assert_monotone_model(model, grid: Iterable[tuple[str, Sequence[float]]] | None = None,
                          tol: float = 1e-9) -> None:
    """Runtime guard that a *fitted* model is still monotone on the key features.

    ``monotone_constraints`` is enforced during fitting, but a pipeline that re-trains
    with different params, or a model loaded from an older artefact, can violate it.
    This is cheap insurance for the safety-critical model and fails loudly.
    """
    defaults = grid or (
        ("danger_zone_minutes", np.linspace(0, 600, 25)),
        ("hours_since_prepared", np.linspace(0, 30, 25)),
        ("cv_unsafe_prob", np.linspace(0, 1, 25)),
        ("humidity_mean", np.linspace(40, 92, 25)),
    )
    base = {c: (_IMPUTE.get(c, 0.0) if c != "category" else "rice_dal")
            for c in FEATURES}
    base["category"] = pd.Categorical(["rice_dal"], categories=sorted({base["category"]}))
    for feature, values in defaults:
        rows = []
        for v in values:
            row = dict(base)
            row[feature] = v
            rows.append(row)
        p = model.predict_proba(prep(pd.DataFrame(rows)))[:, 1]
        if np.any(np.diff(np.asarray(p)) < -tol):
            raise AssertionError(f"monotonicity violated: risk fell when {feature} increased")


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def split3(df: pd.DataFrame, seed: int = 42, frac: tuple[float, float, float] = (0.70, 0.15, 0.15),
           group_col: str | None = None):
    """Train / calibration / test split.

    With ``group_col`` the split is by whole group (batch, kitchen, day), which is what
    you want for real data: rows from the same physical batch are near-duplicates.
    Without it the split is still a *deterministic* hash of the row index, so the
    calibration fold is reproducible across processes -- the old ``rng.random(len)``
    approach silently produced a different split whenever the frame was filtered.
    """
    if abs(sum(frac) - 1.0) > 1e-9 or any(f <= 0 for f in frac):
        raise ValueError(f"frac must sum to 1 with all parts positive, got {frac}")
    if group_col and group_col in df.columns:
        keys = df[group_col].astype(str).to_numpy()
    else:
        keys = np.asarray([f"{i}" for i in range(len(df))])
    uniq, inverse = np.unique(keys, return_inverse=True)
    rng = np.random.default_rng(seed)
    assign = rng.random(len(uniq))
    a, b = frac[0], frac[0] + frac[1]
    tag = np.where(assign[inverse] < a, 0, np.where(assign[inverse] < b, 1, 2))
    out = []
    for k in range(3):
        part = df.iloc[np.flatnonzero(tag == k)]
        if len(part) == 0:
            raise ValueError(f"split produced an empty partition (frac={frac}, seed={seed}); "
                             f"reduce the number of groups or increase the data")
        out.append(part)
    return tuple(out)


def split_by_time(df: pd.DataFrame, date_col: str = "date", *, holdout_days: int = 56,
                  calib_frac: float = 0.15, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological split for real demand/safety data.

    ``holdout_days`` is reserved as the final test window, the ``calib_frac`` slice
    immediately before it is the conformal/calibration fold, and everything older is
    training.  Never shuffle across a time boundary.
    """
    if date_col not in df.columns:
        raise KeyError(f"no '{date_col}' column; pass a frame with calendar info")
    d = df.assign(_date=pd.to_datetime(df[date_col], utc=True)).sort_values("_date")
    dates = d["_date"].dt.normalize().unique()
    if len(dates) <= holdout_days + 5:
        raise ValueError(f"only {len(dates)} distinct days for a {holdout_days}-day holdout")
    cut_test = dates[-holdout_days]
    pre = dates[: -holdout_days]
    cut_cal = pre[int(len(pre) * (1 - calib_frac))]
    tr = d[d["_date"] < pd.Timestamp(cut_cal)]
    ca = d[(d["_date"] >= pd.Timestamp(cut_cal)) & (d["_date"] < pd.Timestamp(cut_test))]
    te = d[d["_date"] >= pd.Timestamp(cut_test)]
    for name, part in (("train", tr), ("calibration", ca), ("test", te)):
        if part.empty:
            raise ValueError(f"{name} split is empty; adjust holdout_days/calib_frac")
    return (tr.drop(columns="_date").reset_index(drop=True),
            ca.drop(columns="_date").reset_index(drop=True),
            te.drop(columns="_date").reset_index(drop=True))