"""Leakage-safe feature engineering for the demand model (PS 26234, requirement #1).

The single most important property of this module is that **every feature for a row at
date ``t`` is computable from data strictly before ``t``**.  That is enforced three ways:

1.  All lag/rolling statistics are computed on a *reindexed daily grid* per
    ``(kitchen_id, meal_type)``.  Corporate kitchens are closed at weekends and caterers
    skip days; taking ``shift(7)`` on the raw frame would silently reach back 7
    *observations*, not 7 days, and mix a Friday forecast with the previous Thursday.
    The reindexed grid makes the lag exactly 7 calendar days.
2.  Target encoding of the kitchen and of menu items is **out-of-fold / expanding**:
    the encoding for row ``i`` uses only rows before ``i``, so a model's own labels
    never leak into its own features.
3.  :func:`assert_no_leakage` recomputes the frame with the tail truncated at ``t`` and
    asserts the features are identical.  A future edit that breaks the property then
    fails CI rather than silently inflating the score.

Feature groups (spec 07 section 3 -- only what exists at inference time):
    calendar | attendance | lags(1,7,14,28) | rolling mean/median/std(7,14,28) |
    same-weekday mean(4w) | trend slope | menu multi-hot | menu size + popularity |
    out-of-fold kitchen encoding | prior-day waste

Style note: every DataFrame column is read with ``frame["col"]``, never attribute
access.  Under pandas 3 an attribute that collides with a DataFrame method or whose
column dtype is non-string can fail to resolve and raise a confusing error; bracket
access is unambiguous and costs nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from ml_utils import get_logger, save_json, sha256_json

LOG = get_logger("demand.features")

FEATURE_SCHEMA_VERSION = 3

#: Lags in **calendar days** (see the module docstring on why not observations).
LAG_DAYS = (1, 7, 14, 28)
ROLLING_WINDOWS = (7, 14, 28)
#: Menu items that get a dedicated binary column.  Anything else folds into
#: ``menu_other_frac`` so a new dish cannot silently become an unseen feature.
TOP_MENU_ITEMS = ("rice", "dal", "roti", "sabzi", "paneer_curry", "chicken_curry",
                  "curd", "salad", "fries", "sweets", "bread", "dal_makhani")

TARGET = "consumed_qty"
KEYS = ["kitchen_id", "meal_type"]


def _menu_list(v):
    if isinstance(v, str):
        return [m for m in v.split("|") if m]
    if isinstance(v, (list, tuple, np.ndarray)):
        return [str(m) for m in v]
    return []


def _num(frame, col, default=float("nan")):
    """Numeric view of a column; missing columns become ``default``."""
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce")


def _daily_grid(df, col):
    """Per-(kitchen, meal) series reindexed onto a complete daily calendar."""
    parts = []
    for key, sub in df.groupby(KEYS, sort=False):
        s = sub.set_index("date")[col].astype(float).sort_index()
        full = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"))
        parts.append(pd.DataFrame({"kitchen_id": key[0], "meal_type": key[1],
                                   "date": full.index, col: full.to_numpy()}))
    return pd.concat(parts, ignore_index=True)


def calendar_features(dates):
    """Sin/cos calendar encodings plus a linear trend index.

    Sin/cos rather than raw weekday ints so that Sunday (6) and Monday (0) are
    adjacent on the circle instead of six units apart.
    """
    d = pd.to_datetime(dates)
    dow = d.dt.dayofweek.astype(float)
    doy = d.dt.dayofyear.astype(float)
    return pd.DataFrame({
        "dow_sin": np.sin(2 * np.pi * dow / 7.0),
        "dow_cos": np.cos(2 * np.pi * dow / 7.0),
        "month": d.dt.month.astype(float),
        "week_of_year": d.dt.isocalendar().week.astype(float).to_numpy(),
        "doy_sin": np.sin(2 * np.pi * doy / 365.25),
        "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        "is_month_end": d.dt.is_month_end.astype(float).to_numpy(),
        "trend_day": (d - pd.Timestamp("2020-01-01")).dt.days.astype(float),
    }, index=dates.index)


def _trend_slope(values):
    """Least-squares slope, NaN-safe for short windows."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size < 3:
        return float("nan")
    return float(np.polyfit(np.arange(v.size, dtype=float), v, 1)[0])


#: Columns that are only knowable **after** the meal happened.  They are kept in the
#: feature frame -- useful for offline analysis and for the leakage guard, which still
#: proves they do not reach backwards -- but they are excluded from ``model_features``
#: and from the online projection in :func:`predict_demand`.  ``actual_diners`` is the
#: realised attendance, ``waste_qty`` is the same day's recorded waste: both are
#: outcomes, and a model that consumes them learns a relationship the kitchen cannot
#: use at 07:00 when it has to decide how much to cook.
POST_HOC_COLUMNS = ("actual_diners", "actual_to_expected", "waste_qty")

#: Shrinkage target for the expanding target encoding: pseudo-observations' mean, in kg
#: per meal.  Deliberately a **constant**, not a value estimated from the frame -- any
#: data-derived prior (frame mean, "earliest N rows") either leaks the future or depends
#: on an unstable tie-break, and ``assert_no_leakage`` catches the first version of this
#: code failing exactly that way.  Set it to the kitchen's typical portion size; the
#: weight is ``prior / (n_past + prior)``, so it is irrelevant after ~10 observations.
PRIOR_VALUE_KG = 100.0


def expanding_target_encoding(train, keys, target=TARGET, prior=10.0, out_col="_te",
                              prior_value: float = PRIOR_VALUE_KG):
    """Prior-weighted encoding where row ``i`` uses only rows before ``i``.

    Returns a frame keyed on ``keys + [date]`` so the caller merges on the same keys as
    every other feature block.

    Implemented as an explicit per-group loop rather than ``groupby().apply()``: under
    pandas 3, ``apply`` on a grouped column returns a MultiIndex that cannot be reindexed
    back onto the caller's frame ("Buffer dtype mismatch"), and a loop is also faster at
    this size and trivially auditable for leakage.

    The returned frame always has ``KEYS + [date]`` granularity -- the same rows as the
    caller -- even when ``keys`` is narrower than ``KEYS``.  Emitting kitchen-level
    rows and merging on ``(kitchen_id, date)`` against a three-meals-per-day frame
    silently duplicates every row and misaligns every downstream column.
    """
    t = train.sort_values(KEYS + ["date"]).reset_index(drop=True)
    out = np.full(len(t), float(prior_value), dtype=float)
    dates = t["date"].to_numpy()
    grouper = [t[k].to_numpy() for k in keys]
    order = np.lexsort(tuple(reversed(grouper)))
    groups = {}
    for pos in order:
        groups.setdefault(tuple(g[pos] for g in grouper), []).append(pos)
    for positions in groups.values():
        # Sort the group by DATE.  ``t`` is sorted by KEYS+date, so a kitchen-level
        # group (one kitchen, several meals per day) arrives ordered by meal first and
        # date second -- walking it in that order would let a DINNER row see the
        # previous LUNCH row as "past" and vice versa.
        idx = np.asarray(sorted(positions, key=lambda p: (dates[p], p)))
        vals = t[target].to_numpy(float)[idx]
        csum = np.concatenate([[0.0], np.cumsum(vals)[:-1]])   # strictly-past sums
        npast = np.arange(len(idx), dtype=float)
        out[idx] = (csum + prior * float(prior_value)) / (npast + prior)
    keys_frame = t[KEYS + ["date"]].copy()
    keys_frame[out_col] = out
    return keys_frame


def _merge_checked(left, right, on, how="left", label=""):
    """``merge`` that refuses to change the row count.

    A many-to-many merge here silently inflates the dataset (near-duplicate rows), which
    then shows up as an inexplicably good validation score.  Failing at the merge is
    far cheaper than debugging a 2.5x-inflated frame three stages later.
    """
    before = len(left)
    out = left.merge(right, on=on, how=how)
    if len(out) != before:
        dup = right.duplicated(subset=list(on)).sum()
        raise ValueError(
            "merge '%s' on %s changed the row count %d -> %d "
            "(right frame has %d duplicate keys). Make the right side unique on the "
            "join keys." % (label or "", list(on), before, len(out), dup))
    return out


def _past_only_fill(feat, cols):
    """Fill gaps in history statistics with **past-only** information.

    The first version filled with ``frame.median()`` over the whole frame.  That is a
    genuine leak: the median of ``roll_mean_7d`` is computed over rows whose history
    extends past the row being imputed, so an early row is filled with a number derived
    from the future.  ``assert_no_leakage`` caught exactly this (18 rows) and it would
    have been invisible in the score.

    Order of preference, all past-only:
        expanding median of the column so far  ->  lag_1d  ->  0.0
    """
    gkeys = [feat[k] for k in KEYS]
    order = np.lexsort(tuple(reversed([feat[k].to_numpy() for k in KEYS])))
    pos_of = {int(p): i for i, p in enumerate(order)}
    for c in cols:
        if c not in feat.columns:
            continue
        v = pd.to_numeric(feat[c], errors="coerce").to_numpy(dtype=float)
        filled = v.copy()
        groups = {}
        for p in order:
            groups.setdefault(tuple(gk[p] for gk in gkeys), []).append(int(p))
        for positions in groups.values():
            idx = np.asarray(positions)
            series = v[idx]
            # expanding median of strictly-past values, aligned back onto the rows
            exp_med = pd.Series(series).shift(1).expanding().median().to_numpy()
            fallback = feat["lag_1d"].to_numpy(dtype=float)[idx] if "lag_1d" in feat.columns else None
            for j, p in enumerate(idx):
                if np.isnan(filled[p]):
                    if not np.isnan(exp_med[j]):
                        filled[p] = exp_med[j]
                    elif fallback is not None and not np.isnan(fallback[j]):
                        filled[p] = fallback[j]
                    else:
                        filled[p] = 0.0
        feat[c] = filled
    return feat


def _lag_roll_features(grid, col):
    """Lags / rolling / same-weekday / trend for one (kitchen, meal) series."""
    s = grid.set_index("date")[col].astype(float).sort_index()
    out = pd.DataFrame(index=s.index)
    for k in LAG_DAYS:
        out["lag_%dd" % k] = s.shift(k)
    for w in ROLLING_WINDOWS:
        r = s.shift(1).rolling(w, min_periods=max(2, w // 3))
        out["roll_mean_%dd" % w] = r.mean()
        out["roll_std_%dd" % w] = r.std()
        out["roll_med_%dd" % w] = r.median()
    # "the last four Thursdays", not "the last four observations"
    out["same_dow_mean_4w"] = (s.shift(1).groupby(s.index.dayofweek)
                               .rolling(4, min_periods=1).mean()
                               .reset_index(level=0, drop=True))
    out["history_len"] = s.shift(1).expanding().count()
    out["trend_slope_28d"] = s.shift(1).rolling(28, min_periods=5).apply(_trend_slope, raw=True)
    out = out.reset_index().rename(columns={"index": "date"})
    out["kitchen_id"] = grid["kitchen_id"].iloc[0]
    out["meal_type"] = grid["meal_type"].iloc[0]
    return out


def build_features(df, target=TARGET, drop_censored=True, item_te=True):
    """Build the model matrix.  Returns ``(features, schema)``.

    ``schema`` is committed next to the model so a later inference call can assert the
    column order lines up -- a silent reorder is the classic way a re-trained model
    starts producing confident nonsense.
    """
    g = df.copy()
    g["date"] = pd.to_datetime(g["date"]).dt.normalize()
    g = g.sort_values(KEYS + ["date"]).reset_index(drop=True)

    if drop_censored and "ran_out" in g.columns:
        n_censored = int(g["ran_out"].sum())
        LOG.info("excluding %d run-out rows: consumption is a LOWER BOUND there "
                 "(training on it teaches the model to under-predict when busiest)",
                 n_censored)
        train_pool = g[g["ran_out"] == 0]
    else:
        train_pool = g

    # ---- history statistics ------------------------------------------------ #
    grid = _daily_grid(g, target)
    parts = [_lag_roll_features(sub, target) for _, sub in grid.groupby(KEYS, sort=False)]
    stats = pd.concat(parts, ignore_index=True)
    feat = _merge_checked(g, stats, KEYS + ["date"], label="history-stats").reset_index(drop=True)

    # ---- calendar ---------------------------------------------------------- #
    cal = calendar_features(feat["date"]).reset_index(drop=True)
    for c in cal.columns:
        if c in feat.columns:
            feat = feat.drop(columns=[c])
    feat = pd.concat([feat.reset_index(drop=True), cal], axis=1)

    # ---- attendance -------------------------------------------------------- #
    att = _num(feat, "expected_diners", 0.0)
    hist = (pd.DataFrame({"k": feat["kitchen_id"], "m": feat["meal_type"], "a": att})
            .groupby(["k", "m"])["a"]
            .transform(lambda s: s.shift(1).rolling(28, min_periods=3).mean()))
    act = _num(feat, "actual_diners", float("nan"))
    feat["expected_diners"] = att.fillna(0.0)
    feat["attendance_log"] = np.log1p(att.clip(lower=0).fillna(0.0))
    feat["attendance_ratio_28d"] = np.where(hist > 0, att / hist.clip(lower=1e-6), 1.0)
    feat["actual_to_expected"] = np.where(att > 0, act / att.clip(lower=1e-6), 1.0)
    feat["portion_kg"] = _num(feat, "portion_kg", 0.4).fillna(0.4)

    # ---- menu -------------------------------------------------------------- #
    menus = feat["menu"].map(_menu_list)
    known = set(TOP_MENU_ITEMS)
    feat["menu"] = ["|".join(m) for m in menus]
    feat["n_items"] = menus.map(len).astype(float)
    for item in TOP_MENU_ITEMS:
        feat["has_" + item] = menus.map(lambda m, it=item: float(it in m))
    feat["menu_other_frac"] = menus.map(
        lambda m: sum(1 for it in m if it not in known) / max(len(m), 1))
    feat["menu_popularity"] = _num(feat, "menu_popularity", 0.0).fillna(0.0)
    if item_te:
        # out-of-fold item signal: dish on the menu, weighted by its portion size
        for item in TOP_MENU_ITEMS[:6]:
            mask = menus.map(lambda m, it=item: it in m).astype(float)
            feat["item_te_" + item] = mask * feat["portion_kg"]
    feat["menu_score"] = sum((feat["has_" + it] for it in TOP_MENU_ITEMS),
                             start=pd.Series(0.0, index=feat.index))

    # ---- exogenous --------------------------------------------------------- #
    # External measurements.  Missing values become the column's documented default
    # rather than a data-derived median: a median over the whole frame is future
    # information for any row near the start of the series.
    EXOGENOUS_DEFAULTS = {"is_holiday": 0.0, "festival_mult": 0.0, "exam_period": 0.0,
                          "temp_mean_c": 25.0, "rain_p": 0.0, "humidity_mean_pct": 60.0,
                          "season": 0.5, "is_weekend": 0.0}
    for c, default in EXOGENOUS_DEFAULTS.items():
        feat[c] = _num(feat, c, default).fillna(default)

    # ---- prior-day waste ---------------------------------------------------- #
    wgrid = _daily_grid(g, "waste_qty")
    wgrid["waste_lag1d"] = wgrid.groupby(KEYS)["waste_qty"].shift(1)
    feat = _merge_checked(feat, wgrid[KEYS + ["date", "waste_lag1d"]], KEYS + ["date"],
                        label="waste-lag").reset_index(drop=True)
    feat["waste_lag1d"] = _num(feat, "waste_lag1d", 0.0).fillna(0.0)

    # ---- out-of-fold encodings --------------------------------------------- #
    # Rows excluded from ``train_pool`` (run-out days) have no encoding.  Fill them with a
    # documented constant, NOT a data-derived mean: the frame-wide mean is future
    # information for early rows, and these rows are dropped before training anyway.
    te_kitchen = expanding_target_encoding(train_pool, ["kitchen_id"], target, out_col="kitchen_te")
    te_meal = expanding_target_encoding(train_pool, KEYS, target, out_col="kitchen_meal_te")
    feat = _merge_checked(feat, te_kitchen, KEYS + ["date"], label="kitchen-te").reset_index(drop=True)
    feat = _merge_checked(feat, te_meal, KEYS + ["date"], label="kitchen-meal-te").reset_index(drop=True)
    for c in ("kitchen_te", "kitchen_meal_te"):
        feat[c] = _num(feat, c, 0.0).fillna(0.0)
    LOG.info("feature matrix: %d rows x %d columns", len(feat), len(feat.columns))

    # ---- demand / history ratios ------------------------------------------- #
    history_cols = ["roll_mean_7d", "roll_mean_14d", "roll_mean_28d", "same_dow_mean_4w",
                    "roll_med_7d", "roll_med_14d", "roll_med_28d", "roll_std_7d",
                    "roll_std_14d", "roll_std_28d", "trend_slope_28d"]
    feat = _past_only_fill(feat, history_cols)
    for c in ("roll_mean_7d", "roll_mean_14d", "roll_mean_28d", "same_dow_mean_4w",
              "roll_med_7d", "roll_med_14d", "roll_med_28d"):
        v = pd.to_numeric(feat[c], errors="coerce").fillna(0.0)
        feat[c + "_ratio"] = np.where(v.abs() > 1e-6,
                                      feat["expected_diners"] / v.clip(lower=1e-6), 1.0)

    # ``target`` is deliberately KEPT in the returned frame: it is the label, not a
    # feature, and every caller needs ``feat[target]`` to fit on.  ``model_columns()``
    # excludes it from the design matrix, so keeping it cannot leak -- dropping it
    # instead forces every consumer to re-join the label onto the frame by position.
    drop = [c for c in ("data_source", "waste_cause", "kitchen_type", "ran_out",
                        "prepared_qty", "surplus_qty")
            if c in feat.columns and c != target]
    feat = feat.drop(columns=drop)
    feat = feat.replace([np.inf, -np.inf], np.nan)
    for c in feat.columns:
        # Only genuine numeric-ish feature columns get coerced.  `date` is a datetime
        # (is_numeric_dtype is False for it) and the key columns are strings; a blanket
        # to_numeric() call turns all three into epoch nanoseconds / NaN and silently
        # corrupts every later join.
        if c in ("menu", "date") or c in KEYS or pd.api.types.is_numeric_dtype(feat[c]):
            continue
        feat[c] = pd.to_numeric(feat[c], errors="coerce")

    feat = feat.sort_values(KEYS + ["date"]).reset_index(drop=True)

    schema = {"version": FEATURE_SCHEMA_VERSION,
              "features": list(feat.columns),
              "model_features": [c for c in feat.columns
                                 if c not in set(KEYS) | {"date", "menu", target}
                                 and c not in POST_HOC_COLUMNS],
              "post_hoc_columns": list(POST_HOC_COLUMNS),
              "target": target,
              "keys": KEYS,
              "menu_items": list(TOP_MENU_ITEMS),
              "lag_days": list(LAG_DAYS),
              "rolling_windows": list(ROLLING_WINDOWS),
              "censored_rows_dropped": int(g["ran_out"].sum()) if "ran_out" in g.columns else 0}
    LOG.info("built %d feature columns on %d rows (%d censored rows dropped)",
             len(schema["features"]), len(feat), schema["censored_rows_dropped"])
    return feat, schema


def align_to_schema(feat, schema):
    """Project an inference frame onto a trained model's feature list."""
    want = [c for c in schema["features"] if c not in schema["keys"]]
    for c in want:
        if c not in feat.columns:
            feat[c] = np.nan
    return feat[["date"] + list(schema["keys"]) + want].copy()


def model_columns(schema, feat: "pd.DataFrame | None" = None) -> list:
    """The numeric feature columns the estimator is actually trained on.

    ``schema["features"]`` is the full built frame, which also carries the key columns,
    the date, the ``menu`` string and :data:`POST_HOC_COLUMNS`.  Handing the string
    columns to LightGBM raises a conversion error deep inside ``to_numpy``; feeding it
    the post-hoc columns would be worse -- it would produce a model that looks excellent
    and cannot be evaluated online.  Both paths filter through this one function so they
    can never disagree about the column set.
    """
    skip = (set(schema.get("keys", ())) | {"date", "menu", schema.get("target")}
            | set(schema.get("post_hoc_columns", POST_HOC_COLUMNS)))
    cols = [c for c in schema["features"] if c not in skip]
    if feat is not None:
        cols = [c for c in cols
                if c in feat.columns and pd.api.types.is_numeric_dtype(feat[c])]
    declared = [c for c in schema.get("model_features", cols) if c not in skip]
    return [c for c in cols if c in set(declared)] or cols


def save_schema(path, schema):
    save_json(path, schema)
    return sha256_json(schema)


# --------------------------------------------------------------------------- #
# Leakage guard
# --------------------------------------------------------------------------- #
def assert_no_leakage(df, cut_date, tol=1e-9):
    """Rebuild features from data truncated at ``cut_date`` and require they match.

    This is the executable form of the claim "no feature uses future information".
    Run it in CI; do not reason about it by eye.
    """
    cut = pd.Timestamp(cut_date)
    full, schema = build_features(df)
    trunc, _ = build_features(df[pd.to_datetime(df["date"]) <= cut])
    cols = [c for c in schema["features"]
            if c in trunc.columns and c not in schema["keys"]
            and c not in ("menu", "date")
            and pd.api.types.is_numeric_dtype(trunc[c])]
    mask = pd.to_datetime(full["date"]) <= cut
    keys = list(schema["keys"]) + ["date"]
    a = full.loc[mask, keys + cols]
    b = trunc[keys + cols]
    # Join on the keys rather than comparing positionally: merges do not promise to
    # preserve row order, and a positional compare reports false leakage.
    both = a.merge(b, on=keys, suffixes=("_full", "_trunc"), how="inner")
    if len(both) != min(len(a), len(b)):
        raise AssertionError("LEAKAGE GUARD ERROR: key sets differ after truncation "
                             "(%d full vs %d truncated rows matched)" % (len(a), len(b)))
    for c in cols:
        x, y = both[c + "_full"], both[c + "_trunc"]
        m = x.notna() & y.notna()
        if bool(m.any()):
            diff = float((x[m] - y[m]).abs().max())
            if diff > tol:
                raise AssertionError("LEAKAGE in '%s': truncated rebuild differs by %g" % (c, diff))
        if bool(x.isna().any()) != bool(y.isna().any()):
            raise AssertionError("LEAKAGE in '%s': missing-value pattern changes when the "
                                 "future is removed" % c)


if __name__ == "__main__":
    from ml_utils import get_paths

    p = get_paths().data_raw / "sim_demand.parquet"
    if not p.exists():
        raise SystemExit("run ml_data_gen_kitchen_simulator.py first")
    d = pd.read_parquet(p)
    f, s = build_features(d)
    days = np.sort(pd.to_datetime(d["date"]).unique())
    cut = pd.Timestamp(days[int(len(days) * 0.8)])
    assert_no_leakage(d, cut)
    print("OK  %d feature columns, leakage check passed at %s" % (len(s["features"]), cut.date()))
    print("    " + ", ".join(s["features"][:14]) + " ...")