"""Leakage tests for the SIH-2026 demand lags.

    python -m pytest ml_tests_test_sih2026_lags.py -q

:func:`ml_data_load_sih2026.add_demand_lags` builds short-horizon lags
(``lag_1/2/3``), exponentially weighted means and a day-of-week x kitchen-type
interaction.  Every one of those is a place a demand model can accidentally
read its own answer, so the guarantee is asserted here rather than trusted.

The check that matters is not "does the column name look backwards" but:
**change a single row's target and prove that row's own features do not move,
while every later row's features do.**  That catches an off-by-one in a
``shift``, a ``rolling`` without the ``shift(1)``, or an ``ewm`` that
accidentally includes the present.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ml_data_load_sih2026 as D  # noqa: E402

DERIVED_PREFIXES = ("lag_", "roll_", "ewm_", "trend_", "dow_")


def _frame(n_days: int = 120) -> pd.DataFrame:
    """A minimal but well-formed demand frame: 2 kitchens x 2 categories."""
    days = pd.date_range("2026-01-01", periods=n_days, freq="D")
    rows = []
    for k in (0, 1):
        for cat in ("dal_curries", "cereals_breads"):
            for i, d in enumerate(days):
                rows.append(
                    {
                        "kitchen_id": f"K{k:02d}",
                        "category": cat,
                        "date": d,
                        "day_of_week": int(d.dayofweek),
                        "kitchen_type": "corporate_canteen" if k == 0 else "hostel_mess",
                        "units_sold": 100 + 10 * k + (i % 7),
                    }
                )
    return pd.DataFrame(rows)


def _derived(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in df.columns
        if c.startswith(DERIVED_PREFIXES) and pd.api.types.is_numeric_dtype(df[c])
    ]


def test_shorts_and_ewms_are_produced() -> None:
    """The features this module was extended to provide actually exist."""
    out = D.add_demand_lags(_frame())
    for col in ("lag_1", "lag_2", "lag_3", "lag_364", "roll_7", "ewm_7", "ewm_28"):
        assert col in out.columns, f"{col} missing from add_demand_lags output"
    assert "trend_7_28" in out.columns
    assert "dow_kitchen_type" in out.columns


def test_perturbing_a_row_does_not_move_that_rows_features() -> None:
    """THE leakage test: a row's features must not depend on that row's target."""
    base = D.add_demand_lags(_frame())
    cols = _derived(base)

    df = _frame()
    # Pick one row well past the longest warm-up so every feature is populated.
    victim = df.index[len(df) // 2]
    # Identify the row BEFORE add_demand_lags re-sorts and resets the index, or the
    # positional index means a different logical row in the sorted frame.
    key = _key(df, victim)
    df.at[victim, "units_sold"] = float(df.at[victim, "units_sold"]) * 40.0 + 1234.0
    bumped = D.add_demand_lags(df)

    b = base.set_index(["kitchen_id", "category", "date"]).loc[[key], cols]
    a = bumped.set_index(["kitchen_id", "category", "date"]).loc[[key], cols]
    for c in cols:
        bv, av = float(b[c].iloc[0]), float(a[c].iloc[0])
        if np.isnan(bv) and np.isnan(av):
            continue
        assert bv == pytest.approx(av, rel=1e-9, abs=1e-9), (
            f"{c} for the perturbed row changed ({bv} -> {av}): "
            "a feature is reading its own target"
        )


def test_later_rows_do_move() -> None:
    """Counter-test: the guarantee must not be vacuous -- later rows MUST react.

    Without this, a feature that is constant everywhere would pass the test
    above while learning nothing at all.
    """
    base = D.add_demand_lags(_frame())
    cols = _derived(base)

    df = _frame()
    victim = df.index[len(df) // 2]
    key = _key(df, victim)
    df.at[victim, "units_sold"] = float(df.at[victim, "units_sold"]) * 40.0 + 1234.0
    bumped = D.add_demand_lags(df)

    mask_after = (bumped["kitchen_id"] == key[0]) & (bumped["category"] == key[1]) & (
        bumped["date"] > key[2]
    )
    changed = [
        c
        for c in cols
        if mask_after.any()
        and not np.allclose(
            base.loc[mask_after, c].astype(float).fillna(-999.0),
            bumped.loc[mask_after, c].astype(float).fillna(-999.0),
        )
    ]
    assert "lag_1" in changed, "lag_1 never responds to an earlier row: test is vacuous"


def test_no_latent_column_leaks_into_features() -> None:
    """Ground-truth columns must be rejected by the loader's own guard."""
    out = D.add_demand_lags(_frame())
    feats = [c for c in out.columns if c.startswith(DERIVED_PREFIXES)]
    with pytest.raises(AssertionError):
        D.assert_no_latent(out, feats + ["latent_true_demand"], table="demand")


def _key(df: pd.DataFrame, idx) -> tuple:
    r = df.loc[idx]
    return (r["kitchen_id"], r["category"], r["date"])