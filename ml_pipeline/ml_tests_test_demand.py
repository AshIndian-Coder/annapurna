"""Executable checks for the demand half (spec 07 sections 5-6).

    python ml_tests_test_demand.py            # no pytest needed
    pytest ml_tests_test_demand.py            # also works

These are the properties that are easy to break with a well-meaning edit and impossible
to notice in a score: interval ordering, calibration correctness, the newsvendor
arithmetic, baseline alignment, the leakage guard, and the frozen request/response
contract that ``mlserving`` hands to Go.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from ml_demand_features import (
    KEYS,
    POST_HOC_COLUMNS,
    assert_no_leakage,
    build_features,
    model_columns,
)
from ml_demand_model import (
    BAND,
    MLError,
    QuantileEnsemble,
    _monotonise,
    _query_keys,
    _single_row_frame,
    apply_cqr,
    conformalize,
    cqr_scores,
    critical_ratio,
    expected_surplus_quantiles,
    operating_quantile,
    predict_demand,
    same_weekday_median,
    seasonal_naive,
    surplus_risk,
    wape,
)

RAW = Path(__file__).resolve().parent / "data" / "raw" / "sim_demand.parquet"


# --------------------------------------------------------------------------- #
# Quantiles and intervals
# --------------------------------------------------------------------------- #
def test_quantile_curves_are_ordered():
    """Independently fitted quantile models cross; the wrapper must un-cross them."""
    rng = np.random.default_rng(0)
    pred = {0.1: rng.normal(100, 20, 200), 0.5: rng.normal(110, 25, 200),
            0.9: rng.normal(120, 30, 200)}
    # guarantee crossings in the input
    pred[0.1][:20] = pred[0.9][:20] + 50.0
    fixed = _monotonise(pred)
    assert np.all(fixed[0.1] <= fixed[0.5] + 1e-9), "p10 exceeded p50"
    assert np.all(fixed[0.5] <= fixed[0.9] + 1e-9), "p50 exceeded p90"
    for k in fixed:
        assert len(fixed[k]) == len(pred[k]), "row count changed"
    # the un-crossing must be exactly a per-row sort of the stacked levels -- the
    # strongest available statement that no row was permuted or value invented
    keys = sorted(pred)
    expected = np.sort(np.column_stack([pred[k] for k in keys]), axis=1)
    for i, k in enumerate(keys):
        assert np.allclose(fixed[k], expected[:, i]), f"level {k} is not the per-row sort"


def test_apply_cqr_never_inverts_the_interval():
    rng = np.random.default_rng(1)
    pred = {0.1: rng.uniform(0, 10, 50), 0.5: rng.uniform(0, 10, 50), 0.9: rng.uniform(0, 10, 50)}
    for qhat in (0.0, 5.0, 100.0):
        lo, hi = apply_cqr(pred, {"qhat": qhat})
        assert np.all(lo >= 0.0), "conformalised lower bound went negative"
        assert np.all(hi >= lo), "upper bound fell below the lower bound"


def test_cqr_scores_and_finite_sample_correction():
    """``E_i = max(lo - y, y - hi)``, and the ceil((n+1)*alpha) order statistic."""
    y = np.array([1.0, 2.0, 3.0, 100.0])
    lo = np.array([0.0, 0.0, 0.0, 0.0])
    hi = np.array([2.0, 2.0, 2.0, 2.0])
    # scores may be negative when the point is strictly inside the interval; the
    # correction is floored at zero so it can only ever widen the band.
    assert np.allclose(cqr_scores(y, lo, hi), [-1.0, 0.0, 1.0, 98.0])

    n = 100
    rng = np.random.default_rng(2)
    pred = rng.normal(100, 10, n)          # model output
    ycal = pred + rng.normal(0, 10, n)      # realised outcome
    conf = conformalize(pred - 2.0, ycal, pred + 2.0, coverage=0.80)
    assert conf["n_calibration"] == n
    s = np.sort(cqr_scores(ycal, pred - 2.0, pred + 2.0))
    k = int(np.ceil((n + 1) * 0.80))        # = 81
    assert conf["qhat"] == max(float(s[k - 1]), 0.0)
    assert conf["qhat"] > 0.0, "an under-wide interval must earn a positive correction"


def test_conformal_coverage_holds_on_fresh_data():
    """The finite-sample guarantee, on data the calibration fold never saw.

    Fixture: a model whose point prediction is off by N(0, 10) against an interval
    that is only +-2 wide.  CQR must widen the band until fresh coverage hits 0.80.
    """
    rng = np.random.default_rng(3)
    p_cal = rng.normal(100, 10, 2000)
    y_cal = p_cal + rng.normal(0, 10, 2000)
    conf = conformalize(p_cal - 2.0, y_cal, p_cal + 2.0, coverage=0.80)
    assert conf["qhat"] > 2.0, "correction did not widen the band at all"

    p_new = rng.normal(100, 10, 20000)
    y_new = p_new + rng.normal(0, 10, 20000)
    lo, hi = apply_cqr({0.1: p_new - 2.0, 0.5: p_new, 0.9: p_new + 2.0}, conf)
    cov = float(np.mean((y_new >= lo) & (y_new <= hi)))
    assert 0.78 <= cov <= 0.82, f"CQR coverage {cov:.3f} outside 0.80 +- 0.02"


def test_empty_calibration_fold_does_not_crash():
    conf = conformalize(np.array([]), np.array([]), np.array([]), coverage=0.8)
    lo, hi = apply_cqr({0.1: np.array([1.0]), 0.5: np.array([2.0]), 0.9: np.array([3.0])}, conf)
    assert np.allclose(lo, [1.0]) and np.allclose(hi, [3.0])


# --------------------------------------------------------------------------- #
# Newsvendor
# --------------------------------------------------------------------------- #
def test_critical_ratio():
    assert critical_ratio(1.0, 1.0) == 0.5
    assert critical_ratio(3.0, 1.0) == 0.75
    for bad in ((0.0, 1.0), (1.0, 0.0), (-1.0, 1.0)):
        try:
            critical_ratio(*bad)
        except ValueError:
            continue
        raise AssertionError(f"critical_ratio{bad} should have raised")


def test_expected_surplus_matches_a_hand_computed_case():
    """A symmetric predictive distribution: q*=0.5 must give a mean-reverting plan."""
    qs = [0.1, 0.5, 0.75, 0.9, 0.95]
    vals = np.array([50.0, 100.0, 120.0, 150.0, 170.0])
    plan = expected_surplus_quantiles({q: np.array([v]) for q, v in zip(qs, vals)}, 0.5)
    assert abs(float(np.atleast_1d(plan["produce_kg"])[0]) - 100.0) < 1e-6
    over = float(np.atleast_1d(plan["expected_surplus_kg"])[0])
    under = float(np.atleast_1d(plan["expected_shortage_kg"])[0])
    assert over > 0 and under > 0, "a centred distribution must have both tails"
    # a higher quantile must produce more, never less
    hi = expected_surplus_quantiles({q: np.array([v]) for q, v in zip(qs, vals)}, 0.9)
    assert float(np.atleast_1d(hi["produce_kg"])[0]) > 100.0


def test_expected_surplus_is_row_wise():
    pred = {0.1: np.array([10.0, 20.0, 30.0]), 0.5: np.array([20.0, 30.0, 40.0]),
            0.9: np.array([30.0, 40.0, 50.0])}
    plan = expected_surplus_quantiles(pred, 0.5)
    assert np.asarray(plan["produce_kg"]).shape == (3,), plan["produce_kg"]


def test_operating_point_reports_the_binding_constraint():
    q_cost, why_cost = operating_quantile(3.0, 1.0, service_level=0.0)
    assert abs(q_cost - 0.75) < 1e-9 and why_cost == "newsvendor_cost"
    q_floor, why_floor = operating_quantile(3.0, 1.0, service_level=0.90)
    assert abs(q_floor - 0.90) < 1e-9 and why_floor == "service_floor"
    q2, why2 = operating_quantile(9.0, 1.0, service_level=0.90)
    assert abs(q2 - 0.90) < 1e-9 and why2 == "newsvendor_cost", (
        "a 9:1 cost ratio and a 90 % service floor agree; the cost rule should win")


def test_surplus_risk_thresholds():
    assert surplus_risk(1.0, 100.0) == "LOW"
    assert surplus_risk(8.0, 100.0) == "MEDIUM"
    assert surplus_risk(30.0, 100.0) == "HIGH"


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def test_baselines_are_positionally_aligned():
    """The exact defect this test exists for: a query frame, not a date list."""
    hist = pd.DataFrame({
        "kitchen_id": ["a"] * 14 + ["b"] * 14,
        "meal_type": ["LUNCH"] * 28,
        "date": list(pd.date_range("2024-01-01", periods=14)) * 2,
        "consumed_qty": [10.0] * 14 + [200.0] * 14,
    })
    query = pd.DataFrame({
        "kitchen_id": ["b", "a"],
        "meal_type": ["LUNCH", "LUNCH"],
        "date": [pd.Timestamp("2024-01-08"), pd.Timestamp("2024-01-08")],
    })
    sn = seasonal_naive(hist, query)
    assert sn[0] == 200.0 and sn[1] == 10.0, f"seasonal naive crossed kitchens: {sn}"
    sm = same_weekday_median(hist, query)
    assert sm[0] == 200.0 and sm[1] == 10.0, f"weekday median crossed kitchens: {sm}"
    assert wape([200.0, 10.0], sn) == 0.0


def test_baseline_query_requires_keys():
    """A bare date list must fail loudly, not silently return the wrong kitchen."""
    hist = pd.DataFrame({"kitchen_id": ["a"], "meal_type": ["LUNCH"],
                         "date": [pd.Timestamp("2024-01-01")], "consumed_qty": [5.0]})
    try:
        seasonal_naive(hist, [pd.Timestamp("2024-01-08")])
    except KeyError:
        return
    raise AssertionError("seasonal_naive accepted a query frame with no keys")


def test_baselines_never_use_the_future():
    # 2024-01-01 is a Monday.  Rows run daily to 01-08, then jump to 01-17 with an
    # extreme value on the SAME weekday as the query (Wednesday 01-10).
    hist = pd.DataFrame({
        "kitchen_id": ["a"] * 9,
        "meal_type": ["LUNCH"] * 9,
        "date": list(pd.date_range("2024-01-01", periods=8)) + [pd.Timestamp("2024-01-17")],
        "consumed_qty": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 900.0],
    })
    query = pd.DataFrame({"kitchen_id": ["a"], "meal_type": ["LUNCH"],
                          "date": [pd.Timestamp("2024-01-10")]})   # Wednesday
    assert seasonal_naive(hist, query)[0] == 3.0, "seasonal naive looked forward"
    # the only strictly-past Wednesday is 01-03 (value 3.0); 01-17 must be ignored
    got = same_weekday_median(hist, query)[0]
    assert got == 3.0, f"weekday median leaked a future observation: {got}"


# --------------------------------------------------------------------------- #
# Features and the leakage guard
# --------------------------------------------------------------------------- #
def test_post_hoc_columns_are_never_model_inputs():
    if not RAW.exists():
        return
    raw = pd.read_parquet(RAW)
    feat, schema = build_features(raw)
    cols = model_columns(schema, feat)
    for c in POST_HOC_COLUMNS:
        assert c not in cols, f"post-hoc column '{c}' is a model input"
        assert c in schema["post_hoc_columns"]
    assert "expected_diners" in cols, "the one attendance field that IS known in advance got dropped"


def test_model_columns_drops_strings_and_keys():
    schema = {"features": ["a", "b", "menu", "date", "kitchen_id", "meal_type", "consumed_qty"],
              "keys": ["kitchen_id", "meal_type"], "target": "consumed_qty",
              "model_features": ["a", "b"]}
    feat = pd.DataFrame({"a": [1.0], "b": [2.0], "menu": ["x|y"], "date": ["2024-01-01"],
                         "kitchen_id": ["k"], "meal_type": ["LUNCH"],
                         "consumed_qty": [3.0]})
    assert model_columns(schema, feat) == ["a", "b"]


def test_no_leakage_guard_catches_a_planted_leak():
    """The guard must FAIL on a frame that does leak, or it proves nothing."""
    import ml_demand_features as F

    if not RAW.exists():
        return
    raw = pd.read_parquet(RAW).head(400)
    original = F.expanding_target_encoding

    def leaky(train, keys, target=F.TARGET, prior=10.0, out_col="_te", prior_value=1.0):
        out = original(train, keys, target, prior, out_col, prior_value)
        # plant the leak: the *final* value of the frame, which a truncated rebuild cannot see
        out[out_col] = float(out[out_col].max()) + 1000.0
        return out

    F.expanding_target_encoding = leaky
    try:
        assert_no_leakage(raw, pd.to_datetime(raw["date"]).max() - pd.Timedelta(days=30))
    except AssertionError as exc:
        assert "LEAKAGE" in str(exc), exc
    else:
        raise AssertionError("the leakage guard passed a deliberately leaky frame")
    finally:
        F.expanding_target_encoding = original


def test_leakage_guard_passes_on_real_data():
    if not RAW.exists():
        return
    raw = pd.read_parquet(RAW)
    days = np.sort(pd.to_datetime(raw["date"]).dt.normalize().unique())
    assert_no_leakage(raw, days[int(len(days) * 0.75)])


# --------------------------------------------------------------------------- #
# The frozen contract
# --------------------------------------------------------------------------- #
def _toy_bundle() -> dict:
    rng = np.random.default_rng(7)
    n = 800
    X = pd.DataFrame({
        "day_of_week": rng.integers(0, 7, n).astype(float),
        "expected_diners": rng.uniform(50, 500, n),
        "roll_mean_7d": rng.uniform(50, 500, n),
        "kitchen_te": rng.uniform(50, 500, n),
    })
    y = X["expected_diners"].to_numpy() * 0.8 + rng.normal(0, 10, n)
    m = QuantileEnsemble(n_estimators=60, num_leaves=15).fit(X, y)
    schema = {"features": list(X.columns), "model_features": list(X.columns),
              "keys": ["kitchen_id", "meal_type"], "target": "consumed_qty",
              "post_hoc_columns": list(POST_HOC_COLUMNS)}
    return {"model": m, "feature_schema": schema, "conformal": {"qhat": 5.0, "coverage_target": 0.8},
            "feature_frame": X.head(1), "model_version": "demand-test",
            "data_source": "SYNTHETIC", "cost_under": 3.0, "cost_over": 1.0,
            "service_level": 0.90}


def test_predict_demand_contract_shape():
    b = _toy_bundle()
    r = predict_demand(b, {"attendance": 300, "meal_type": "LUNCH", "day_of_week": 3,
                            "menu": ["rice", "dal"],
                            "historical_consumption": [200, 210, 190, 205, 215]})
    for k in ("predicted_consumption", "recommended_production", "expected_surplus",
              "surplus_risk", "prediction_interval", "recommended_quantile", "top_drivers",
              "model_version", "data_source"):
        assert k in r, f"contract field '{k}' missing from the response"
    pi = r["prediction_interval"]
    assert pi["p10"] <= pi["p50"] <= pi["p90"], pi
    assert r["surplus_risk"] in ("LOW", "MEDIUM", "HIGH")
    assert r["history_used"] is True
    assert r["data_source"] == "SYNTHETIC"


def test_predict_demand_rejects_bad_input_with_stable_codes():
    b = _toy_bundle()
    cases = [
        ({"attendance": -1, "meal_type": "LUNCH", "day_of_week": 1}, "NEGATIVE_ATTENDANCE"),
        ({"attendance": 10, "meal_type": "SUPPER", "day_of_week": 1}, "UNKNOWN_MEAL_TYPE"),
        ({"attendance": 10, "meal_type": "LUNCH", "day_of_week": 7}, "BAD_DOW"),
        ({"meal_type": "LUNCH", "day_of_week": 1}, "NEGATIVE_ATTENDANCE"),
    ]
    for payload, code in cases:
        try:
            predict_demand(b, payload)
        except MLError as exc:
            assert exc.code == code, f"{payload} -> {exc.code}, expected {code}"
        else:
            raise AssertionError(f"payload {payload} should have raised {code}")


def test_single_row_projection_uses_the_request_history():
    b = _toy_bundle()
    schema, frame = b["feature_schema"], b["feature_frame"]
    low = _single_row_frame(frame, schema, {"attendance": 300, "day_of_week": 1})
    high = _single_row_frame(frame, schema, {"attendance": 300, "day_of_week": 1,
                                             "historical_consumption": [400.0] * 28})
    assert high["roll_mean_7d"].iloc[0] == 400.0
    assert high["kitchen_te"].iloc[0] == 400.0
    assert low["roll_mean_7d"].iloc[0] != high["roll_mean_7d"].iloc[0]
    for name, row in (("low", low), ("high", high)):
        assert np.isfinite(row.to_numpy()).all(), f"{name} projection produced NaN/inf"
        assert list(row.columns) == [c for c in schema["model_features"]], name


def test_single_row_projection_survives_garbage_history():
    b = _toy_bundle()
    row = _single_row_frame(b["feature_frame"], b["feature_schema"],
                            {"attendance": 100, "day_of_week": 0,
                             "historical_consumption": ["a", None, float("nan"), 5.0]})
    assert np.isfinite(row.to_numpy()).all()


def test_data_source_is_never_claimed_real_by_accident():
    b = _toy_bundle()
    b["data_source"] = ["SYNTHETIC", "REAL"]
    assert predict_demand(b, {"attendance": 10, "meal_type": "LUNCH",
                              "day_of_week": 0})["data_source"] == "MIXED"
    b["data_source"] = ["REAL"]
    assert predict_demand(b, {"attendance": 10, "meal_type": "LUNCH",
                              "day_of_week": 0})["data_source"] == "REAL"


# --------------------------------------------------------------------------- #
def _main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed.append((name, exc))
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
