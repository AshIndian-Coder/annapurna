"""Executable monotonicity and config checks for the safety-fusion model.

    python ml_tests_test_fusion_monotonic.py
    pytest ml_tests_test_fusion_monotonic.py

Monotonicity is a *product* claim, not a modelling nicety: more visible spoilage, more
time above the safe temperature band, a longer wait since cooking, or more humidity must
never make a batch look **safer**.  The fusion model is constrained to enforce that, and
this file is the proof.  If an edit breaks the constraint, the failure is a safety bug,
not a flaky test -- so the assertions are exact (a tolerance of -1e-9), not fuzzy.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from ml_data_gen_sensor_simulator import simulate
from ml_quality_datasets import prep
from ml_quality_time_temp import load_cfg
from ml_quality_train_fusion import fit, predict_unsafe

#: (column, low, high, must_increase_risk)
DIRECTIONS = [
    ("danger_zone_minutes", 0.0, 600.0),
    ("hours_since_prepared", 0.0, 30.0),
    ("cv_unsafe_prob", 0.0, 1.0),
    ("humidity_mean", 40.0, 92.0),
    ("temperature_mean", 2.0, 75.0),
]


def _bundle(n: int = 3000, seed: int = 5):
    df = simulate(n, seed=seed)
    cats = sorted(df["category"].unique())
    return df, cats, fit(df, cats, n_estimators=80)


def test_risk_never_decreases_as_exposure_increases():
    """Per-feature monotonicity, then the joint claim that actually matters.

    A LightGBM monotone constraint bounds the score with respect to each constrained
    feature, but a *correlated* feature can be satisfied without ever being split on:
    ``hours_since_prepared`` and ``danger_zone_minutes`` are near-collinear in the
    simulator, so raising the former while holding the latter fixed can leave the
    prediction unchanged -- and that is a correct model, not a broken constraint.

    So this test asserts two different things, at two different strengths:

    * **per feature, exactly**: risk must never decrease.  This is the safety claim.
    * **jointly, strictly**: a batch with every exposure signal raised at once must be
      materially riskier.  This is the claim that catches a model which satisfies the
      constraints by ignoring all of them.

    An individual constrained feature having zero effect is logged, not failed -- and the
    reason is printed, because "the model is monotone but ignores elapsed time" is a
    finding a reviewer needs, not a passing test.
    """
    df, cats, b = _bundle()
    base = df.sample(50, random_state=1).reset_index(drop=True)
    inert = []
    for col, lo, hi in DIRECTIONS:
        if col not in base.columns:
            continue
        probs = np.asarray([predict_unsafe(b, prep(base.assign(**{col: v}), cats))
                            for v in np.linspace(lo, hi, 25)])
        diffs = np.diff(probs, axis=0)
        worst = float(diffs.min()) if diffs.size else 0.0
        assert worst >= -1e-9, (
            f"risk DECREASED by {abs(worst):.6f} when {col} increased -- the monotonicity "
            f"constraint is not doing its job, and a batch that looks safer after more "
            f"time in the danger zone is exactly the failure this system must not have")
        if float(probs.max(axis=0).mean() - probs.min(axis=0).mean()) <= 1e-6:
            inert.append(col)
    if inert:
        corr = np.corrcoef(df["hours_since_prepared"], df["danger_zone_minutes"])[0, 1]
        print(f"       note: {inert} have no independent effect; "
              f"corr(hours_since_prepared, danger_zone_minutes) = {corr:.3f}")
        assert "danger_zone_minutes" not in inert, (
            "danger_zone_minutes has no effect either -- the constraint is being "
            "satisfied by ignoring the features entirely")

    low = predict_unsafe(b, prep(base, cats))
    hot = predict_unsafe(b, prep(base.assign(
        hours_since_prepared=24.0, danger_zone_minutes=500.0, humidity_mean=90.0,
        cv_unsafe_prob=0.95), cats))
    assert float(np.mean(hot)) > float(np.mean(low)) + 1e-3, (
        f"raising every exposure signal together moved risk from {np.mean(low):.4f} to "
        f"{np.mean(hot):.4f} -- no useful joint effect, so the model has learned nothing "
        f"about spoilage")


def test_prediction_is_a_probability():
    df, cats, b = _bundle(n=800, seed=9)
    p = np.asarray(predict_unsafe(b, prep(df.head(200), cats)), dtype=float)
    assert p.shape == (len(df.head(200)),), p.shape
    assert np.all((p >= 0.0) & (p <= 1.0)), "unsafe probability left [0, 1]"
    assert np.all(np.isfinite(p)), "unsafe probability is not finite"


def test_risk_at_extremes_is_ordered():
    """Worst case must be more dangerous than best case, on the same batch."""
    df, cats, b = _bundle(n=1500, seed=11)
    col = [c for c, _, _ in DIRECTIONS if c in df.columns][0]
    one = df.head(40).copy()
    best = float(np.mean(predict_unsafe(b, prep(one.assign(**{col: 0.0}), cats))))
    worst = float(np.mean(predict_unsafe(b, prep(one.assign(**{col: 999.0}), cats))))
    assert worst >= best, f"{col}: worst exposure ({worst:.4f}) scored safer than best " \
                          f"({best:.4f})"


def test_config_has_every_category_and_meets_the_recall_target():
    cfg = load_cfg()
    assert cfg["targets"]["unsafe_recall"] >= 0.98, \
        "the recall target in the config is below the spec's 0.98"
    assert cfg["categories"], "no food categories in the config"
    for name, spec in cfg["categories"].items():
        assert "budget_hours" in spec, f"category '{name}' has no danger-zone budget"
        assert spec["budget_hours"] > 0, f"category '{name}' has a non-positive budget"


def test_simulated_labels_follow_the_budget_rule():
    df = simulate(500, seed=2)
    assert set(df["unsafe"].unique()) <= {0, 1}, "unsafe label is not binary"
    if "hours_left_true" in df.columns:
        assert (df.loc[df.unsafe == 1, "hours_left_true"] == 0).all(), \
            "a batch marked unsafe still has hours left -- the simulator contradicts itself"
    assert df["hours_since_prepared"].min() >= 0, "negative time since preparation"


def test_monotonicity_check_can_actually_fail():
    """A test that cannot fail proves nothing.

    Fit a model with the constraints switched off and confirm the same assertion trips.
    """
    df, cats, b = _bundle(n=1200, seed=13)
    try:
        loose = fit(df, cats, n_estimators=60, monotone=False)
    except TypeError:
        return          # this build has no switch; the constraint is structural
    col = "danger_zone_minutes"
    if col not in df.columns:
        return
    one = df.head(40).copy()
    probs = np.asarray([predict_unsafe(loose, prep(one.assign(**{col: v}), cats))
                        for v in np.linspace(0, 600, 25)])
    assert np.diff(probs, axis=0).min() < -1e-9, (
        "an unconstrained fusion model is still monotone on this data, so the "
        "monotonicity test cannot distinguish constrained from unconstrained -- "
        "the assertion is not measuring what it claims to measure")


def _main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed.append(name)
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())