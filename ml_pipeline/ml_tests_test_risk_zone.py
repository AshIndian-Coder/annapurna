"""Executable checks for the RISK middle zone.

    python ml_tests_test_risk_zone.py
    pytest ml_tests_test_risk_zone.py

RISK is a *decision* zone, not a learned class: a batch whose release score
``s = 1 - P(GOOD)`` is neither confidently releasable nor confidently rejectable
must be held for a human, never silently forced into GOOD or REJECTED.  The safety
contract (``flagged == s >= tau``) must not move by one row when the zone is
switched on -- the tests below assert exactly that.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from ml_cv_common import C2I, CLASSES, decide, evaluate, pick_reject_threshold


def _probs(p_good, p_risk, p_rejected, p_nf=0.0):
    p = np.array([p_good, p_risk, p_rejected, p_nf], dtype=float)
    p = p / p.sum()
    return p.reshape(1, -1)


def test_middle_band_routes_to_risk_hold():
    """Scores between tau and tau_reject are RISK, not GOOD and not REJECTED."""
    tau, tau_reject = 0.3, 0.6
    rows = np.vstack([
        _probs(0.95, 0.03, 0.02),   # s = 0.05 -> auto_accept
        _probs(0.60, 0.35, 0.05),   # s = 0.40 -> risk_hold (the middle)
        _probs(0.25, 0.25, 0.50),   # s = 0.75 -> reject
    ])
    action = decide(rows, tau, tau_reject)
    assert action[0] == "auto_accept", action
    assert action[1] == "risk_hold", action
    assert action[2] == "reject", action


def test_no_tau_reject_collapses_to_the_old_binary_rule():
    """tau_reject=None must reproduce the pre-RISK behaviour exactly."""
    rng = np.random.default_rng(0)
    probs = rng.dirichlet([2, 2, 2, 2], size=500)
    tau = 0.35
    action = decide(probs, tau, None)
    flagged = (1 - probs[:, C2I["GOOD"]]) >= tau
    assert np.array_equal(action != "auto_accept", flagged), \
        "the binary hold rule changed when tau_reject is None"
    assert not (action == "reject").any()


def test_flag_contract_is_unchanged_by_the_reject_zone():
    """The 0.98-unsafe-recall flag set must be identical with and without a zone."""
    rng = np.random.default_rng(1)
    probs = rng.dirichlet([2, 2, 2, 2], size=800)
    tau, tau_reject = 0.3, 0.55
    m_old = evaluate(probs, np.full(800, C2I["GOOD"]), tau)
    m_new = evaluate(probs, np.full(800, C2I["GOOD"]), tau, tau_reject=tau_reject)
    for key in ("unsafe_recall", "false_accepts", "good_auto_accept_rate",
                "macro_f1", "not_food_recall", "nll", "ece"):
        assert m_old[key] == m_new[key], f"{key} moved when the RISK zone switched on"


def test_reject_zone_only_widens_the_hold_set():
    """Reject-zone rows must be a subset of the binary hold rows (never released)."""
    rng = np.random.default_rng(2)
    probs = rng.dirichlet([2, 2, 2, 2], size=600)
    tau, tau_reject = 0.3, 0.55
    assert tau_reject > tau
    hold_old = (1 - probs[:, C2I["GOOD"]]) >= tau
    action = decide(probs, tau, tau_reject)
    hold_new = action != "auto_accept"
    assert np.array_equal(hold_new, hold_old), \
        "the reject zone changed who is held -- the safety contract moved"


def test_changing_tau_reject_never_touches_auto_accept():
    """Raising or lowering tau_reject only redistributes rows between reject and
    risk_hold; the auto-accept set (the release set) must be identical for every
    tau_reject, and shrinking the zone moves rows risk_hold -> reject only."""
    rng = np.random.default_rng(3)
    probs = rng.dirichlet([2, 2, 2, 2], size=500)
    tau = 0.3
    a_hi = decide(probs, tau, 0.8)   # narrow reject zone
    a_lo = decide(probs, tau, 0.45)  # wide reject zone
    # auto-accept set invariant in both directions
    assert np.array_equal(a_hi == "auto_accept", a_lo == "auto_accept"),         "tau_reject changed who is auto-accepted -- the release set moved"
    # widening the reject zone: rows can only go risk_hold -> reject
    widened = (a_hi == "risk_hold") & (a_lo == "reject")
    assert (widened | (a_hi == a_lo)).all(),         "rows moved somewhere other than risk_hold -> reject when the zone widened"
    assert widened.any(), "the two zones did not differ; the test measures nothing"


def test_pick_reject_threshold_hits_the_recall_target():
    rng = np.random.default_rng(4)
    n = 2000
    probs = rng.dirichlet([2, 2, 2, 2], size=n)
    y = np.full(n, C2I["GOOD"])
    y[:400] = C2I["REJECTED"]
    tau_reject = pick_reject_threshold(probs, y, target=0.90)
    s = 1 - probs[:, C2I["GOOD"]]
    recall = float((s[y == C2I["REJECTED"]] >= tau_reject).mean())
    assert recall >= 0.90, f"tau_reject only catches {recall:.3f} of REJECTED rows"
    assert 0.0 <= tau_reject <= 1.0


def test_pick_reject_threshold_refuses_without_rejected_rows():
    rng = np.random.default_rng(5)
    probs = rng.dirichlet([2, 2, 2, 2], size=100)
    y = np.full(100, C2I["GOOD"])
    try:
        pick_reject_threshold(probs, y)
    except ValueError:
        return
    raise AssertionError("invented a reject threshold with no REJECTED rows to calibrate on")


def test_evaluate_reports_risk_zone_metrics():
    rng = np.random.default_rng(6)
    n = 900
    probs = rng.dirichlet([2, 2, 2, 2], size=n)
    y = rng.choice([C2I["GOOD"], C2I["RISK"], C2I["REJECTED"], C2I["NOT_FOOD"]], size=n)
    m = evaluate(probs, y, tau=0.3, tau_reject=0.55)
    assert set(m["actions"]) == {"auto_accept", "risk_hold", "reject"}
    total = sum(m["actions"].values())
    assert total == n, "action counts do not add up to the split size"
    assert m["tau_reject"] == 0.55
    assert 0.0 <= m["risk_hold_rate"] <= 1.0
    if m["risk_zone_recall"] is not None:
        assert 0.0 <= m["risk_zone_recall"] <= 1.0
    # binary fallback still records tau_reject=None and an empty reject zone
    m_bin = evaluate(probs, y, tau=0.3)
    assert m_bin["tau_reject"] is None
    assert m_bin["actions"]["reject"] == 0
    assert m_bin["actions"]["risk_hold"] == m_bin["n"] - m_bin["actions"]["auto_accept"]


def test_tau_reject_at_or_below_tau_is_inert():
    """A degenerate zone (tau_reject <= tau) must behave exactly like tau_reject=None."""
    rng = np.random.default_rng(7)
    probs = rng.dirichlet([2, 2, 2, 2], size=300)
    for tr in (0.3, 0.1):
        a = decide(probs, 0.4, tr)
        b = decide(probs, 0.4, None)
        assert np.array_equal(a, b), f"tau_reject={tr} was not inert"
        assert not (a == "reject").any()


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
