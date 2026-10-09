"""Surplus food -> recipients: feasibility filter + linear-programme allocation.

    python ml_logistics_match.py          # self-check / benchmark

No model is trained here, deliberately.  Matching is a constrained allocation problem
and pretending otherwise would be dressing an LP up as AI (spec 07 section 3.5).  What
*is* learned later is "will this NGO actually accept" -- and that needs real
accept/decline history, which in a pilot is worth collecting and in a demo is not worth
inventing.

Design notes
------------
* **Feasibility is a hard filter, not a penalty.**  Travel time plus handling buffer
  must fit inside the food's *lower-bound* safe-until time, the NGO must be open, must
  have free capacity, must accept the category, and must have cold storage if the batch
  needs it.  Anything else is not a low-scoring match, it is not a match.
* **The objective is priority-weighted portions.**  Maximising "kg delivered" alone
  starves the highest-need NGOs when capacity is tight.
* **The LP is a max-flow.**  ``scipy.optimize.linprog`` with HiGHS solves it in
  microseconds at this size and returns an *optimal* allocation, so there is nothing for
  a neural model to improve on here.
* **Infeasibility is a result, not a crash.**  ``linprog`` returns ``x = None`` when no
  feasible solution exists; the previous code did ``np.floor(res.x + 1e-9)`` on ``None``
  and died with ``AttributeError``.  It now returns an empty plan with a reason that
  the API maps to ``NO_ELIGIBLE_RECIPIENT``.

Travel time is haversine x road_factor / speed.  Those two constants are ASSUMPTIONS
until replaced by a routing API or GPS traces from real drivers, and every artefact
that uses them records that fact.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from ml_utils import get_logger

LOG = get_logger("logistics.match")

#: Score weights, matching the frozen backend contract (03_BACKEND section 4:
#: "need 20, capacity 20, distance 25, window 15, shelf-life 10, fairness 10").
SCORE_WEIGHTS = {"need": 20, "capacity": 20, "distance": 25, "window": 15,
                 "shelf_life": 10, "fairness": 10}

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def travel_minutes(d_km, speed_kmph: float = 25.0, road_factor: float = 1.3):
    """Minutes to cover ``d_km``.  Speed 25 km/h and road factor 1.3 are ASSUMPTIONS."""
    if speed_kmph <= 0:
        raise ValueError("speed_kmph must be > 0")
    return np.asarray(d_km, dtype=float) * road_factor / speed_kmph * 60.0


def score_pairs(offers: pd.DataFrame, recips: pd.DataFrame, pairs: pd.DataFrame,
                now_min: float = 0.0) -> pd.DataFrame:
    """Explainable 0-100 score per feasible pair, using the contract's weights.

    Returns the same rows as :func:`feasible_pairs` plus ``score``, ``score_breakdown``
    and ``reasons`` -- the exact fields ``POST /match`` returns, so the API, the app and
    the benchmark all speak the same language.
    """
    if pairs.empty:
        return pairs.assign(score=[], score_breakdown=[], reasons=[])
    o = offers.loc[pairs["i"].to_numpy()].reset_index(drop=True)
    r = recips.loc[pairs["j"].to_numpy()].reset_index(drop=True)
    tm = pairs["travel_min"].to_numpy(float)
    buf = float(pairs["buffer_min"].iloc[0]) if len(pairs) else 0.0

    slack_min = np.maximum(o["safe_minutes"].to_numpy(float) - tm - buf, 0.0)
    shelf_slack = np.where(o["safe_minutes"].to_numpy(float) > 0,
                           slack_min / np.maximum(o["safe_minutes"].to_numpy(float), 1e-9), 0.0)
    window_slack = np.maximum(r["close_min"].to_numpy(float) - (now_min + tm), 0.0)
    window_score = np.clip(window_slack / np.maximum(r["close_min"].to_numpy(float), 1e-9), 0, 1)

    need = np.clip(r["priority"].to_numpy(float) / 3.0, 0, 1)
    # Pre-allocation estimate: how much of this recipient's capacity the offer would use.
# The exact split is only known after the LP solves, so this ranks pairs, it does not
# replace the capacity constraint.
    cap_used = np.clip(o["portions"].to_numpy(float) / np.maximum(r["capacity"].to_numpy(float), 1e-9), 0, 1)
    # Distance decays with travel time as a fraction of the food's remaining life:
    # 10 minutes out of a 4-hour window is cheap, out of a 40-minute window is not.
    dist_score = np.exp(-tm / np.maximum(o["safe_minutes"].to_numpy(float), 1.0) * 2.0)
    # Fairness: penalise an NGO that already received in the last 48 h.
    since_h = r["hours_since_last"].to_numpy(float) if "hours_since_last" in r else np.full(len(r), 72.0)
    fair = np.clip(since_h / 48.0, 0.0, 1.0)

    parts = {"need": SCORE_WEIGHTS["need"] * need,
             "capacity": SCORE_WEIGHTS["capacity"] * (1.0 - cap_used * 0.5),
             "distance": SCORE_WEIGHTS["distance"] * dist_score,
             "window": SCORE_WEIGHTS["window"] * window_score,
             "shelf_life": SCORE_WEIGHTS["shelf_life"] * shelf_slack,
             "fairness": SCORE_WEIGHTS["fairness"] * fair}
    score = sum(parts.values())
    breakdown = [dict(p, total=float(score[k])) for k, p in
                 ((k, {name: float(val[k]) for name, val in parts.items()})
                  for k in range(len(pairs)))]

    reasons = []
    for k in range(len(pairs)):
        row = {name: float(val[k]) for name, val in parts.items()}
        ranked = sorted(row.items(), key=lambda kv: kv[1], reverse=True)[:3]
        reasons.append([f"{name}:{val:.1f}" for name, val in ranked])
    return pairs.assign(score=score, score_breakdown=breakdown, reasons=reasons)


def feasible_pairs(offers: pd.DataFrame, recips: pd.DataFrame, now_min: float = 0.0,
                   buffer_min: float = 20.0) -> pd.DataFrame:
    """Vectorised feasibility filter.

    offers : offer_id, lat, lon, portions, safe_minutes, category, needs_cold
    recips : recip_id, lat, lon, capacity, accepts ('ALL' or 'a,b'), has_cold,
             priority (1-3), close_min, [hours_since_last]
    """
    offers = offers.reset_index(drop=True)
    recips = recips.reset_index(drop=True)
    if offers.empty or recips.empty:
        return pd.DataFrame(columns=["i", "j", "travel_min", "buffer_min"])

    o_lat, o_lon = offers.lat.to_numpy(float)[:, None], offers.lon.to_numpy(float)[:, None]
    r_lat, r_lon = recips.lat.to_numpy(float)[None, :], recips.lon.to_numpy(float)[None, :]
    tm = travel_minutes(haversine_km(o_lat, o_lon, r_lat, r_lon))
    safe = offers.safe_minutes.to_numpy(float)[:, None]
    cap = recips.capacity.to_numpy(float)[None, :]
    close = recips.close_min.to_numpy(float)[None, :]

    ok = np.broadcast_to(
        (tm + buffer_min <= safe)
        & (now_min + tm <= close)
        & (cap > 0)
        & (recips.accepts.to_numpy() == "ALL"), tm.shape).copy()
    # category acceptance (vectorised with a python fallback for the rare mixed list)
    for j, acc in enumerate(recips["accepts"].tolist()):
        if acc != "ALL":
            ok[:, j] &= offers.category.isin(str(acc).split(",")).to_numpy()
    ok &= np.where(recips.has_cold.to_numpy(bool)[None, :], True,
                   ~offers.needs_cold.to_numpy(bool)[:, None])

    i, j = np.nonzero(ok)
    pairs = pd.DataFrame({"i": i, "j": j, "travel_min": tm[i, j], "buffer_min": float(buffer_min)})
    if pairs.empty:
        LOG.info("no feasible (offer, recipient) pair survived the filter")
        return pairs
    # LP objective weight: priority first, then fill the window early.
    prio = recips.priority.to_numpy(float)[pairs.j.to_numpy()]
    horizon = np.maximum(offers.safe_minutes.to_numpy(float)[pairs.i.to_numpy()], 1.0)
    pairs["w"] = prio * (1.0 - pairs.travel_min.to_numpy() / horizon) + 0.1
    return pairs


def _frame(P: pd.DataFrame, offers: pd.DataFrame, recips: pd.DataFrame, x: np.ndarray) -> pd.DataFrame:
    if P.empty or x is None:
        return pd.DataFrame(columns=["offer_id", "recip_id", "portions", "travel_min"])
    out = P.assign(portions=np.floor(np.asarray(x, dtype=float) + 1e-9)).copy()
    out = out[out.portions > 0]
    if out.empty:
        return out.drop(columns=["portions"])
    return pd.DataFrame({
        "offer_id": offers.offer_id.to_numpy()[out.i.to_numpy()],
        "recip_id": recips.recip_id.to_numpy()[out.j.to_numpy()],
        "portions": out.portions.astype(int).to_numpy(),
        "travel_min": out.travel_min.to_numpy(),
        "score": out.score.to_numpy() if "score" in out else np.nan,
        "score_breakdown": list(out.score_breakdown) if "score_breakdown" in out else [],
        "reasons": list(out.reasons) if "reasons" in out else [],
    })


def allocate(offers: pd.DataFrame, recips: pd.DataFrame, now_min: float = 0.0,
             buffer_min: float = 20.0) -> pd.DataFrame:
    """Optimal (max-weighted) allocation via a linear programme.

    A batch may be split across several NGOs; a recipient's row constrains the *total*
    it receives across all offers, which is what the capacity column actually means.
    """
    offers, recips = offers.reset_index(drop=True), recips.reset_index(drop=True)
    P = score_pairs(offers, recips, feasible_pairs(offers, recips, now_min, buffer_min), now_min)
    if P.empty:
        return _frame(P, offers, recips, None)

    n_pairs = len(P)
    A = np.zeros((len(offers) + len(recips), n_pairs))
    A[P.i.to_numpy(), np.arange(n_pairs)] = 1.0
    A[len(offers) + P.j.to_numpy(), np.arange(n_pairs)] = 1.0
    b = np.concatenate([offers.portions.to_numpy(float), recips.capacity.to_numpy(float)])
    res = linprog(-P.w.to_numpy(), A_ub=A, b_ub=b, bounds=(0, None), method="highs")
    if res is None or res.x is None:
        # The old code did np.floor(res.x + 1e-9) here and raised AttributeError on None.
        LOG.warning("no feasible allocation (status=%s); returning an empty plan",
                    getattr(res, "message", "unknown"))
        return _frame(P, offers, recips, None)
    if not res.success:
        LOG.warning("LP did not certify optimality (status=%s); using the best feasible point",
                    res.message)
    return _frame(P, offers, recips, res.x)


def greedy_allocate(offers: pd.DataFrame, recips: pd.DataFrame, now_min: float = 0.0,
                    buffer_min: float = 20.0) -> pd.DataFrame:
    """Baseline: biggest offer first, nearest feasible recipient first."""
    offers, recips = offers.reset_index(drop=True), recips.reset_index(drop=True)
    P = score_pairs(offers, recips, feasible_pairs(offers, recips, now_min, buffer_min), now_min)
    if P.empty:
        return _frame(P, offers, recips, None)
    cap = recips.capacity.to_numpy(float).copy()
    x = np.zeros(len(P))
    order = P.sort_values(["i", "travel_min"], ascending=[False, True])
    for i in offers.sort_values("portions", ascending=False).index:
        left = float(offers.portions[i])
        for k in order.index[order.i == i]:
            j = int(P.j[k])
            q = min(left, cap[j])
            x[P.index.get_loc(k)] = q
            cap[j] -= q
            left -= q
            if left <= 1e-9:
                break
    return _frame(P, offers, recips, x)


def _selftest() -> None:
    offers = pd.DataFrame([{"offer_id": "O1", "lat": 26.91, "lon": 75.79, "portions": 500,
                            "safe_minutes": 200, "category": "rice_dal", "needs_cold": False}])
    recips = pd.DataFrame({
        "recip_id": ["R1", "R2", "R3"], "lat": [26.92, 26.90, 27.10], "lon": [75.80, 75.78, 75.70],
        "capacity": [200, 150, 400], "accepts": ["ALL", "rice_dal", "ALL"],
        "has_cold": [False, False, True], "priority": [3, 2, 1],
        "close_min": [300, 300, 300], "hours_since_last": [72.0, 12.0, 48.0]})
    lp = allocate(offers, recips)
    gr = greedy_allocate(offers, recips)
    print(f"LP     -> {len(lp)} allocations, {int(lp.portions.sum()) if len(lp) else 0} portions")
    print(f"greedy -> {len(gr)} allocations, {int(gr.portions.sum()) if len(gr) else 0} portions")
    far = recips.copy()
    far["close_min"] = 1
    print(f"infeasible -> {len(allocate(offers, far))} allocations (expected 0, no crash)")


if __name__ == "__main__":
    _selftest()