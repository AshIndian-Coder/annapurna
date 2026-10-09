"""Benchmark matching + routing vs simple baselines on SIMULATED Jaipur-like instances.

    python ml_logistics_benchmark.py

Speed (25 km/h) and road factor (1.3) are assumptions. Results are a pipeline check, not field evidence.

**Why this benchmark scores on-time portions, not just portions delivered.**

The first version of this file compared ``served_portions`` between OR-Tools and a
nearest-neighbour walk, and the two tied at 500/500 on every instance: with a 2x300
fleet against 500 portions, fleet capacity -- not route quality -- decided the answer, so
the benchmark could not distinguish a real optimiser from a greedy one.  Tying the fleet
tighter (see ``--vehicles`` / ``--capacity``) does not fix it either, because both
methods then drop the same stops for the same reason.

What actually separates a good route from a bad one in food rescue is **arriving before
the food spoils**.  ``solve_vrptw`` already records ``deadline_ok`` per stop and this
benchmark now counts it, so the headline numbers are:

* ``ontime_portions``  -- portions delivered inside their expiry window (the scoreboard)
* ``late_portions``    -- portions delivered but already spoiled (the failure that matters)
* ``served_portions``  -- total delivered, capacity-bound, kept for continuity

The instance generator also stores the generated instances to ``data/raw/sim_logistics``
with a ``.meta.json`` beside them, matching how the other simulators record provenance,
so a benchmark run is reproducible from the file rather than only from the seed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from ml_logistics_match import allocate, greedy_allocate
from ml_logistics_route import nearest_neighbour, solve_vrptw
from ml_utils import get_logger, get_paths, save_json

LOG = get_logger("logistics.benchmark")

#: Jaipur city centre -- the demo city for this deployment.
CENTER = (26.9124, 75.7873)
#: ASSUMPTIONS, not measurements. Replace with an OSRM/ORS distance matrix or driver GPS
#: traces before quoting any delivery-time number to a judge.
SPEED_KMPH = 25.0
ROAD_FACTOR = 1.3


def make_instance(rng, n_recips=12, portions=500):
    offers = pd.DataFrame([{"offer_id": "O1", "lat": CENTER[0], "lon": CENTER[1], "portions": portions,
                            "safe_minutes": int(rng.integers(150, 300)), "category": "rice_dal",
                            "needs_cold": False}])
    recips = pd.DataFrame({
        "recip_id": [f"R{i}" for i in range(n_recips)],
        "lat": CENTER[0] + rng.normal(0, 0.05, n_recips), "lon": CENTER[1] + rng.normal(0, 0.05, n_recips),
        "capacity": rng.integers(40, 160, n_recips), "accepts": "ALL", "has_cold": rng.random(n_recips) < 0.5,
        "priority": rng.integers(1, 4, n_recips), "close_min": rng.integers(120, 360, n_recips),
        "hours_since_last": rng.uniform(0, 96, n_recips)})
    return offers, recips


def to_stops(alloc, recips, offer, buffer_min=20):
    m = alloc.merge(recips, on="recip_id")
    return pd.DataFrame({"lat": m.lat, "lon": m.lon, "demand": m.portions.astype(int), "ready": 0,
                         "latest": np.minimum(m.close_min, offer.safe_minutes - buffer_min).astype(int), "service": 10})


def on_time_portions(result: dict, stops: pd.DataFrame) -> tuple[int, int]:
    """Split the delivered portions into (on-time, late) using each stop's deadline.

    ``solve_vrptw`` emits an explicit ``deadline_ok`` per visited stop.  ``nearest_neighbour``
    does not: it only ever selects a stop it can reach inside the window, so its arrival
    time has to be re-checked against ``stops.latest`` here.  Without that fallback the
    baseline silently scores every portion as late and the benchmark reports a fake
    500-vs-0 win -- which is exactly the kind of number that must not survive review.

    Either way the test is the same: ``arrive_min <= latest[stop]``.  A portion arriving
    after its safe window is not rescued, it is only moved.
    """
    demand = stops.demand.to_numpy()
    latest = stops.latest.to_numpy()
    ontime = late = 0
    for route in result.get("routes", []):
        for s in route.get("stops", []):
            i = int(s["stop"])
            if i < 0 or i >= len(demand):
                continue
            qty = int(demand[i])
            if "deadline_ok" in s:
                ok = bool(s["deadline_ok"])
            else:
                ok = int(s.get("arrive_min", 1 << 30)) <= int(latest[i])
            if ok:
                ontime += qty
            else:
                late += qty
    return ontime, late


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Benchmark matching + routing against baselines.")
    ap.add_argument("--instances", type=int, default=20)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--vehicles", type=int, default=2)
    ap.add_argument("--capacity", type=int, default=200,
                    help="portions per vehicle. The fleet must be tight enough that route "
                         "QUALITY decides the outcome: at 300/vehicle the fleet carries all "
                         "500 portions and every solver ties at 500 on-time, which measures "
                         "the fleet, not the router.")
    ap.add_argument("--time-limit-s", type=int, default=3,
                    help="solver time limit; OR-Tools requires an integer here")
    ap.add_argument("--save-instances", action="store_true",
                    help="write the generated instances to data/raw/sim_logistics/ with a "
                         ".meta.json, like the other simulators")
    a = ap.parse_args(argv)
    rng = np.random.default_rng(a.seed)
    rows = []
    inst_ledger = []
    for k in range(a.instances):
        offers, recips = make_instance(rng)
        o = offers.iloc[0]
        depot = {"lat": CENTER[0], "lon": CENTER[1]}
        lp, gr = allocate(offers, recips), greedy_allocate(offers, recips)
        r = {"instance": k, "offered": int(o.portions),
             "lp_portions": int(lp.portions.sum()) if len(lp) else 0,
             "greedy_portions": int(gr.portions.sum()) if len(gr) else 0}
        if a.save_instances:
            inst_ledger.append({"instance": k,
                                "offer": o.to_dict(),
                                "recipients": recips.to_dict(orient="records")})
        if len(lp):
            s_lp = to_stops(lp, recips, o)
            ort = solve_vrptw(depot, s_lp, n_vehicles=a.vehicles, capacity=a.capacity,
                              speed=SPEED_KMPH, road_factor=ROAD_FACTOR, time_limit_s=a.time_limit_s)
            nn = nearest_neighbour(depot, s_lp, n_vehicles=a.vehicles, capacity=a.capacity,
                                   speed=SPEED_KMPH, road_factor=ROAD_FACTOR)
            o_ok, o_late = on_time_portions(ort, s_lp)
            n_ok, n_late = on_time_portions(nn, s_lp)
            r.update(ortools_delivered=ort["served_portions"], nn_delivered=nn["served_portions"],
                     ortools_ontime=o_ok, ortools_late=o_late,
                     nn_ontime=n_ok, nn_late=n_late,
                     ortools_minutes=ort["total_minutes"], nn_minutes=nn["total_minutes"],
                     ortools_dropped=len(ort["dropped"]), nn_dropped=len(nn["dropped"]))
        rows.append(r)
    df = pd.DataFrame(rows)
    summ = df.drop(columns="instance").mean().round(2).to_dict()

    # Headline verdict, stated in the artefact so a reader cannot mistake a tie for a win.
    scoreboard = None
    if "ortools_ontime" in df.columns and len(df):
        o_ok = float(df["ortools_ontime"].mean())
        n_ok = float(df["nn_ontime"].mean())
        o_late = float(df["ortools_late"].mean())
        n_late = float(df["nn_late"].mean())
        if o_ok > n_ok:
            verdict = "OR-TOOLS WINS"
        elif abs(o_ok - n_ok) < 1e-9:
            verdict = "TIE -- this fleet does not discriminate; tighten --vehicles/--capacity"
        else:
            verdict = "NEAREST-NEIGHBOUR WINS -- investigate the solver configuration"
        scoreboard = {
            "ortools_ontime_portions": round(o_ok, 2),
            "nn_ontime_portions": round(n_ok, 2),
            "ortools_late_portions": round(o_late, 2),
            "nn_late_portions": round(n_late, 2),
            "ontime_advantage": round(o_ok - n_ok, 2),
            "late_avoided": round(n_late - o_late, 2),
            "verdict": verdict,
            "note": ("On-time portions are the scoreboard. served_portions is "
                     "capacity-bound and ties between solvers whenever the fleet has "
                     "slack; a portion that arrives late is not rescued."),
        }

    out = get_paths().reports / "logistics_benchmark.json"
    save_json(out, {
        "mean": summ, "n_instances": a.instances, "seed": a.seed,
        "fleet": {"vehicles": a.vehicles, "capacity_per_vehicle": a.capacity,
                  "fleet_capacity": a.vehicles * a.capacity},
        "scoreboard": scoreboard,
        "assumptions": {"speed_kmph": SPEED_KMPH, "road_factor": ROAD_FACTOR,
                        "note": "ASSUMPTIONS; replace with a routing API or driver GPS traces"},
        "caveat": "simulated Jaipur-like instances; results are a pipeline check, not field evidence",
        "per_instance": df.to_dict(orient="records"),
    })
    if a.save_instances and inst_ledger:
        raw = get_paths().data_raw / "sim_logistics"
        raw.mkdir(parents=True, exist_ok=True)
        inst_path = raw / f"instances_seed{a.seed}_n{a.instances}.json"
        save_json(inst_path, {"seed": a.seed, "n_instances": a.instances,
                              "instances": inst_ledger})
        save_json(raw / "sim_logistics.meta.json", {
            "rows": sum(len(i["recipients"]) for i in inst_ledger),
            "instances": len(inst_ledger),
            "seed": a.seed,
            "city": "Jaipur (simulated)",
            "fleet": {"vehicles": a.vehicles, "capacity_per_vehicle": a.capacity},
            "assumptions": {"speed_kmph": SPEED_KMPH, "road_factor": ROAD_FACTOR},
            "caveat": ("SYNTHETIC routing instances. The on-time advantage is a pipeline "
                       "check on the solver, not evidence about real Jaipur traffic."),
        })
        LOG.info("instances -> %s", inst_path)

    print(df.round(1).to_string(index=False))
    print("\nmean over instances:", summ)
    if scoreboard:
        print("\n=== SCOREBOARD (on-time portions; late = arrived spoiled) ===")
        print(f"  OR-Tools      on-time {scoreboard['ortools_ontime_portions']:>8.1f}"
              f"   late {scoreboard['ortools_late_portions']:>7.1f}")
        print(f"  nearest-neigh on-time {scoreboard['nn_ontime_portions']:>8.1f}"
              f"   late {scoreboard['nn_late_portions']:>7.1f}")
        print(f"  advantage: {scoreboard['ontime_advantage']:+.1f} portions on time, "
              f"{scoreboard['late_avoided']:+.1f} spoiled portions avoided")
        print(f"  verdict: {scoreboard['verdict']}")
    print("Delivered on time = portions reaching NGOs inside their window AND before the "
          "food's safe-until lower bound.")
    print(f"artifacts -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
