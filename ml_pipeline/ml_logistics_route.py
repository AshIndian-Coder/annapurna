"""Delivery routing with time windows and capacity (OR-Tools), plus a nearest-neighbour baseline.

Every stop has [ready, latest] in minutes from now. `latest` must already be min(NGO closing time,
food safe-until lower bound - handling buffer). Stops that cannot be served are DROPPED with a penalty
and reported, never silently late.
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from ml_logistics_match import haversine_km, travel_minutes
from ml_utils import get_logger

LOG = get_logger("logistics.route")

REQUIRED_STOP_COLS = ("lat", "lon", "demand", "ready", "latest", "service")


def validate_stops(stops: pd.DataFrame) -> pd.DataFrame:
    """Fail loudly on a stop table the solver would silently mis-handle.

    OR-Tools does not validate ``SetRange(lo, hi)`` with ``hi < lo``: the solver simply
    finds no solution and returns ``None``, which the caller reports as "infeasible"
    even when the real problem was a bad deadline in the input.  That is the difference
    between a fixable API bug and a dead demo.
    """
    missing = [c for c in REQUIRED_STOP_COLS if c not in stops.columns]
    if missing:
        raise KeyError(f"stops frame is missing columns: {missing}")
    out = stops.reset_index(drop=True)
    latest = pd.to_numeric(out["latest"], errors="coerce")
    ready = pd.to_numeric(out["ready"], errors="coerce")
    bad = out.index[(ready > latest).to_numpy() | latest.isna().to_numpy()].tolist()
    if bad:
        raise ValueError(f"stop windows are empty or NaN at rows {bad[:10]}; "
                         f"ready must be <= latest and both must be finite")
    for c in ("demand", "service", "ready", "latest"):
        out[c] = pd.to_numeric(out[c]).astype(int)
    if (out["demand"] < 0).any():
        raise ValueError("stop demand must be >= 0")
    return out


def _tm(depot: dict, stops: pd.DataFrame, speed: float, road: float) -> np.ndarray:
    lat = np.r_[depot["lat"], stops.lat.to_numpy(float)]
    lon = np.r_[depot["lon"], stops.lon.to_numpy(float)]
    d = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    return np.round(travel_minutes(d, speed, road)).astype(int)


def solve_vrptw(depot: dict, stops: pd.DataFrame, n_vehicles: int, capacity: int, speed: float = 25.0,
                road_factor: float = 1.3, horizon: int | None = None, time_limit_s: int = 5,
                drop_penalty: int | None = None) -> dict:
    """Capacity + time-window VRPTW with explicit drop penalties.

    ``stops`` columns: ``lat, lon, demand, ready, latest, service`` (ints except lat/lon).
    ``latest`` must already be ``min(NGO closing time, food safe-until lower bound -
    handling buffer)``.

    Two fixes over the previous version:

    * the Time dimension's capacity is now derived from the data
      (``max(latest) + max(travel) + slack``) instead of a hard-coded 600 minutes, so an
      instance whose deadlines sit past 10:00 no longer becomes spuriously infeasible;
    * ``drop_penalty`` defaults to the largest travel cost in the instance, so it always
      dominates "skipping" a stop relative to a very long route.
    """
    stops = validate_stops(stops)
    tm = _tm(depot, stops, speed, road_factor)
    service = np.r_[0, stops.service.to_numpy(int)]
    demand = np.r_[0, stops.demand.to_numpy(int)]
    n = len(stops) + 1
    if capacity <= 0:
        raise ValueError("vehicle capacity must be > 0")
    if demand.sum() > capacity * n_vehicles:
        LOG.warning("total demand %d exceeds fleet capacity %d; some stops must be dropped",
                    int(demand.sum()), capacity * n_vehicles)
    slack = int(max(tm.max(), 1) * 2 + 60)
    horizon = int(max(horizon or 0, int(stops["latest"].max()) + slack))
    drop_penalty = int(drop_penalty if drop_penalty is not None else max(tm.max() * 3, 1000))

    mgr = pywrapcp.RoutingIndexManager(n, n_vehicles, 0)
    routing = pywrapcp.RoutingModel(mgr)

    def time_cb(i, j):
        a, b = mgr.IndexToNode(i), mgr.IndexToNode(j)
        return int(tm[a][b] + service[a])

    t_idx = routing.RegisterTransitCallback(time_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(t_idx)
    routing.AddDimension(t_idx, horizon, horizon, True, "Time")
    tdim = routing.GetDimensionOrDie("Time")
    for s in range(len(stops)):
        idx = mgr.NodeToIndex(s + 1)
        tdim.CumulVar(idx).SetRange(int(stops.ready[s]), int(stops.latest[s]))
        routing.AddDisjunction([idx], drop_penalty)
    d_idx = routing.RegisterUnaryTransitCallback(lambda i: int(demand[mgr.IndexToNode(i)]))
    routing.AddDimensionWithVehicleCapacity(d_idx, 0, [int(capacity)] * n_vehicles, True, "Load")

    p = pywrapcp.DefaultRoutingSearchParameters()
    p.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    p.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    p.time_limit.seconds = time_limit_s
    sol = routing.SolveWithParameters(p)
    if sol is None:
        LOG.warning("OR-Tools found no feasible solution; every stop is dropped")
        return {"routes": [], "dropped": list(range(len(stops))), "total_minutes": 0,
                "served_portions": 0, "solver": "ortools-vrptw", "feasible": False}

    routes, total, served = [], 0, 0
    for v in range(n_vehicles):
        idx, seq = routing.Start(v), []
        while not routing.IsEnd(idx):
            node = mgr.IndexToNode(idx)
            if node:
                seq.append({"stop": node - 1,
                            "arrive_min": int(sol.Min(tdim.CumulVar(idx))),
                            "deadline_ok": bool(sol.Min(tdim.CumulVar(idx)) <= int(stops.latest[node - 1]))})
            idx = sol.Value(routing.NextVar(idx))
        if seq:
            total += sol.Min(tdim.CumulVar(idx))
            served += int(sum(demand[s["stop"] + 1] for s in seq))
            routes.append({"vehicle": v, "stops": seq, "return_min": int(sol.Min(tdim.CumulVar(idx)))})
    dropped = [s for s in range(len(stops))
               if sol.Value(routing.NextVar(mgr.NodeToIndex(s + 1))) == mgr.NodeToIndex(s + 1)]
    return {"routes": routes, "dropped": dropped, "total_minutes": int(total),
            "served_portions": served, "solver": "ortools-vrptw",
            "horizon_minutes": horizon, "drop_penalty": drop_penalty}


def nearest_neighbour(depot: dict, stops: pd.DataFrame, n_vehicles: int, capacity: int, speed: float = 25.0,
                      road_factor: float = 1.3) -> dict:
    """Baseline: each vehicle drives to the nearest stop it can still reach in-window."""
    stops = validate_stops(stops)
    tm = _tm(depot, stops, speed, road_factor)
    left, routes, total, served = set(range(len(stops))), [], 0, 0
    for v in range(n_vehicles):
        pos, t, load, seq = 0, 0, 0, []
        while True:
            best = None
            for s in left:
                arrive = max(t + tm[pos][s + 1], int(stops.ready[s]))
                if arrive <= stops.latest[s] and load + stops.demand[s] <= capacity:
                    if best is None or tm[pos][s + 1] < best[1]:
                        best = (s, tm[pos][s + 1], arrive)
            if best is None:
                break
            s, _, arrive = best
            seq.append({"stop": s, "arrive_min": int(arrive)})
            t, pos, load = arrive + int(stops.service[s]), s + 1, load + int(stops.demand[s])
            left.discard(s)
        if seq:
            back = t + tm[pos][0]
            total += back
            served += load
            routes.append({"vehicle": v, "stops": seq, "return_min": int(back)})
    return {"routes": routes, "dropped": sorted(left), "total_minutes": int(total), "served_portions": served,
            "solver": "nearest_neighbour"}
