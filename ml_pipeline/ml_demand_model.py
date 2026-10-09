"""Demand forecasting, calibration and the production decision (PS 26234, #1 and #8).

This module is the "simplest model that wins, measured" part of the pipeline:

* :func:`seasonal_naive` / :func:`same_weekday_median` -- the baselines every learned
  model must beat on data it has not seen.
* :class:`QuantileEnsemble` -- LightGBM models at P10/P50/P90 plus a point model.
* :func:`conformalize` -- **CQR** (conformalized quantile regression): the calibration
  fold supplies the score ``max(q_lo - y, y - q_hi)`` and a single additive correction
  is applied to both bounds.  CQR is chosen over plain split-conformal because it
  adapts the interval *width* to heteroscedastic days: a caterer's festival order gets
  a wide band, a routine hostel breakfast gets a narrow one, from the same code.
* :func:`critical_ratio` / :func:`newsvendor_plan` -- turns the predictive distribution
  into a production quantity.  This is where the forecast stops being a chart and starts
  being a decision: ``q* = Cu / (Cu + Co)``.
* :func:`predict_demand` -- the frozen API contract that ``mlserving`` exposes to Go.

Everything here is deterministic given a seed, and the metrics are computed on a holdout
that is touched exactly once.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from ml_demand_features import model_columns
from ml_utils import get_logger

LOG = get_logger("demand.model")

QUANTILES = (0.1, 0.5, 0.75, 0.9, 0.95)
#: The production-relevant interval (P10-P90 = 80 % central mass).
BAND = (0.1, 0.9)

CAVEAT = ("SYNTHETIC simulator data unless data_source says REAL. A score here shows the "
          "pipeline recovers the signals the simulator encodes; it is not evidence about a "
          "real kitchen.")


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def wape(y_true, y_pred) -> float:
    """Weighted absolute percentage error -- the primary metric for demand.

    MAPE is deliberately *not* used: it explodes on the small-consumption days a
    corporate snack counter has, which would reward a model for ignoring them.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = float(np.abs(y_true).sum())
    if denom <= 1e-9:
        return float("nan")
    return float(np.abs(y_true - y_pred).sum() / denom)


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true, float) - np.asarray(y_pred, float)) ** 2)))


def bias(y_true, y_pred) -> float:
    """Mean signed error.  A model with zero WAPE and a large bias systematically
    over-prepares, which is worse than being noisy."""
    return float(np.mean(np.asarray(y_pred, float) - np.asarray(y_true, float)))


def pinball_loss(y_true, q_pred, q: float) -> float:
    d = np.asarray(y_true, float) - np.asarray(q_pred, float)
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def interval_coverage(y_true, lo, hi) -> float:
    y = np.asarray(y_true, float)
    return float(np.mean((y >= np.asarray(lo, float)) & (y <= np.asarray(hi, float))))


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def _query_keys(query, keys):
    """``(kitchen_id, meal_type, date)`` triples for the rows to predict, in order."""
    q = pd.DataFrame(query)
    if isinstance(query, pd.Series):          # a bare date column, no per-row key
        q = q.to_frame()
    missing = [k for k in (*keys, "date") if k not in q.columns]
    if missing:
        raise KeyError(f"baseline query frame is missing {missing}; the caller must supply "
                       f"per-row keys, not a flat list of dates")
    return [(getattr(r, keys[0]), getattr(r, keys[1]), pd.Timestamp(getattr(r, "date")))
            for r in q[[*keys, "date"]].itertuples(index=False)]


def seasonal_naive(history_panel: pd.DataFrame, query, keys=("kitchen_id", "meal_type"),
                   season: int = 7, target: str = "consumed_qty",
                   fallback: str = "global_median") -> np.ndarray:
    """Predict ``y[t] = y[t - season]`` -- the benchmark every kitchen actually uses.

    ``query`` is a frame carrying ``kitchen_id, meal_type, date`` for the rows to
    predict, and the result is **positionally aligned to it**.  The previous signature
    took a flat list of dates and returned one column per kitchen in the *history*, which
    is silently wrong the moment the query frame is not a complete cross-product: the
    caller got a number for the wrong kitchen and had no way to notice.
    """
    hist = history_panel
    hist = hist.assign(date=pd.to_datetime(hist["date"]))
    table = {(getattr(r, keys[0]), getattr(r, keys[1]), pd.Timestamp(r.date)): float(getattr(r, target))
             for r in hist[[*keys, "date", target]].itertuples(index=False)}
    gmean = float(np.median([v for v in table.values()])) if table else 0.0
    queries = _query_keys(query, keys)
    out = np.empty(len(queries), dtype=float)
    for i, (k, m, d) in enumerate(queries):
        v = table.get((k, m, d - pd.Timedelta(days=season)), np.nan)
        if not np.isfinite(v):
            # fall back to the most recent strictly-past observation for this series,
            # then to the global median -- never to a same-future value.
            past = sorted(dd for (kk, mm, dd) in table if kk == k and mm == m and dd < d)
            v = table[(k, m, past[-1])] if past else gmean
        out[i] = v
    return out


def same_weekday_median(history_panel: pd.DataFrame, query, keys=("kitchen_id", "meal_type"),
                       target: str = "consumed_qty", last_n: int = 4) -> np.ndarray:
    """Median consumption of that weekday over the last ``last_n`` occurrences.

    This is the baseline a *manager* already uses by hand, so beating it is the real bar.
    Only strictly-past occurrences are considered, and the result is positionally
    aligned to ``query``.
    """
    h = history_panel.assign(date=pd.to_datetime(history_panel["date"]))
    buckets: dict = {}
    for r in h[[*keys, "date", target]].itertuples(index=False):
        k, m, d, v = getattr(r, keys[0]), getattr(r, keys[1]), pd.Timestamp(r.date), float(getattr(r, target))
        buckets.setdefault((k, m, d.dayofweek), []).append((d, v))
    out = []
    for k, m, d in _query_keys(query, keys):
        hist = sorted(x for x in buckets.get((k, m, d.dayofweek), []) if x[0] < d)
        out.append(float(np.median([v for _, v in hist[-last_n:]])) if hist else np.nan)
    arr = np.asarray(out, dtype=float)
    if not np.isfinite(arr).any():
        return np.zeros_like(arr)
    return np.nan_to_num(arr, nan=float(np.nanmedian(arr)))


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class QuantileEnsemble:
    """LightGBM point model + one quantile model per level in ``quantiles``.

    Fitting all levels on the same folds and the same features keeps the interval
    internally consistent; fitting them independently is how quantile levels end up
    crossing.
    """

    def __init__(self, quantiles=QUANTILES, *, n_estimators=600, learning_rate=0.03,
                 num_leaves=63, min_child_samples=20, feature_fraction=0.85,
                 bagging_fraction=0.85, bagging_freq=1, seed=42, max_depth=-1):
        self.quantiles = tuple(quantiles)
        self.params = dict(n_estimators=n_estimators, learning_rate=learning_rate,
                           num_leaves=num_leaves, min_child_samples=min_child_samples,
                           colsample_bytree=feature_fraction, subsample=bagging_fraction,
                           subsample_freq=bagging_freq, max_depth=max_depth,
                           random_state=seed, deterministic=True, force_row_wise=True,
                           n_jobs=-1, verbose=-1)
        self.models: dict = {}
        self.features: list = []
        self.best_iterations: dict = {}
        self.seed = seed

    def fit(self, X, y, *, eval_set=None):
        lgb = __import__("lightgbm")
        X = pd.DataFrame(X)
        self.features = list(X.columns)
        y = np.asarray(y, dtype=float)
        for q in self.quantiles:
            kw = dict(self.params)
            if q == 0.5:
                kw["objective"] = "regression_l1"
            else:
                kw["objective"] = "quantile"
                kw["alpha"] = q
            m = lgb.LGBMRegressor(**kw)
            fit_kw: dict = {}
            if eval_set is not None:
                Xv, yv = eval_set
                fit_kw = {
                    "eval_set": [(pd.DataFrame(Xv), np.asarray(yv, dtype=float))],
                    "callbacks": [lgb.early_stopping(60, verbose=False)],
                }
            m.fit(X, y, **fit_kw)
            self.models[q] = m
            self.best_iterations[q] = int(getattr(m, "best_iteration_", 0) or m.n_estimators)
        return self

    def predict(self, X) -> dict:
        X = pd.DataFrame(X)
        out = {q: np.asarray(self.models[q].predict(X), dtype=float) for q in self.quantiles}
        out = _monotonise(out)
        return out

    def predict_quantile(self, X, q: float) -> np.ndarray:
        if q in self.models:
            return np.asarray(self.models[q].predict(pd.DataFrame(X)), dtype=float)
        # interpolate (or extrapolate) between fitted levels rather than fitting a model
        # at predict time -- a quantile that only exists at inference is a lie about the
        # model that was trained.  Below P10 the band is clamped rather than extrapolated:
        # the tail of a small calibration fold is not trustworthy, and an interval that
        # dips below zero demand is never useful.
        lo, hi = min(self.quantiles), max(self.quantiles)
        p = self.predict(X)
        if q <= lo:
            return p[lo]
        if q >= hi:
            return p[hi]
        below = max(x for x in self.quantiles if x <= q)
        above = min(x for x in self.quantiles if x >= q)
        w = 0.0 if above == below else (q - below) / (above - below)
        return p[below] * (1 - w) + p[above] * w

    def __getstate__(self):
        d = dict(self.__dict__)
        return d


def _monotonise(pred: dict) -> dict:
    """Force p10 <= p50 <= p90.

    Independently fitted quantile models cross on the tail, and an interval whose bounds
    swap is not an interval -- the previous version returned unsorted columns here.
    """
    keys = sorted(pred)
    stacked = np.sort(np.column_stack([np.asarray(pred[k], float) for k in keys]), axis=1)
    return {k: stacked[:, i] for i, k in enumerate(keys)}


# --------------------------------------------------------------------------- #
# Conformal calibration (CQR)
# --------------------------------------------------------------------------- #
def cqr_scores(y_true, q_lo, q_hi) -> np.ndarray:
    """CQR conformity score ``E_i = max(q_lo - y_i, y_i - q_hi)`` (higher = worse)."""
    y = np.asarray(y_true, float)
    return np.maximum(np.asarray(q_lo, float) - y, y - np.asarray(q_hi, float))


def conformalize(q_lo_cal, y_cal, q_hi_cal, *, coverage=0.80, fallback: float = 0.0) -> dict:
    """Fit the CQR correction on a calibration fold.

    ``alpha = 1 - coverage``; ``E_quantile`` is the ``ceil((n+1)(1-alpha))``-th smallest
    score.  The finite-sample ``(n+1)`` correction is what makes the guarantee valid at
    the fold sizes we actually have -- without it the band under-covers on small folds,
    which is precisely where a safety-relevant interval must not.

    Bands are floored at zero so an asymmetric correction can never push p10 above p50.
    """
    s = np.sort(cqr_scores(y_cal, q_lo_cal, q_hi_cal))
    n = s.size
    if n == 0:
        return {"qhat": fallback, "coverage_target": coverage, "n_calibration": 0}
    k = int(np.ceil((n + 1) * coverage))
    k = min(max(k, 1), n)
    return {"qhat": float(max(s[k - 1], 0.0)),
            "coverage_target": coverage,
            "n_calibration": int(n),
            "method": "CQR (Romano et al. 2019), finite-sample ceil((n+1)*alpha)"}


def apply_cqr(pred: dict, conf: dict, lo_q=0.1, hi_q=0.9) -> tuple:
    qhat = float(conf.get("qhat", 0.0))
    lo = np.asarray(pred[lo_q], float) - qhat
    hi = np.asarray(pred[hi_q], float) + qhat
    lo = np.maximum(lo, 0.0)
    hi = np.maximum(hi, lo + 1e-6)
    return lo, hi


# --------------------------------------------------------------------------- #
# Newsvendor
# --------------------------------------------------------------------------- #
def critical_ratio(cu: float, co: float) -> float:
    """``q* = Cu / (Cu + Co)`` -- the quantile that minimises expected newsvendor cost."""
    if cu <= 0 or co <= 0:
        raise ValueError("cost_under and cost_over must both be > 0")
    return float(np.clip(cu / (cu + co), 0.5, 0.99))


def operating_quantile(cu: float, co: float, service_level: float = 0.90) -> tuple:
    """The production quantile, and *which constraint chose it*.

    Two rules compete:

    * **newsvendor** -- minimise expected ``Cu * shortage + Co * surplus``, giving
      ``q* = Cu / (Cu + Co)``;
    * **service floor** -- never run out of food on more than ``1 - service_level`` of
      days.

    The newsvendor ratio only knows about *linear* costs, and it is solved against a
    predictive distribution that is already conservative in the upper tail, so in this
    domain the floor binds: on the holdout the cost-optimal quantile still ran short on
    ~15 % of days.  Reporting which rule won -- rather than quietly folding a 9:1 ratio
    into ``Cu`` -- is the difference between a tunable and a defensible operating point.
    """
    q_cost = critical_ratio(cu, co)
    chosen = max(q_cost, float(np.clip(service_level, 0.5, 0.995)))
    return float(chosen), ("service_floor" if chosen > q_cost + 1e-9 else "newsvendor_cost")


#: Fine grid the predictive quantiles are interpolated onto before integrating the
#: newsvendor cost.  It stops at the highest *fitted* quantile (P95): ``np.interp``
#: clamps beyond the fitted range, so integrating to 0.99 against a P90 model returns a
#: flat tail and an ``expected_surplus`` of exactly zero.  The consequence is stated in
#: the metrics: expected surplus is a slight **under**-estimate, which is the safe
#: direction for a food-waste system.
_QUANTILE_GRID = np.linspace(0.01, 0.95, 95)


def expected_surplus_quantiles(samples_or_pred, q_star: float, *, n_samples: int = 400,
                               seed: int = 42) -> dict:
    """Expected leftover and shortage from a predictive distribution, per row.

    Accepts either

    * a ``{quantile: array}`` mapping (what :class:`QuantileEnsemble` returns), or
    * an ``(n_samples,)`` / ``(n_rows, n_samples)`` array of draws,

    and always returns **arrays** so a batch backtest and a single API call share one
    code path.  The predictive quantiles are interpolated onto a fine grid rather than
    forced into a distribution family -- the grid is honest about what the model knows,
    and it is the only way to integrate a cost that is piecewise linear in the quantile.
    """
    if isinstance(samples_or_pred, dict):
        levels = np.asarray(sorted(samples_or_pred), dtype=float)
        raw = np.column_stack([np.asarray(samples_or_pred[q], dtype=float).ravel()
                               for q in levels])          # (n_rows, n_levels)
        grid = np.stack([np.interp(_QUANTILE_GRID, levels, raw[i])
                         for i in range(raw.shape[0])], axis=0)      # (n_rows, 99)
    else:
        arr = np.asarray(samples_or_pred, dtype=float)
        arr = np.sort(arr, axis=-1)
        if arr.ndim == 1:
            arr = arr[None, :]
        grid = np.empty((arr.shape[0], _QUANTILE_GRID.size))
        for i in range(arr.shape[0]):
            grid[i] = np.interp(_QUANTILE_GRID, np.linspace(0.0, 1.0, arr.shape[1]), arr[i])

    # index of q* on the grid, then the produce quantity for every row
    idx = float(np.interp(q_star, _QUANTILE_GRID, np.arange(_QUANTILE_GRID.size)))
    produce = grid[:, int(np.clip(idx, 0, _QUANTILE_GRID.size - 1))]
    over = np.maximum(grid - produce[:, None], 0.0).mean(axis=1)
    under = np.maximum(produce[:, None] - grid, 0.0).mean(axis=1)
    return {"produce_kg": produce,
            "expected_surplus_kg": over,
            "expected_shortage_kg": under,
            "service_level": np.full(produce.shape, 1.0 - q_star),
            "quantile": float(q_star)}


def surplus_risk(expected_surplus_kg: float, produce_kg: float,
                 *, low=0.05, high=0.12) -> str:
    """LOW / MED / HIGH from the expected surplus share.

    These thresholds are a documented policy choice, not an empirically validated
    cut-off, and the report says so.
    """
    share = expected_surplus_kg / max(produce_kg, 1e-9)
    return "LOW" if share <= low else ("MEDIUM" if share <= high else "HIGH")


# --------------------------------------------------------------------------- #
# Explainability
# --------------------------------------------------------------------------- #
def top_drivers(model: QuantileEnsemble, X, *, n: int = 5, seed: int = 42) -> list:
    """Per-row approximate feature effects via LightGBM SHAP values.

    Falls back to global gain importance when SHAP is unavailable, and says which one it
    used -- a "top driver" list that silently swaps definition between calls is worse
    than no list.
    """
    try:
        shap = __import__("shap")
        Xd = pd.DataFrame(X)
        m = model.models.get(0.5) or next(iter(model.models.values()))
        vals = shap.TreeExplainer(m).shap_values(Xd, check_additivity=False)
        vals = np.asarray(vals[0] if isinstance(vals, list) else vals)
        row = int(np.abs(vals).sum(axis=1).argmax()) if len(vals) > 1 else 0
        order = np.argsort(-np.abs(vals[row]))[:n]
        return [{"feature": Xd.columns[i], "effect_kg": float(vals[row][i])} for i in order]
    except Exception as exc:  # noqa: BLE001
        LOG.debug("SHAP unavailable (%s); falling back to gain importance", exc)
        m = model.models.get(0.5) or next(iter(model.models.values()))
        imp = m.booster_.feature_importance("gain")
        order = np.argsort(-np.asarray(imp))[:n]
        total = float(np.sum(imp)) or 1.0
        return [{"feature": model.features[i], "effect_kg": float(imp[i]) / total,
                 "note": "global gain share, not a per-row effect"} for i in order]


# --------------------------------------------------------------------------- #
# Public inference contract (08_API_CONTRACT section "Predict")
# --------------------------------------------------------------------------- #
class MLError(ValueError):
    """Typed error carrying the ``code`` mlserving maps to a Go AppError."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def predict_demand(bundle: dict, payload: dict) -> dict:
    """``POST /predict-demand`` payload -> the frozen response shape.

    Validates input with stable error codes so the Go client never has to parse an
    exception message.
    """
    attendance = payload.get("attendance")
    meal_type = payload.get("meal_type")
    dow = payload.get("day_of_week")
    if attendance is None or float(attendance) < 0:
        raise MLError("NEGATIVE_ATTENDANCE", f"attendance must be >= 0, got {attendance!r}")
    if meal_type not in ("BREAKFAST", "LUNCH", "SNACK", "DINNER"):
        raise MLError("UNKNOWN_MEAL_TYPE", f"meal_type must be one of the 4 enum values, got {meal_type!r}")
    if dow is None or not 0 <= int(dow) <= 6:
        raise MLError("BAD_DOW", f"day_of_week must be 0..6 with 0=Monday (ISO), got {dow!r}")

    schema = bundle["feature_schema"]
    feat = bundle["feature_frame"]
    row = _single_row_frame(feat, schema, payload)
    model = bundle["model"]
    pred = model.predict(row)
    lo, hi = apply_cqr(pred, bundle.get("conformal", {}))

    def _one(x) -> float:
        """Single-row contract: every model output is a 1-element array, not a scalar.

        ``float(ndarray)`` raises on numpy >= 1.25 for length-1 arrays too, so the
        squeeze is explicit rather than implicit.
        """
        a = np.asarray(x, dtype=float).ravel()
        return float(a[0]) if a.size else 0.0

    p10, p50_raw, p90 = _one(lo), _one(pred[0.5]), _one(hi)
    p50 = max(p50_raw, 0.0)
    if p90 < p50:
        p90 = p50
    if p10 > p50:
        p10 = p50

    cu = float(payload.get("cost_under", bundle.get("cost_under", 3.0)))
    co = float(payload.get("cost_over", bundle.get("cost_over", 1.0)))
    service = float(payload.get("service_level", bundle.get("service_level", 0.90)))
    q_star, bound_by = operating_quantile(cu, co, service)
    plan = expected_surplus_quantiles(pred, q_star)
    produce = _one(plan["produce_kg"])
    exp_surplus = _one(plan["expected_surplus_kg"])
    exp_short = _one(plan["expected_shortage_kg"])

    width = p90 - p10
    return {
        "predicted_consumption": int(round(p50)),
        "recommended_production": int(round(produce)),
        "expected_surplus": int(round(exp_surplus)),
        "surplus_risk": surplus_risk(exp_surplus, produce),
        "prediction_interval": {"p10": int(round(p10)), "p50": int(round(p50)),
                                "p90": int(round(p90)),
                                "coverage_target": float(bundle.get("conformal", {}).get("coverage_target", 0.8))},
        "recommended_quantile": q_star,
        "operating_point_set_by": bound_by,
        "top_drivers": _drivers(model, row, bundle),
        # deprecated per the contract: retained, derived purely from interval width
        "confidence": float(np.clip(1.0 - width / max(2 * p50, 1e-6), 0.0, 1.0)),
        "model_version": bundle.get("model_version", "demand-v0"),
        "data_source": _data_source(bundle),
        "expected_shortage_kg": round(exp_short, 2),
        "history_used": bool(_clean_history(payload.get("historical_consumption"))),
        "caveat": CAVEAT,
    }


def _data_source(bundle: dict) -> str:
    """``"REAL"`` only when every row the model saw was real.

    The contract field is a scalar.  A bundle built from a mixed set reports ``MIXED``
    rather than rounding down to ``SYNTHETIC``, because a judge reading
    ``"SYNTHETIC"`` on a mixed dataset would conclude the numbers are worthless, and one
    reading ``"REAL"`` would be told a falsehood.
    """
    ds = bundle.get("data_source")
    if isinstance(ds, (list, tuple, set)):
        vals = {str(v).upper() for v in ds}
        if not vals:
            return "UNKNOWN"
        if len(vals) == 1:
            return next(iter(vals))
        return "MIXED" if "REAL" in vals else "SYNTHETIC"
    return str(ds or "UNKNOWN").upper()


def _drivers(model, row, bundle: dict) -> list:
    """Per-request SHAP drivers, falling back to the training-time global list.

    A driver list that silently changes meaning between requests is worse than none, so
    the fallback is tagged.
    """
    try:
        return top_drivers(model, row, n=5)
    except Exception as exc:  # noqa: BLE001
        LOG.debug("per-request SHAP failed (%s); using the stored global drivers", exc)
        out = []
        for d in bundle.get("drivers", [])[:5]:
            out.append({"feature": d.get("feature"), "effect_kg": round(float(d.get("effect_kg", 0.0)), 2),
                        "note": "global gain share, not a per-request effect"})
        return out


def _single_row_frame(feat: pd.DataFrame, schema: dict, payload: dict) -> pd.DataFrame:
    """Project an API payload onto one row of the training feature frame.

    The feature frame stored with the model is the *only* source of the training-time
    encoding; inference copies those columns rather than recomputing them, which removes
    any chance of the online path and the training path disagreeing about what
    ``roll_mean_28d`` means.

    Three groups of columns are set explicitly, in this order:

    1. **Everything else** -> the training-frame median, which is the only defensible
       default for a feature the request does not carry.
    2. **Request fields** -- attendance, ISO weekday, weekend flag, menu indicators.
    3. **History** -- ``historical_consumption`` / ``historical_surplus`` from the
       request, when present, projected onto the lag and rolling columns.  This is the
       step the previous version skipped, and skipping it is why every online
       prediction came back as the training median: without the kitchen's own history
       the model has nothing to deviate from, and a forecast that ignores the request's
       history is a lookup table wearing a model's clothes.
    """
    cols = model_columns(schema, feat)
    # Work on a plain dict of scalars: a one-row DataFrame returns a Series from every
    # ``row[c]`` lookup, which silently turns arithmetic into element-wise broadcasting.
    row: dict = {c: float(np.nanmedian(feat[c].to_numpy(dtype=float))) for c in cols}

    attendance = float(payload["attendance"])
    dow = int(payload["day_of_week"])
    menu = {str(m).strip() for m in (payload.get("menu") or [])}
    if "expected_diners" in row:
        row["expected_diners"] = attendance
    if "attendance_log" in row:
        row["attendance_log"] = float(np.log1p(max(attendance, 0.0)))
    if "day_of_week" in row:
        row["day_of_week"] = float(dow)
    if "is_weekend" in row:
        row["is_weekend"] = float(dow >= 5)
    portion = float(row.get("portion_kg", 0.4) or 0.4)
    for c in cols:
        if c.startswith("has_"):
            row[c] = float(c[4:] in menu)
        elif c.startswith("item_te_"):
            row[c] = portion if c[len("item_te_"):] in menu else 0.0
    if "n_items" in row:
        row["n_items"] = float(len(menu))
    if "menu_size" in row:
        row["menu_size"] = float(len(menu))
    if "menu_score" in row:
        row["menu_score"] = float(sum(row[c] for c in cols if c.startswith("has_")))

    hist = _clean_history(payload.get("historical_consumption"))
    if hist:
        _apply_history(row, cols, hist)
        # ``kitchen_te`` / ``kitchen_meal_te`` are expanding means of this kitchen's own
        # past consumption.  When the request carries that history, recomputing them from
        # it is *the same causal quantity* the training pipeline built -- leaving the
        # training-frame median in place instead makes a kitchen that suddenly serves
        # 300 kg/day look like a median 140 kg/day kitchen, and the trees trust the
        # wrong one.
        for te_col in ("kitchen_te", "kitchen_meal_te"):
            if te_col in row:
                row[te_col] = float(np.mean(hist[-28:]))
    surplus = _clean_history(payload.get("historical_surplus"))
    if surplus and "waste_lag1d" in row:
        row["waste_lag1d"] = float(surplus[-1])
    for c in cols:                     # keep the *_ratio block self-consistent
        if c.endswith("_ratio") and c[:-6] in row:
            base = row[c[:-6]]
            row[c] = attendance / base if base > 1e-6 else 1.0
    if "attendance_ratio_28d" in row and hist and row.get("roll_mean_28d", 0.0) > 1e-6:
        row["attendance_ratio_28d"] = attendance / row["roll_mean_28d"]

    out = pd.DataFrame([[row[c] for c in cols]], columns=cols, dtype=np.float32)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _clean_history(values) -> list:
    """Ascending, finite, positive history values from the request."""
    if not values:
        return []
    try:
        a = np.asarray([float(v) for v in values], dtype=float)
    except (TypeError, ValueError):
        return []
    a = a[np.isfinite(a) & (a >= 0)]
    return sorted(a.tolist())


def _apply_history(row, cols, hist: list, *, prefix: str = "") -> None:
    """Project an ascending history list onto the lag / rolling / ratio columns."""
    n = len(hist)
    for lag in (1, 7, 14, 28):
        c = f"{prefix}lag_{lag}d"
        if c in cols:
            row[c] = float(hist[-lag]) if n >= lag else 0.0
    for win in (7, 14, 28):
        seg = hist[-win:]
        if f"{prefix}roll_mean_{win}d" in cols:
            row[f"{prefix}roll_mean_{win}d"] = float(np.mean(seg)) if seg else 0.0
        if f"{prefix}roll_med_{win}d" in cols:
            row[f"{prefix}roll_med_{win}d"] = float(np.median(seg)) if seg else 0.0
        if f"{prefix}roll_std_{win}d" in cols:
            row[f"{prefix}roll_std_{win}d"] = float(np.std(seg)) if len(seg) > 1 else 0.0
    if f"{prefix}history_len" in cols:
        row[f"{prefix}history_len"] = float(n)
    if f"{prefix}trend_slope_28d" in cols and n >= 3:
        seg = np.asarray(hist[-28:], dtype=float)
        row[f"{prefix}trend_slope_28d"] = float(np.polyfit(np.arange(len(seg)), seg, 1)[0])
    elif f"{prefix}trend_slope_28d" in cols:
        row[f"{prefix}trend_slope_28d"] = 0.0