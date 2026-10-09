"""``assess_safety()``: hard rules first, then the fusion model, then the shelf-life range.

Status is ``GOOD | RISK | REJECTED``.  ``hours_left`` is a conservative lower bound plus
P10/P50/P90, capped by the label expiry.

This is the function the Go backend calls through ``mlserving`` (spec 07 section 15),
so two properties matter more than speed:

* **Thread safety.**  ``mlserving`` runs2 under a threadpool with a semaphore, and the
  previous module-level dict cache could have two threads load (and unpickle) the same
  400 MB LightGBM bundle concurrently -- a real thundering-herd problem the first time
  two requests arrive together after a deploy.  Loading is now guarded by a lock and
  the instance is published atomically.
* **D20 compliance.**  ``device_preview`` must never reach this function.  The previous
  signature accepted arbitrary ``**kwargs``, so nothing stopped a caller from passing
  the phone's advisory preview in as a feature.  The signature is now explicit and
  unknown keys raise.

Order of evaluation, and why:

1. Expiry and danger-minute budget are **hard rules**.  They come first and they
   override the model -- a batch past its label date is rejected regardless of what the
   classifier believes.
2. The calibrated fusion model supplies RISK/GOOD.
3. The shelf-life model supplies the deadline used by routing.

The output is decision support.  ``requires_human_approval`` is always ``True``.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ml_quality_datasets import FEATURES, prep
from ml_quality_time_temp import load_cfg, summarize
from ml_quality_train_fusion import predict_unsafe, status_from_p
from ml_quality_train_shelf_life import predict_range
from ml_utils import get_logger, get_paths

LOG = get_logger("quality.inference")

_cache: dict[str, Any] = {}
_cache_lock = threading.Lock()


class SafetyInputError(ValueError):
    """Raised with a stable ``code`` that mlserving maps to the Go error contract."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _load(kind: str) -> dict:
    """Lazily load a registry bundle exactly once, thread-safely."""
    if kind in _cache:
        return _cache[kind]
    with _cache_lock:  # double-checked: the second waiter finds it already published
        if kind in _cache:
            return _cache[kind]
        import joblib

        paths = get_paths()
        path = paths.latest_model_dir(kind) / "model.joblib"
        LOG.info("loading %s model from %s", kind, path)
        bundle = joblib.load(path)
        missing = [f for f in FEATURES if f not in bundle.get("features", FEATURES)]
        if missing:
            raise RuntimeError(f"{kind} bundle is missing features {missing}; it was trained "
                               f"with an older feature contract -- retrain it")
        _cache[kind] = bundle
        return bundle


def warmup() -> dict:
    """Load both bundles and run one dummy row.  Called by mlserving lifespan."""
    dummy = dict(category="rice_dal", hours_since_prepared=1.0, danger_zone_minutes=0.0,
                 hot_hold_minutes=60.0, chilled_minutes=0.0, current_temp_c=70.0,
                 mean_temp_c=65.0, humidity_mean=55.0, cv_unsafe_prob=0.1)
    out = {}
    for kind in ("fusion", "shelf_life"):
        _load(kind)
        out[kind] = "loaded"
    assess_safety(category="rice_dal", hours_since_prepared=1.0, temps=[70.0, 70.0, 69.0],
                  step_min=10.0, humidity_mean=55.0, cv_unsafe_prob=0.1)
    out["smoke"] = "ok"
    return out


def assess_safety(*, category: str, hours_since_prepared: float, temps: Sequence[float],
                  step_min: float, humidity_mean: float, cv_unsafe_prob: float,
                  hours_to_expiry: float | None = None,
                  temps_dt_min: Sequence[float] | None = None) -> dict:
    """Fuse the image risk with time-temperature exposure and return a safety decision.

    Parameters
    ----------
    category
        One of the keys in ``ml_configs_fusion.yaml`` -> ``categories``.
    temps
        Temperature trace in degrees C.  May contain NaN for dropped samples.
    temps_dt_min
        Per-sample durations.  Pass this whenever the trace is irregular -- see
        ``ml_quality_time_temp.summarize``; omitting it on a gappy trace under-counts
        danger-zone minutes and can release an unsafe batch.

    ``device_preview`` is deliberately **not** a parameter (D20).
    """
    cfg = load_cfg()
    if category not in cfg["categories"]:
        raise SafetyInputError("ML_INVALID_INPUT",
                               f"UNKNOWN_CATEGORY: {category} (expected one of "
                               f"{sorted(cfg['categories'])})")
    if not 0.0 <= float(cv_unsafe_prob) <= 1.0:
        raise SafetyInputError("ML_INVALID_INPUT",
                               f"cv_unsafe_prob must be in [0,1], got {cv_unsafe_prob}")
    if hours_to_expiry is not None and float(hours_to_expiry) > 24 * 30:
        raise SafetyInputError("ML_INVALID_INPUT",
                               f"hours_to_expiry={hours_to_expiry} is implausible; "
                               f"check the batch timestamps")

    feats = summarize(temps, float(step_min), cfg, dt_min=temps_dt_min)
    row = pd.DataFrame([{"category": category,
                         "hours_since_prepared": float(hours_since_prepared),
                         "humidity_mean": float(humidity_mean),
                         "cv_unsafe_prob": float(cv_unsafe_prob), **feats}])

    fusion = _load("fusion")
    shelf = _load("shelf_life")
    p = float(predict_unsafe(fusion, prep(row, fusion["categories"],
                                          medians=fusion.get("medians")))[0])
    status = str(status_from_p(np.array([p]), fusion["t_hold"], fusion["t_reject"])[0])

    # ---- hard rules (override the model) ---------------------------------- #
    reasons: list[str] = []
    dz = float(feats["danger_zone_minutes"])
    if hours_to_expiry is not None and hours_to_expiry <= 0:
        status, reasons = "REJECTED", ["EXPIRED"]
    elif dz > cfg["danger_minutes_reject"]:
        status, reasons = "REJECTED", [f"DANGER_MINUTES_{int(dz)}"]
    elif dz > cfg["danger_minutes_hold"] and status == "GOOD":
        status, reasons = "RISK", [f"DANGER_MINUTES_{int(dz)}"]
    elif dz > cfg["danger_minutes_hold"]:
        reasons.append(f"DANGER_MINUTES_{int(dz)}")
    if cv_unsafe_prob >= 0.5:
        reasons.append("CV_RISK")
    if hours_to_expiry is not None and 0 < hours_to_expiry < 2:
        reasons.append("EXPIRY_NEAR")
    if cfg["categories"][category]["risk"] >= 0.7 and status != "GOOD":
        reasons.append("CATEGORY_HIGH_RISK")
    if feats.get("unobserved_minutes", 0) > cfg["danger_minutes_hold"]:
        reasons.append("SENSOR_GAP_UNOBSERVED")
    if not reasons and status != "GOOD":
        reasons.append("FUSION_MODEL")

    # ---- remaining safe hours -------------------------------------------- #
    # ``predict_range`` returns [p_low, p50, p_high, lower_bound], already conformalised
    # and already clipped at zero.  The previous version recomputed the lower bound here
    # from a raw prediction and its own copy of ``qhat``; two copies of a conformal
    # constant is one more than the system should have.
    rng = predict_range(shelf, prep(row, shelf["categories"], medians=shelf.get("medians")))[0]
    cap = float("inf") if hours_to_expiry is None else max(float(hours_to_expiry), 0.0)
    if status == "REJECTED":
        hl = {"lower_bound": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0}
    else:
        hl = {"lower_bound": float(min(rng[3], cap)), "p10": float(min(rng[0], cap)),
              "p50": float(min(rng[1], cap)), "p90": float(min(rng[2], cap))}

    return {
        "status": status,
        "p_unsafe": p,
        "reasons": reasons,
        "danger_zone_minutes": dz,
        "unobserved_minutes": float(feats.get("unobserved_minutes", 0.0)),
        "hours_left": hl,
        "display": (f"about {hl['lower_bound']:.0f}-{hl['p50']:.0f} h left "
                    f"(decision support, not certification)"),
        "hours_to_expiry": None if hours_to_expiry is None else float(hours_to_expiry),
        "requires_human_approval": True,
        "model_version": f"fusion-{fusion.get('metrics_version', '?')}",
        "model_versions": {"fusion": f"fusion-{fusion.get('metrics_version', '?')}",
                           "shelf_life": f"shelf-life-{shelf.get('metrics_version', '?')}"},
        "caveat": "rule-derived safety model fused with a server-side image model; "
                  "decision support only, a human approves every redistribution",
    }


def clear_cache() -> None:
    """Drop loaded bundles (used by tests and by a mlserving reload endpoint)."""
    with _cache_lock:
        _cache.clear()