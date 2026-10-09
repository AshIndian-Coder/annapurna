#!/usr/bin/env python3
"""Sanity-check the synthetic data by training the project's model types on it.
If a model cannot learn from the data (or learns it trivially), the generator needs tuning.
Usage: python validate_annapurna_data.py --data data
"""
import argparse
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score

warnings.filterwarnings("ignore")


def demand(path):
    print("\n=== 1. DEMAND: LightGBM quantile + CQR (time-based split) ===")
    d = pd.read_csv(f"{path}/demand_daily.csv", parse_dates=["date"])
    d = d[(d.dq_flag.isin(["ok"]))].sort_values(["kitchen_id", "category", "date"]).reset_index(drop=True)
    g = d.groupby(["kitchen_id", "category"])["units_sold"]
    for L in (7, 14, 21, 28):
        d[f"lag_{L}"] = g.shift(L)
    d["roll_same_dow"] = d[["lag_7", "lag_14", "lag_21", "lag_28"]].mean(axis=1)
    d["roll_28"] = g.transform(lambda s: s.shift(1).rolling(28, min_periods=7).mean())
    d = d.dropna(subset=["roll_same_dow", "roll_28", "units_sold"])
    d = d[(d.is_closed == 0) & (d.roll_28 > 0)].copy()
    for c in ("kitchen_id", "category", "kitchen_type", "holiday_type", "city"):
        d[c] = d[c].astype("category")
    feats = ["day_of_week", "day_of_month", "month", "is_holiday", "is_holiday_eve", "days_to_next_holiday",
             "days_since_last_holiday", "ipl_match_day", "academic_break", "temp_c", "humidity_pct", "rain_mm",
             "promo_discount_pct", "special_menu_flag", "unit_price_inr", "lag_7", "lag_14", "lag_28", "roll_same_dow",
             "roll_28", "kitchen_id", "category", "kitchen_type", "holiday_type"]
    for c in ("lag_7", "lag_14", "lag_28", "roll_same_dow"):
        d[c + "_r"] = d[c] / d["roll_28"]
    feats += [c + "_r" for c in ("lag_7", "lag_14", "lag_28", "roll_same_dow")]
    d["y"] = d["units_sold"] / d["roll_28"]
    t_end = d.date.max()
    test = d[d.date > t_end - pd.Timedelta(days=90)]
    cal = d[(d.date > t_end - pd.Timedelta(days=150)) & (d.date <= t_end - pd.Timedelta(days=90))]
    tr = d[d.date <= t_end - pd.Timedelta(days=150)]
    qs = {}
    for a in (0.1, 0.5, 0.9):
        m = lgb.LGBMRegressor(objective="quantile", alpha=a, n_estimators=400, learning_rate=0.05, num_leaves=31,
                              min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
        m.fit(tr[feats], tr["y"])
        qs[a] = m
    pred = {a: qs[a].predict(test[feats]) for a in qs}
    pc = {a: qs[a].predict(cal[feats]) for a in qs}
    scale = test["roll_28"].values
    y = test["units_sold"].values
    p50 = pred[0.5] * scale
    mape = lambda p: np.mean(np.abs(p - y)) / np.mean(y)
    print(f"rows train/cal/test = {len(tr):,}/{len(cal):,}/{len(test):,}")
    print(f"WAPE  LightGBM p50: {mape(p50):.3f} | seasonal naive (lag_7): {mape(test['lag_7'].values):.3f} | "
          f"same-dow mean: {mape(test['roll_same_dow'].values):.3f} | kitchen planner (prepared_units): {mape(test['prepared_units'].values):.3f}")
    cov_raw = np.mean((y >= pred[0.1] * scale) & (y <= pred[0.9] * scale))
    E = np.maximum(pc[0.1] - cal["y"].values, cal["y"].values - pc[0.9])
    qhat = np.quantile(E, min(1, np.ceil((len(E) + 1) * 0.8) / len(E)))
    cov_cqr = np.mean((y >= (pred[0.1] - qhat) * scale) & (y <= (pred[0.9] + qhat) * scale))
    print(f"80% interval coverage on units_sold: raw quantiles {cov_raw:.3f} -> after CQR {cov_cqr:.3f} (target 0.80)")
    dem = test["latent_true_demand"].values
    cov_true = np.mean((dem >= (pred[0.1] - qhat) * scale) & (dem <= (pred[0.9] + qhat) * scale))
    print(f"coverage of true (uncensored) demand: {cov_true:.3f}  <- lower than nominal because units_sold is censored by stockouts")
    imp = pd.Series(qs[0.5].feature_importances_, feats).sort_values(ascending=False)
    print("top features (p50):", ", ".join(imp.index[:8]))


def safety_and_shelf(path):
    b = pd.read_csv(f"{path}/food_batches.csv")
    b = b[b.dq_flag == "ok"].copy()
    for c in ("kitchen_type", "category", "dish", "storage_type", "packaging", "cooling_method", "source_stage", "meal_period", "photo_quality"):
        b[c] = b[c].astype("category")
    base = ["hours_since_prep", "minutes_in_danger_zone", "storage_temp_c_max", "storage_temp_c_mean", "cooling_time_hr",
            "reheat_count", "handler_hygiene_score", "kitchen_audit_score", "packaging_integrity", "cv_spoilage_prob",
            "voc_index", "odor_score", "cooked_core_temp_c", "quantity_kg", "ambient_temp_c", "humidity_pct", "water_activity",
            "ph", "manual_probe_temp_c", "freeze_thaw_cycles", "hour_of_day", "month"]
    cats = ["storage_type", "packaging", "cooling_method", "source_stage", "category", "dish", "kitchen_type", "photo_quality"]
    feats = base + cats
    mono = {"hours_since_prep": 1, "minutes_in_danger_zone": 1, "storage_temp_c_max": 1, "cooling_time_hr": 1, "reheat_count": 1,
            "handler_hygiene_score": -1, "kitchen_audit_score": -1, "packaging_integrity": -1, "cv_spoilage_prob": 1,
            "voc_index": 1, "odor_score": 1}
    print("\n=== 2. SAFETY FUSION: LightGBM (monotone) + isotonic calibration ===")
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(b))
    n = len(b)
    tr, ca, te = b.iloc[idx[: int(.6 * n)]], b.iloc[idx[int(.6 * n): int(.8 * n)]], b.iloc[idx[int(.8 * n):]]
    print(f"unsafe rate {b.unsafe_for_redistribution.mean():.3f}; rows {len(b):,}")
    cons = [mono.get(f, 0) for f in feats]
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.04, num_leaves=31, min_child_samples=40, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.8, monotone_constraints=cons, verbose=-1)
    m.fit(tr[feats], tr["unsafe_for_redistribution"])
    p_raw = m.predict_proba(te[feats])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(m.predict_proba(ca[feats])[:, 1], ca["unsafe_for_redistribution"])
    p_cal = iso.predict(p_raw)
    y = te["unsafe_for_redistribution"].values
    print(f"AUC fusion {roc_auc_score(y, p_raw):.3f} | Brier raw {brier_score_loss(y, p_raw):.4f} -> calibrated {brier_score_loss(y, p_cal):.4f}")
    print(f"AUC CV-score alone {roc_auc_score(y[te.cv_spoilage_prob.notna()], te.cv_spoilage_prob.dropna()):.3f} (pathogens are mostly invisible, so fusion should beat it)")
    lat = te["latent_p_unsafe"].values
    print(f"mean |pred - true latent probability|: raw {np.mean(np.abs(p_raw - lat)):.4f}, calibrated {np.mean(np.abs(p_cal - lat)):.4f}")
    print("\n=== 3. SHELF LIFE: LightGBM L2 (log1p target) + split conformal ===")
    s = b[b.source_stage != "plate_waste"]
    idx = rng.permutation(len(s)); n = len(s)
    tr, ca, te = s.iloc[idx[: int(.6 * n)]], s.iloc[idx[int(.6 * n): int(.8 * n)]], s.iloc[idx[int(.8 * n):]]
    sfe = [f for f in feats if f not in ("cv_spoilage_prob", "voc_index", "odor_score")]
    r = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.04, num_leaves=31, min_child_samples=30, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    r.fit(tr[sfe], np.log1p(tr["remaining_shelf_life_hr"]))
    pt = r.predict(te[sfe]); yt = np.log1p(te["remaining_shelf_life_hr"].values)
    q = np.quantile(np.abs(np.log1p(ca["remaining_shelf_life_hr"].values) - r.predict(ca[sfe])), np.ceil((len(ca) + 1) * .9) / len(ca))
    cov = np.mean(np.abs(yt - pt) <= q)
    ss = 1 - np.sum((yt - pt) ** 2) / np.sum((yt - yt.mean()) ** 2)
    print(f"R2 (log1p hours) {ss:.3f} | MAE {np.mean(np.abs(np.expm1(pt) - np.expm1(yt))):.1f} h | median abs err {np.median(np.abs(np.expm1(pt) - np.expm1(yt))):.1f} h")
    print(f"90% split-conformal coverage {cov:.3f} (target 0.90) | rows with 0h remaining: {(s.remaining_shelf_life_hr == 0).mean():.3f}")


def anomalies(path):
    print("\n=== 4. ANOMALY DETECTION: EWMA / CUSUM / IsolationForest ===")
    s = pd.read_csv(f"{path}/sensor_readings.csv", parse_dates=["timestamp"])
    res = []
    for sid, g in s.groupby("sensor_id"):
        g = g.sort_values("timestamp").copy()
        t = g["temp_c"].interpolate(limit_direction="both")
        med = t.rolling(32, center=False, min_periods=8).median().bfill()
        resid = (t - med)
        sig = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        z = (resid / sig).values
        ew = np.zeros(len(z)); lam = 0.2
        for i in range(1, len(z)):
            ew[i] = lam * z[i] + (1 - lam) * ew[i - 1]
        ewma_flag = np.abs(ew) > 3 * math_sqrt(lam / (2 - lam))
        cp = cn = 0.0; cus = np.zeros(len(z), bool)
        for i, zi in enumerate(z):
            cp = max(0, cp + zi - 0.5); cn = max(0, cn - zi - 0.5)
            cus[i] = (cp > 5) or (cn > 5)
            if cus[i]:
                cp = cn = 0.0
        X = pd.DataFrame(dict(resid=resid, level=t - t.mean(), d1=t.diff().fillna(0), cur=g.compressor_current_a, door=g.door_open_frac,
                              defrost=g.defrost_active, power=g.power_ok, flat=(t.diff().abs() < 1e-9).rolling(6).sum().fillna(0)))
        iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=0).fit(X)
        if_flag = iso.predict(X) == -1
        g["ewma"], g["cusum"], g["iforest"] = ewma_flag, cus, if_flag
        res.append(g)
    r = pd.concat(res)
    y = r.is_anomaly.values
    for name in ("ewma", "cusum", "iforest"):
        f = r[name].values
        print(f"{name:8s} precision {precision_score(y, f):.2f} recall {recall_score(y, f):.2f} F1 {f1_score(y, f):.2f}")
    u = (r.ewma | r.cusum | r.iforest).values
    print(f"union    precision {precision_score(y, u):.2f} recall {recall_score(y, u):.2f} F1 {f1_score(y, u):.2f}")
    print("recall of the union by anomaly type:", r[r.is_anomaly == 1].assign(u=u[y == 1]).groupby("anomaly_type").u.mean().round(2).to_dict())
    print(f"false-alarm rate during benign defrost cycles: {r[(r.defrost_active == 1) & (r.is_anomaly == 0)][['ewma', 'cusum', 'iforest']].mean().round(2).to_dict()}")


def math_sqrt(x):
    return float(np.sqrt(x))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    a = ap.parse_args()
    demand(a.data)
    safety_and_shelf(a.data)
    anomalies(a.data)
