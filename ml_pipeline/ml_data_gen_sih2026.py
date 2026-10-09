#!/usr/bin/env python3
"""
Annapurna (SIH 2026, PS 26234) - synthetic data generator.

Produces 5 CSVs (+ data_dictionary.md) that feed the predictive-analysis models:

  kitchens.csv         12 institutional / commercial kitchens (master table)
  demand_daily.csv     ~35k rows  -> LightGBM quantile + CQR (demand forecasting)
  food_batches.csv     36k rows   -> safety fusion (monotone LGBM + isotonic) AND
                                     shelf-life (LGBM L2 + split conformal)
  sensor_readings.csv  36k rows   -> CUSUM / EWMA / IsolationForest (anomaly detection)
  routing_nodes.csv    ~1.7k rows -> OR-Tools VRPTW vs nearest-neighbour benchmark

Everything is generated from explicit, documented mechanisms (queueing-style demand
drivers, Ratkowsky-type bacterial growth, Q10-style spoilage, HACCP rules).
Columns starting with `latent_` are ground truth for EVALUATION ONLY - never features.

Usage:  python generate_annapurna_data.py --out data --seed 2026
"""
import argparse
import math
import os

import numpy as np
import pandas as pd
from scipy.stats import norm

START = pd.Timestamp("2025-02-01")
N_DAYS = 600
DATES = pd.date_range(START, periods=N_DAYS, freq="D")
CATS = ["cereals_breads", "dal_curries", "non_veg_dishes", "snacks_bakery", "desserts_dairy"]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# --------------------------------------------------------------------------- kitchens
# id, city, type, capacity (meals/day), lat, lon, overproduction bias, hygiene base, temp logger, VOC sensor
KITCHENS = [
    ("K01", "Mumbai", "corporate_canteen", 2500, 19.1136, 72.8697, 0.10, 86, 1, 1),
    ("K02", "Mumbai", "hotel_banquet", 1800, 18.9330, 72.8350, 0.18, 90, 1, 1),
    ("K03", "Pune", "hostel_mess", 1500, 18.5074, 73.8077, 0.14, 74, 0, 0),
    ("K04", "Pune", "cloud_kitchen", 900, 18.5912, 73.7389, 0.09, 80, 1, 0),
    ("K05", "Delhi", "hospital_kitchen", 1200, 28.5672, 77.2100, 0.07, 92, 1, 1),
    ("K06", "Delhi", "central_kitchen", 6000, 28.5355, 77.2750, 0.06, 93, 1, 1),
    ("K07", "Bengaluru", "corporate_canteen", 3000, 12.9698, 77.7500, 0.11, 88, 1, 0),
    ("K08", "Bengaluru", "restaurant", 700, 12.9784, 77.6408, 0.16, 78, 0, 0),
    ("K09", "Hyderabad", "school_midday", 2500, 17.4399, 78.4983, 0.12, 70, 0, 0),
    ("K10", "Chennai", "hostel_mess", 1800, 13.0067, 80.2206, 0.15, 72, 0, 0),
    ("K11", "Kolkata", "hotel_banquet", 1500, 22.5535, 88.3520, 0.20, 82, 1, 0),
    ("K12", "Ahmedabad", "central_kitchen", 5000, 23.0963, 72.6550, 0.07, 89, 1, 1),
]
KCOLS = ["kitchen_id", "city", "kitchen_type", "capacity_meals_day", "lat", "lon",
         "overproduction_bias", "hygiene_baseline", "has_temp_logger", "has_voc_sensor"]

SHARE = {  # portions per day per category as a fraction of capacity
    "corporate_canteen": [1.0, 0.9, 0.35, 0.45, 0.30],
    "hostel_mess": [1.2, 1.0, 0.30, 0.25, 0.20],
    "hospital_kitchen": [1.0, 0.95, 0.25, 0.15, 0.30],
    "hotel_banquet": [0.9, 0.7, 0.8, 0.35, 0.6],
    "cloud_kitchen": [0.8, 0.5, 0.9, 0.45, 0.35],
    "restaurant": [0.8, 0.6, 0.85, 0.4, 0.4],
    "central_kitchen": [1.1, 1.0, 0.4, 0.3, 0.25],
    "school_midday": [1.1, 1.0, 0.0, 0.12, 0.0],
}
NONVEG_CITY = {"Ahmedabad": 0.15, "Hyderabad": 1.3, "Kolkata": 1.4, "Chennai": 1.1}
DOW = {  # Mon..Sun
    "corporate_canteen": [0.88, 1.0, 1.04, 1.04, 0.82, 0.06, 0.02],
    "hostel_mess": [1, 1, 1, 1, 1, 1.0, 1.08],
    "hospital_kitchen": [1, 1, 1, 1, 1, 0.97, 0.95],
    "hotel_banquet": [0.7, 0.7, 0.75, 0.85, 1.15, 1.5, 1.3],
    "cloud_kitchen": [0.85, 0.85, 0.9, 0.95, 1.2, 1.3, 1.25],
    "restaurant": [0.75, 0.75, 0.8, 0.9, 1.2, 1.4, 1.3],
    "central_kitchen": [1.05, 1.1, 1.1, 1.1, 1.1, 0.8, 0.05],
    "school_midday": [1, 1, 1, 1, 1, 0.5, 0.0],
}
CAT_DOW = {
    "non_veg_dishes": [0.95, 0.9, 0.95, 0.9, 1.05, 1.15, 1.2],
    "desserts_dairy": [0.95, 0.95, 1, 1, 1.05, 1.1, 1.15],
}
HOL = {  # (national, major festival, eve)  demand multipliers
    "corporate_canteen": (0.04, 0.05, 0.85), "school_midday": (0.0, 0.0, 0.92),
    "hostel_mess": (0.8, 0.55, 0.85), "hospital_kitchen": (0.98, 1.0, 1.0),
    "hotel_banquet": (1.25, 1.4, 1.15), "cloud_kitchen": (1.2, 1.3, 1.1),
    "restaurant": (1.2, 1.35, 1.1), "central_kitchen": (0.5, 0.4, 0.9),
}
DINING = {"hotel_banquet", "cloud_kitchen", "restaurant"}
GROWTH = {"cloud_kitchen": 0.25, "restaurant": 0.08, "hotel_banquet": 0.06}  # per-year trend
CAT_WASTE = [1.0, 1.1, 0.9, 1.2, 0.8]
PORTION_G = [250, 200, 180, 120, 100]
BASE_PRICE = [60, 90, 220, 70, 80]
PRICE_MULT = {"corporate_canteen": 0.8, "hostel_mess": 0.6, "hospital_kitchen": 0.5, "hotel_banquet": 3.0,
              "cloud_kitchen": 1.2, "restaurant": 2.0, "central_kitchen": 0.4, "school_midday": 0.3}
ELAST = [0.006, 0.007, 0.011, 0.013, 0.012]

# monthly climatology (approximate): mean temp C, RH %, rain-day probability
CLIM = {
    "Mumbai": ([24, 25, 28, 30, 31, 29, 27, 27, 27, 29, 28, 26], [62, 62, 66, 68, 70, 82, 87, 86, 82, 72, 64, 60],
               [.01, .01, .01, .02, .05, .65, .90, .85, .70, .20, .04, .01]),
    "Pune": ([21, 23, 27, 30, 30, 27, 25, 25, 25, 25, 23, 21], [48, 40, 35, 38, 48, 72, 82, 82, 78, 65, 55, 50],
             [.01, .01, .02, .04, .12, .45, .65, .60, .55, .25, .08, .02]),
    "Delhi": ([13, 17, 23, 29, 33, 34, 31, 30, 29, 25, 19, 14], [72, 62, 50, 38, 38, 52, 75, 78, 68, 56, 58, 70],
              [.08, .10, .08, .06, .08, .18, .45, .50, .30, .05, .02, .05]),
    "Bengaluru": ([22, 24, 27, 28, 27, 25, 24, 24, 24, 24, 22, 21], [65, 55, 50, 55, 65, 72, 78, 78, 78, 76, 72, 68],
                  [.03, .03, .05, .20, .35, .40, .45, .50, .55, .50, .25, .08]),
    "Hyderabad": ([23, 26, 30, 33, 34, 30, 27, 27, 27, 27, 24, 22], [55, 45, 40, 40, 42, 62, 75, 76, 72, 62, 55, 55],
                  [.02, .03, .04, .07, .10, .30, .45, .50, .45, .25, .07, .02]),
    "Chennai": ([25, 26, 28, 31, 33, 33, 31, 30, 30, 28, 26, 25], [74, 72, 72, 70, 66, 62, 64, 66, 70, 78, 80, 78],
                [.10, .04, .03, .05, .08, .12, .20, .25, .25, .50, .55, .35]),
    "Kolkata": ([20, 24, 28, 31, 31, 30, 29, 29, 29, 28, 24, 20], [75, 68, 62, 70, 75, 82, 86, 87, 86, 80, 72, 74],
                [.04, .08, .10, .20, .30, .65, .80, .80, .70, .30, .08, .03]),
    "Ahmedabad": ([20, 23, 28, 33, 35, 34, 30, 29, 30, 30, 25, 21], [50, 42, 35, 32, 40, 58, 75, 80, 68, 48, 45, 50],
                  [.01, .01, .01, .01, .03, .20, .50, .45, .30, .05, .02, .01]),
}

# (start, end, name, kind, nonveg_mult, dessert_mult, dining_mult, cities or None)   dates approximate (+-1 day)
FEST = [
    ("2025-02-14", "2025-02-14", "valentines_day", "minor", 1.0, 1.10, 1.30, None),
    ("2025-02-26", "2025-02-26", "maha_shivratri", "major", 0.55, 1.0, 1.0, None),
    ("2025-03-14", "2025-03-14", "holi", "major", 1.15, 1.50, 1.10, None),
    ("2025-03-30", "2025-03-30", "gudi_padwa_ugadi", "major", 0.9, 1.30, 1.10, ["Mumbai", "Pune", "Hyderabad", "Bengaluru"]),
    ("2025-03-31", "2025-03-31", "eid_ul_fitr", "major", 1.50, 1.40, 1.20, None),
    ("2025-04-06", "2025-04-06", "ram_navami", "minor", 0.60, 1.0, 1.0, None),
    ("2025-04-14", "2025-04-14", "ambedkar_jayanti", "national", 1.0, 1.0, 1.0, None),
    ("2025-04-18", "2025-04-18", "good_friday", "national", 0.80, 1.0, 1.0, None),
    ("2025-06-07", "2025-06-07", "eid_ul_adha", "major", 1.60, 1.20, 1.15, None),
    ("2025-08-09", "2025-08-09", "raksha_bandhan", "minor", 1.0, 1.40, 1.10, None),
    ("2025-08-15", "2025-08-15", "independence_day", "national", 1.0, 1.0, 1.0, None),
    ("2025-08-16", "2025-08-16", "janmashtami", "minor", 0.60, 1.30, 1.0, None),
    ("2025-08-27", "2025-08-27", "ganesh_chaturthi", "major", 0.55, 1.50, 1.15, ["Mumbai", "Pune"]),
    ("2025-08-28", "2025-09-05", "ganesh_festival", "minor", 0.70, 1.30, 1.10, ["Mumbai", "Pune"]),
    ("2025-09-06", "2025-09-06", "anant_chaturdashi", "major", 0.7, 1.0, 1.0, ["Mumbai", "Pune"]),
    ("2025-09-22", "2025-10-01", "navratri", "minor", 0.60, 1.20, 1.10, ["Mumbai", "Pune", "Ahmedabad", "Delhi"]),
    ("2025-09-28", "2025-10-02", "durga_puja", "minor", 0.8, 1.30, 1.40, ["Kolkata"]),
    ("2025-10-02", "2025-10-02", "gandhi_jayanti_dussehra", "national", 1.0, 1.2, 1.1, None),
    ("2025-10-18", "2025-10-19", "diwali_run_up", "minor", 1.0, 1.50, 1.15, None),
    ("2025-10-20", "2025-10-20", "diwali", "major", 0.8, 1.60, 1.20, None),
    ("2025-10-21", "2025-10-23", "diwali_after", "minor", 0.9, 1.40, 1.10, None),
    ("2025-10-22", "2025-10-22", "govardhan_puja", "major", 0.9, 1.2, 1.0, ["Mumbai", "Pune", "Ahmedabad"]),
    ("2025-11-05", "2025-11-05", "guru_nanak_jayanti", "national", 0.7, 1.1, 1.0, None),
    ("2025-12-25", "2025-12-25", "christmas", "major", 1.30, 1.40, 1.15, None),
    ("2025-12-31", "2025-12-31", "new_years_eve", "minor", 1.2, 1.2, 1.50, None),
    ("2026-01-01", "2026-01-01", "new_year", "minor", 1.0, 1.1, 1.10, None),
    ("2026-01-14", "2026-01-15", "sankranti_pongal", "minor", 0.8, 1.30, 1.0, ["Chennai", "Hyderabad", "Bengaluru", "Ahmedabad"]),
    ("2026-01-26", "2026-01-26", "republic_day", "national", 1.0, 1.0, 1.0, None),
    ("2026-02-14", "2026-02-14", "valentines_day", "minor", 1.0, 1.10, 1.30, None),
    ("2026-02-15", "2026-02-15", "maha_shivratri", "major", 0.55, 1.0, 1.0, None),
    ("2026-03-04", "2026-03-04", "holi", "major", 1.15, 1.50, 1.10, None),
    ("2026-03-19", "2026-03-19", "gudi_padwa_ugadi", "major", 0.9, 1.30, 1.10, ["Mumbai", "Pune", "Hyderabad", "Bengaluru"]),
    ("2026-03-21", "2026-03-21", "eid_ul_fitr", "major", 1.50, 1.40, 1.20, None),
    ("2026-03-26", "2026-03-26", "ram_navami", "minor", 0.60, 1.0, 1.0, None),
    ("2026-04-03", "2026-04-03", "good_friday", "national", 0.80, 1.0, 1.0, None),
    ("2026-04-14", "2026-04-14", "ambedkar_jayanti", "national", 1.0, 1.0, 1.0, None),
    ("2026-05-27", "2026-05-27", "eid_ul_adha", "major", 1.60, 1.20, 1.15, None),
    ("2026-08-15", "2026-08-15", "independence_day", "national", 1.0, 1.0, 1.0, None),
    ("2026-08-28", "2026-08-28", "raksha_bandhan", "minor", 1.0, 1.40, 1.10, None),
    ("2026-09-04", "2026-09-04", "janmashtami", "minor", 0.60, 1.30, 1.0, None),
    ("2026-09-14", "2026-09-14", "ganesh_chaturthi", "major", 0.55, 1.50, 1.15, ["Mumbai", "Pune"]),
    ("2026-09-15", "2026-09-22", "ganesh_festival", "minor", 0.70, 1.30, 1.10, ["Mumbai", "Pune"]),
]


def build_calendar():
    """Per-city calendar arrays aligned to DATES."""
    cal = {}
    for city in CLIM:
        n = N_DAYS
        htype = np.array(["none"] * n, dtype=object)
        names = [[] for _ in range(n)]
        nv = np.ones(n)
        ds = np.ones(n)
        dine = np.ones(n)
        rank = {"none": 0, "minor": 1, "national": 2, "major": 3}
        for s, e, name, kind, a, b, c, cities in FEST:
            if cities is not None and city not in cities:
                continue
            for dt in pd.date_range(s, e):
                if dt < START or dt > DATES[-1]:
                    continue
                i = (dt - START).days
                names[i].append(name)
                nv[i] *= a
                ds[i] *= b
                dine[i] *= c
                if rank[kind] > rank[htype[i]]:
                    htype[i] = kind
        is_hol = np.isin(htype, ["national", "major"])
        eve = np.zeros(n, bool)
        eve[:-1] = is_hol[1:]
        to_next = np.full(n, 30)
        since = np.full(n, 30)
        last = -999
        for i in range(n):
            if is_hol[i]:
                last = i
            since[i] = min(30, i - last) if last > -999 else 30
        nxt = 10 ** 6
        for i in range(n - 1, -1, -1):
            if is_hol[i]:
                nxt = i
            to_next[i] = min(30, nxt - i) if nxt < 10 ** 6 else 30
        cal[city] = dict(htype=htype, name=["+".join(x) if x else "" for x in names], nv=nv, ds=ds, dine=dine,
                         is_hol=is_hol, eve=eve, to_next=to_next, since=since)
    return cal


def build_flags(rng):
    def span(ranges, val, base):
        a = np.full(N_DAYS, base, float)
        for s, e, v in ranges:
            m = (DATES >= s) & (DATES <= e)
            a[m] = v
        return a
    school_closed = span([("2025-04-25", "2025-06-15", 1), ("2025-10-18", "2025-10-28", 1),
                          ("2025-12-24", "2026-01-02", 1), ("2026-05-01", "2026-06-14", 1)], 1, 0) > 0
    hostel_occ = span([("2025-05-10", "2025-07-05", 0.35), ("2025-10-17", "2025-10-26", 0.40),
                       ("2025-12-20", "2026-01-03", 0.60), ("2026-05-05", "2026-07-05", 0.35)], 0, 1.0)
    ipl = np.zeros(N_DAYS, bool)
    for s, e in [("2025-03-22", "2025-06-03"), ("2026-03-28", "2026-05-31")]:
        m = (DATES >= s) & (DATES <= e)
        ipl[m] = rng.random(m.sum()) < 0.72
    return school_closed, hostel_occ, ipl


# --------------------------------------------------------------------------- weather
def build_weather(rng):
    mid = np.array([15.5, 45, 74.5, 105, 135.5, 166, 196.5, 227.5, 258, 288.5, 319, 349.5])
    x = np.concatenate([mid - 365, mid, mid + 365])
    doy = DATES.dayofyear.values
    out = {}
    for city, (T, H, P) in CLIM.items():
        tm = np.interp(doy, x, np.tile(T, 3))
        hm = np.interp(doy, x, np.tile(H, 3))
        pm = np.interp(doy, x, np.tile(P, 3))
        z = np.zeros(N_DAYS)
        z[0] = rng.normal()
        for i in range(1, N_DAYS):
            z[i] = 0.6 * z[i - 1] + math.sqrt(1 - 0.36) * rng.normal()
        rain = z < norm.ppf(np.clip(pm, 1e-4, 0.999))
        mm = np.where(rain, rng.lognormal(np.log(5 + 14 * pm), 0.9, N_DAYS), 0.0)
        a = np.zeros(N_DAYS)
        a[0] = rng.normal(0, 1.8)
        for i in range(1, N_DAYS):
            a[i] = 0.7 * a[i - 1] + rng.normal(0, 1.8 * math.sqrt(1 - 0.49))
        temp = tm + a - 1.8 * rain
        hum = np.clip(hm - 1.5 * a + 8 * rain + rng.normal(0, 4, N_DAYS), 15, 100)
        out[city] = dict(temp=temp, hum=hum, rain=mm)
    return out


# --------------------------------------------------------------------------- demand
def gen_demand(rng, weather, cal, school_closed, hostel_occ, ipl):
    rows = []
    t_idx = np.arange(N_DAYS)
    dow = DATES.dayofweek.values
    dom = DATES.day.values
    month = DATES.month.values
    for kid, city, ktype, cap, lat, lon, bias, hyg, logger, voc in KITCHENS:
        w, c = weather[city], cal[city]
        outage = rng.random(N_DAYS) < 0.008
        unplanned_closed = rng.random(N_DAYS) < 0.005
        # kitchen level slowly varying drivers
        phi = rng.uniform(0, 6.28)
        if ktype == "corporate_canteen":
            occ = 0.93 + 0.05 * np.sin(2 * np.pi * t_idx / 200 + phi) + rng.normal(0, 0.01, N_DAYS)
        elif ktype == "hospital_kitchen":
            occ = 0.95 + 0.04 * np.sin(2 * np.pi * t_idx / 365 + phi) + 0.06 * np.isin(month, [7, 8, 9])
        elif ktype == "hostel_mess":
            occ = hostel_occ
        else:
            occ = np.ones(N_DAYS)
        # campaigns / promotions (dining types only)
        promo = np.zeros(N_DAYS)
        if ktype in DINING:
            wk = t_idx // 7
            camp = rng.random(wk.max() + 1) < 0.10
            disc = rng.choice([10, 15, 20, 30, 40, 50], wk.max() + 1, p=[.2, .2, .25, .2, .1, .05])
            promo = np.where(camp[wk], disc[wk], 0.0)
            flash = rng.random(N_DAYS) < 0.03
            promo = np.maximum(promo, np.where(flash, rng.choice([15, 25, 35], N_DAYS), 0))
        special = (rng.random(N_DAYS) < 0.05).astype(int) if ktype not in DINING else np.zeros(N_DAYS, int)
        g = GROWTH.get(ktype, 0.02)
        trend = (1 + g) ** (t_idx / 365.0)
        hol_n, hol_m, hol_e = HOL[ktype]
        hf = np.ones(N_DAYS)
        hf = np.where(c["htype"] == "national", hol_n, hf)
        hf = np.where(c["htype"] == "major", hol_m, hf)
        hf = np.where(c["eve"] & ~c["is_hol"], hol_e, hf)
        payday = np.where(dom <= 5, 1.08, np.where(dom >= 28, 1.04, 1.0)) if ktype in DINING else np.ones(N_DAYS)
        wedding = np.where(np.isin(month, [11, 12, 1, 2, 4, 5]), 1.15, 1.0) if ktype == "hotel_banquet" else np.ones(N_DAYS)
        ipl_m = np.ones(N_DAYS)
        if ktype == "cloud_kitchen":
            ipl_m = np.where(ipl, 1.22, 1.0)
        elif ktype == "restaurant":
            ipl_m = np.where(ipl, 0.90, 1.0)
        elif ktype == "corporate_canteen":
            ipl_m = np.where(ipl, 0.97, 1.0)
        rain = w["rain"]
        rain_f = np.ones(N_DAYS)
        if ktype == "cloud_kitchen":
            rain_f = 1 + 0.25 * np.minimum(rain / 40, 1) * (rain > 2)
        elif ktype == "restaurant":
            rain_f = 1 - 0.12 * np.minimum(rain / 40, 1) * (rain > 2)
        elif ktype == "corporate_canteen":
            rain_f = 1 - 0.07 * np.minimum(rain / 40, 1) * (rain > 2)
        for ci, cat in enumerate(CATS):
            share = SHARE[ktype][ci] * (NONVEG_CITY.get(city, 1.0) if ci == 2 else 1.0)
            if share <= 0:
                continue
            f = np.array(DOW[ktype])[dow] * (np.array(CAT_DOW[cat])[dow] if cat in CAT_DOW else 1.0)
            temp = w["temp"]
            wx = np.ones(N_DAYS)
            if ci == 4:
                wx = 1 + 0.015 * np.maximum(temp - 28, 0)
            elif ci == 1:
                wx = 1 - 0.008 * np.maximum(temp - 30, 0)
            elif ci == 3:
                wx = 1 + 0.012 * np.maximum(18 - temp, 0)
            fest = np.ones(N_DAYS)
            if ci == 2:
                fest = c["nv"]
            if ci == 4:
                fest = c["ds"]
            if ktype in DINING:
                fest = fest * c["dine"]
            sp_eff = 1 + 0.07 * special
            price = 1 + ELAST[ci] * promo
            mean = (cap * share * f * hf * occ * trend * wx * fest * rain_f * payday * wedding * ipl_m * price * sp_eff)
            planned_closed = (f < 0.1) | ((hf < 0.1) & (c["htype"] != "none"))
            if ktype == "school_midday":
                planned_closed |= school_closed
            r = 40 if ktype in DINING else 150
            lam = mean * rng.gamma(r, 1.0 / r, N_DAYS)
            dem = rng.poisson(np.maximum(lam, 0)).astype(float)
            if ktype in ("hotel_banquet", "cloud_kitchen", "central_kitchen", "corporate_canteen"):
                bulk = rng.random(N_DAYS) < 0.012
                dem = np.where(bulk, dem * rng.uniform(1.5, 2.5, N_DAYS), dem)
            dip = rng.random(N_DAYS) < 0.01
            dem = np.where(dip, dem * rng.uniform(0.4, 0.7, N_DAYS), dem)
            closed = planned_closed | unplanned_closed
            dem = np.where(closed, 0, np.round(dem))
            # planner (what a kitchen manager would prepare): trailing same-weekday mean + bias + partial holiday awareness
            prep = np.zeros(N_DAYS)
            sold = np.zeros(N_DAYS)
            hist = {d: [] for d in range(7)}
            b_eff = bias * CAT_WASTE[ci]
            for i in range(N_DAYS):
                if closed[i]:
                    continue
                h = hist[dow[i]][-4:]
                fc = (0.65 * np.mean(h) + 0.35 * np.max(h)) if len(h) >= 2 else cap * share * DOW[ktype][dow[i]] * 0.92
                if (c["is_hol"][i] or c["eve"][i]) and rng.random() < 0.55:
                    fc *= hf[i] * rng.uniform(0.85, 1.1)
                prep[i] = max(0, round(fc * (1 + b_eff + 0.05 + rng.normal(0, 0.04))))
                s = min(dem[i], prep[i])
                if outage[i]:
                    s = round(s * rng.uniform(0.45, 0.85))
                sold[i] = s
                hist[dow[i]].append(s)
            leftover = prep - sold
            base_price = np.round(BASE_PRICE[ci] * PRICE_MULT[ktype] * (1 + 0.06 * t_idx / 365.0))
            df = pd.DataFrame(dict(
                date=DATES, kitchen_id=kid, city=city, kitchen_type=ktype, category=cat, capacity_meals_day=cap,
                day_of_week=dow, day_of_month=dom, month=month, week_of_year=DATES.isocalendar().week.values.astype(int),
                is_weekend=(dow >= 5).astype(int), is_holiday=c["is_hol"].astype(int), holiday_type=c["htype"],
                festival_name=c["name"], is_holiday_eve=(c["eve"] & ~c["is_hol"]).astype(int),
                days_to_next_holiday=c["to_next"], days_since_last_holiday=c["since"],
                ipl_match_day=ipl.astype(int), academic_break=(school_closed.astype(int) if ktype == "school_midday"
                                                                else (occ < 0.7).astype(int) if ktype == "hostel_mess" else 0),
                temp_c=np.round(temp, 1), humidity_pct=np.round(w["hum"], 0), rain_mm=np.round(rain, 1),
                promo_discount_pct=promo.astype(int), special_menu_flag=special,
                base_price_inr=base_price, unit_price_inr=np.round(base_price * (1 - promo / 100.0)),
                # ---- outcomes (not available ahead of time) ----
                prepared_units=prep.astype(int), units_sold=sold.astype(float), leftover_units=leftover.astype(int),
                leftover_kg=np.round(leftover * PORTION_G[ci] / 1000.0, 1),
                stockout_flag=((dem > prep) & ~closed).astype(int), is_closed=closed.astype(int),
                outage_flag=outage.astype(int), latent_true_demand=dem.astype(int),
            ))
            rows.append(df)
    df = pd.concat(rows, ignore_index=True)
    # weather sensor gaps
    miss = rng.random(len(df)) < 0.01
    df.loc[miss, ["temp_c", "humidity_pct", "rain_mm"]] = np.nan
    # data quality problems (flagged so the cleaning pipeline can be demonstrated and tested)
    df["dq_flag"] = "ok"
    n = len(df)
    open_rows = np.where(df["is_closed"].values == 0)[0]
    pick = lambda p: rng.choice(open_rows, int(n * p), replace=False)
    i1 = pick(0.005)
    df.loc[i1, ["units_sold", "leftover_units", "leftover_kg"]] = np.nan
    df.loc[i1, "dq_flag"] = "missing_sales"
    i2 = pick(0.002)
    df.loc[i2, "units_sold"] = df.loc[i2, "units_sold"] * 10
    df.loc[i2, "dq_flag"] = "unit_error_x10"
    i3 = pick(0.003)
    prev = df["units_sold"].shift(1)
    ok = df.loc[i3, "kitchen_id"].values == df["kitchen_id"].shift(1).loc[i3].values
    i3 = i3[ok & (df.loc[i3, "category"].values == df["category"].shift(1).loc[i3].values)]
    df.loc[i3, "units_sold"] = prev.loc[i3]
    df.loc[i3, "dq_flag"] = "stale_copy"
    dup = df.sample(frac=0.003, random_state=1).copy()
    dup["dq_flag"] = "duplicate"
    df = pd.concat([df, dup], ignore_index=True).sort_values(["kitchen_id", "category", "date"]).reset_index(drop=True)
    df.insert(0, "row_id", np.arange(len(df)))
    return df


# --------------------------------------------------------------------------- batches (safety + shelf life)
# category, dish, aw, pH, spoil_mult, protein, dairy, rice, core_target_C, cooked, hot_limit_hr, frozen_limit_days
DISHES = [
    ("cereals_breads", "steamed_rice", .97, 6.6, 1.3, 0, 0, 1, 63, 1, 5, 60),
    ("cereals_breads", "jeera_rice", .97, 6.4, 1.3, 0, 0, 1, 63, 1, 5, 60),
    ("cereals_breads", "veg_pulao", .97, 6.2, 1.3, 0, 0, 1, 63, 1, 5, 60),
    ("cereals_breads", "chapati", .90, 6.0, 0.5, 0, 0, 0, 63, 1, 6, 90),
    ("cereals_breads", "paratha", .91, 6.0, 0.55, 0, 0, 0, 63, 1, 6, 90),
    ("cereals_breads", "idli", .96, 5.0, 0.9, 0, 0, 0, 63, 1, 5, 60),
    ("cereals_breads", "poha_upma", .95, 5.8, 0.9, 0, 0, 0, 63, 1, 4, 45),
    ("dal_curries", "dal_tadka", .98, 6.0, 1.0, 0, 0, 0, 63, 1, 6, 75),
    ("dal_curries", "rajma", .98, 6.2, 1.0, 0, 0, 0, 63, 1, 6, 75),
    ("dal_curries", "chole", .97, 5.8, 1.0, 0, 0, 0, 63, 1, 6, 75),
    ("dal_curries", "paneer_masala", .98, 6.0, 1.4, 0, 1, 0, 63, 1, 5, 60),
    ("dal_curries", "mixed_veg_curry", .97, 6.0, 1.0, 0, 0, 0, 63, 1, 6, 75),
    ("dal_curries", "sambar", .98, 4.8, 0.8, 0, 0, 0, 63, 1, 7, 75),
    ("dal_curries", "kadhi", .98, 4.5, 0.8, 0, 1, 0, 63, 1, 6, 60),
    ("non_veg_dishes", "chicken_curry", .98, 6.2, 1.5, 1, 0, 0, 74, 1, 4, 90),
    ("non_veg_dishes", "chicken_biryani", .97, 6.3, 1.6, 1, 0, 1, 74, 1, 4, 60),
    ("non_veg_dishes", "mutton_curry", .98, 6.1, 1.4, 1, 0, 0, 71, 1, 4, 90),
    ("non_veg_dishes", "fish_curry", .98, 6.5, 2.0, 1, 0, 0, 63, 1, 3, 60),
    ("non_veg_dishes", "egg_curry", .98, 6.8, 1.5, 1, 0, 0, 71, 1, 4, 45),
    ("non_veg_dishes", "butter_chicken", .98, 6.0, 1.5, 1, 1, 0, 74, 1, 4, 90),
    ("snacks_bakery", "samosa", .92, 6.0, 0.6, 0, 0, 0, 63, 1, 3, 90),
    ("snacks_bakery", "veg_sandwich", .96, 5.8, 1.2, 0, 0, 0, 63, 0, 0, 30),
    ("snacks_bakery", "veg_puff", .90, 6.0, 0.5, 0, 0, 0, 63, 1, 3, 60),
    ("snacks_bakery", "pav_bhaji", .97, 5.4, 1.0, 0, 0, 0, 63, 1, 5, 60),
    ("snacks_bakery", "cake_slice", .88, 6.5, 0.4, 0, 1, 0, 63, 1, 0, 90),
    ("snacks_bakery", "dhokla", .96, 4.6, 0.8, 0, 0, 0, 63, 1, 4, 45),
    ("desserts_dairy", "gulab_jamun", .85, 6.0, 0.25, 0, 1, 0, 63, 1, 8, 90),
    ("desserts_dairy", "kheer", .97, 6.4, 1.5, 0, 1, 0, 63, 1, 4, 30),
    ("desserts_dairy", "rasmalai", .98, 6.3, 2.0, 0, 1, 0, 63, 0, 0, 30),
    ("desserts_dairy", "curd_raita", .98, 4.4, 1.3, 0, 1, 0, 63, 0, 0, 20),
    ("desserts_dairy", "lassi", .99, 4.3, 1.4, 0, 1, 0, 63, 0, 0, 15),
    ("desserts_dairy", "halwa", .87, 6.0, 0.4, 0, 0, 0, 63, 1, 8, 90),
]
DC = ["category", "dish", "aw", "ph", "mult", "protein", "dairy", "rice", "core", "cooked", "hotlim", "frzdays"]
STORAGE_P = {  # hot_hold, chilled, ambient, frozen, insulated_box
    "cereals_breads": [.35, .30, .25, .02, .08], "dal_curries": [.38, .30, .18, .04, .10],
    "non_veg_dishes": [.35, .40, .10, .05, .10], "snacks_bakery": [.20, .20, .50, .02, .08],
    "desserts_dairy": [.05, .65, .20, .05, .05],
}
STORAGES = ["hot_hold", "chilled", "ambient", "frozen", "insulated_box"]
PACK = ["sealed_container", "covered_tray", "vacuum_pack", "foil_wrapped", "open_tray"]
PACK_P = {"hot_hold": [.10, .50, 0, .10, .30], "insulated_box": [.45, .25, 0, .25, .05],
          "chilled": [.40, .35, .08, .07, .10], "ambient": [.15, .30, 0, .20, .35], "frozen": [.50, 0, .30, .20, 0]}
PACK_MULT = {"vacuum_pack": 0.55, "sealed_container": 0.8, "foil_wrapped": 0.95, "covered_tray": 0.9, "open_tray": 1.25}


def mu_path(T):
    T = np.asarray(T, float)
    Te = np.minimum(T, 37.0)
    base = np.where(Te > 4, (0.029 * (Te - 4)) ** 2, 0.0)
    return np.where(T > 37, base * np.clip(1 - (T - 37) / 15, 0, 1), base)


def mu_spoil(T, tmin):
    T = np.asarray(T, float)
    Te = np.minimum(T, 37.0)
    base = np.where(Te > tmin, (0.033 * (Te - tmin)) ** 2, 0.0)
    return np.where(T > 37, base * np.clip(1 - (T - 37) / 13, 0, 1), base)


PATHOGEN_THRESHOLD = 4.5   # log10 CFU/g midpoint of the dose-response logistic (tuned to ~20% unsafe)


def gen_batches(rng, weather, n):
    K = pd.DataFrame(KITCHENS, columns=KCOLS)
    D = pd.DataFrame(DISHES, columns=DC)
    w = K["capacity_meals_day"].values ** 0.6
    kidx = rng.choice(len(K), n, p=w / w.sum())
    kit = K.iloc[kidx].reset_index(drop=True)
    # category + dish
    cat = np.empty(n, dtype=object)
    for k in range(len(K)):
        m = np.where(kidx == k)[0]
        p = np.array(SHARE[K.loc[k, "kitchen_type"]], float)
        p[2] *= NONVEG_CITY.get(K.loc[k, "city"], 1.0)
        cat[m] = rng.choice(CATS, len(m), p=p / p.sum())
    dish_idx = np.zeros(n, int)
    for c in CATS:
        m = np.where(cat == c)[0]
        pool = D.index[D["category"] == c].values
        dish_idx[m] = rng.choice(pool, len(m))
    dd = D.iloc[dish_idx].reset_index(drop=True)
    # timing + weather
    day = rng.integers(0, N_DAYS, n)
    meal = rng.choice(["breakfast", "lunch", "snack", "dinner"], n, p=[.20, .35, .10, .35])
    base_h = pd.Series(meal).map({"breakfast": 7.0, "lunch": 12.0, "snack": 16.5, "dinner": 19.5}).values
    hour = np.clip(base_h + rng.normal(0, 0.6, n), 5, 23)
    prep_ts = DATES[day] + pd.to_timedelta(hour, unit="h")
    tday = np.zeros(n)
    hday = np.zeros(n)
    for city in CLIM:
        m = (kit["city"].values == city)
        tday[m] = weather[city]["temp"][day[m]]
        hday[m] = weather[city]["hum"][day[m]]
    amb = tday + 4 * np.sin(2 * np.pi * (hour - 9) / 24) + rng.normal(0, 1.2, n)
    hum = np.clip(hday + rng.normal(0, 5, n), 15, 100)
    qty = np.clip(np.exp(rng.normal(np.log(3 + kit["capacity_meals_day"].values / 500), 0.7)), 0.5, 120)
    source = rng.choice(["unserved_bulk", "service_line_return", "plate_waste"], n, p=[.91, .07, .02])
    # storage
    storage = np.empty(n, dtype=object)
    for c in CATS:
        m = np.where(cat == c)[0]
        storage[m] = rng.choice(STORAGES, len(m), p=STORAGE_P[c])
    # hours since prep at assessment
    H = np.zeros(n)
    for s, f in {"hot_hold": lambda k: np.clip(rng.gamma(2.0, 1.2, k) + 0.3, 0.3, 14),
                 "insulated_box": lambda k: np.clip(rng.gamma(2.5, 1.4, k) + 0.5, 0.5, 12),
                 "ambient": lambda k: np.clip(rng.gamma(2.0, 1.8, k) + 0.3, 0.3, 14),
                 "chilled": lambda k: np.clip(np.exp(rng.normal(np.log(20), 0.9, k)), 1, 240),
                 "frozen": lambda k: np.clip(np.exp(rng.normal(np.log(240), 1.0, k)), 24, 2000)}.items():
        m = storage == s
        H[m] = f(m.sum())
    # cooking / cooling
    cooked = dd["cooked"].values == 1
    undercooked_roll = rng.random(n) < 0.035
    core = np.where(undercooked_roll, rng.normal(60, 5, n), rng.normal(83, 5, n))
    undercooked = cooked & (core < dd["core"].values)
    core_obs = np.where(cooked, core, np.nan)
    cool_m = np.empty(n, dtype=object)
    cool_t = np.zeros(n)
    sf = (qty / 12) ** 0.2
    chill_like = np.isin(storage, ["chilled", "frozen"])
    meth = rng.choice(["blast_chiller", "shallow_trays", "ice_bath", "ambient_cooling"], n, p=[.35, .30, .15, .20])
    ct = np.where(meth == "blast_chiller", rng.normal(1.6, .4, n), np.where(meth == "shallow_trays", rng.normal(2.8, .8, n),
                  np.where(meth == "ice_bath", rng.normal(2.2, .7, n), rng.normal(6.5, 2.0, n))))
    cool_m[:] = "none"
    cool_m[chill_like] = meth[chill_like]
    cool_t[chill_like] = np.clip(ct[chill_like] * sf[chill_like], 0.6, 14)
    am = storage == "ambient"
    cool_m[am] = "ambient_cooling"
    cool_t[am] = np.clip(rng.normal(2.5, 0.8, am.sum()) * sf[am], 0.5, 8)
    cool_t = np.minimum(cool_t, H)
    t_stor = np.maximum(H - cool_t, 0)
    # storage temperatures (recorded) and true temperatures
    mean = np.zeros(n)
    mx = np.zeros(n)
    mn = np.zeros(n)
    thaw = np.zeros(n)
    cold_break = np.zeros(n, bool)
    m = storage == "hot_hold"
    mean[m] = rng.normal(64, 6, m.sum()); mx[m] = mean[m] + np.abs(rng.normal(3, 2, m.sum())); mn[m] = mean[m] - np.abs(rng.normal(5, 3, m.sum()))
    m = storage == "chilled"
    mean[m] = np.clip(rng.normal(3.8, 1.6, m.sum()), 0.2, None)
    cb = rng.random(m.sum()) < 0.04
    mean[m] = mean[m] + cb * rng.uniform(5, 14, m.sum())
    cold_break[np.where(m)[0][cb]] = True
    mx[m] = mean[m] + np.abs(rng.normal(2.0, 1.5, m.sum())) + cb * rng.uniform(2, 8, m.sum()); mn[m] = mean[m] - np.abs(rng.normal(1.5, 1, m.sum()))
    m = storage == "ambient"
    mean[m] = amb[m] + rng.normal(0, 1.0, m.sum()); mx[m] = mean[m] + np.abs(rng.normal(2, 1, m.sum())); mn[m] = mean[m] - np.abs(rng.normal(2, 1, m.sum()))
    m = storage == "frozen"
    mean[m] = rng.normal(-18, 2.0, m.sum())
    thaw[m] = rng.poisson(0.15, m.sum())
    cb = rng.random(m.sum()) < 0.03
    cold_break[np.where(m)[0][cb]] = True
    mean[m] = mean[m] + cb * rng.uniform(8, 18, m.sum())
    mx[m] = mean[m] + np.abs(rng.normal(2, 1.2, m.sum())) + cb * rng.uniform(3, 10, m.sum()); mn[m] = mean[m] - np.abs(rng.normal(1, 0.8, m.sum()))
    thaw = thaw + cold_break * (storage == "frozen")
    m = storage == "insulated_box"
    a_m = amb[m]
    Hm = np.maximum(t_stor[m], 0.3)
    mean[m] = a_m + (68 - a_m) * (6.0 / Hm) * (1 - np.exp(-Hm / 6.0))
    mx[m] = 68.0 - rng.uniform(0, 2, m.sum()); mn[m] = a_m + (68 - a_m) * np.exp(-Hm / 6.0)
    # danger-zone minutes
    f_hold = np.clip(norm.cdf((60 - mean) / 3.0), 0, 1)
    danger = np.zeros(n)
    danger += 0.9 * cool_t * 60 * (chill_like | am)
    m = storage == "hot_hold"; danger[m] += t_stor[m] * 60 * f_hold[m]
    m = storage == "chilled"; danger[m] += t_stor[m] * 60 * np.clip(norm.cdf((mean[m] - 5) / 0.8), 0, 1)
    m = storage == "ambient"; danger[m] += t_stor[m] * 60
    m = storage == "frozen"; danger[m] += thaw[m] * rng.uniform(30, 240, m.sum())
    m = storage == "insulated_box"
    a_m = amb[m]; t60 = 6.0 * np.log((68 - a_m) / np.maximum(60 - a_m, 0.5))
    danger[m] += np.maximum(t_stor[m] - np.minimum(t60, t_stor[m]), 0) * 60
    dang_h = danger / 60.0
    # hygiene
    audit = np.clip(kit["hygiene_baseline"].values + rng.normal(0, 4, n), 40, 100)
    handler = np.clip(audit + rng.normal(0, 8, n), 20, 100)
    # reheating
    reheat = rng.choice([0, 1, 2], n, p=[.78, .17, .05]) * (cooked & (storage != "hot_hold") & (storage != "insulated_box"))
    reheat_t = np.where(reheat > 0, rng.normal(78, 7, n), np.nan)
    # packaging
    pack = np.empty(n, dtype=object)
    for s in STORAGES:
        m = storage == s
        pack[m] = rng.choice(PACK, m.sum(), p=PACK_P[s])
    integ = np.clip(rng.beta(9, 1.6, n), 0, 1)
    dmg = rng.random(n) < 0.03
    integ = np.where(dmg, rng.uniform(0.1, 0.5, n), integ)
    # ---------------- pathogen model (latent) ----------------
    prot = dd["protein"].values
    n0 = (1.3 + 0.5 * prot + 0.3 * dd["rice"].values + (100 - handler) / 45 + (100 - audit) / 70
          + 1.5 * (source == "service_line_return") + 2.2 * undercooked + rng.normal(0, 0.45, n))
    true_mean = mean + np.where(np.isin(storage, ["chilled", "ambient"]), rng.normal(0, 0.8, n), 0)
    t_eff = np.where(storage == "chilled", true_mean + 0.5 * (mx - mean),
                     np.where(storage == "ambient", true_mean + 0.3 * (mx - mean), true_mean))
    g_cool = 0.16 * cool_t * np.where(prot == 1, 1.0, 0.8) * (chill_like | am)
    mu_p = mu_path(t_eff)
    lag_p = np.minimum(24, 0.5 / np.maximum(mu_p, 0.0208))
    g_stor = np.where(np.isin(storage, ["chilled", "ambient"]), mu_p * np.maximum(0, t_stor - lag_p), 0.0)
    g_hold = np.where(np.isin(storage, ["hot_hold", "insulated_box"]), mu_path(45.0) * np.maximum(0, dang_h - 0.8), 0.0)
    g_thaw = np.where(storage == "frozen", mu_path(25.0) * thaw * rng.uniform(0.5, 4, n), 0.0)
    reheat_eff = np.where(reheat > 0, np.where(reheat_t < 74, 0.6, -1.0) * reheat, 0.0)
    logn = n0 + g_cool + g_stor + g_hold + g_thaw + reheat_eff
    p_patho = sigmoid((logn - PATHOGEN_THRESHOLD) / 0.6)
    # ---------------- spoilage / shelf life (latent) ----------------
    tmin_sp = np.where((prot == 1) | (dd["dairy"].values == 1), -3.0, -1.0)
    aw_m = np.where(dd["aw"].values < 0.88, 0.35, np.where(dd["aw"].values < 0.93, 0.65, 1.0))
    ph_m = np.where(dd["ph"].values < 4.6, 0.55, 1.0)
    pm = pd.Series(pack).map(PACK_MULT).values * (1 + (1 - integ) * 0.8)
    mult = dd["mult"].values * pm * aw_m * ph_m * (1 + 0.1 * ((hum > 80) & (pack == "open_tray")))
    g_req = np.clip(rng.normal(4.0, 0.4, n), 2.8, 5.5)
    mu_s = mu_spoil(t_eff, tmin_sp) * mult
    g_cool_s = 0.6 * cool_t * mu_spoil(22.0, tmin_sp) * mult * (chill_like | am)
    lag_s = np.minimum(24, 0.7 / np.maximum(mu_s, 0.03))
    g_s = mu_s * np.maximum(0, t_stor - lag_s)
    cons = (g_cool_s + g_s) / g_req
    rem = np.minimum(np.maximum(0, (g_req - g_cool_s - g_s) / np.maximum(mu_s, 1e-4)) + np.maximum(0, lag_s - t_stor), 2160)
    # hot / insulated: quality time limit; frozen: freezer limit
    lim = dd["hotlim"].values * np.exp(rng.normal(0, 0.12, n))
    lim = np.where(storage == "hot_hold", lim * np.where(mean < 60, 0.7, 1.0), lim * 1.3)
    hot = np.isin(storage, ["hot_hold", "insulated_box"]) & (dd["hotlim"].values > 0)
    hot0 = np.isin(storage, ["hot_hold", "insulated_box"]) & (dd["hotlim"].values == 0)  # no hot life: treat like ambient
    cons = np.where(hot, H / np.maximum(lim, 1e-3), cons)
    cons_safety = np.where(hot, cons / 1.8, cons)
    rem = np.where(hot, np.maximum(0, lim - H), rem)
    flim = dd["frzdays"].values * 24 * np.exp(rng.normal(0, 0.1, n)) * np.maximum(1 - 0.12 * thaw, 0.2)
    fz = storage == "frozen"
    cons = np.where(fz, H / flim, cons)
    rem = np.where(fz, np.maximum(0, flim - H), rem)
    rem = np.where(source == "service_line_return", rem * 0.6, rem)
    rem = np.where(source == "plate_waste", 0.0, rem)
    rem_obs = np.clip(rem * np.exp(rng.normal(0, 0.08, n)), 0, 4320)
    p_spoil = 0.6 * sigmoid((cons_safety - 1.3) / 0.12)
    p_unsafe = 1 - (1 - p_patho) * (1 - p_spoil)
    p_unsafe = np.where(source == "plate_waste", 0.995, p_unsafe)
    y = (rng.random(n) < p_unsafe).astype(int)
    flip = rng.random(n) < 0.012
    y = np.where(flip, 1 - y, y)
    # ---------------- observed sensing: CV, gas, smell ----------------
    pq = rng.choice(["good", "low_light", "blurry", "occluded", "no_photo"], n, p=[.72, .10, .08, .04, .06])
    vis = sigmoid(6 * (np.minimum(cons, 1.6) - 0.85))
    sd = np.where(pq == "good", 0.07, 0.18)
    cv = np.clip(0.04 + 0.9 * vis + rng.normal(0, 1, n) * sd, 0, 1)
    wrong = rng.random(n) < 0.05
    cv = np.where(wrong, rng.random(n), cv)
    cv = np.where(pq == "no_photo", np.nan, np.round(cv, 3))
    voc_ok = (kit["has_voc_sensor"].values == 1) & (rng.random(n) > 0.05)
    voc = np.clip(5 + 90 * sigmoid(4 * (np.minimum(cons, 1.6) - 0.9)) + rng.normal(0, 6, n) + 0.15 * (amb - 28), 0, 100)
    voc = np.where(voc_ok, np.round(voc, 1), np.nan)
    odor = np.round(2 * np.clip(10 * sigmoid(5 * (np.minimum(cons, 1.6) - 0.9)) + rng.normal(0, 1.2, n), 0, 10)) / 2
    odor = np.where(rng.random(n) < 0.80, odor, np.nan)
    probe = np.where(rng.random(n) < 0.60, np.round(true_mean + rng.normal(0, 1.5, n), 1), np.nan)
    # recorded temperature channels (loggers missing / glitching)
    logger = kit["has_temp_logger"].values == 1
    miss = np.where(logger, rng.random(n) < 0.03, rng.random(n) < 0.85)
    rec_mean = np.where(miss, np.nan, np.round(mean, 1))
    rec_max = np.where(miss, np.nan, np.round(mx, 1))
    rec_min = np.where(miss, np.nan, np.round(mn, 1))
    rec_danger = np.where(miss, np.nan, np.round(danger, 0))
    dq = np.array(["ok"] * n, dtype=object)
    g_i = np.where(~miss)[0]
    gl = rng.choice(g_i, int(n * 0.003), replace=False)
    rec_max[gl] = rng.uniform(85, 99, len(gl)); dq[gl] = "sensor_glitch"
    fi = rng.choice(np.setdiff1d(g_i, gl), int(n * 0.002), replace=False)
    rec_mean[fi] = np.round(rec_mean[fi] * 1.8 + 32, 1); dq[fi] = "unit_error_fahrenheit"
    df = pd.DataFrame(dict(
        batch_id=[f"B{i:06d}" for i in range(n)], kitchen_id=kit["kitchen_id"].values, city=kit["city"].values,
        kitchen_type=kit["kitchen_type"].values, category=cat, dish=dd["dish"].values, meal_period=meal,
        prep_timestamp=prep_ts, month=prep_ts.month, hour_of_day=np.round(hour, 2), source_stage=source,
        quantity_kg=np.round(qty, 1), storage_type=storage, packaging=pack, packaging_integrity=np.round(integ, 2),
        cooked_core_temp_c=np.round(core_obs, 1), cooling_method=cool_m, cooling_time_hr=np.round(cool_t, 2),
        reheat_count=reheat, reheat_core_temp_c=np.round(reheat_t, 1), freeze_thaw_cycles=thaw.astype(int),
        hours_since_prep=np.round(H, 2), ambient_temp_c=np.round(amb, 1), humidity_pct=np.round(hum, 0),
        water_activity=np.round(np.clip(dd["aw"].values + rng.normal(0, 0.01, n), 0.7, 1.0), 3),
        ph=np.round(np.clip(dd["ph"].values + rng.normal(0, 0.15, n), 3.5, 7.5), 2),
        storage_temp_c_mean=rec_mean, storage_temp_c_max=rec_max, storage_temp_c_min=rec_min,
        minutes_in_danger_zone=rec_danger, manual_probe_temp_c=probe,
        handler_hygiene_score=np.round(handler, 0), kitchen_audit_score=np.round(audit, 0),
        photo_quality=pq, cv_spoilage_prob=cv, voc_index=voc, odor_score=odor,
        # ---- targets ----
        unsafe_for_redistribution=y, remaining_shelf_life_hr=np.round(rem_obs, 1),
        # ---- evaluation only ----
        latent_p_unsafe=np.round(np.where(source == "plate_waste", 0.995, p_unsafe), 4),
        latent_spoilage_consumed=np.round(cons, 3), latent_log10_pathogen=np.round(logn, 2),
        cold_chain_break=cold_break.astype(int), dq_flag=dq,
    ))
    return df


# --------------------------------------------------------------------------- sensors
SENSORS = [("S01", "K01", "walk_in_cooler"), ("S02", "K01", "freezer"), ("S03", "K02", "hot_hold_cabinet"),
           ("S04", "K03", "reach_in_fridge"), ("S05", "K05", "walk_in_cooler"), ("S06", "K06", "walk_in_cooler"),
           ("S07", "K06", "freezer"), ("S08", "K07", "reach_in_fridge"), ("S09", "K10", "walk_in_cooler"),
           ("S10", "K12", "freezer")]
UNIT = {"walk_in_cooler": dict(sp=2.5, amp=0.5, door_k=0.9, cur=7.0, defrost=1.5),
        "reach_in_fridge": dict(sp=3.5, amp=0.7, door_k=1.3, cur=5.0, defrost=1.5),
        "freezer": dict(sp=-19.0, amp=0.9, door_k=1.1, cur=9.0, defrost=2.8),
        "hot_hold_cabinet": dict(sp=66.0, amp=1.5, door_k=-1.6, cur=8.0, defrost=0.0)}
KC = {k[0]: k[1] for k in KITCHENS}


def gen_sensors(rng, n_steps=3600):
    out = []
    start = pd.Timestamp("2026-08-01")
    for sid, kid, utype in SENSORS:
        P = UNIT[utype]
        city = KC[kid]
        t = np.arange(n_steps)
        ts = start + pd.to_timedelta(15 * t, unit="min")
        hr = ts.hour.values + ts.minute.values / 60
        act = (np.exp(-((hr - 8) / 1.3) ** 2) + 1.1 * np.exp(-((hr - 12.8) / 1.5) ** 2) + 0.9 * np.exp(-((hr - 19.5) / 1.6) ** 2))
        door = np.clip(0.02 + 0.22 * act * rng.uniform(0.6, 1.4, n_steps) + rng.normal(0, 0.01, n_steps), 0, 1)
        amb_m = np.mean(CLIM[city][0][7:9])
        amb = amb_m + 4 * np.sin(2 * np.pi * (hr - 9) / 24) + rng.normal(0, 0.8, n_steps)
        period = rng.uniform(7.3, 10.7)
        wave = np.sin(2 * np.pi * t / period + rng.uniform(0, 6.28))
        ar = np.zeros(n_steps)
        for i in range(1, n_steps):
            ar[i] = 0.6 * ar[i - 1] + rng.normal(0, 0.18)
        temp = P["sp"] + P["door_k"] * door * 4 + P["amp"] * wave + ar + (0.03 * (amb - 28) if utype != "hot_hold_cabinet" else 0)
        on = wave > -0.2
        cur = np.where(on, rng.normal(P["cur"], 0.4, n_steps), rng.normal(1.3, 0.2, n_steps))
        defrost = np.zeros(n_steps, int)
        if P["defrost"] > 0:
            off = rng.integers(0, 32)
            defrost = ((t + off) % 32 < 2).astype(int)
            temp = temp + P["defrost"] * defrost * np.where((t + off) % 32 == 0, 1.0, 0.6)
            cur = cur + 3 * defrost
        power = np.ones(n_steps, int)
        label = np.zeros(n_steps, int)
        atype = np.array(["none"] * n_steps, dtype=object)
        occupied = np.zeros(n_steps, bool)
        sign_cold = -1.0 if utype == "hot_hold_cabinet" else 1.0

        def place(dur, pad=8):
            for _ in range(60):
                s = int(rng.integers(40, n_steps - dur - pad - 5))
                if not occupied[max(0, s - 6): s + dur + pad].any():
                    occupied[s: s + dur + pad] = True
                    return s
            return None
        # long events first (only ~half the sensors have one)
        if rng.random() < 0.35:
            kind = rng.choice(["sensor_drift", "gasket_degradation"]) if utype != "hot_hold_cabinet" else "sensor_drift"
            dur = int(rng.integers(200, 450)); s = place(dur)
            if s is not None:
                if kind == "sensor_drift":
                    sg = rng.choice([-1, 1]); bias = sg * rng.uniform(0.012, 0.03) * np.arange(dur)
                    temp[s:s + dur] += bias
                    m = np.abs(bias) > 1.0
                else:
                    off_t = np.linspace(0, rng.uniform(1.8, 3.5), dur)
                    temp[s:s + dur] += off_t; cur[s:s + dur] += np.linspace(0, 1.6, dur)
                    m = off_t > 1.0
                label[s:s + dur] = m; atype[s:s + dur][m] = kind
        kinds = ["door_left_open", "equipment_failure", "power_outage", "stuck_sensor", "spike_outlier"]
        probs = [.22, .12, .08, .15, .43]
        for _ in range(int(rng.integers(4, 8))):
            kind = rng.choice(kinds, p=probs)
            if kind == "door_left_open":
                dur = int(rng.integers(4, 13)); s = place(dur)
                if s is None: continue
                amp = sign_cold * (rng.uniform(5, 11) if utype == "freezer" else rng.uniform(4, 9)) if sign_cold > 0 else -rng.uniform(8, 18)
                ramp = amp * (1 - np.exp(-(np.arange(dur) + 1) / 2.5))
                temp[s:s + dur] += ramp
                temp[s + dur:s + dur + 6] += ramp[-1] * np.exp(-(np.arange(6) + 1) / 2.5)
                door[s:s + dur] = rng.uniform(0.7, 1.0, dur); cur[s:s + dur] += 1.5
                label[s:s + dur + 2] = 1; atype[s:s + dur + 2] = kind
            elif kind == "equipment_failure":
                dur = int(rng.integers(24, 97)); s = place(dur, 10)
                if s is None: continue
                rise = min((amb_m - P["sp"]) * 0.65, 25) if utype != "hot_hold_cabinet" else -(P["sp"] - amb_m) * 0.5
                temp[s:s + dur] += rise * (1 - np.exp(-np.arange(dur) / (dur / 3)))
                temp[s + dur:s + dur + 8] += rise * np.exp(-(np.arange(8) + 1) / 2.0)
                cur[s:s + dur] = rng.normal(0.4, 0.15, dur)
                label[s:s + dur] = 1; atype[s:s + dur] = kind
            elif kind == "power_outage":
                dur = int(rng.integers(8, 41)); s = place(dur, 8)
                if s is None: continue
                rise = min((amb_m - P["sp"]) * 0.5, 20) if utype != "hot_hold_cabinet" else -(P["sp"] - amb_m) * 0.5
                temp[s:s + dur] += rise * (1 - np.exp(-np.arange(dur) / (dur / 2.5)))
                temp[s + dur:s + dur + 8] += rise * np.exp(-(np.arange(8) + 1) / 2.0)
                cur[s:s + dur] = 0; power[s:s + dur] = 0
                label[s:s + dur] = 1; atype[s:s + dur] = kind
            elif kind == "stuck_sensor":
                dur = int(rng.integers(12, 49)); s = place(dur)
                if s is None: continue
                temp[s:s + dur] = temp[s - 1]
                label[s:s + dur] = 1; atype[s:s + dur] = kind
            else:
                s = place(1, 2)
                if s is None: continue
                temp[s] += rng.choice([-1, 1]) * rng.uniform(8, 20)
                label[s] = 1; atype[s] = kind
        cur = np.maximum(cur, 0)
        dq = np.array(["ok"] * n_steps, dtype=object)
        temp_o = np.round(temp, 2).astype(float)
        for _ in range(int(rng.integers(0, 4))):
            s = int(rng.integers(5, n_steps - 15)); d = int(rng.integers(1, 11))
            temp_o[s:s + d] = np.nan; dq[s:s + d] = "data_gap"
        out.append(pd.DataFrame(dict(
            sensor_id=sid, kitchen_id=kid, unit_type=utype, timestamp=ts, temp_c=temp_o, door_open_frac=np.round(door, 3),
            compressor_current_a=np.round(cur, 2), power_ok=power, defrost_active=defrost, ambient_temp_c=np.round(amb, 1),
            hour_of_day=np.round(hr, 2), day_of_week=ts.dayofweek.values, is_anomaly=label, anomaly_type=atype, dq_flag=dq)))
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------- routing instances
def gen_routing(rng, per_size=20):
    rows = []
    iid = 0
    for n_c in (10, 25, 50):
        for _ in range(per_size):
            itype = rng.choice(["R", "C", "RC"])
            depot = (rng.uniform(18.95, 19.25), rng.uniform(72.82, 72.98))
            if itype == "R":
                pts = np.column_stack([rng.uniform(18.90, 19.30, n_c), rng.uniform(72.80, 73.00, n_c)])
            else:
                k = int(rng.integers(3, 5)); cen = np.column_stack([rng.uniform(18.95, 19.25, k), rng.uniform(72.82, 72.98, k)])
                a = cen[rng.integers(0, k, n_c)] + rng.normal(0, 0.02, (n_c, 2))
                if itype == "RC":
                    h = n_c // 2
                    a[:h] = np.column_stack([rng.uniform(18.90, 19.30, h), rng.uniform(72.80, 73.00, h)])
                pts = a
            kg = np.clip(np.exp(rng.normal(np.log(25), 0.7, n_c)), 3, 180)
            cap = float(rng.choice([150, 300, 500]))
            edge = rng.random() < 0.05   # edge case: one donor larger than a vehicle
            if edge:
                kg[0] = cap * 1.3
            else:
                kg = np.minimum(kg, cap * 0.9)
            n_veh = max(int(math.ceil(kg.sum() / (cap * 0.8))), int(math.ceil(n_c / 4.5))) + int(rng.integers(0, 3))
            meal = rng.choice(["lunch", "dinner", "breakfast"], n_c, p=[.5, .4, .1])
            ready = pd.Series(meal).map({"lunch": 840, "dinner": 1230, "breakfast": 480}).values + rng.integers(0, 90, n_c)
            width = np.clip(rng.normal(120, 40, n_c), 60, 240)
            urgent = rng.random(n_c) < 0.15
            deadline = ready + np.where(urgent, rng.uniform(120, 180, n_c), np.minimum(np.exp(rng.normal(np.log(300), 0.4, n_c)), 480))
            deadline = np.minimum(deadline, 1700)
            service = np.round(6 + kg / 15 + rng.normal(3, 1.5, n_c), 1).clip(4, 30)
            off_sp, pk_sp = rng.normal(26, 2), rng.normal(15, 2)
            common = dict(instance_id=f"I{iid:03d}", instance_type=itype, n_customers=n_c, n_vehicles=n_veh,
                          vehicle_capacity_kg=cap, speed_offpeak_kmph=round(off_sp, 1), speed_peak_kmph=round(pk_sp, 1),
                          road_circuity=1.35)
            rows.append(dict(common, node_id=0, node_type="depot_hub", lat=depot[0], lon=depot[1], pickup_kg=0.0,
                             service_min=0.0, tw_start_min=0, tw_end_min=1800, food_deadline_min=1800, urgent=0,
                             meal_period="none"))
            for j in range(n_c):
                rows.append(dict(common, node_id=j + 1, node_type="donor_kitchen", lat=round(pts[j, 0], 5),
                                 lon=round(pts[j, 1], 5), pickup_kg=round(kg[j], 1), service_min=service[j],
                                 tw_start_min=int(ready[j]), tw_end_min=int(min(ready[j] + width[j], 1700)),
                                 food_deadline_min=int(min(deadline[j], 1700)), urgent=int(urgent[j]), meal_period=meal[j]))
            iid += 1
    df = pd.DataFrame(rows)
    df["lat"] = df["lat"].round(5); df["lon"] = df["lon"].round(5)
    return df


DICT_MD = """# Annapurna synthetic data - data dictionary

All data is synthetic, generated from explicit mechanisms (see generate_annapurna_data.py). It is realistic in
structure and in the *kinds* of messiness it contains, but it is NOT real kitchen data: models trained on it learn
the generator's assumptions. Say so in the SIH deck (as "synthetic data seeded from published HACCP / climatology
assumptions, to be replaced by pilot data"). Festival dates are approximate (+-1 day).

Rule: columns starting with `latent_` are ground truth for evaluation. Never use them as features.

## demand_daily.csv  (kitchen x category x day, 2025-02-01 .. 2026-09-23)
Features known ahead of time: date parts, is_holiday, holiday_type, festival_name, is_holiday_eve,
days_to_next/since_last_holiday, ipl_match_day, academic_break, promo_discount_pct, special_menu_flag, prices,
kitchen/category attributes. Weather columns are *observed* values: in production use the forecast instead.
Outcomes (NOT features): prepared_units, units_sold, leftover_units, leftover_kg, stockout_flag, is_closed,
outage_flag, latent_true_demand.
Target: `units_sold` is censored by stock (stockout_flag=1 rows under-state demand: drop or treat as censored).
Edge cases: planned closures (weekends, holidays) and unplanned closures (is_closed), power outages (sales cut,
waste up), stockouts, festival spikes and eve effects, regional festivals, school/hostel vacations, rain/heat
effects, IPL days, promo campaigns + flash sales, bulk catering orders, unexplained dips, price inflation,
growth trends, missing weather (1%).
dq_flag: ok | missing_sales | unit_error_x10 | stale_copy | duplicate  -> clean before training.

## food_batches.csv  (one row = one surplus batch assessed for redistribution)
Targets: unsafe_for_redistribution (1 = unsafe), remaining_shelf_life_hr (>=0, right-skewed: model log1p).
Suggested monotone constraints (safety): +hours_since_prep, +minutes_in_danger_zone, +storage_temp_c_max,
+cooling_time_hr, +reheat_count, -handler_hygiene_score, -kitchen_audit_score, -packaging_integrity,
+cv_spoilage_prob, +voc_index, +odor_score.
Evaluation only: latent_p_unsafe (true probability - use it to check isotonic calibration),
latent_spoilage_consumed, latent_log10_pathogen, cold_chain_break.
Edge cases: undercooked food, slow cooling, hot-hold below 60C, ambient storage, insulated boxes, frozen with
thaw cycles, cold-chain breaks, reheating, service-line returns, plate waste (always unsafe), damaged packaging,
photo quality failures (cv missing/noisy), CV that is confidently wrong (5%), kitchens without loggers (temps
~85% missing) or VOC sensors, manual probe readings, 1.2% label noise (inspector error).
dq_flag: ok | sensor_glitch (impossible max temp) | unit_error_fahrenheit.
Note: pathogens are mostly invisible, so the CV score alone is a weak safety predictor - that is why fusion helps.

## sensor_readings.csv  (10 units x 3600 readings, 15-minute interval, 2026-08-01 onward)
is_anomaly / anomaly_type are ground truth: door_left_open, equipment_failure, power_outage, stuck_sensor,
spike_outlier, sensor_drift (reading only), gasket_degradation (temp creeps up AND compressor current rises).
Benign look-alikes that must NOT be flagged: defrost cycles (defrost_active=1), meal-rush door openings,
compressor cycling. dq_flag=data_gap rows have NaN temp and are not anomalies.

## routing_nodes.csv  (60 instances: 20 each with 10 / 25 / 50 donor kitchens + 1 depot hub, Mumbai bounding box)
Pick up surplus (pickup_kg) from donors within [tw_start_min, tw_end_min] (minutes after midnight; values above 1440 mean the next day), service for
service_min, return to the hub before food_deadline_min. Vehicles: n_vehicles x vehicle_capacity_kg.
Travel time = haversine_km x road_circuity / speed, with speed_peak_kmph during 08:00-11:00 and 17:00-21:00 and
speed_offpeak_kmph otherwise. Edge cases: urgent donors, clustered/random/mixed layouts, one-donor-bigger-than-
vehicle instances (~5%), dinner/breakfast/lunch surplus windows.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--batches", type=int, default=36000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    cal = build_calendar()
    school_closed, hostel_occ, ipl = build_flags(rng)
    weather = build_weather(rng)
    pd.DataFrame(KITCHENS, columns=KCOLS).to_csv(f"{a.out}/kitchens.csv", index=False)
    demand = gen_demand(rng, weather, cal, school_closed, hostel_occ, ipl)
    demand.to_csv(f"{a.out}/demand_daily.csv", index=False)
    batches = gen_batches(rng, weather, a.batches)
    batches.to_csv(f"{a.out}/food_batches.csv", index=False)
    sensors = gen_sensors(rng)
    sensors.to_csv(f"{a.out}/sensor_readings.csv", index=False)
    routing = gen_routing(rng)
    routing.to_csv(f"{a.out}/routing_nodes.csv", index=False)
    with open(f"{a.out}/data_dictionary.md", "w") as f:
        f.write(DICT_MD)
    for name, d in [("demand_daily", demand), ("food_batches", batches), ("sensor_readings", sensors), ("routing_nodes", routing)]:
        print(f"{name}: {len(d):,} rows x {d.shape[1]} cols")


if __name__ == "__main__":
    main()
