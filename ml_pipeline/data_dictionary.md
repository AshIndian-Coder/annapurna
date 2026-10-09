# Annapurna synthetic data - data dictionary

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
