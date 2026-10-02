# 04 — DATABASE SPEC (v3 FINAL, PostgreSQL 16, golang-migrate)

Audience: the AI/engineer building `database/`. Build exactly this; ask before adding tables. Schema is identical to v2 — only the migration tooling changed (Alembic → plain SQL, D14 v3).

## 1. Conventions
UUID PKs (`gen_random_uuid()`, enable `pgcrypto`), `created_at timestamptz default now()`, `updated_at` with trigger, all times UTC (`timestamptz`), quantities `numeric(10,2)` with CHECK ≥ 0, soft delete not used (status columns instead), enums as PG ENUM types created in 0002, snake_case plural table names, FK `ON DELETE RESTRICT` unless stated. Optional extension: PostGIS; default is plain lat/lng + haversine, so PostGIS is not required. Migrations are raw SQL `{version}_{name}.{up,down}.sql` run by golang-migrate; Go embeds nothing — `database/migrations` is the single location.

## 2. Tables (locked core + approved additions)
| Table | Columns (type) | Notes |
|---|---|---|
| users | id, name, email UNIQUE, password_hash, role role_enum, kitchen_id FK null, recipient_id FK null, is_active bool, created_at | role NGO links to recipient; KITCHEN to kitchen |
| kitchens | id, name, type (HOSTEL/CORPORATE/HOSPITAL/CATERER/PROCESSING), latitude, longitude, timezone | |
| meals | id, kitchen_id FK, date, meal_type, menu jsonb, UNIQUE(kitchen_id,date,meal_type) | menu = string array |
| attendance | id, meal_id FK UNIQUE, expected_diners int ≥0, actual_diners int null | |
| production | id, meal_id FK UNIQUE, prepared_qty numeric, consumed_qty numeric null, unit text default 'kg' | consumed filled by feedback |
| waste | id, meal_id FK null, kitchen_id FK, quantity_kg, cause waste_cause, note, client_event_id text UNIQUE null (D24), created_at | offline-created rows dedupe on client_event_id |
| surplus_batches | id, batch_code UNIQUE, meal_id FK null, kitchen_id FK, food_name, food_category, quantity_kg, prepared_at, expiry_at, safety_status (PENDING/ELIGIBLE/HOLD/REJECTED), status surplus_status, approved_by FK users null, approved_at | CHECK expiry_at > prepared_at |
| recipients | id, name, type (NGO/FOOD_BANK/SHELTER/COMMUNITY_KITCHEN/BUYER/FEED_FARM/COMPOST), capacity_kg, latitude, longitude, pickup_window_start time, pickup_window_end time, accepts_categories text[], last_received_at, active | |
| matches | id, batch_id FK, recipient_id FK, score numeric, score_breakdown jsonb, reasons text[], status (OFFERED/ACCEPTED/DECLINED/EXPIRED), responded_at | UNIQUE(batch_id,recipient_id) |
| routes | id, batch_id FK, distance_km, eta_minutes, solver text, created_at | |
| route_stops | id, route_id FK CASCADE, recipient_id FK, sequence int, eta timestamptz | UNIQUE(route_id,sequence) |
| qr_events | id, batch_id FK, event_type qr_event, actor_id FK users, lat, lng, evidence_hash text, prev_hash text, hash text NOT NULL, client_event_id text UNIQUE null, client_ts timestamptz null (D24), created_at | chain per batch; trigger forbids UPDATE/DELETE; hash uses server order + server ts; client_ts is provenance only |
| processing_metrics | id, kitchen_id FK, date, raw_material_kg, output_kg, waste_kg, downtime_min, runtime_min, energy_kwh, UNIQUE(kitchen_id,date) | CHECK output ≤ raw, downtime ≤ runtime |
| sensor_readings | id bigserial, sensor_id, location_id, batch_id null, ts, temperature_c, humidity_pct, energy_kwh | partition by month if > 1M rows |
| alerts | id, kitchen_id, type, severity, message, payload jsonb, acknowledged_by, acknowledged_at, created_at | |
| quality_checks | id, batch_id FK, image_path, visual_status, risk_level, cv_confidence, detections jsonb, cv_model_version, safety_decision, safety_reasons text[], fusion_model_version, danger_zone_minutes, temperature_c, device_preview jsonb null, client_meta jsonb null (D20), created_by, created_at | source of evidence_hash; device block telemetry only |
| prediction_log | id, kitchen_id, meal_id null, request jsonb, response jsonb, model_version, created_at | enables drift and backtests |
| outcome_feedback | id, meal_id FK, predicted_p50, actual_consumed, prepared, created_at | |
| diversions | id, batch_id FK, stream (ANIMAL_FEED/COMPOST/BIOGAS), recipient_id null, quantity_kg, created_at | |
| audit_log | id, user_id, action, entity, entity_id, before jsonb, after jsonb, ip, created_at | append-only |
| refresh_tokens | id, user_id FK CASCADE, family_id UUID, token_hash text UNIQUE, expires_at, revoked_at null, created_at, last_used_at | rotation: new row per refresh; reuse of revoked → revoke family (D22); sha256(token) ONLY, never the token |
| device_tokens | id, user_id FK CASCADE, platform (ANDROID/IOS), token text UNIQUE (FCM registration token), app_version, active bool, created_at, last_seen_at | push best-effort; prune on FCM invalid-token errors (D23) |

## 3. Hash chain (owned by Go, D30)
`hash = sha256(prev_hash || batch_id || event_type || actor_id || created_at_iso || evidence_hash)` — ASCII byte concatenation, language-neutral. First event prev_hash = `GENESIS`. `evidence_hash = sha256(canonical_json(quality_check row essentials))` (fixed field order, Go json.Marshal of a dedicated struct). Go `internal/qrchain` computes; DB trigger `qr_events_immutable` blocks UPDATE/DELETE. `verify` recomputes. **`client_ts` and `client_event_id` are NEVER part of the hash input (D24).** `cmd/seedcheck` generates the seeded chains and validates them (golden vectors committed for the Python-side tests).

## 4. Indexes (verify against real queries)
users(email); meals(kitchen_id,date DESC); attendance(meal_id); production(meal_id); waste(kitchen_id,created_at); surplus_batches(status), (kitchen_id,status), (expiry_at) WHERE status IN ('AVAILABLE','PENDING_SAFETY','HOLD'); matches(batch_id), (recipient_id,status); routes(batch_id); route_stops(route_id,sequence); qr_events(batch_id,created_at); qr_events(client_event_id) WHERE client_event_id IS NOT NULL; processing_metrics(kitchen_id,date DESC); sensor_readings(location_id,ts DESC), (batch_id,ts); alerts(kitchen_id,created_at DESC) WHERE acknowledged_at IS NULL; prediction_log(kitchen_id,created_at DESC); quality_checks(batch_id,created_at DESC); refresh_tokens(user_id), (token_hash), (expires_at); device_tokens(user_id) WHERE active.

## 5. Views
`v_training_set` joins meals+attendance+production+waste+calendar for ML export (date, kitchen_id, day_of_week ISO, meal_type, menu, attendance, prepared_qty, consumed_qty, surplus_qty, waste_qty). `v_daily_waste`, `v_impact_summary` (waste avoided, redistributed, diverted by day).

## 6. Seeds (demo story)
Users (password `demo123` bcrypt): kitchen@, ngo@, logistics@, admin@example.com, plus `sensor@system`. 1 hostel kitchen, 1 processing unit. 10 recipients around a real Indian city centre (mark as demo). 60 days × 3 meals with weekday pattern. Waste rows across all 7 causes. 8 surplus batches in different statuses. 2 completed QR chains with valid hashes (**generated via `cmd/seedcheck`** — the Go function, not hand-written SQL; one chain's events carry client_ts values to demo offline provenance). 30 processing days, days 12 and 23 anomalous. Sensor stream with excursion at a fixed timestamp. Two demo device tokens (stub strings, pruned by the push worker in mock mode).

## 7. Performance and ops
Connection pool 10–20 (pgx pool config); `statement_timeout` 5 s for API role; separate read-only role for ML export; nightly `pg_dump`; `ANALYZE` after seed; enable `pg_stat_statements` in prod compose.

## 8. Tests (in the backend Go suite, run against a testcontainers Postgres)
Empty DB → migrate up to head; migrate down one step works; every FK rejects orphan; CHECKs reject negatives; UPDATE on qr_events raises; seed minimums (users 5, recipients ≥8, meals ≥60×3, waste causes =7, surplus ≥8, qr full chains ≥2, processing ≥30, sensors ≥1 excursion); `v_training_set` non-empty; hash chain verifies for seeded batches; duplicate client_event_id insert → unique violation; refresh token reuse-family revoke cascade; deleting a user cascades refresh_tokens+device_tokens but RESTRICTS content rows.

## 9. Handoffs
To P1: `schema.md`, ERD, reset command, connection string template. To P4: `v_training_set` + `export_training_data.py`. To P5: `sensor_readings` and `quality_checks` columns. Any change → "SCHEMA CONFLICT / CURRENT / PROPOSED CHANGE" note.

## 10. Definition of done
Fresh clone → `make reset` → Go backend starts → every endpoint in 08_API_CONTRACT has its table; schema.md matches `\d+`; CI green including mobile-constraint tests.