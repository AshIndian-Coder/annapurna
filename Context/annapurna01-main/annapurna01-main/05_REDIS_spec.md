# 05 — REDIS SPEC (v3 FINAL — go-redis + Asynq; semantics identical to v2)

## 1. Principle
Redis accelerates and coordinates; **PostgreSQL remains the system of record**. Every Redis use degrades gracefully: if `REDIS_URL` is unset or down, the Go API still works (in-process LRU cache, in-memory limiter, synchronous report generation, direct DB insert of sensors, direct FCM send, DB-unique as the only sync dedupe). `/ready` reports `redis: degraded`, not failed. This is deliberate so a Redis hiccup can't kill the demo.

## 2. Databases
DB0 = cache, rate limits, latest sensor values, idempotency, deny-list (all with TTL, eviction OK). DB1 = Asynq queues + Streams (persistent, `noeviction`; `asynq:*` keys owned by Asynq). Use `appendonly yes`, `appendfsync everysec`.

## 3. Key catalogue
| Key pattern | Type | TTL | Purpose |
|---|---|---|---|
| `rl:login:{ip}` + `rl:login:{device}` | counter | 60 s | 10 login attempts/min per IP AND per device id |
| `rl:quality:{user_id}` | counter | 60 s | 20 CV calls/min |
| `rl:api:{user_id}` | counter | 60 s | 300 req/min default |
| `rl:sync:{user_id}` | counter | 60 s | 120 sync batches/min |
| `jwt:deny:{jti}` | string | token remaining life | logout / revoke |
| `cache:predict:{sha256(payload+model_version)}` | string JSON | 60 s | repeat forecasts |
| `cache:impact:{kitchen_id}` | string JSON | 30 s | dashboard impact |
| `cache:recipients:{kitchen_id}` | string JSON | 300 s | invalidated on recipient change |
| `cache:overview:{kitchen_id}` | string JSON | 15 s | kitchen overview |
| `cache:appconfig` | string JSON | 60 s | /app/config response |
| `sensor:latest:{location_id}` | hash (temp,hum,energy,ts) | 10 min | live monitor without DB hit |
| `dz:{batch_id}` | hash (minutes, last_ts, above) | until batch terminal + 1 d | running danger-zone minutes |
| `stream:sensor` | Stream | trimmed MAXLEN ~ 100k | ingest buffer; group `g_db` |
| `idem:{user}:{key}` | string (status+body) | 24 h | safe retries of POST |
| `sync:dedupe:{user_id}:{client_event_id}` | string (result ref) | 24 h | fast-path offline replay dedupe (DB UNIQUE is the final guard) |
| `lock:expiry_sweep` | string NX | 55 s | one sweeper across replicas |
| `lock:report:{hash}` | string NX | 120 s | dedupe report jobs |
| `chan:kitchen:{id}` | pub/sub | — | SSE fan-out (alerts, status, QR) |
| `chan:system` | pub/sub | — | app-config / bundle-pointer refresh |
| `job:{id}` | hash | 1 h | report job status |
| `model:active:{name}` | string | none | active registry version pointer (pub/sub refresh) |
| `model:bundle:active` | string | none | active CV bundle version + sha256s (D21) |
| `asynq:*` (asynq queues, schedules, states) | various | managed | **owned by Asynq** — documented in KEYS.md, never flushed manually |

## 4. Patterns
**Cache-aside** `GET → miss → compute → SET EX`. Keys include model_version so a promotion invalidates implicitly. Never cache safety decisions or approvals.
**Danger-zone clock** on each sensor reading (Lua pipeline): if temp outside safe band, `HINCRBYFLOAT dz:{batch} minutes dt`, dt capped at 15 min; persisted to `quality_checks.danger_zone_minutes` at decision time; recomputed from `sensor_readings` on cache miss (Redis loss doesn't change decisions).
**Sensor ingest**: `POST /sensors/readings` → validate → `XADD stream:sensor` + `HSET sensor:latest` + publish `chan:kitchen` → Asynq `jobs_sensors` XREADGROUP consumer bulk-COPYs into Postgres every 1 s/500 rows. Fallback without Redis: direct insert.
**SSE**: each API instance subscribes to `chan:kitchen:*` for connected clients; events `{type,data,ts}`; `Last-Event-ID` resume via `events:recent:{kitchen}` (LTRIM 200). Foreground-only on mobile (D23); the Flutter app polls by default.
**Idempotency**: middleware reads `Idempotency-Key`; stores `{status,body}` after first success; replays identical; mismatched payload hash → 422 `IDEMPOTENCY_KEY_REUSED`.
**Sync dedupe**: `SET sync:dedupe:{user}:{client_event_id} result NX` before replay; on hit return DUPLICATE from the stored result ref (or re-fetch by client_event_id from DB). Redis down → rely on DB UNIQUE client_event_id alone.
**Push queue**: `pushx.Send` enqueues an Asynq task with {user_ids, event_type, payload}; `jobs_push` chunks tokens ≤500 per FCM call, honors 429 `Retry-After`, prunes `device_tokens` on UNREGISTERED/INVALID_CREDENTIALS. Push failures are metric-only. Redis down → inline send attempt, else drop (best-effort, D23).
**Rate limit**: sliding window via ZSET + Lua (same scripts as v2); return 429 with `Retry-After`.
**Locks**: `SET key token NX PX ttl` + compare-and-delete Lua on release.

## 5. Config (`redis.conf` highlights)
`appendonly yes`, `maxmemory 256mb`, `maxmemory-policy noeviction` on queue instance (or two instances: `redis-cache` with `allkeys-lru`, `redis-queue` with `noeviction` — preferred in prod compose), `requirepass` from env, bind to compose network only, `timeout 0`, `tcp-keepalive 60`.

## 6. Security and ops
Password + private network; never expose 6379; keys contain ids, never PII or tokens (store only jti; refresh-token HASHES live in Postgres, never Redis); metrics via `redis_exporter`; `make flush-demo` resets cache but not queues.

## 7. Tests
Limiter blocks the 11th login; cache hit/miss and invalidation; idempotent replay; lock exclusion between two tasks; stream consumer inserts rows and acks; sync dedupe returns DUPLICATE on replay; push job chunking + receipt pruning with stubbed FCM; **app works with Redis stopped** (CI job runs the Go test suite with `REDIS_URL=` empty); SSE delivers an event within 1 s.

## 8. Definition of done
All keys above implemented or explicitly deferred in KEYS.md (incl. `asynq:*`); degrade-mode test green; Grafana shows cache hit ratio, queue depth, sync-reject rate and push failures.