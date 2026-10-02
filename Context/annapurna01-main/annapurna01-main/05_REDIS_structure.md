# 05 — REDIS STRUCTURE

```
redis/
├── redis.conf              # appendonly yes, maxmemory 256mb, policy noeviction (queue DB) ; cache DB uses app-level TTLs
├── KEYS.md                 # catalogue of keys, TTLs, owners (incl. asynq:* keys — owned by Asynq, listed not managed)
└── scripts/flush_demo.sh   # clears cache/limits, keeps queues

backend/internal/
├── redisx/                 # go-redis v9 client (DB0 cache/limits, DB1 queue/streams), health ping, graceful degrade
│   ├── cache.go            # @Cached-style helper (ttl, keyFn), invalidate(prefix)
│   ├── ratelimit.go        # sliding-window limiter (Lua, same scripts as v2)
│   ├── idempotency.go      # Idempotency-Key store/replay
│   ├── locks.go            # SET NX PX distributed lock + compare-and-delete Lua release
│   ├── pubsub.go           # publish(event)/subscribe — SSE fan-out + app-config refresh
│   ├── syncdedupe.go       # client_event_id dedupe fast path (DB UNIQUE is the final guard)
│   └── dangerzone.go       # Lua pipeline danger-zone clock per batch
├── asynqx/                 # Asynq task definitions + handlers (DB1)
│   ├── worker.go           # server config, queues, graceful drain ≤10 s
│   ├── jobs_reports.go     # ESG PDF (maroto)
│   ├── jobs_push.go        # FCM sends (chunks ≤500, 429 Retry-After) + invalid-token pruning
│   ├── jobs_sensors.go     # XREADGROUP consumer → pgx COPY bulk insert (1 s/500 rows)
│   └── cron.go             # expiry_sweep (locked), offer_expiry, stale_sensor_check, drift_check→mlserving
└── sse/                    # chi handler: subscribe chan:kitchen:*, Last-Event-ID via events:recent:{kitchen}

backend/internal/**/*_test.go  # limiter, cache, idempotency, lock, sync-dedupe, push-queue, sensor-stream, degrade-without-redis
```
Compose service: `redis:7-alpine`, healthcheck `redis-cli ping`, volume `redis-data`. Asynq stores its queues in DB1 with `asynq:*` keys — documented in KEYS.md, never manually flushed.