# Project Annapurna — Comprehensive Progress & Status Report
**Project Identification:** SIH26234 (Ministry of Food Processing Industries / Food Waste Mitigation)  
**System:** High-Performance Go Backend, Real-Time Surplus Redistribution & Quality Assurance Engine  
**Date of Report:** October 2026  
**Status:** Core Backend Operational, Authenticated Database Integration Verified, Phase 1 & 2 Milestones Implemented  

---

## 1. Executive Summary & Project Goal

**Annapurna** is an AI-driven, mobile-first institutional food surplus prediction, quality verification, and redistribution platform. In large-scale food service operations (university messes, corporate cafeterias, institutional kitchens, catering facilities), substantial quantities of prepared meals are routinely discarded due to demand-forecasting errors, rigid safety uncertainties, and logistics friction.

### Core Objectives:
1. **Institutional Kitchen Demand & Surplus Prediction:** Forecast production requirements using quantile regression to minimize waste before cooking begins.
2. **AI & Computer Vision Food Quality Assurance:** Verify food freshness using on-device image gating and server-side CV inference combined with time-temperature fusion sensors.
3. **Real-Time Fair NGO Redistribution:** Instantly match verified edible surplus with nearby verified NGOs and shelters based on need, capacity, and travel-time windows.
4. **Tamper-Evident Custody Verification:** Maintain a cryptographic SHA-256 QR hash chain across every step of physical custody (Kitchen $\rightarrow$ Transporter $\rightarrow$ NGO).
5. **Offline-First Resilience:** Ensure field staff in low-connectivity areas can log waste, scan QR handoffs, and queue transactions with guaranteed idempotency upon reconnection.
6. **ESG & Sustainability Analytics:** Quantify greenhouse gas emissions avoided ($kg\ CO_2e$), meal equivalents salvaged, and landfill diversion for regulatory compliance and donor reporting.

---

## 2. Architecture & Technology Stack

The backend has been re-architected into a high-throughput, memory-safe Go service conforming strictly to frozen API contracts.

```
                   +------------------------------------+
                   |     Flutter Mobile App (Clients)   |
                   |   Kitchen | NGO | Driver | Admin   |
                   +-----------------+------------------+
                                     |
                                     | HTTPS / REST / SSE
                                     v
+-------------------------------------------------------------------------------+
|                             Go Backend (Chi Router)                           |
|                                                                               |
|  [Middleware: RequestID | Recoverer | RealIP | Auth/JWT | RBAC | Logger]     |
|                                                                               |
|  +--------------------+   +--------------------+   +----------------------+   |
|  |   Auth Service     |   |  Surplus Service   |   |   Quality Service    |   |
|  | (Bcrypt, JWT, RBAC)|   |  (State Machine)   |   |  (CV + Sensor Fusion)|   |
|  +--------------------+   +--------------------+   +----------------------+   |
|  |  QR Chain Service  |   |    Sync Service    |   |     SSE Event Hub    |   |
|  |  (SHA-256 Custody) |   | (Offline Outbox)   |   | (Real-Time Streams)  |   |
|  +--------------------+   +--------------------+   +----------------------+   |
+-------------------+--------------------+--------------------+-----------------+
                    |                    |                    |
                    v                    v                    v
          +------------------+  +------------------+  +------------------+
          |    PostgreSQL    |  |     Redis 7      |  | Python Sidecar   |
          |   with PostGIS   |  | DB0: Cache/SSE   |  | (mlserving:8001) |
          | (Spatial Tables) |  | DB1: Asynq Queue |  | ML & OR-Tools    |
          +------------------+  +------------------+  +------------------+
                                         ^
                                         |
                                +------------------+
                                |   Asynq Worker   |
                                | (cmd/worker/main)|
                                |  Cron / Sweepers |
                                +------------------+
```

### Technology Matrix:
| Layer | Technology | Purpose |
|---|---|---|
| **Language & Runtime** | Go 1.23+ | High concurrency, zero GC pauses on critical paths, single binary distribution. |
| **HTTP Framework** | `go-chi/chi/v5` | Lightweight, idiomatic HTTP router with zero external allocations. |
| **Database** | PostgreSQL 16 + PostGIS | Relational integrity, spatial GIS indexing for proximity matching, jsonb telemetry. |
| **Connection Pooling** | `jackc/pgx/v5` | High-performance PostgreSQL driver and connection pooling. |
| **In-Memory & Cache** | Redis 7 | DB 0: Distributed locks, rate-limiting, session blacklisting, SSE pub/sub.<br>DB 1: Asynq queue storage. |
| **Async Task Worker** | `hibiken/asynq` | Reliable distributed background task queue and cron management. |
| **Security & Auth** | `golang-jwt/v5`, `bcrypt` | Signed HS256 JWT access tokens, salted passwords, strict RBAC enforcement. |
| **Tamper-Evident Proof** | Custom `internal/qrchain` | Cryptographic hash chaining: $H(prev\_hash \parallel batch\_id \parallel event \parallel actor \parallel ts \parallel evidence)$. |
| **Observability** | `log/slog`, Prometheus | Structured JSON logging with trace context, `/metrics`, `/health`, `/ready`. |
| **ML Inference Bridge** | `internal/mlclient` | Non-blocking HTTP client with circuit breaker (`sony/gobreaker`) and mock fallbacks. |

---

## 3. What Has Been Completed (Current Progress)

### A. Infrastructure & Database Initialization
- [x] **Docker Compose Stack:** PostgreSQL 16 with PostGIS extensions and Redis 7 containerized with healthy persistent volumes (`docker-compose.yml`).
- [x] **Configuration Management:** Environment loader (`.env` and `.env.example`) parsing database credentials, JWT parameters, redis URLs, and feature flags without overwriting system variables.
- [x] **Database Schema Migration & Alignment:**
  - Configured and executed PostgreSQL schema defining `organizations`, `users`, `kitchens`, `surplus_batches`, `quality_assessments`, `distribution_offers`, `routes`, `qr_events`, and `audit_logs`.
  - Resolved table constraint mismatches: added missing `batch_code`, `meal_id`, `food_name`, `food_category`, and `prepared_at` columns.
  - Aligned column naming (`expiry_at`) and foreign key relationships (`surplus_batches_kitchen_id_fkey`).
- [x] **Database Connection Pooling:** Implemented thread-safe `store.NewPool` utilizing `pgxpool` with automatic reconnection retry logic.

### B. Authentication, Security & RBAC
- [x] **Production Password Hashing:** Implemented Bcrypt hashing (cost factor 10) for secure password storage.
- [x] **Real Database Authentication:**
  - Migrated from temporary mock authentication stubs to real PostgreSQL credential lookups.
  - Active user verification (`is_active` validation) and structured credential validation.
- [x] **JWT Token Generation & Verification:**
  - Signed HS256 claims containing User UUID, Organization UUID, and User Role (`KITCHEN`, `NGO`, `ADMIN`, `LOGISTICS`).
  - `/api/v1/auth/login` issuing fresh access tokens.
  - `/api/v1/auth/me` returning current authenticated user identity and role.
  - Token revocation and logout endpoints (`/auth/logout`, `/auth/logout-all`).
- [x] **Seeded Demo Accounts:** Reset and verified working credentials in the local database:
  - **Kitchen:** `kitchen@example.com` / `demo123`
  - **NGO:** `ngo@example.com` / `demo123`
  - **Admin:** `admin@example.com` / `demo123`

### C. Surplus Batch Management
- [x] **Batch Lifecycle State Machine:** Enforced state transitions (`PENDING_SAFETY` $\rightarrow$ `AVAILABLE` $\rightarrow$ `MATCHED` $\rightarrow$ `IN_TRANSIT` $\rightarrow$ `DELIVERED`, plus `HOLD` and `DIVERTED`).
- [x] **Dynamic Claim Extraction:** Updated `internal/httpapi/handlers/surplus.go` to parse user and kitchen UUIDs directly from validated JWT claims instead of hardcoded strings.
- [x] **Creation & Listing:**
  - `POST /api/v1/surplus`: Create new surplus records with food category, quantity ($kg$), preparation timestamp, and expiry deadline.
  - `GET /api/v1/surplus`: List batches with multi-tenant filtering (kitchen isolation).
  - `GET /api/v1/surplus/{id}`: Detailed batch status query.
- [x] **Approval & Diversion:**
  - `POST /api/v1/surplus/{id}/approve`: Authorized human review transition.
  - `POST /api/v1/surplus/{id}/divert`: Route surplus to secondary channels (composting/animal feed/biogas) if unsafe for direct consumption.

### D. Quality Verification & AI Client
- [x] **Quality Assessment Endpoint (`POST /api/v1/quality/check`):**
  - Accepts image evidence and sensor readings.
  - Performs multi-factor assessment combining visual confidence and temperature thresholding.
- [x] **ML Serving Client with Circuit Breaker:**
  - Built `internal/mlclient` targeting the Python inference sidecar (`:8001`).
  - Fallback logic: When ML sidecar is unavailable, graceful mock fallbacks return contract-valid responses with safety flags set to require human inspection, ensuring zero system crashes.

### E. Cryptographic QR Custody & Tamper Evidence
- [x] **Cryptographic Hash Chaining (`internal/qrchain`):**
  - Generates immutable hashes per handoff: $H_n = \text{SHA256}(H_{n-1} + \text{batch\_id} + \text{event\_type} + \text{actor\_id} + \text{timestamp} + \text{evidence\_hash})$.
  - Verification algorithm detecting any broken links, missing steps, or out-of-order handoffs.

### F. Offline-First Synchronization Engine
- [x] **Batch Sync Engine (`POST /api/v1/sync/batch`):**
  - Replays offline client actions (created while in airplane mode or network dead-zones).
  - Enforces client event UUIDv7 deduplication using Redis cache keys and database uniqueness.
  - FIFO event sequencing with per-item status response (`ACCEPTED`, `DUPLICATE`, `REJECTED`).

### G. Real-Time Streaming & Background Workers
- [x] **Server-Sent Events (SSE) Hub (`GET /api/v1/events/stream`):**
  - Thread-safe pub/sub hub broadcasting live operational updates (status changes, new surplus batches, match alerts) to connected dashboards.
- [x] **Asynq Worker Service (`cmd/worker/main.go`):**
  - Background worker process connected to Redis DB 1.
  - Multi-queue priority configuration (`critical`, `default`, `low`).
  - Task handlers registered for `TaskExpirySweep` and `TaskOfferExpiry`.

### H. Health & Observability
- [x] **Health Check (`GET /health`):** Lightweight liveness probe.
- [x] **Readiness Check (`GET /ready`):** Deep dependency check verifying PostgreSQL connectivity and Redis responsiveness.
- [x] **Prometheus Metrics (`GET /metrics`):** Exposing request counts, latencies, and service metrics via standard Prometheus exposition format.

---

## 4. What Is In-Progress & Remaining (Next Steps)

| Module / Feature | Current Status | Remaining Work |
|---|---|---|
| **Geospatial Matching Engine** | Architecture designed; service stubbed (`internal/services/matching.go`) | Implement PostGIS nearest-neighbor radius query, scoring algorithm (distance decay, NGO capacity, shelf-life slack, historical fairness balance), and offer creation. |
| **FCM Push Notification Dispatch** | Push service wrapper created (`internal/pushx`) | Wire Firebase Admin SDK credentials and hook background `asynqx` push tasks on new match offers and critical temperature alerts. |
| **Vehicle Routing Problem (VRP) Integration** | Interface defined in `mlclient` | Connect to Python OR-Tools sidecar (`/v1/build-route`) for multi-stop vehicle route optimization with time windows (VRPTW). |
| **Background Cron Schedulers** | Task handlers registered | Activate periodic 1-minute cron for `TaskExpirySweep` (flagging expired food batches under distributed Redis lock) and 30-minute offer expiry timeout. |
| **ESG & Sustainability Report Generation** | Specifications drafted | Integrate `maroto/v2` PDF generation service to render branded ESG compliance certificates and impact reports. |
| **Rate Limiting & Security Hardening** | Basic middleware in place | Implement Redis token-bucket rate limiter per IP/user and enforce `X-App-Version` gate checking. |
| **Mobile Client End-to-End Testing** | Mobile specs frozen | Connect Flutter client to backend API; validate offline sync replay from physical mobile devices. |
| **Automated Test Suite & Load Testing** | Scratch tests verified | Write comprehensive unit and integration test suite using `testcontainers-go`; execute k6 benchmark for 50 concurrent virtual users. |

---

## 5. Active API Reference

| Endpoint | Method | Auth | Description | Status |
|---|---|---|---|---|
| `/health` | GET | None | Basic process liveness probe | ✅ Operational |
| `/ready` | GET | None | Deep health check (DB + Redis connectivity) | ✅ Operational |
| `/metrics` | GET | None | Prometheus telemetry metrics | ✅ Operational |
| `/api/v1/auth/login` | POST | None | Authenticate with email/password; returns JWT | ✅ Operational |
| `/api/v1/auth/me` | GET | Bearer | Retrieve authenticated user profile and role | ✅ Operational |
| `/api/v1/auth/refresh` | POST | None | Refresh expired JWT using refresh token | ✅ Operational |
| `/api/v1/auth/logout` | POST | Bearer | Revoke current token / session | ✅ Operational |
| `/api/v1/surplus` | POST | Bearer | Create a new surplus food batch | ✅ Operational |
| `/api/v1/surplus` | GET | Bearer | List surplus batches with role filtering | ✅ Operational |
| `/api/v1/surplus/{id}` | GET | Bearer | Get surplus batch details and status | ✅ Operational |
| `/api/v1/surplus/{id}/approve` | POST | Bearer | Human approval of inspected surplus | ✅ Operational |
| `/api/v1/surplus/{id}/divert` | POST | Bearer | Divert batch to secondary non-human channels | ✅ Operational |
| `/api/v1/quality/check` | POST | Bearer | Evaluate food quality & safety | ✅ Operational |
| `/api/v1/sync/batch` | POST | Bearer | Offline event batch synchronization | ✅ Operational |
| `/api/v1/events/stream` | GET | Bearer | SSE real-time event subscription stream | ✅ Operational |

---

## 6. How to Run & Test Locally

### 1. Start Database & Cache
```powershell
docker compose up -d
```

### 2. Build & Launch Backend Server
```powershell
go build -o bin/server.exe ./cmd/server
.\bin\server.exe
```

### 3. Launch Background Worker (In a separate terminal)
```powershell
go build -o bin/worker.exe ./cmd/worker
.\bin\worker.exe
```

### 4. Verification Workflow (PowerShell)
```powershell
# 1. Login as Institutional Kitchen User
$response = Invoke-RestMethod -Method POST -Uri "http://localhost:8080/api/v1/auth/login" `
  -ContentType "application/json" `
  -Body '{"email":"kitchen@example.com","password":"demo123"}'
$token = $response.access_token
Write-Host "Logged in. Token acquired."

# 2. Check User Identity
Invoke-RestMethod -Uri "http://localhost:8080/api/v1/auth/me" `
  -Headers @{Authorization="Bearer $token"}

# 3. Create a Surplus Food Batch
$batchPayload = @{
    food_name     = "Dal & Basmati Rice"
    food_category = "cooked"
    quantity_kg   = 15.0
    expiry_at     = (Get-Date).AddHours(6).ToString("yyyy-MM-ddTHH:mm:ssZ")
    prepared_at   = (Get-Date).AddHours(-1).ToString("yyyy-MM-ddTHH:mm:ssZ")
} | ConvertTo-Json

Invoke-RestMethod -Method POST -Uri "http://localhost:8080/api/v1/surplus" `
  -Headers @{Authorization="Bearer $token"} `
  -ContentType "application/json" `
  -Body $batchPayload

# 4. List All Surplus Batches
Invoke-RestMethod -Uri "http://localhost:8080/api/v1/surplus" `
  -Headers @{Authorization="Bearer $token"}
```

---
*Report generated and archived in `KnowMore/` for internal tracking, SIH evaluation, and project handover.*
