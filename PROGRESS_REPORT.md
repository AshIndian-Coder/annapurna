# Project Annapurna — Progress Report
**System:** High-Performance Go Backend, Real-Time Surplus Food Redistribution & Quality Assurance Engine  
**Project Reference:** SIH26234 (Ministry of Food Processing Industries / Food Waste Mitigation)  
**Last Updated:** October 2026  

---

## 1. What We Are Trying to Build (The Goal & Vision)

**Annapurna** is an AI-powered, mobile-first institutional food surplus prediction, quality verification, and redistribution platform. In large-scale dining operations (university messes, corporate cafeterias, institutional kitchens, hospitals, and large events), massive quantities of edible food are routinely discarded due to inaccurate demand forecasting, rigid safety uncertainty, and logistical friction.

### Core Objectives & System Vision:
1. **Demand Forecasting & Waste Prevention:** Predict production requirements before cooking using quantile regression, reducing overproduction at the source.
2. **AI & Computer Vision Food Quality Assurance:** Verify food freshness and safety using on-device image gating and server-side CV inference combined with time-temperature IoT sensor fusion.
3. **Automated Fair NGO Redistribution:** Instantly match verified edible surplus with nearby NGOs, shelters, and orphanages based on real-time need, capacity, and travel-time windows.
4. **Tamper-Evident Custody Verification:** Maintain a cryptographic SHA-256 QR hash chain across every step of physical custody (Kitchen $\rightarrow$ Transporter $\rightarrow$ NGO).
5. **Offline-First Resilience:** Ensure field staff in low-connectivity areas can log waste, scan QR handoffs, and queue transactions with guaranteed idempotency upon reconnection.
6. **Secondary Recovery Hierarchy:** Automatically divert food unsafe for direct human consumption to animal feed, composting, or biogas production.
7. **ESG & Sustainability Analytics:** Quantify greenhouse gas emissions avoided ($kg\ CO_2e$), meal equivalents salvaged, and landfill diversion for regulatory compliance and donor reporting.

---

## 2. What Is Done (Completed & Verified)

### A. Infrastructure & Database Layer
- [x] **PostgreSQL 16 & PostGIS:** Relational data store containerized via Docker with spatial indexing for geo-proximity calculations.
- [x] **Redis 7 Cache & Message Broker:** DB 0 configured for caching, distributed locks, and real-time SSE streaming; DB 1 configured for Asynq background workers.
- [x] **Database Schema & Migrations Aligned:**
  - Designed and created core tables: `organizations`, `users`, `kitchens`, `surplus_batches`, `quality_checks`, `audit_log`, `qr_events`, `distribution_offers`, `routes`.
  - Added and verified required columns: `batch_code`, `food_name`, `meal_id`, `food_category`, `prepared_at`, `expiry_at`.
  - Configured and tested `quality_checks` schema with visual status, risk level, confidence scores, and safety decision fields.
  - Resolved foreign-key constraints linking surplus batches to registered kitchens.
- [x] **Connection Pooling:** High-throughput `pgxpool` thread-safe connection pool with automatic retry logic.

### B. Authentication, Security & RBAC
- [x] **Database Credential Authentication:** Real PostgreSQL authentication using salted Bcrypt password hashing.
- [x] **JWT Access Token Issuance:** Signed HS256 tokens carrying user UUID, organization UUID, and role claims (`KITCHEN`, `NGO`, `ADMIN`, `LOGISTICS`).
- [x] **Identity Verification (`/api/v1/auth/me`):** Authenticated endpoint returning user profile and role.
- [x] **Seed Credentials Verified:** Pre-seeded demo accounts operational (`kitchen@example.com` / `demo123`).

### C. Surplus Batch Lifecycle State Machine
- [x] **Surplus Batch Creation (`POST /api/v1/surplus`):** Inserts batch records into PostgreSQL with dynamic UUID parsing from token claims.
- [x] **Surplus Listing (`GET /api/v1/surplus`):** Returns active batches with role-based multi-tenant scoping.
- [x] **Surplus Detail (`GET /api/v1/surplus/{id}`):** Retrieves batch details, expiration deadlines, and approval history.
- [x] **Food Safety Quality Check (`POST /api/v1/quality/check`):** Multi-factor evaluation combining food image analysis and temperature telemetry.
- [x] **Human Inspection Approval (`POST /api/v1/surplus/{id}/approve`):** Human review transition advancing food status from `PENDING_SAFETY` to `AVAILABLE`, generating cryptographic SHA-256 audit proof.
- [x] **Food Diversion (`POST /api/v1/surplus/{id}/divert`):** Transitions unsafe food to `DIVERTED` for composting or biogas.

### D. Offline-First Synchronization & Background Architecture
- [x] **Batch Synchronization (`POST /api/v1/sync/batch`):** Replays offline field transactions with client UUIDv7 deduplication and FIFO order.
- [x] **Real-Time SSE Event Hub (`GET /api/v1/events/stream`):** Live Server-Sent Events hub broadcasting status changes and alerts.
- [x] **Asynq Worker Skeleton (`cmd/worker/main.go`):** Background task worker configured with prioritized queues (`critical`, `default`, `low`).
- [x] **Health & Monitoring Probes:** Operational `/health`, `/ready`, and `/metrics` (Prometheus) endpoints.
- [x] **Automated Test Suite:** Created `test_suite.ps1` running 10 comprehensive tests verifying all core workflows live.

---

## 3. What Is Remaining (Roadmap & Next Steps)

| Module / Milestone | Description | Priority |
|---|---|---|
| **Geospatial Matching Engine** | Complete PostGIS nearest-neighbor radius query to rank and match available surplus with candidate NGOs based on travel distance, capacity, shelf-life slack, and 48-hour fairness penalty. | **High** |
| **Vehicle Routing Problem (VRP)** | Wire client to the Python OR-Tools sidecar (`/v1/build-route`) to generate multi-stop route plans with time windows for logistics drivers. | **High** |
| **Active Worker Schedulers** | Enable 1-minute cron for `TaskExpirySweep` (flagging expired batches under distributed Redis lock) and 30-minute offer timeout check. | **Medium** |
| **FCM Push Notification Dispatch** | Connect Firebase Admin SDK credentials in `internal/pushx` for instant mobile push delivery when new surplus matches are generated. | **Medium** |
| **ESG PDF Report Generator** | Implement `maroto/v2` PDF generation service to produce downloadable ESG compliance certificates and sustainability analytics. | **Medium** |
| **Flutter Mobile App Integration** | Hook up the single Flutter client (Kitchen, NGO, Driver, Admin) to the live Go API endpoints and test offline outbox replay from physical devices. | **Medium** |
| **Rate Limiting & Security Hardening** | Implement Redis token-bucket rate limiter per IP/client token and enforce minimum app version checks via `X-App-Version`. | **Low** |
