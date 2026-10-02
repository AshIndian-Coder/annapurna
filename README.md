# Annapurna Backend — AI-Powered Food Waste Mitigation & Surplus Redistribution Platform
**Project Reference:** SIH26234 | Ministry of Food Processing Industries  
**Stack:** Go 1.23+ (Chi Router), PostgreSQL 16 + PostGIS, Redis 7, Asynq, Docker  

---

## Part 1: Quick Setup & Getting Started Guide

If you want to run and test the backend on your own machine, follow these simple steps:

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (running)
- [Go 1.23+](https://go.dev/dl/) installed
- Windows PowerShell (or bash on Linux/macOS)

---

### Step 1: Start Database & Cache Containers
Spin up PostgreSQL (with PostGIS) on port `5433` and Redis on port `6379`:
```powershell
docker compose up -d
```
Verify the containers are healthy:
```powershell
docker ps
```

---

### Step 2: Environment Configuration
Copy the sample environment file to `.env`:
```powershell
Copy-Item .env.example .env
```
*(The default `.env` is already pre-configured to match the Docker container ports, JWT secrets, and demo settings).*

---

### Step 3: Build and Launch the Backend Server
Compile the Go API server and run the binary:
```powershell
# Build binary
go build -o bin/server.exe ./cmd/server

# Run server (runs on http://localhost:8080)
.\bin\server.exe
```

When started, you should see:
```text
{"time":"...","level":"INFO","msg":"connected to postgres"}
{"time":"...","level":"INFO","msg":"connected to redis"}
{"time":"...","level":"INFO","msg":"Go backend server starting","port":8080}
```

---

### Step 4: Run the Automated End-to-End Test Suite
In a **second PowerShell terminal**, run the automated test script:
```powershell
.\test_suite.ps1
```
This script executes all 10 core API workflows automatically and displays live green status checks.

---

### Demo User Accounts (Pre-Seeded)
| Role | Email | Password | Scope |
|---|---|---|---|
| **Kitchen** | `kitchen@example.com` | `demo123` | Institutional food production, surplus logging, quality checks |
| **NGO** | `ngo@example.com` | `demo123` | Surplus claim, delivery acceptance, distribution tracking |
| **Admin** | `admin@example.com` | `demo123` | Safety overrides, system audits, reporting |

---

## Part 2: API Testing Report

### 1. What Testing We Have Done
We performed end-to-end integration testing covering the complete critical path of the food surplus lifecycle:
1. **Liveness & Dependency Readiness:** Verified process uptime and active database/cache connections.
2. **Authentication & Token Lifecycle:** Verified password bcrypt verification, JWT claims encoding, and profile retrieval.
3. **Surplus Food Batch Logging:** Created real surplus batches in PostgreSQL with timestamps, weight ($kg$), and safety statuses.
4. **Food Safety Quality Assurance:** Evaluated temperature and visual checks against safety thresholds.
5. **Human Inspection & Approval:** Approved food batches to transition status from `PENDING_SAFETY` to `AVAILABLE`, generating cryptographic SHA-256 custody chain hashes.
6. **Food Diversion:** Tested diversion of expired or spoiled food to non-human channels (`DIVERTED`).
7. **Offline-First Synchronization:** Replayed batch transactions simulated from offline field environments with UUIDv7 deduplication.

---

### 2. Live Test Results Table

| # | Test Case / Workflow | HTTP Method & Route | Expected Result | Actual Result | Status |
|---|---|---|---|---|---|
| **1** | Process Health Probe | `GET /health` | HTTP 200 `{"status":"ok"}` | `{"status":"ok"}` | **PASS ✅** |
| **2** | Deep Dependency Probe | `GET /ready` | HTTP 200, DB=True, Redis=ok | `{"database":true,"redis":"ok","status":"ready"}` | **PASS ✅** |
| **3** | Kitchen Login | `POST /api/v1/auth/login` | HTTP 200, Signed JWT token | JWT access token received | **PASS ✅** |
| **4** | User Profile Identity | `GET /api/v1/auth/me` | HTTP 200, Role=KITCHEN | User: `kitchen@example.com`, Role: `KITCHEN` | **PASS ✅** |
| **5** | Create Surplus Batch | `POST /api/v1/surplus` | HTTP 201, Status=PENDING_SAFETY | Batch created (ID: `e19c242c-...`, 15.0 kg) | **PASS ✅** |
| **6** | Quality Safety Check | `POST /api/v1/quality/check` | HTTP 200, Visual & Temp evaluation | Visual: `GOOD`, Check stored in DB | **PASS ✅** |
| **7** | Human Batch Approval | `POST /api/v1/surplus/{id}/approve` | HTTP 200, Status=APPROVED | Batch transitioned to `AVAILABLE` with SHA-256 hash | **PASS ✅** |
| **8** | Food Diversion Route | `POST /api/v1/surplus/{id}/divert` | HTTP 200, Status=DIVERTED | Batch transitioned to `DIVERTED` | **PASS ✅** |
| **9** | Multi-Tenant Surplus List | `GET /api/v1/surplus` | HTTP 200, List of batches | Returns all active kitchen batches | **PASS ✅** |
| **10** | Offline Batch Replay | `POST /api/v1/sync/batch` | HTTP 200, Applied=1, Duplicates=0 | Applied: 1, Duplicates: 0 | **PASS ✅** |

---

### 3. Live Execution Terminal Output
```text
==========================================================
         ANNAPURNA BACKEND END-TO-END TEST SUITE          
==========================================================
[1/10] GET /health                      => PASS (Status: ok)
[2/10] GET /ready                       => PASS (DB: True, Redis: ok)
[3/10] POST /api/v1/auth/login          => PASS (JWT Token Acquired)
[4/10] GET /api/v1/auth/me              => PASS (User: kitchen@example.com, Role: KITCHEN)
[5/10] POST /api/v1/surplus (Batch 1)   => PASS (ID: e19c242c-b846-4411-9a88-0daf717e5741, Status: PENDING_SAFETY)
[6/10] POST /api/v1/quality/check       => PASS (Visual: GOOD)
[7/10] POST /api/v1/surplus/{id}/approve => PASS (New Status: APPROVED)
       Verification: Batch 1 Detail     => Status: AVAILABLE (ApprovedBy: 11111111-1111-1111-1111-111111111111)
[8/10] POST /api/v1/surplus/{id}/divert  => PASS (Diverted ID: 369e9fc4-b829-4eab-be2c-24c9ec3561b7, Status: DIVERTED)
[9/10] GET /api/v1/surplus (List)        => PASS (Found 4 batches total)
[10/10] POST /api/v1/sync/batch          => PASS (Applied: 1, Duplicates: 0)
==========================================================
          ALL 10 END-TO-END TESTS PASSED 100%!            
==========================================================
```

---

### 4. Remaining Testing & Roadmap
The following secondary and integration endpoints are scheduled for subsequent test phases:
1. **Geospatial Proximity Matching:** Testing PostGIS radius queries matching `AVAILABLE` surplus batches with candidate NGOs.
2. **Vehicle Routing Problem (VRP) Integration:** Testing multi-stop driver route generation via the Python OR-Tools sidecar.
3. **FCM Push Notification Dispatch:** Testing mobile device token registration and push delivery over Firebase Cloud Messaging.
4. **Automated Asynq Expiry Sweep:** Testing the 1-minute background cron job that auto-expires uncollected batches.
5. **ESG Compliance PDF Generation:** Testing downloadable sustainability reports generated via `maroto/v2`.
