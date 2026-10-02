# Project Annapurna — API Testing & Verification Report
**Target Environment:** Localhost (`http://localhost:8080`)  
**Automated Test Suite:** `test_suite.ps1`  
**Execution Date:** October 2026  
**Status:** 10/10 Tests Passed (100% Success Rate)  

---

## Part 1: Live Test Results & Verification Summary

Every core endpoint of the backend has been thoroughly tested and verified against the live PostgreSQL database and Redis engine.

### Verified Results Table

| # | Test Case / Workflow | HTTP Method & Route | Expected Result | Actual Result | Status |
|---|---|---|---|---|---|
| **1** | Process Health Probe | `GET /health` | HTTP 200 `{"status":"ok"}` | `{"status":"ok"}` | **PASS ✅** |
| **2** | Deep Dependency Probe | `GET /ready` | HTTP 200, DB=True, Redis=ok | `{"database":true,"redis":"ok","status":"ready"}` | **PASS ✅** |
| **3** | Kitchen User Authentication | `POST /api/v1/auth/login` | HTTP 200, Signed JWT token | JWT access token received | **PASS ✅** |
| **4** | User Identity Profile | `GET /api/v1/auth/me` | HTTP 200, Role=KITCHEN | User: `kitchen@example.com`, Role: `KITCHEN` | **PASS ✅** |
| **5** | Log Surplus Food Batch | `POST /api/v1/surplus` | HTTP 201, Status=PENDING_SAFETY | Batch created (ID: `e19c242c-...`, 15.0 kg) | **PASS ✅** |
| **6** | Food Safety Quality Check | `POST /api/v1/quality/check` | HTTP 200, Visual & Temp evaluation | Visual: `GOOD`, Check stored in DB | **PASS ✅** |
| **7** | Human Inspection Approval | `POST /api/v1/surplus/{id}/approve` | HTTP 200, Status=APPROVED | Batch transitioned to `AVAILABLE` with SHA-256 hash | **PASS ✅** |
| **8** | Secondary Food Diversion | `POST /api/v1/surplus/{id}/divert` | HTTP 200, Status=DIVERTED | Batch transitioned to `DIVERTED` | **PASS ✅** |
| **9** | Multi-Tenant Surplus List | `GET /api/v1/surplus` | HTTP 200, List of batches | Returns active kitchen batches from DB | **PASS ✅** |
| **10** | Offline Batch Replay | `POST /api/v1/sync/batch` | HTTP 200, Applied=1, Duplicates=0 | Applied: 1, Duplicates: 0 | **PASS ✅** |

---

### Live Execution Terminal Output

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

## Part 2: How to Test the System Yourself (Setup Guide)

If you want to run the tests on your machine, follow these steps:

### 1. Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (must be running)
- [Go 1.23+](https://go.dev/dl/) installed
- Windows PowerShell

### 2. Start Database & Redis
In the root directory of the repository, start the Docker containers:
```powershell
docker compose up -d
```

### 3. Build & Run the Go Backend Server
```powershell
# Compile the binary
go build -o bin/server.exe ./cmd/server

# Start the server (runs on port 8080)
.\bin\server.exe
```
Leave this terminal window open. You will see:
```text
{"level":"INFO","msg":"connected to postgres"}
{"level":"INFO","msg":"connected to redis"}
{"level":"INFO","msg":"Go backend server starting","port":8080}
```

---

## Part 3: Testing the Endpoints One-by-One

Open a **second PowerShell window** and run each command step-by-step:

### 1. Check Server Liveness
```powershell
Invoke-RestMethod -Uri "http://localhost:8080/health"
```
**Expected Output:**
```json
{
  "status": "ok"
}
```

---

### 2. Check Database & Redis Readiness
```powershell
Invoke-RestMethod -Uri "http://localhost:8080/ready"
```
**Expected Output:**
```json
{
  "database": true,
  "redis": "ok",
  "status": "ready"
}
```

---

### 3. Log In & Obtain Access Token
```powershell
$response = Invoke-RestMethod -Method POST -Uri "http://localhost:8080/api/v1/auth/login" `
  -ContentType "application/json" `
  -Body '{"email":"kitchen@example.com","password":"demo123"}'

$token = $response.access_token
Write-Host "Token obtained: $token"
```
**Expected Output:**
A signed JWT token string is returned and saved into the `$token` variable for subsequent requests.

---

### 4. Check Current Authenticated User Profile
```powershell
Invoke-RestMethod -Uri "http://localhost:8080/api/v1/auth/me" `
  -Headers @{Authorization="Bearer $token"}
```
**Expected Output:**
```json
{
  "id": "11111111-1111-1111-1111-111111111111",
  "email": "kitchen@example.com",
  "role": "KITCHEN"
}
```

---

### 5. Create a Surplus Food Batch
```powershell
$batchBody = @{
    food_name     = "Paneer Tikka"
    food_category = "cooked"
    quantity_kg   = 12.0
    expiry_at     = (Get-Date).AddHours(5).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    prepared_at   = (Get-Date).AddHours(-1).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
} | ConvertTo-Json

$batch = Invoke-RestMethod -Method POST -Uri "http://localhost:8080/api/v1/surplus" `
  -Headers @{Authorization="Bearer $token"} `
  -ContentType "application/json" `
  -Body $batchBody

$batchId = $batch.id
Write-Host "Created Batch ID: $batchId"
```
**Expected Output:**
Returns HTTP 201 with `id`, `batch_code` (e.g. `B-0f2e9cfa`), and `status: "PENDING_SAFETY"`.

---

### 6. Perform a Food Quality & Temperature Check
```powershell
$boundary = [System.Guid]::NewGuid().ToString()
$LF = "`r`n"
$body = (
    "--$boundary",
    'Content-Disposition: form-data; name="batch_id"',
    '',
    $batchId,
    "--$boundary",
    'Content-Disposition: form-data; name="temperature_c"',
    '',
    '65.0',
    "--$boundary--",
    ''
) -join $LF

Invoke-RestMethod -Method POST -Uri "http://localhost:8080/api/v1/quality/check" `
  -Headers @{
      Authorization = "Bearer $token"
      "Content-Type" = "multipart/form-data; boundary=$boundary"
  } `
  -Body $body
```
**Expected Output:**
```json
{
  "batch_id": "<batch-id>",
  "visual": {
    "status": "GOOD",
    "risk_level": "LOW",
    "confidence": 0.95
  }
}
```

---

### 7. Approve the Food Batch (Human Inspection)
```powershell
Invoke-RestMethod -Method POST -Uri "http://localhost:8080/api/v1/surplus/$batchId/approve" `
  -Headers @{Authorization="Bearer $token"} `
  -ContentType "application/json" `
  -Body '{"decision":"APPROVE","note":"Food temperature verified safe"}'
```
**Expected Output:**
```json
{
  "batch_id": "<batch-id>",
  "status": "APPROVED"
}
```

---

### 8. Verify the Batch Status is Now `AVAILABLE`
```powershell
Invoke-RestMethod -Uri "http://localhost:8080/api/v1/surplus/$batchId" `
  -Headers @{Authorization="Bearer $token"}
```
**Expected Output:**
```json
{
  "id": "<batch-id>",
  "food_name": "Paneer Tikka",
  "status": "AVAILABLE",
  "approved_by": "11111111-1111-1111-1111-111111111111"
}
```

---

### 9. Divert an Unsafe Batch (Secondary Channel)
```powershell
# Create a second batch to divert
$staleBatch = Invoke-RestMethod -Method POST -Uri "http://localhost:8080/api/v1/surplus" `
  -Headers @{Authorization="Bearer $token"} `
  -ContentType "application/json" `
  -Body (@{
      food_name     = "Expired Soup"
      food_category = "cooked"
      quantity_kg   = 5.0
      expiry_at     = (Get-Date).AddHours(1).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
      prepared_at   = (Get-Date).AddHours(-6).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
  } | ConvertTo-Json)

# Divert the batch
Invoke-RestMethod -Method POST -Uri "http://localhost:8080/api/v1/surplus/$($staleBatch.id)/divert" `
  -Headers @{Authorization="Bearer $token"}
```
**Expected Output:**
```json
{
  "batch_id": "<batch-id>",
  "status": "DIVERTED"
}
```

---

### 10. List All Surplus Batches
```powershell
Invoke-RestMethod -Uri "http://localhost:8080/api/v1/surplus" `
  -Headers @{Authorization="Bearer $token"}
```
**Expected Output:**
Returns a JSON array of all surplus batches logged by this kitchen.

---

### 11. Test Offline Field Event Synchronization
```powershell
$syncPayload = @{
    items = @(
        @{
            client_event_id = [System.Guid]::NewGuid().ToString()
            kind            = "WASTE_LOG"
            payload         = @{
                station_id   = "prep-station-1"
                waste_type   = "organic_peels"
                quantity_kg  = 2.5
                reason       = "preparation trim"
            }
        }
    )
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Method POST -Uri "http://localhost:8080/api/v1/sync/batch" `
  -Headers @{Authorization="Bearer $token"} `
  -ContentType "application/json" `
  -Body $syncPayload
```
**Expected Output:**
```json
{
  "applied": 1,
  "duplicates": 0,
  "results": [
    {
      "client_event_id": "<uuid>",
      "status": "ACCEPTED"
    }
  ]
}
```

---

## Part 4: Don't Want to Check One by One? Run All Tests at Once!

If you don't want to type commands one by one, you can run the automated test suite in **a single command**.

In your PowerShell terminal, simply run:
```powershell
.\test_suite.ps1
```

It executes all 10 core API workflows sequentially in ~3 seconds and displays verified green status checks directly on your screen!
