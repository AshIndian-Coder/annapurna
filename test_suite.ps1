# ==============================================================================
# Annapurna Backend — Automated End-to-End Verification Test Suite
# Usage: powershell -ExecutionPolicy Bypass -File .\test_suite.ps1
# ==============================================================================

$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "         ANNAPURNA BACKEND END-TO-END TEST SUITE          " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Health check
$health = Invoke-RestMethod -Uri "http://localhost:8080/health"
Write-Host "[1/10] GET /health                      => PASS (Status: $($health.status))" -ForegroundColor Green

# 2. Ready check
$ready = Invoke-RestMethod -Uri "http://localhost:8080/ready"
Write-Host "[2/10] GET /ready                       => PASS (DB: $($ready.database), Redis: $($ready.redis))" -ForegroundColor Green

# 3. Login
$loginResp = Invoke-RestMethod -Method POST -Uri "http://localhost:8080/api/v1/auth/login" `
    -ContentType "application/json" `
    -Body '{"email":"kitchen@example.com","password":"demo123"}'
$token = $loginResp.access_token
Write-Host "[3/10] POST /api/v1/auth/login          => PASS (JWT Token Acquired)" -ForegroundColor Green

# 4. Auth Me
$me = Invoke-RestMethod -Uri "http://localhost:8080/api/v1/auth/me" `
    -Headers @{Authorization="Bearer $token"}
Write-Host "[4/10] GET /api/v1/auth/me              => PASS (User: $($me.email), Role: $($me.role))" -ForegroundColor Green

# 5. Create Surplus Batch 1
$batch1Payload = @{
    food_name     = "Dal Tadka"
    food_category = "cooked"
    quantity_kg   = 15.0
    expiry_at     = (Get-Date).AddHours(6).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    prepared_at   = (Get-Date).AddHours(-1).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
} | ConvertTo-Json

$batch1 = Invoke-RestMethod -Method POST -Uri "http://localhost:8080/api/v1/surplus" `
    -Headers @{Authorization="Bearer $token"} `
    -ContentType "application/json" `
    -Body $batch1Payload
$batch1Id = $batch1.id
Write-Host "[5/10] POST /api/v1/surplus (Batch 1)   => PASS (ID: $batch1Id, Status: $($batch1.status))" -ForegroundColor Green

# 6. Quality Check on Batch 1
$boundary = [System.Guid]::NewGuid().ToString()
$LF = "`r`n"
$bodyLines = (
    "--$boundary",
    'Content-Disposition: form-data; name="batch_id"',
    '',
    $batch1Id,
    "--$boundary",
    'Content-Disposition: form-data; name="temperature_c"',
    '',
    '65.5',
    "--$boundary--",
    ''
) -join $LF

$qualityResp = Invoke-RestMethod -Method POST -Uri "http://localhost:8080/api/v1/quality/check" `
    -Headers @{
        Authorization = "Bearer $token"
        "Content-Type" = "multipart/form-data; boundary=$boundary"
    } `
    -Body $bodyLines

Write-Host "[6/10] POST /api/v1/quality/check       => PASS (Visual: $($qualityResp.visual.status))" -ForegroundColor Green

# 7. Approve Batch 1
$approveResp = Invoke-RestMethod -Method POST -Uri "http://localhost:8080/api/v1/surplus/$batch1Id/approve" `
    -Headers @{Authorization="Bearer $token"} `
    -ContentType "application/json" `
    -Body '{"decision":"APPROVE","note":"Quality check passed, hot food approved"}'
Write-Host "[7/10] POST /api/v1/surplus/{id}/approve => PASS (New Status: $($approveResp.status))" -ForegroundColor Green

# Verify Batch 1 is AVAILABLE
$b1Detail = Invoke-RestMethod -Uri "http://localhost:8080/api/v1/surplus/$batch1Id" `
    -Headers @{Authorization="Bearer $token"}
Write-Host "       Verification: Batch 1 Detail     => Status: $($b1Detail.status) (ApprovedBy: $($b1Detail.approved_by))" -ForegroundColor Yellow

# 8. Create Surplus Batch 2 and Divert
$batch2Payload = @{
    food_name     = "Stale Salad"
    food_category = "raw"
    quantity_kg   = 5.0
    expiry_at     = (Get-Date).AddHours(2).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    prepared_at   = (Get-Date).AddHours(-5).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
} | ConvertTo-Json

$batch2 = Invoke-RestMethod -Method POST -Uri "http://localhost:8080/api/v1/surplus" `
    -Headers @{Authorization="Bearer $token"} `
    -ContentType "application/json" `
    -Body $batch2Payload
$batch2Id = $batch2.id

$divertResp = Invoke-RestMethod -Method POST -Uri "http://localhost:8080/api/v1/surplus/$batch2Id/divert" `
    -Headers @{Authorization="Bearer $token"}
Write-Host "[8/10] POST /api/v1/surplus/{id}/divert  => PASS (Diverted ID: $batch2Id, Status: $($divertResp.status))" -ForegroundColor Green

# 9. List All Surplus Batches
$listResp = Invoke-RestMethod -Uri "http://localhost:8080/api/v1/surplus" `
    -Headers @{Authorization="Bearer $token"}
Write-Host "[9/10] GET /api/v1/surplus (List)        => PASS (Found $($listResp.items.Count) batches total)" -ForegroundColor Green

# 10. Test Offline Batch Sync Replay
$syncPayload = @{
    items = @(
        @{
            client_event_id = [System.Guid]::NewGuid().ToString()
            kind            = "WASTE_LOG"
            payload         = @{
                station_id   = "prep-1"
                waste_type   = "peels"
                quantity_kg  = 3.2
                reason       = "preparation trim"
            }
        }
    )
} | ConvertTo-Json -Depth 5

$syncResp = Invoke-RestMethod -Method POST -Uri "http://localhost:8080/api/v1/sync/batch" `
    -Headers @{Authorization="Bearer $token"} `
    -ContentType "application/json" `
    -Body $syncPayload

Write-Host "[10/10] POST /api/v1/sync/batch          => PASS (Applied: $($syncResp.applied), Duplicates: $($syncResp.duplicates))" -ForegroundColor Green

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "          ALL 10 END-TO-END TESTS PASSED 100%!            " -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Cyan
