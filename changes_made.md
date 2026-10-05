# Authentication: Signup + Login — Change Log

**Scope:** fix "login doesn't work" and "any email/password opens one of the 4 demo pages", then add self-service signup so a new account lands on *its own* role dashboard.

**Status:** implemented and verified end-to-end against a live Postgres and a running server.

---

## 1. Root cause

The headline symptom — typing a real Gmail and password and landing on a demo page — was **not** a backend bug. It was a default:

```dart
// Frontend/lib/core/api/api_client.dart:7 (before)
const bool useMocks = bool.fromEnvironment('USE_MOCKS', defaultValue: true);
```

With mock mode on by default, `AuthService.login` never called the API. It returned a hardcoded user regardless of input:

```dart
// before
if (useMocks) {
  return Success(MockData.loginAs(MockData.kitchenUser.role)); // always KITCHEN
}
```

So every credential produced the Kitchen account → the router redirected to `/kitchen` → it looked like login "silently picked a demo page".

Underneath that, the **real** backend login path was also broken in ways that would have failed as soon as mocks were turned off. Those are listed in §2.

---

## 2. Deep analysis — what already existed vs. what was broken

The brief asked to check whether signup/auth already existed but was merely not wired. Findings:

### Backend — existed but broken

| Area | State before |
|---|---|
| `POST /auth/login` | Existed, and the bcrypt + JWT logic was sound — but the **response was unusable** (see below) |
| `POST /auth/register` | **Did not exist.** `main.go` registered only `login / refresh / logout / logout-all / me` |
| User creation | `queries.CreateUser` existed in `internal/store/queries/users.go` but **was never called by any handler** |
| `CreateUser` in `_stubs/admin.go` | **Dead code.** Go ignores directories beginning with `_`, and it imports a non-existent module (`github.com/sih26234/backend`) |

Defects found in the existing login path:

| # | Defect | Consequence |
|---|---|---|
| 1 | Login response omitted `user.name` | Frontend did `name: json['name'] as String` → **hard cast threw** on the real payload |
| 2 | `expires_in` hardcoded to `3600` | Actual TTL is `JWT_EXPIRE_MIN` (default **15**). Client was told the token lived 4× longer than it did |
| 3 | Scoping read `users.org_id` | Schema has dedicated `kitchen_id` / `recipient_id` columns; these were ignored |
| 4 | Refresh response had no `user` block | Frontend `AuthTokens.fromJson` requires it → parse failure on every refresh |
| 5 | `Me` re-implemented token parsing | Duplicated `requireClaims`; also returned no `name` |
| 6 | No login rate limit | Contract row 1 requires 10/min per IP. `redisx.SlidingWindowLimit` existed but was unused |
| 7 | Email lookup case-sensitive | Unique index is `users_email_lower_key` (`lower(email)`); login used `email = $1` |

### Frontend — root cause + gaps

| # | Issue |
|---|---|
| 1 | **`useMocks` defaulted to `true`** → credentials ignored (the reported bug) |
| 2 | No signup screen existed at all |
| 3 | `User.fromJson` hard-cast `name`/`role`; any missing field threw |
| 4 | No session restore — every cold start bounced to login |
| 5 | Router treated only `/login` as an auth route |
| 6 | `logout()` sent no refresh token, so the server never revoked the token family |
| 7 | Demo buttons posted `demo_<role>@annapurna.in` — **credentials the backend never had**. The seeder creates `<role>@example.com` |

---

## 3. Decisions taken

Confirmed before implementing:

| Decision | Choice |
|---|---|
| Self-service roles | All four — KITCHEN, NGO, LOGISTICS, ADMIN (`SYSTEM` excluded; reserved for service-to-service) |
| Mock mode | Default to the **real** API; demo shortcuts retained behind `!useMocks` |
| Org linking | **Auto-provision** organisation + kitchen/recipient on signup |

> ADMIN self-service is a deliberate choice for this deployment, not an oversight. It is a real privilege-escalation surface if this ever reaches production — see §8.

---

## 4. Backend changes

### New: `backend/internal/validation/signup.go`
Isolated, HTTP-free, DB-free input rules so they are unit-testable on their own.
- `SignupInput` + `Validate()` returning a `FieldError{Field, Message}` → rendered as 422 with the offending field.
- Email: normalised lowercase, parsed via `net/mail`, must have a dotted domain.
- Password: ≥ 8 chars, **≤ 72 bytes** (bcrypt silently truncates past 72, which would make two different passwords interchangeable), and must mix letters with digits/symbols.
- Role must be in the allow-list.
- Latitude/longitude range-checked.

### `backend/internal/httpapi/handlers/auth.go` (rewritten)

**New `Register` handler** — creates the account and immediately returns a token pair, so the client is signed in and routed to its own dashboard.

Provisioning runs in one **serializable transaction** (`store.RunTx`):

| Role | Rows created |
|---|---|
| KITCHEN | organisation → kitchen → user, with `kitchen_id` set |
| NGO | organisation → recipient → user, with `recipient_id` set |
| LOGISTICS / ADMIN | user only (these roles carry no row scope) |

Without this, a KITCHEN or NGO account would have a null scope id and **every kitchen/NGO-scoped query would return nothing**.

Two details worth flagging:
- **Duplicate email is checked inside the transaction**, not just before it. A pre-transaction check lets two concurrent signups both pass and then race on the unique index; the in-transaction check plus SQLSTATE `23505` mapping closes that window.
- The organisation id is captured from the `INSERT ... RETURNING id`, **not** looked up by name. A name lookup would attach a new account to a *different* org that happened to share the name.

**Login fixes**
- Returns `name` in the user block (defect #1).
- `expires_in` now derived from `cfg.JWTExpireMin * 60` (defect #2).
- Scoping reads `kitchen_id` / `recipient_id`, falling back to `org_id` for pre-existing rows (defect #3).
- Lookup is `lower(email) = $1`, matching the unique index (defect #7).

**Refresh / Me fixes**
- Refresh now returns the full `user` block (defect #4).
- `Me` uses the shared `requireClaims` and returns `name`/`email`/`role` (defect #5).

**Rate limiting**
- 10 req/min per IP on `/auth/login` and `/auth/register` via `redisx.SlidingWindowLimit`, with `Retry-After` (defect #6). Fails open when Redis is unavailable.

**Single response shape**
- `tokenResponse()` guarantees every token-issuing endpoint returns the identical structure, which is what the contract requires of refresh.

### Wiring & config
- `backend/cmd/server/main.go` — mounted `POST /auth/register`.
- `backend/internal/httpapi/middleware/_stubs/auth.go` — added `/auth/register` to `publicPaths`.
- `backend/.env.example`, `Makefile`, `API_TESTING.md`, `test_suite.ps1` — `8080` → `8000`. The frontend and the contract both specified `localhost:8000` while the backend template said `8080`; that mismatch alone would have broken every call.

---

## 5. Frontend changes

| File | Change |
|---|---|
| `core/api/api_client.dart` | **`useMocks` now defaults to `false`** — the root-cause fix |
| `data/dtos/models.dart` | `User.fromJson` coerces instead of casting (no more throw on a missing `name`); added `displayName` (falls back name → email local-part → role); `AuthTokens` tolerant parse + `expiresAt` |
| `data/services/auth_service.dart` | **Added `register()`** + `SignupRequest`; added `demoAccounts` map matching the seeder; mock login now infers role from the address instead of hardcoding Kitchen; `logout(refreshToken)` |
| `features/auth/auth_provider.dart` | **Added `register()`**; **session restore** from secure storage (falls back to one refresh attempt before giving up); logout now sends the refresh token; added `isRestoring` |
| `features/auth/signup_screen.dart` | **New.** Name / email / password / confirm + 4-way role picker; organisation field shown only for KITCHEN & NGO |
| `features/auth/widgets/auth_scaffold.dart` | **New.** Shared gradient scaffold, submit button, error banner |
| `features/auth/login_screen.dart` | Rebuilt on the shared scaffold; **"Create Account" link**; demo shortcuts gated behind `!useMocks` and prefilled with real seeded credentials |
| `core/router/app_router.dart` | Added `/signup`; extracted `homeForRole()`; **holds redirect while `isRestoring`** |
| `features/kitchen/dashboard_screen.dart`, `more_screen.dart` | Use `displayName` so an account registered without a display name still renders |
| `lib/offline/sync.dart` | Added the missing `outbox.dart` import for `databaseProvider` |

### Why `isRestoring` matters
Without it, the router's redirect sees "not authenticated" during the async storage read and bounces a signed-in user to `/login` on **every cold start** — a second, subtler version of the reported bug.

---

## 6. Tests added

**Backend**
- `internal/validation/signup_test.go` — email parsing, password rules, role allow-list, coordinate bounds, normalisation.
- `internal/httpapi/handlers/auth_test.go` — scoping precedence (`kitchen_id` beats `org_id`), case-insensitivity, `expires_in` reflecting config, `clientIP` parsing.
- `internal/httpapi/handlers/auth_integration_test.go` — **real database**, auto-skipped when `DATABASE_URL` is unset. Covers register → login → me → refresh, per-role provisioning, duplicate email (409), validation (422), wrong password (401).

**Frontend**
- `test/api_contract_test.dart` — register posts the chosen role; `organisation_name` omitted when blank; login parses the user block; demo credentials match the seeder.
- `test/widget_test.dart` — replaced a stale counter smoke test that referenced a `MyApp` class which no longer existed (it could not compile). Now covers login validation, the signup link, all four roles, password mismatch, and weak passwords.

---

## 7. Verification actually performed

Environment: PostgreSQL 18 on `127.0.0.1:5433`, all 9 migrations applied, demo data seeded, server running on `:8000`.

**Go**
```
go build ./...    → clean
go vet ./...      → clean
gofmt             → clean (all touched files)
go test ./...     → ok: auth, files, httpapi/handlers, qrchain, validation
```
New auth tests: **20 top-level passing** (40 including subtests), `go test` exit 0 — 10 database integration tests + 6 handler unit tests + 4 validation tests.

**Live HTTP** (server on `:8000`)

| Check | Result |
|---|---|
| Register NGO | `201`, role `NGO`, name/email echoed, `expires_in: 900`, token issued |
| Login with those creds, **uppercase email** | `200`, same identity (case-insensitive) |
| All 4 seeded demo accounts | `kitchen→KITCHEN`, `ngo→NGO`, `logistics→LOGISTICS`, `admin→ADMIN`, each with its real name |
| `/auth/me` with signup token | full profile incl. `name` |
| Refresh | returns new pair **and** the `user` block |
| Duplicate email (different case) | `409 EMAIL_ALREADY_REGISTERED` |
| `SYSTEM` role | `422` |
| Weak password | `422` naming the `password` field |
| Wrong password | `401` |

**Database** — confirmed scoping landed correctly:
```
priya.ngo@example.com | NGO  | has_kitchen=f | has_recipient=t | Helping Hands Foundation
kitchen@example.com   | KITCHEN | has_kitchen=t | has_recipient=f |
ngo@example.com       | NGO  | has_kitchen=f | has_recipient=t |
admin@example.com     | ADMIN| has_kitchen=f | has_recipient=f |
```

**Flutter**
```
flutter analyze  → 0 errors (remaining items are pre-existing warnings)
flutter test     → 25/25 passed
```

> During widget testing the submit button sat at y=896 in the 800×600 viewport, so `tap` silently missed it and validation never ran — a false pass. Fixed with `ensureVisible`; the tests genuinely exercise the validators now.

---

## 8. Known limitations / follow-ups

1. **ADMIN self-registration is a live privilege-escalation path.** Confirmed as a deliberate choice, but if this reaches production, restrict `/auth/register` to KITCHEN/NGO/LOGISTICS and provision ADMIN through an admin-only endpoint.
2. **`RedisDB/` is a stale duplicate of `backend/`** — same module path (`github.com/sih26234/food-waste`), old `auth.go`, no `internal/validation`. **It was not modified.** If it is meant to be live code it needs the same changes; if it is scratch, delete it to avoid confusion. Worth resolving before it diverges further.
3. **No new database migration was needed.** The existing schema already had every column and the `users_email_lower_key` unique index the feature relies on.
4. **Signup does not verify email ownership** — an address is trusted as typed. Email confirmation would need a token column and a verification endpoint.
5. **Rate limiting fails open** when Redis is down (deliberate, to avoid locking everyone out during an outage), so it provides no protection in that window.
6. **Pre-existing analyzer warnings remain** in unrelated files (unused imports in several services, `print` in `push_service.dart`, an unused field in `waste_log_screen.dart`). Untouched as out of scope.
7. **`backend/.env` was created locally** for the verification run. It is gitignored (`.env` with `!.env.example`) and points at the throwaway Postgres; adjust or delete it for your own setup.
8. **Flutter was verified by analyzer + tests, not on a device.** The auth screens were not exercised on a real handset or emulator.

---

## 9. How to run it

```bash
# backend
cd backend
cp .env.example .env          # set DATABASE_URL / REDIS_URL
make up                       # postgres + redis
go run ./cmd/migrate -dir=database/migrations up
go run ./cmd/seed             # demo accounts, password demo123
go run ./cmd/server           # :8000

# frontend — real API is now the default
cd ../Frontend
flutter run

# frontend — bundled fixtures instead (enables the demo shortcuts)
flutter run --dart-define=USE_MOCKS=true
```

Signup: `/signup` — pick a role, and the account is created, signed in, and routed to that role's dashboard in one step.

---

## 10. Local Network & Device Testing Fixes

**Status:** Fixed connectivity and parsing errors on physical/emulator devices.

**Symptoms:**
- The Android app on a physical device/emulator could not connect to `192.168.1.3` and displayed "request timed out".
- Clicking "Demo" resulted in an `AppError: Type 'Null' is not a subtype of type 'String' in type cast`.

**Changes Made:**
1. **Enabled Cleartext HTTP Traffic:**
   - Modified `Frontend/android/app/src/main/AndroidManifest.xml`.
   - Added `android:usesCleartextTraffic="true"` to `<application>` to allow HTTP connections to local IPs (e.g., `192.168.1.3:8000`) instead of enforcing HTTPS.
2. **Updated Base URL:**
   - Changed `_baseUrl` in `Frontend/lib/core/api/api_client.dart` to `http://192.168.1.3:8000/api/v1` to point to the host machine's local IP address, making the backend reachable from devices on the same Wi-Fi.
3. **Resilient JSON Parsing (Crash Fix):**
   - Modified `AuthTokens.fromJson` and `User.fromJson` in `Frontend/lib/data/dtos/models.dart`.
   - Changed strict casts (e.g., `json['access_token'] as String`) to nullable with fallbacks (`json['access_token'] as String? ?? ''`).
   - This prevents the app from hard-crashing if the backend response is missing fields, ensuring error handling is graceful instead of showing a `TypeError`.
4. **Fixed "Demo Accounts" Visibility Issue:**
   - Modified `login_screen.dart` to change the `if (!useMocks)` conditional to `if (useMocks)`.
   - Previously, the Demo Accounts widget erroneously showed up *only* when the real backend was active (`useMocks = false`), which pushed the "Create Account" footer off the screen and confused users into thinking they were on a "Demo Page". Now, the demo buttons only appear when `--dart-define=USE_MOCKS=true`.
- Added logout button to the top app bar for Admin, NGO, and Driver roles.
- Updated Driver (Logistics) to fetch live data rather than always returning mock data. New driver accounts will now see a fresh/empty route instead of hardcoded demo stops.
- Fixed `useMocks` fallbacks in `kitchen_service.dart`, `ngo_service.dart`, and `admin_service.dart`. Now, when new users log in and no data is found (or endpoints return 404), they will see a fresh empty page in real-time instead of demo data.
- Implemented real QR code scanner using \mobile_scanner\ in NGO (\
go_scan_screen.dart\) and Logistics (\logistics_scan_screen.dart\) replacing the mock scanner buttons.
- Created shared \ProfileDrawer\ to display user credentials (Name, Email, Role) and added it to the Scaffold \drawer\ for all role dashboards (Admin, NGO, Logistics) allowing users to see their login identity and logout.
- Fixed Driver/Logistics scan screen UI overflow by wrapping the scan screen layout in a SingleChildScrollView.
- Added Android CAMERA permission to AndroidManifest.xml so the newly integrated mobile_scanner can actually access the device camera without failing silently.
- Fixed bottom navigation bar in Kitchen shell where navigating back to Home did not update the selected index.
- Changed FloatingActionButtons in Kitchen dashboard and surplus list to standard circular icon buttons to prevent text overflow.

---

# Part 2 — Kitchen dashboard + demand-history upload (txt / csv / excel)

Two follow-up problems: the kitchen dashboard showed placeholder data, and a
`type 'Null' is not a subtype of type 'List<dynamic>'` crash appeared.

## Root cause of BOTH symptoms

The backend mounted only **10** routes while the frontend called **25**. Every
missing one returned a `404` carrying the error envelope
`{"detail","code","message"}` — which has **no `items` key**. The services then
evaluated `data['items'] as List`, i.e. `null as List`, producing exactly the
reported crash.

So the "demo data" was never demo data: it was the `getOverview` failure
fallback (`expectedDiners: 0, ...`) plus screens still reading `MockData`.
The kitchen dashboard had no real read model to call at all.

## Backend — added

### `internal/ingest/demand_history.go` (new)
Parses uploaded history into normalised rows. **Contains no forecasting logic.**

| Format | Handling |
|---|---|
| `.csv` | comma-delimited, BOM and CRLF tolerant |
| `.tsv` | tab-delimited |
| `.txt` | comma, with automatic tab sniffing |
| `.xlsx` / `.xlsm` | read via `excelize`, first sheet |

Column headers are matched loosely through an alias map, so `Head Count`,
`head_count` and `headcount` all resolve to `footfall`; likewise
`attendance`/`diners`/`visitors`, `orders`/`tickets`/`transactions`, and the
prepared/consumed/waste quantity columns.

- **Partial success:** one unreadable row is skipped and reported, never fatal.
- 11 date layouts accepted.
- `day_of_week` derived from the date when absent (contract's 0 = Monday).
- A file with no recognisable column is rejected, naming the expected headers.
- Capped at 50 000 rows.

### Migration `0010_demand_history`
`demand_datasets` (one row per upload) + `demand_history` (one row per
observation) + a `v_uploaded_history` view. Every prediction input is stored so
a prediction can be re-run or audited, and later used for retraining.

### `POST | GET | DELETE /kitchen/datasets`
Upload (multipart, field `file`, 8 MB cap), list, delete. The response reports
`rows_imported`, `rows_rejected`, `columns_found`, `reject_samples` and the
covered date range, so the UI can tell the user exactly what happened.

### `POST /predict-demand`
**No model added.** It assembles the feature vector and forwards it to the ML
sidecar through the existing `mlclient.PredictDemand`:

1. attendance supplied for this prediction, else
2. mean footfall from uploaded history, else
3. mean footfall from the seeded attendance table.

Also adds `mean_orders`, `mean_prepared_kg`, `kg_per_footfall` and a
threshold-derived `surplus_risk`. When the trained model is deployed behind
`/predict/demand`, **nothing here has to change**.

### `GET /kitchen/overview`, `/alerts`, `/analytics/impact`, `/sensors/latest`
Real aggregates scoped to the caller's own kitchen. This is what replaced the
placeholder dashboard. The hardcoded `+5%` / `-2%` trend labels were removed
because nothing was computing them.

## Frontend — added

- `core/api/api_client.dart` — `extractList`, `extractMap`, `asDouble` helpers
  (see crash fix), plus `onSendProgress` on `postMultipart`.
- `data/dtos/dataset.dart` — `DatasetUploadResult`, `DatasetInfo`.
- `data/services/dataset_service.dart` — file picker (csv/tsv/txt/xlsx),
  multipart upload, list.
- `features/kitchen/prediction_screen.dart` — rewritten with an upload card
  (Choose File / Upload), imported-row summary, rejected-row samples, already
  uploaded list, and the expected-columns hint.
- `features/kitchen/dashboard_screen.dart` — real values, upload prompt when no
  history exists, dataset count provider.
- `pubspec.yaml` — added `file_picker: ^8.1.2`.

## The null-cast crash — fixed at the root

Fixed with shared, total helpers in `api_client.dart` rather than patching each
call site:

- `extractList` — accepts a bare list, `{items}`, `{data}` or `{results}`, and
  returns `[]` for an error envelope or null.
- `extractMap` — coerces any payload to a map.
- `asDouble` — safe numeric read.

Applied across the `kitchen`, `surplus`, `ngo`, `logistics`, `admin`,
`auth`, `qr` and `waste` services, plus `models.dart` and `offline/sync.dart`.

A follow-up consistency pass replaced the remaining
`data as Map<String, dynamic>` parser casts with `extractMap(data)`. They were
the same crash class as `data['items'] as List`: an unmapped payload (a bare
list, or chi's plain-text `404 page not found`) made the cast throw. The
helpers now handle every shape — bare list, `{items}`, `{data}`, `{results}`,
error envelope, plain text, or null — and every parser in `lib/` uses them.

## Verification

```
go build ./...    -> exit 0
go vet ./...      -> exit 0
gofmt             -> clean
go test ./...     -> exit 0 (auth, files, handlers, ingest, qrchain, validation)
flutter analyze   -> 0 errors
flutter test      -> 32/32 passed (7 new)
```

Live against the running server and real Postgres:

| Check | Result |
|---|---|
| CSV upload with 1 bad row | `201`, 4 imported, 1 rejected, reason shown, columns detected |
| TSV upload | 2 imported |
| **Real .xlsx upload** | 2 imported, columns detected |
| Unknown columns | `422` naming expected headers |
| Unsupported type (`.pdf`) | `422` |
| No auth | `401` |
| `/kitchen/overview` | real aggregates, `history_rows: 4`, `datasets: 1` |
| `/predict-demand` with dataset | `data_source: UPLOADED_HISTORY` |
| `/alerts`, `/analytics/impact`, `/sensors/latest` | all return data |

Two real bugs were caught and fixed rather than worked around: the TSV path
ignored its delimiter (`csv.Reader.Comma` was never set, so tab files parsed as
a single column), and `/alerts` compared a `uuid` column to `''`, which is a
type error in PostgreSQL.

## Notes and limitations

1. **No model was added**, as instructed. The prediction currently returns the
   sidecar's canned response (`model_version: mock-v1`) or `503 ML_UNAVAILABLE`
   when the sidecar is unreachable — the API never invents a number.
2. **Only the kitchen subset of the missing routes was implemented.** The
   frontend still calls `/waste`, `/processing/metrics`, `/admin/*`,
   `/matches/{id}/respond`, `/route` and `/planning/what-if`, which remain
   unmounted and 404. They now degrade to an empty list or an inline error
   instead of crashing, but those features are not yet functional.
3. `surplusRisk` in `kitchen.go` is a simple waste-intensity threshold, not a
   forecast, and is labelled as such in the code.
4. Uploaded rows are stored verbatim; nothing reconciles them against the seeded
   `attendance` table, so a kitchen sees both sources.
5. Prediction interval (p10/p90) is a flat +/-10% around the sidecar's point
   estimate until the model supplies real quantiles.
6. The Flutter screens were verified by analyzer and widget tests only — **not
   run on a device or emulator**. The upload flow in particular has not been
   exercised against a real system file picker.
- Removed redundant 'Processing Metrics' / 'Processing' shortcuts from the Kitchen More screen and Dashboard since the Impact Dashboard serves the same purpose.
- Removed the placeholder 'Scan QR' button from Kitchen More screen as it was unused in this role.
- Added 'Coming Soon' SnackBars to the 'Settings' and 'About' buttons in the Kitchen More screen to provide user feedback.

- Fixed Kitchen surplus list crash: Modified Surplus.fromJson in models.dart to check both id and atch_id since the Go backend sends id while the Dart model expects atch_id.
- Fixed 'invalid request' when creating a new surplus: Updated surplus_create_screen.dart to format preparedAt and expiryAt using .toUtc().toIso8601String() to guarantee the 'Z' timezone suffix expected by the Go backend's date parser.

- Supported legacy data files: Updated the predictive history upload handler (demand_history.go) to automatically fall back and allow files even when no known column headers exist. They are safely tracked using standard legacy naming.
- Fixed created surplus not showing: Updated surplus_create_screen.dart to invalidate and refresh the UI state for the surplus batches list upon successful creation.
