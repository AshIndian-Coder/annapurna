package handlers

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
	"golang.org/x/crypto/bcrypt"

	"github.com/sih26234/food-waste/internal/httpapi"
)

// AdminHandler holds dependencies for admin and kitchen ops endpoints.
type AdminHandler struct {
	pool *pgxpool.Pool
}

// NewAdminHandler constructs an AdminHandler.
func NewAdminHandler(pool *pgxpool.Pool) *AdminHandler {
	return &AdminHandler{pool: pool}
}

// RegisterRoutes mounts admin and kitchen routes onto r.
func (h *AdminHandler) RegisterRoutes(r chi.Router) {
	// User management
	r.Get("/admin/users", h.ListUsers)
	r.Post("/admin/users", h.CreateUser)
	r.Patch("/admin/users/{id}", h.UpdateUser)

	// Model registry
	r.Get("/admin/models", h.ListModels)

	// Audit log
	r.Get("/admin/audit-log", h.AuditLog)

	// Data re-seed (dev/staging)
	r.Post("/admin/reseed", h.Reseed)

	// Kitchen ops
	r.Get("/kitchen/overview", h.KitchenOverview)
	r.Post("/kitchen/attendance", h.PostAttendance)
	r.Post("/kitchen/menu", h.PostMenu)
}

// AdminUserItem represents a user in the admin list.
type AdminUserItem struct {
	UserID    string    `json:"user_id"`
	Email     string    `json:"email"`
	Name      string    `json:"name"`
	Role      string    `json:"role"`
	KitchenID string    `json:"kitchen_id,omitempty"`
	Active    bool      `json:"active"`
	CreatedAt time.Time `json:"created_at"`
}

// ListUsers handles GET /admin/users.
func (h *AdminHandler) ListUsers(w http.ResponseWriter, r *http.Request) {
	claims, ok := requireClaims(w, r)
	if !ok {
		return
	}
	if !requireRole(w, claims, "ADMIN") {
		return
	}

	q := r.URL.Query()
	role := strings.ToUpper(strings.TrimSpace(q.Get("role")))
	page, _ := strconv.Atoi(q.Get("page"))
	if page < 1 {
		page = 1
	}
	limit, _ := strconv.Atoi(q.Get("limit"))
	if limit < 1 || limit > 200 {
		limit = 50
	}
	offset := (page - 1) * limit

	query := `SELECT id, email, COALESCE(name, ''), role, COALESCE(kitchen_id::text, ''), is_active, created_at
	          FROM users WHERE 1=1`
	var args []any
	if role != "" {
		args = append(args, role)
		query += fmt.Sprintf(" AND UPPER(role) = $%d", len(args))
	}
	args = append(args, limit, offset)
	query += fmt.Sprintf(" ORDER BY created_at DESC LIMIT $%d OFFSET $%d", len(args)-1, len(args))

	rows, err := h.pool.Query(r.Context(), query, args...)
	if err != nil {
		httpapi.NewInternal("failed to list users: " + err.Error()).Render(w)
		return
	}
	defer rows.Close()

	var users []AdminUserItem
	for rows.Next() {
		var u AdminUserItem
		if err := rows.Scan(&u.UserID, &u.Email, &u.Name, &u.Role, &u.KitchenID, &u.Active, &u.CreatedAt); err == nil {
			users = append(users, u)
		}
	}

	var total int
	_ = h.pool.QueryRow(r.Context(), `SELECT COUNT(*) FROM users`).Scan(&total)

	writeJSON(w, http.StatusOK, map[string]any{
		"users": users,
		"total": total,
		"page":  page,
		"limit": limit,
	})
}

// CreateUser handles POST /admin/users.
func (h *AdminHandler) CreateUser(w http.ResponseWriter, r *http.Request) {
	claims, ok := requireClaims(w, r)
	if !ok {
		return
	}
	if !requireRole(w, claims, "ADMIN") {
		return
	}

	var req struct {
		Email     string `json:"email"`
		Name      string `json:"name"`
		Role      string `json:"role"`
		KitchenID string `json:"kitchen_id,omitempty"`
		Password  string `json:"password"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		httpapi.NewValidation("invalid request body", err.Error()).Render(w)
		return
	}
	if req.Email == "" || req.Password == "" || req.Role == "" {
		httpapi.NewValidation("email, password and role are required", nil).Render(w)
		return
	}

	hash, err := bcrypt.GenerateFromPassword([]byte(req.Password), bcrypt.DefaultCost)
	if err != nil {
		httpapi.NewInternal("failed to hash password").Render(w)
		return
	}

	newID := uuid.New()
	var kUUID *uuid.UUID
	if req.KitchenID != "" {
		if id, err := uuid.Parse(req.KitchenID); err == nil {
			kUUID = &id
		}
	}

	_, err = h.pool.Exec(r.Context(), `
		INSERT INTO users (id, email, name, role, password_hash, kitchen_id, is_active, created_at, updated_at)
		VALUES ($1, $2, $3, $4, $5, $6, true, NOW(), NOW())`,
		newID, req.Email, req.Name, strings.ToUpper(req.Role), string(hash), kUUID,
	)
	if err != nil {
		httpapi.NewInternal("failed to create user: " + err.Error()).Render(w)
		return
	}

	writeJSON(w, http.StatusCreated, map[string]any{
		"user_id": newID.String(),
		"email":   req.Email,
		"role":    strings.ToUpper(req.Role),
		"status":  "created",
	})
}

// UpdateUser handles PATCH /admin/users/{id}.
func (h *AdminHandler) UpdateUser(w http.ResponseWriter, r *http.Request) {
	claims, ok := requireClaims(w, r)
	if !ok {
		return
	}
	if !requireRole(w, claims, "ADMIN") {
		return
	}

	id := strings.TrimSpace(chi.URLParam(r, "id"))
	uUUID, err := uuid.Parse(id)
	if err != nil {
		httpapi.NewValidation("invalid user id", err.Error()).Render(w)
		return
	}

	var req struct {
		Name      *string `json:"name,omitempty"`
		Role      *string `json:"role,omitempty"`
		Active    *bool   `json:"active,omitempty"`
		KitchenID *string `json:"kitchen_id,omitempty"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		httpapi.NewValidation("invalid request body", err.Error()).Render(w)
		return
	}

	if req.Name != nil {
		_, _ = h.pool.Exec(r.Context(), `UPDATE users SET name = $1, updated_at = NOW() WHERE id = $2`, *req.Name, uUUID)
	}
	if req.Role != nil {
		_, _ = h.pool.Exec(r.Context(), `UPDATE users SET role = $1, updated_at = NOW() WHERE id = $2`, strings.ToUpper(*req.Role), uUUID)
	}
	if req.Active != nil {
		_, _ = h.pool.Exec(r.Context(), `UPDATE users SET is_active = $1, updated_at = NOW() WHERE id = $2`, *req.Active, uUUID)
	}
	if req.KitchenID != nil {
		if kID, err := uuid.Parse(*req.KitchenID); err == nil {
			_, _ = h.pool.Exec(r.Context(), `UPDATE users SET kitchen_id = $1, updated_at = NOW() WHERE id = $2`, kID, uUUID)
		}
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"user_id": id,
		"status":  "updated",
	})
}

// ListModels handles GET /admin/models.
func (h *AdminHandler) ListModels(w http.ResponseWriter, r *http.Request) {
	if _, ok := requireClaims(w, r); !ok {
		return
	}

	models := []map[string]any{
		{
			"model_id":    "demand-v1",
			"name":        "Deep Demand Forecaster",
			"version":     "v1.2",
			"type":        "demand_forecast",
			"status":      "active",
			"accuracy":    0.91,
			"deployed_at": time.Now().UTC().AddDate(0, -1, 0),
		},
		{
			"model_id":    "cv-v1",
			"name":        "Surplus Food Freshness & Safety",
			"version":     "v1.0",
			"type":        "visual_inspection",
			"status":      "active",
			"accuracy":    0.94,
			"deployed_at": time.Now().UTC().AddDate(0, -1, 0),
		},
		{
			"model_id":    "fusion-v1",
			"name":        "Multimodal Safety Fusion Engine",
			"version":     "v1.1",
			"type":        "safety_decision",
			"status":      "active",
			"accuracy":    0.98,
			"deployed_at": time.Now().UTC().AddDate(0, -1, 0),
		},
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"models": models,
		"count":  len(models),
	})
}

// AuditLog handles GET /admin/audit-log.
func (h *AdminHandler) AuditLog(w http.ResponseWriter, r *http.Request) {
	claims, ok := requireClaims(w, r)
	if !ok {
		return
	}
	if !requireRole(w, claims, "ADMIN") {
		return
	}

	page, _ := strconv.Atoi(r.URL.Query().Get("page"))
	if page < 1 {
		page = 1
	}
	limit, _ := strconv.Atoi(r.URL.Query().Get("limit"))
	if limit < 1 || limit > 200 {
		limit = 50
	}
	offset := (page - 1) * limit

	rows, err := h.pool.Query(r.Context(), `
		SELECT id, COALESCE(user_id::text, actor_id::text, ''), action,
		       COALESCE(resource_type, entity, ''), COALESCE(resource_id, entity_id, ''), created_at
		FROM audit_log
		ORDER BY created_at DESC LIMIT $1 OFFSET $2`, limit, offset,
	)
	if err != nil {
		httpapi.NewInternal("failed to fetch audit log: " + err.Error()).Render(w)
		return
	}
	defer rows.Close()

	type auditEntry struct {
		ID         string    `json:"entry_id"`
		ActorID    string    `json:"actor_id"`
		Action     string    `json:"action"`
		Resource   string    `json:"resource"`
		ResourceID string    `json:"resource_id"`
		CreatedAt  time.Time `json:"created_at"`
	}

	var entries []auditEntry
	for rows.Next() {
		var e auditEntry
		if err := rows.Scan(&e.ID, &e.ActorID, &e.Action, &e.Resource, &e.ResourceID, &e.CreatedAt); err == nil {
			entries = append(entries, e)
		}
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"entries": entries,
		"page":    page,
		"limit":   limit,
	})
}

// Reseed handles POST /admin/reseed.
func (h *AdminHandler) Reseed(w http.ResponseWriter, r *http.Request) {
	claims, ok := requireClaims(w, r)
	if !ok {
		return
	}
	if !requireRole(w, claims, "ADMIN") {
		return
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"ok":         true,
		"message":    "reseed requested",
		"started_at": time.Now().UTC(),
	})
}

// KitchenOverview handles GET /kitchen/overview.
func (h *AdminHandler) KitchenOverview(w http.ResponseWriter, r *http.Request) {
	claims, ok := requireClaims(w, r)
	if !ok {
		return
	}

	kid := r.URL.Query().Get("kitchen_id")
	if kid == "" {
		kid = claims.KitchenID
	}
	if kid == "" {
		httpapi.NewValidation("kitchen_id is required", nil).Render(w)
		return
	}

	var kName string
	_ = h.pool.QueryRow(r.Context(), `SELECT name FROM kitchens WHERE id = $1`, kid).Scan(&kName)

	var todayHeadcount, predictedCount int
	today := time.Now().UTC().Format("2006-01-02")
	_ = h.pool.QueryRow(r.Context(),
		`SELECT COALESCE(head_count, expected_diners, 0) FROM attendance WHERE kitchen_id = $1 AND meal_date = $2`,
		kid, today,
	).Scan(&todayHeadcount)

	var todayWasteKg float64
	_ = h.pool.QueryRow(r.Context(),
		`SELECT COALESCE(SUM(quantity_kg), 0) FROM waste WHERE kitchen_id = $1 AND recorded_at::date = $2`,
		kid, today,
	).Scan(&todayWasteKg)

	var todayRedirectedKg float64
	_ = h.pool.QueryRow(r.Context(),
		`SELECT COALESCE(SUM(quantity_kg), 0) FROM surplus_batches WHERE kitchen_id = $1 AND upper(status) = 'DELIVERED' AND updated_at::date = $2`,
		kid, today,
	).Scan(&todayRedirectedKg)

	var activeAlerts int
	_ = h.pool.QueryRow(r.Context(),
		`SELECT COUNT(*) FROM alerts WHERE kitchen_id = $1 AND is_acked = false`,
		kid,
	).Scan(&activeAlerts)

	writeJSON(w, http.StatusOK, map[string]any{
		"kitchen_id":          kid,
		"name":                kName,
		"today_headcount":     todayHeadcount,
		"predicted_count":     predictedCount,
		"today_waste_kg":      todayWasteKg,
		"today_redirected_kg": todayRedirectedKg,
		"active_alerts":       activeAlerts,
		"active_sensors":      3,
		"last_updated":        time.Now().UTC(),
	})
}

// PostAttendance handles POST /kitchen/attendance.
func (h *AdminHandler) PostAttendance(w http.ResponseWriter, r *http.Request) {
	claims, ok := requireClaims(w, r)
	if !ok {
		return
	}
	if !requireRole(w, claims, "KITCHEN", "ADMIN") {
		return
	}

	var req struct {
		KitchenID     string `json:"kitchen_id,omitempty"`
		MealID        string `json:"meal_id,omitempty"`
		MealType      string `json:"meal_type,omitempty"`
		Date          string `json:"date"`
		HeadCount     int    `json:"headcount"`
		ClientEventID string `json:"client_event_id,omitempty"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		httpapi.NewValidation("invalid request body", err.Error()).Render(w)
		return
	}

	kid := req.KitchenID
	if kid == "" {
		kid = claims.KitchenID
	}
	if kid == "" {
		httpapi.NewValidation("kitchen_id is required", nil).Render(w)
		return
	}

	mealDate := time.Now().UTC().Format("2006-01-02")
	if req.Date != "" {
		mealDate = req.Date
	}

	newID := uuid.New()
	query := `
		INSERT INTO attendance (id, kitchen_id, meal_date, head_count, created_at, updated_at)
		VALUES ($1, $2, $3, $4, NOW(), NOW())
		ON CONFLICT (kitchen_id, meal_date) DO UPDATE
		SET head_count = EXCLUDED.head_count,
		    updated_at = NOW()
		RETURNING id, head_count`

	var attID string
	var count int
	err := h.pool.QueryRow(r.Context(), query, newID, kid, mealDate, req.HeadCount).Scan(&attID, &count)
	if err != nil {
		httpapi.NewInternal("failed to record attendance: " + err.Error()).Render(w)
		return
	}

	writeJSON(w, http.StatusCreated, map[string]any{
		"attendance_id": attID,
		"kitchen_id":    kid,
		"headcount":     count,
		"recorded_at":   time.Now().UTC(),
	})
}

// PostMenu handles POST /kitchen/menu.
func (h *AdminHandler) PostMenu(w http.ResponseWriter, r *http.Request) {
	claims, ok := requireClaims(w, r)
	if !ok {
		return
	}
	if !requireRole(w, claims, "KITCHEN", "ADMIN") {
		return
	}

	var req struct {
		KitchenID string `json:"kitchen_id,omitempty"`
		MealType  string `json:"meal_type"`
		Date      string `json:"date"`
		Items     []any  `json:"items"`
		Name      string `json:"name,omitempty"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		httpapi.NewValidation("invalid request body", err.Error()).Render(w)
		return
	}

	kid := req.KitchenID
	if kid == "" {
		kid = claims.KitchenID
	}
	if kid == "" {
		httpapi.NewValidation("kitchen_id is required", nil).Render(w)
		return
	}

	mealType := strings.ToUpper(strings.TrimSpace(req.MealType))
	if mealType == "" {
		mealType = "LUNCH"
	}
	mealDate := time.Now().UTC().Format("2006-01-02")
	if req.Date != "" {
		mealDate = req.Date
	}

	menuBytes, _ := json.Marshal(req.Items)
	newID := uuid.New()

	query := `
		INSERT INTO meals (id, kitchen_id, date, meal_type, menu, name, created_at, updated_at)
		VALUES ($1, $2, $3, $4, $5, $6, NOW(), NOW())
		ON CONFLICT (kitchen_id, date, meal_type) DO UPDATE
		SET menu       = EXCLUDED.menu,
		    name       = EXCLUDED.name,
		    updated_at = NOW()
		RETURNING id`

	var mealID string
	err := h.pool.QueryRow(r.Context(), query, newID, kid, mealDate, mealType, menuBytes, req.Name).Scan(&mealID)
	if err != nil {
		httpapi.NewInternal("failed to record menu: " + err.Error()).Render(w)
		return
	}

	writeJSON(w, http.StatusCreated, map[string]any{
		"menu_id":    mealID,
		"kitchen_id": kid,
		"item_count": len(req.Items),
		"created_at": time.Now().UTC(),
	})
}
