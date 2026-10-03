package handlers

import (
	"encoding/json"
	"net/http"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/sih26234/food-waste/internal/httpapi"
)

// DevicesHandler holds dependencies for device registration endpoints.
type DevicesHandler struct {
	pool *pgxpool.Pool
}

// NewDevicesHandler constructs a DevicesHandler.
func NewDevicesHandler(pool *pgxpool.Pool) *DevicesHandler {
	return &DevicesHandler{pool: pool}
}

// RegisterRoutes mounts device routes onto r.
func (h *DevicesHandler) RegisterRoutes(r chi.Router) {
	r.Post("/devices", h.Register)
	r.Delete("/devices/{id}", h.Deregister)
}

// RegisterDeviceRequest is the body for POST /devices.
type RegisterDeviceRequest struct {
	Token       string `json:"token"`
	DeviceToken string `json:"device_token,omitempty"`
	Platform    string `json:"platform"`
	AppVersion  string `json:"app_version,omitempty"`
}

// Register handles POST /devices.
func (h *DevicesHandler) Register(w http.ResponseWriter, r *http.Request) {
	claims, ok := requireClaims(w, r)
	if !ok {
		return
	}

	var req RegisterDeviceRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		httpapi.NewValidation("invalid request body", err.Error()).Render(w)
		return
	}

	token := strings.TrimSpace(req.Token)
	if token == "" {
		token = strings.TrimSpace(req.DeviceToken)
	}
	platform := strings.ToUpper(strings.TrimSpace(req.Platform))

	if token == "" || (platform != "ANDROID" && platform != "IOS") {
		httpapi.NewValidation("token and platform ('ANDROID' or 'IOS') are required", nil).Render(w)
		return
	}

	userUUID, err := uuid.Parse(claims.Subject)
	if err != nil {
		httpapi.NewValidation("invalid user ID", err.Error()).Render(w)
		return
	}

	deviceID := uuid.New()
	query := `
		INSERT INTO device_tokens (id, user_id, platform, token, app_version, is_valid, last_seen_at, created_at, updated_at)
		VALUES ($1, $2, $3, $4, $5, true, NOW(), NOW(), NOW())
		ON CONFLICT (token) DO UPDATE
		SET user_id      = EXCLUDED.user_id,
		    platform     = EXCLUDED.platform,
		    app_version  = EXCLUDED.app_version,
		    is_valid     = true,
		    last_seen_at = NOW(),
		    updated_at   = NOW()
		RETURNING id`

	err = h.pool.QueryRow(r.Context(), query,
		deviceID, userUUID, platform, token, req.AppVersion,
	).Scan(&deviceID)
	if err != nil {
		httpapi.NewInternal("failed to register device token: " + err.Error()).Render(w)
		return
	}

	writeJSON(w, http.StatusCreated, map[string]any{
		"device_id": deviceID.String(),
		"active":    true,
	})
}

// Deregister handles DELETE /devices/{id}.
func (h *DevicesHandler) Deregister(w http.ResponseWriter, r *http.Request) {
	_, ok := requireClaims(w, r)
	if !ok {
		return
	}

	id := strings.TrimSpace(chi.URLParam(r, "id"))
	if id == "" {
		httpapi.NewValidation("device id is required", nil).Render(w)
		return
	}

	dUUID, err := uuid.Parse(id)
	if err != nil {
		// Treat id as raw token string if not UUID
		_, _ = h.pool.Exec(r.Context(), `UPDATE device_tokens SET is_valid = false, updated_at = NOW() WHERE token = $1`, id)
	} else {
		_, _ = h.pool.Exec(r.Context(), `UPDATE device_tokens SET is_valid = false, updated_at = NOW() WHERE id = $1`, dUUID)
	}

	w.WriteHeader(http.StatusNoContent)
}
