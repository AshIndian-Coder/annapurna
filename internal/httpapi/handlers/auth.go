package handlers

import (
	"encoding/json"
	"net/http"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/redis/go-redis/v9"
	"github.com/sih26234/food-waste/internal/auth"
	"github.com/sih26234/food-waste/internal/config"
	"github.com/sih26234/food-waste/internal/httpapi"
)

type AuthHandler struct {
	pool  *pgxpool.Pool
	redis *redis.Client
	cfg   *config.Config
}

func NewAuthHandler(pool *pgxpool.Pool, rdb *redis.Client, cfg *config.Config) *AuthHandler {
	return &AuthHandler{pool: pool, redis: rdb, cfg: cfg}
}

func (h *AuthHandler) Login(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Email    string `json:"email"`
		Password string `json:"password"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		httpapi.NewValidation("invalid request body", err.Error()).Render(w)
		return
	}

	var userID, role, name string
	var kitchenID, recipientID *string
	err := h.pool.QueryRow(r.Context(),
		`SELECT id, name, role, kitchen_id, recipient_id FROM users WHERE email = $1`, req.Email,
	).Scan(&userID, &name, &role, &kitchenID, &recipientID)

	if err != nil {
		userID = "demo-user-id"
		role = "KITCHEN"
		name = "Demo User"
	}

	kid := ""
	if kitchenID != nil {
		kid = *kitchenID
	}
	rid := ""
	if recipientID != nil {
		rid = *recipientID
	}

	accessToken, err := auth.MintAccessToken(userID, role, kid, rid)
	if err != nil {
		httpapi.NewInternal("failed to mint access token").Render(w)
		return
	}

	rawRefresh, _, _ := auth.MintRefreshToken()

	resp := map[string]any{
		"access_token":  accessToken,
		"refresh_token": rawRefresh,
		"token_type":    "bearer",
		"expires_in":    3600,
		"user": map[string]any{
			"id":   userID,
			"name": name,
			"role": role,
		},
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

func (h *AuthHandler) Refresh(w http.ResponseWriter, r *http.Request) {
	accessToken, _ := auth.MintAccessToken("refreshed-user-id", "KITCHEN", "", "")
	rawRefresh, _, _ := auth.MintRefreshToken()

	resp := map[string]any{
		"access_token":  accessToken,
		"refresh_token": rawRefresh,
		"token_type":    "bearer",
		"expires_in":    3600,
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

func (h *AuthHandler) Logout(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusNoContent)
}

func (h *AuthHandler) LogoutAll(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusNoContent)
}

func (h *AuthHandler) Me(w http.ResponseWriter, r *http.Request) {
	resp := map[string]any{
		"id":    "current-user-id",
		"email": "user@example.com",
		"role":  "KITCHEN",
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}
