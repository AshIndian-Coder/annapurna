package handlers

import (
	"encoding/json"
	"errors"
	"net/http"
	"strings"

	"github.com/jackc/pgx/v5"
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
	req.Email = strings.TrimSpace(strings.ToLower(req.Email))
	if req.Email == "" || req.Password == "" {
		httpapi.NewValidation("email and password are required", "").Render(w)
		return
	}

	// ── fetch user from DB ────────────────────────────────────────────────────
	var userID, passwordHash, role string
	var orgID *string
	var isActive bool

	err := h.pool.QueryRow(r.Context(),
		`SELECT id, password_hash, role, org_id, is_active
		 FROM users WHERE email = $1 LIMIT 1`,
		req.Email,
	).Scan(&userID, &passwordHash, &role, &orgID, &isActive)

	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			httpapi.NewUnauthorized("invalid credentials").Render(w)
			return
		}
		httpapi.NewInternal("database error").Render(w)
		return
	}

	if !isActive {
		httpapi.NewUnauthorized("account is disabled").Render(w)
		return
	}

	// ── verify password ───────────────────────────────────────────────────────
	if err := auth.CheckPassword(passwordHash, req.Password); err != nil {
		httpapi.NewUnauthorized("invalid credentials").Render(w)
		return
	}

	// ── mint tokens ───────────────────────────────────────────────────────────
	kitchenID := ""
	if orgID != nil && strings.EqualFold(role, "kitchen") {
		kitchenID = *orgID
	}
	recipientID := ""
	if orgID != nil && strings.EqualFold(role, "ngo") {
		recipientID = *orgID
	}

	accessToken, err := auth.MintAccessToken(userID, strings.ToUpper(role), kitchenID, recipientID)
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
			"id":    userID,
			"email": req.Email,
			"role":  strings.ToUpper(role),
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
	// Extract bearer token from Authorization header
	authHeader := r.Header.Get("Authorization")
	tokenStr := strings.TrimPrefix(authHeader, "Bearer ")

	claims, err := auth.VerifyAccessToken(tokenStr)
	if err != nil {
		httpapi.NewUnauthorized("invalid or expired token").Render(w)
		return
	}

	// Fetch real user from DB
	var email, role string
	var isActive bool
	dbErr := h.pool.QueryRow(r.Context(),
		`SELECT email, role, is_active FROM users WHERE id = $1 LIMIT 1`,
		claims.Subject,
	).Scan(&email, &role, &isActive)

	if dbErr != nil || !isActive {
		httpapi.NewUnauthorized("user not found or disabled").Render(w)
		return
	}

	resp := map[string]any{
		"id":    claims.Subject,
		"email": email,
		"role":  strings.ToUpper(role),
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}
