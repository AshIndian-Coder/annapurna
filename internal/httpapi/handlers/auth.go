package handlers

import (
	"context"
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
	"github.com/sih26234/food-waste/internal/store"
)

type AuthHandler struct {
	pool    *pgxpool.Pool
	redis   *redis.Client
	cfg     *config.Config
	refresh *store.RefreshTokenStore
}

func NewAuthHandler(pool *pgxpool.Pool, rdb *redis.Client, cfg *config.Config) *AuthHandler {
	return &AuthHandler{pool: pool, redis: rdb, cfg: cfg, refresh: store.NewRefreshTokenStore(pool)}
}

// issueTokens persists a refresh token (hash only) and mints a matching access
// token for the user. The raw refresh token is returned to the caller once.
func (h *AuthHandler) issueTokens(ctx context.Context, userID, role, kitchenID, recipientID string) (accessToken, refreshToken string, err error) {
	accessToken, err = auth.MintAccessToken(userID, role, kitchenID, recipientID)
	if err != nil {
		return "", "", err
	}
	refreshToken, err = auth.CreateRefreshToken(ctx, h.refresh, userID)
	if err != nil {
		return "", "", err
	}
	return accessToken, refreshToken, nil
}

// identityFor loads the role/org of a user and derives the token scoping ids.
func (h *AuthHandler) identityFor(ctx context.Context, userID string) (role, kitchenID, recipientID string, err error) {
	var orgID *string
	var rawRole string
	var isActive bool
	err = h.pool.QueryRow(ctx,
		`SELECT role, org_id, is_active FROM users WHERE id = $1 LIMIT 1`, userID).
		Scan(&rawRole, &orgID, &isActive)
	if err != nil {
		return "", "", "", err
	}
	if !isActive {
		return "", "", "", errors.New("account is disabled")
	}
	role = strings.ToUpper(rawRole)
	if orgID != nil {
		switch role {
		case "KITCHEN":
			kitchenID = *orgID
		case "NGO":
			recipientID = *orgID
		}
	}
	return role, kitchenID, recipientID, nil
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

	accessToken, rawRefresh, err := h.issueTokens(r.Context(), userID, strings.ToUpper(role), kitchenID, recipientID)
	if err != nil {
		httpapi.NewInternal("failed to issue tokens").Render(w)
		return
	}

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

// refreshTokenFromRequest accepts the token in the JSON body (mobile) or in
// the refresh_token cookie (web).
func refreshTokenFromRequest(r *http.Request) string {
	var body struct {
		RefreshToken string `json:"refresh_token"`
	}
	_ = json.NewDecoder(r.Body).Decode(&body)
	if body.RefreshToken != "" {
		return body.RefreshToken
	}
	if c, err := r.Cookie("refresh_token"); err == nil {
		return c.Value
	}
	return ""
}

// Refresh rotates the refresh token (single use) and mints a new access token.
// Presenting an already-rotated token is treated as theft: the whole family is
// revoked and the caller gets 401 REFRESH_TOKEN_REUSED.
func (h *AuthHandler) Refresh(w http.ResponseWriter, r *http.Request) {
	presented := refreshTokenFromRequest(r)
	if presented == "" {
		httpapi.NewUnauthorized("refresh_token is required").Render(w)
		return
	}

	newRaw, userID, err := auth.RotateRefreshToken(r.Context(), h.refresh, presented)
	if err != nil {
		switch {
		case errors.Is(err, auth.ErrRefreshTokenRevoked):
			httpapi.NewAppError(httpapi.CodeRefreshTokenReused,
				"refresh token reuse detected; session revoked", http.StatusUnauthorized).Render(w)
		case errors.Is(err, auth.ErrRefreshTokenExpired), errors.Is(err, auth.ErrRefreshTokenNotFound):
			httpapi.NewUnauthorized("invalid or expired refresh token").Render(w)
		default:
			httpapi.NewInternal("could not rotate refresh token").Render(w)
		}
		return
	}

	role, kitchenID, recipientID, err := h.identityFor(r.Context(), userID)
	if err != nil {
		httpapi.NewUnauthorized("user not found or disabled").Render(w)
		return
	}
	accessToken, err := auth.MintAccessToken(userID, role, kitchenID, recipientID)
	if err != nil {
		httpapi.NewInternal("failed to mint access token").Render(w)
		return
	}
	h.refresh.TouchLastUsed(r.Context(), presented)

	writeJSON(w, http.StatusOK, map[string]any{
		"access_token":  accessToken,
		"refresh_token": newRaw,
		"token_type":    "bearer",
		"expires_in":    3600,
	})
}

// Logout revokes the presented refresh token's family (idempotent).
func (h *AuthHandler) Logout(w http.ResponseWriter, r *http.Request) {
	if presented := refreshTokenFromRequest(r); presented != "" {
		if err := auth.RevokeFamily(r.Context(), h.refresh, presented); err != nil {
			httpapi.NewInternal("could not revoke refresh token").Render(w)
			return
		}
	}
	w.WriteHeader(http.StatusNoContent)
}

// LogoutAll revokes every refresh family owned by the authenticated user.
func (h *AuthHandler) LogoutAll(w http.ResponseWriter, r *http.Request) {
	claims, ok := requireClaims(w, r)
	if !ok {
		return
	}
	if err := h.refresh.RevokeAllForUser(r.Context(), claims.Subject); err != nil {
		httpapi.NewInternal("could not revoke sessions").Render(w)
		return
	}
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
