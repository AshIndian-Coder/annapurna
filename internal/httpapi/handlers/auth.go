package handlers

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"net/http"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/redis/go-redis/v9"
	"golang.org/x/crypto/bcrypt"

	"github.com/sih26234/food-waste/internal/auth"
	"github.com/sih26234/food-waste/internal/config"
	"github.com/sih26234/food-waste/internal/httpapi"
	mw "github.com/sih26234/food-waste/internal/httpapi/middleware"
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

	var userID, role, name, passwordHash string
	var kitchenID, recipientID *string
	var isActive bool

	err := h.pool.QueryRow(r.Context(),
		`SELECT id, name, role, password_hash, kitchen_id, recipient_id, is_active 
		 FROM users WHERE email = $1`, req.Email,
	).Scan(&userID, &name, &role, &passwordHash, &kitchenID, &recipientID, &isActive)

	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			httpapi.NewUnauthorized("invalid email or password").Render(w)
			return
		}
		httpapi.NewInternal("database error during authentication").Render(w)
		return
	}

	if !isActive {
		httpapi.NewForbidden("user account is inactive").Render(w)
		return
	}

	if err := bcrypt.CompareHashAndPassword([]byte(passwordHash), []byte(req.Password)); err != nil {
		httpapi.NewUnauthorized("invalid email or password").Render(w)
		return
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

	rawRefresh, refreshHash, err := auth.MintRefreshToken()
	if err != nil {
		httpapi.NewInternal("failed to mint refresh token").Render(w)
		return
	}

	familyID := uuid.New()
	expiresAt := time.Now().UTC().Add(30 * 24 * time.Hour)

	_, err = h.pool.Exec(r.Context(),
		`INSERT INTO refresh_tokens (id, user_id, family_id, token_hash, expires_at, created_at, last_used_at)
		 VALUES ($1, $2, $3, $4, $5, $6, $6)`,
		uuid.New(), userID, familyID, refreshHash, expiresAt, time.Now().UTC(),
	)
	if err != nil {
		httpapi.NewInternal("failed to persist refresh token").Render(w)
		return
	}

	resp := map[string]any{
		"access_token":  accessToken,
		"refresh_token": rawRefresh,
		"token_type":    "bearer",
		"expires_in":    3600,
		"user": map[string]any{
			"id":           userID,
			"name":         name,
			"role":         role,
			"kitchen_id":   kitchenID,
			"recipient_id": recipientID,
		},
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

func (h *AuthHandler) Refresh(w http.ResponseWriter, r *http.Request) {
	var req struct {
		RefreshToken string `json:"refresh_token"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.RefreshToken == "" {
		httpapi.NewValidation("refresh_token is required", nil).Render(w)
		return
	}

	sum := sha256.Sum256([]byte(req.RefreshToken))
	tokenHash := hex.EncodeToString(sum[:])

	var tokenID, userID string
	var familyID uuid.UUID
	var expiresAt time.Time
	var revokedAt *time.Time

	err := h.pool.QueryRow(r.Context(),
		`SELECT id, user_id, family_id, expires_at, revoked_at 
		 FROM refresh_tokens WHERE token_hash = $1`, tokenHash,
	).Scan(&tokenID, &userID, &familyID, &expiresAt, &revokedAt)

	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			httpapi.NewUnauthorized("invalid refresh token").Render(w)
			return
		}
		httpapi.NewInternal("database query error").Render(w)
		return
	}

	if revokedAt != nil {
		_, _ = h.pool.Exec(r.Context(),
			`UPDATE refresh_tokens SET revoked_at = $1 WHERE family_id = $2 AND revoked_at IS NULL`,
			time.Now().UTC(), familyID,
		)
		httpapi.ErrRefreshTokenReused().Render(w)
		return
	}

	if time.Now().UTC().After(expiresAt) {
		httpapi.NewUnauthorized("refresh token expired").Render(w)
		return
	}

	now := time.Now().UTC()
	_, _ = h.pool.Exec(r.Context(),
		`UPDATE refresh_tokens SET revoked_at = $1, last_used_at = $1 WHERE id = $2`,
		now, tokenID,
	)

	var role string
	var kitchenID, recipientID *string
	err = h.pool.QueryRow(r.Context(),
		`SELECT role, kitchen_id, recipient_id FROM users WHERE id = $1 AND is_active = true`, userID,
	).Scan(&role, &kitchenID, &recipientID)

	if err != nil {
		httpapi.NewUnauthorized("user not found or inactive").Render(w)
		return
	}

	kid := ""
	if kitchenID != nil {
		kid = *kitchenID
	}
	rid := ""
	if recipientID != nil {
		rid = *recipientID
	}

	newAccess, err := auth.MintAccessToken(userID, role, kid, rid)
	if err != nil {
		httpapi.NewInternal("failed to mint new access token").Render(w)
		return
	}

	newRawRefresh, newRefreshHash, err := auth.MintRefreshToken()
	if err != nil {
		httpapi.NewInternal("failed to mint new refresh token").Render(w)
		return
	}

	newExpiresAt := now.Add(30 * 24 * time.Hour)
	_, err = h.pool.Exec(r.Context(),
		`INSERT INTO refresh_tokens (id, user_id, family_id, token_hash, expires_at, created_at, last_used_at)
		 VALUES ($1, $2, $3, $4, $5, $6, $6)`,
		uuid.New(), userID, familyID, newRefreshHash, newExpiresAt, now,
	)
	if err != nil {
		httpapi.NewInternal("failed to persist rotated refresh token").Render(w)
		return
	}

	resp := map[string]any{
		"access_token":  newAccess,
		"refresh_token": newRawRefresh,
		"token_type":    "bearer",
		"expires_in":    3600,
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

func (h *AuthHandler) Logout(w http.ResponseWriter, r *http.Request) {
	claims := mw.ClaimsFromCtx(r.Context())
	if claims != nil && claims.ID != "" && h.redis != nil {
		remaining := time.Until(claims.ExpiresAt.Time)
		if remaining > 0 {
			_ = h.redis.Set(r.Context(), "jwt:deny:"+claims.ID, "revoked", remaining).Err()
		}
	}
	w.WriteHeader(http.StatusNoContent)
}

func (h *AuthHandler) LogoutAll(w http.ResponseWriter, r *http.Request) {
	claims := mw.ClaimsFromCtx(r.Context())
	if claims == nil || claims.Subject == "" {
		httpapi.NewUnauthorized("unauthorized").Render(w)
		return
	}

	_, err := h.pool.Exec(r.Context(),
		`UPDATE refresh_tokens SET revoked_at = $1 WHERE user_id = $2 AND revoked_at IS NULL`,
		time.Now().UTC(), claims.Subject,
	)
	if err != nil {
		httpapi.NewInternal("failed to revoke tokens").Render(w)
		return
	}

	if claims.ID != "" && h.redis != nil {
		remaining := time.Until(claims.ExpiresAt.Time)
		if remaining > 0 {
			_ = h.redis.Set(r.Context(), "jwt:deny:"+claims.ID, "revoked", remaining).Err()
		}
	}

	w.WriteHeader(http.StatusNoContent)
}

func (h *AuthHandler) Me(w http.ResponseWriter, r *http.Request) {
	claims := mw.ClaimsFromCtx(r.Context())
	if claims == nil || claims.Subject == "" {
		httpapi.NewUnauthorized("unauthorized").Render(w)
		return
	}

	var user struct {
		ID          string  `json:"id"`
		Name        string  `json:"name"`
		Email       string  `json:"email"`
		Role        string  `json:"role"`
		KitchenID   *string `json:"kitchen_id,omitempty"`
		RecipientID *string `json:"recipient_id,omitempty"`
	}

	err := h.pool.QueryRow(r.Context(),
		`SELECT id, name, email, role, kitchen_id, recipient_id 
		 FROM users WHERE id = $1`, claims.Subject,
	).Scan(&user.ID, &user.Name, &user.Email, &user.Role, &user.KitchenID, &user.RecipientID)

	if err != nil {
		httpapi.NewNotFound("user").Render(w)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(user)
}
