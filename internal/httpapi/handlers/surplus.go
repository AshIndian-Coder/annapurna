package handlers

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/go-chi/chi/v5"
	"github.com/sih26234/food-waste/internal/auth"
	"github.com/sih26234/food-waste/internal/httpapi"
	"github.com/sih26234/food-waste/internal/services"
)

type SurplusHandler struct {
	surplus *services.SurplusService
	approve *services.ApproveService
}

func NewSurplusHandler(s *services.SurplusService, a *services.ApproveService) *SurplusHandler {
	return &SurplusHandler{surplus: s, approve: a}
}

// claimsFromRequest extracts and verifies the JWT from the Authorization header.
func claimsFromRequest(r *http.Request) (*auth.Claims, error) {
	header := r.Header.Get("Authorization")
	tokenStr := strings.TrimPrefix(header, "Bearer ")
	return auth.VerifyAccessToken(tokenStr)
}

func (h *SurplusHandler) Create(w http.ResponseWriter, r *http.Request) {
	claims, err := claimsFromRequest(r)
	if err != nil {
		httpapi.NewUnauthorized("invalid or missing token").Render(w)
		return
	}

	var req services.CreateSurplusRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		httpapi.NewValidation("invalid request", err.Error()).Render(w)
		return
	}
	// KitchenID from token scope; empty → service will insert NULL
	batch, err := h.surplus.CreateSurplus(r.Context(), claims.KitchenID, claims.Subject, req)
	if err != nil {
		httpapi.NewInternal(err.Error()).Render(w)
		return
	}
	w.WriteHeader(http.StatusCreated)
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(batch)
}

func (h *SurplusHandler) List(w http.ResponseWriter, r *http.Request) {
	result, err := h.surplus.ListSurplus(r.Context(), services.ListSurplusParams{Limit: 20})
	if err != nil {
		httpapi.NewInternal(err.Error()).Render(w)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(result)
}

func (h *SurplusHandler) Get(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	batch, err := h.surplus.GetSurplus(r.Context(), id)
	if err != nil {
		httpapi.NewNotFound("surplus batch").Render(w)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(batch)
}

func (h *SurplusHandler) Approve(w http.ResponseWriter, r *http.Request) {
	claims, err := claimsFromRequest(r)
	if err != nil {
		httpapi.NewUnauthorized("invalid or missing token").Render(w)
		return
	}
	id := chi.URLParam(r, "id")
	var req services.ApproveRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		httpapi.NewValidation("invalid request", err.Error()).Render(w)
		return
	}
	err = h.approve.Approve(r.Context(), id, claims.Subject, claims.Role, req)
	if err != nil {
		httpapi.NewConflict("APPROVE_FAILED", err.Error()).Render(w)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"batch_id": id, "status": "APPROVED"})
}

func (h *SurplusHandler) Divert(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	err := h.surplus.Transition(r.Context(), id, services.StatusDiverted)
	if err != nil {
		httpapi.NewConflict("DIVERT_FAILED", err.Error()).Render(w)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"batch_id": id, "status": "DIVERTED"})
}
