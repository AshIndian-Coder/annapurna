package handlers

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/go-chi/chi/v5"
	"github.com/sih26234/food-waste/internal/httpapi"
	"github.com/sih26234/food-waste/internal/services"
)

// MatchingHandler holds dependencies for matching endpoints.
type MatchingHandler struct {
	matchSvc *services.MatchingService
}

// NewMatchingHandler constructs a MatchingHandler.
func NewMatchingHandler(matchSvc *services.MatchingService) *MatchingHandler {
	return &MatchingHandler{matchSvc: matchSvc}
}

// CreateMatchRequest is the body for POST /match.
type CreateMatchRequest struct {
	BatchID    string `json:"batch_id"`
	SurplusID  string `json:"surplus_id,omitempty"`
	MaxResults int    `json:"max_results,omitempty"`
}

// RespondRequest is the body for POST /matches/{id}/respond.
type RespondRequest struct {
	Decision string `json:"decision,omitempty"`
	Accept   *bool  `json:"accept,omitempty"`
	Reason   string `json:"reason,omitempty"`
}

// CreateMatch handles POST /match.
func (h *MatchingHandler) CreateMatch(w http.ResponseWriter, r *http.Request) {
	claims, ok := requireClaims(w, r)
	if !ok {
		return
	}
	if !requireRole(w, claims, "KITCHEN", "ADMIN") {
		return
	}

	var req CreateMatchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		httpapi.NewValidation("invalid request body", err.Error()).Render(w)
		return
	}

	batchID := strings.TrimSpace(req.BatchID)
	if batchID == "" {
		batchID = strings.TrimSpace(req.SurplusID)
	}
	if batchID == "" {
		httpapi.NewValidation("batch_id is required", nil).Render(w)
		return
	}

	if req.MaxResults <= 0 {
		req.MaxResults = 5
	}

	results, err := h.matchSvc.Match(r.Context(), batchID, req.MaxResults, claims.KitchenID)
	if err != nil {
		renderServiceError(w, err)
		return
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"matches": results,
		"count":   len(results),
	})
}

// Respond handles POST /matches/{id}/respond.
func (h *MatchingHandler) Respond(w http.ResponseWriter, r *http.Request) {
	claims, ok := requireClaims(w, r)
	if !ok {
		return
	}
	if !requireRole(w, claims, "NGO", "ADMIN") {
		return
	}

	matchID := strings.TrimSpace(chi.URLParam(r, "id"))
	if matchID == "" {
		httpapi.NewValidation("match id is required", nil).Render(w)
		return
	}

	var req RespondRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		httpapi.NewValidation("invalid request body", err.Error()).Render(w)
		return
	}

	decision := strings.ToUpper(strings.TrimSpace(req.Decision))
	if decision == "" {
		if req.Accept != nil && *req.Accept {
			decision = "ACCEPTED"
		} else {
			decision = "DECLINED"
		}
	}

	if decision != "ACCEPTED" && decision != "DECLINED" {
		httpapi.NewValidation("decision must be ACCEPTED or DECLINED", nil).Render(w)
		return
	}

	err := h.matchSvc.Respond(r.Context(), matchID, decision)
	if err != nil {
		renderServiceError(w, err)
		return
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"match_id": matchID,
		"status":   decision,
		"message":  "match response recorded successfully",
	})
}
