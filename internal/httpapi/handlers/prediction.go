package handlers

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/go-chi/chi/v5"
	"github.com/sih26234/food-waste/internal/httpapi"
	"github.com/sih26234/food-waste/internal/services"
)

// PredictionHandler holds dependencies for prediction endpoints.
type PredictionHandler struct {
	predSvc *services.PredictionService
}

// NewPredictionHandler constructs a PredictionHandler.
func NewPredictionHandler(predSvc *services.PredictionService) *PredictionHandler {
	return &PredictionHandler{predSvc: predSvc}
}

// RegisterRoutes mounts prediction routes onto r.
func (h *PredictionHandler) RegisterRoutes(r chi.Router) {
	r.Post("/predict-demand", h.PredictDemand)
	r.Post("/planning/what-if", h.WhatIf)
	r.Post("/feedback/outcome", h.RecordOutcome)
}

// PredictDemand handles POST /predict-demand.
func (h *PredictionHandler) PredictDemand(w http.ResponseWriter, r *http.Request) {
	claims, ok := requireClaims(w, r)
	if !ok {
		return
	}

	var req services.PredictDemandInput
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		httpapi.NewValidation("invalid request body", err.Error()).Render(w)
		return
	}

	if req.KitchenID == "" {
		req.KitchenID = claims.KitchenID
	}

	out, err := h.predSvc.PredictDemand(r.Context(), req)
	if err != nil {
		httpapi.NewInternal("failed to generate demand prediction: " + err.Error()).Render(w)
		return
	}

	writeJSON(w, http.StatusOK, out)
}

// WhatIf handles POST /planning/what-if.
func (h *PredictionHandler) WhatIf(w http.ResponseWriter, r *http.Request) {
	claims, ok := requireClaims(w, r)
	if !ok {
		return
	}

	var req services.PredictDemandInput
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		httpapi.NewValidation("invalid request body", err.Error()).Render(w)
		return
	}

	if req.KitchenID == "" {
		req.KitchenID = claims.KitchenID
	}

	out, err := h.predSvc.PredictDemand(r.Context(), req)
	if err != nil {
		httpapi.NewInternal("failed to run what-if simulation: " + err.Error()).Render(w)
		return
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"simulation":           out,
		"baseline_consumption": out.PredictedConsumption,
		"status":               "simulated",
	})
}

// RecordOutcome handles POST /feedback/outcome.
func (h *FeedbackOutcomeRequest) RecordOutcome(w http.ResponseWriter, r *http.Request) {
	// struct helper if needed
}

// RecordOutcome handles POST /feedback/outcome on PredictionHandler.
func (h *PredictionHandler) RecordOutcome(w http.ResponseWriter, r *http.Request) {
	claims, ok := requireClaims(w, r)
	if !ok {
		return
	}

	var req services.FeedbackOutcomeInput
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		httpapi.NewValidation("invalid request body", err.Error()).Render(w)
		return
	}

	if strings.TrimSpace(req.MealID) == "" && strings.TrimSpace(req.MatchID) == "" {
		httpapi.NewValidation("meal_id or match_id is required", nil).Render(w)
		return
	}

	err := h.predSvc.RecordOutcome(r.Context(), req)
	if err != nil {
		httpapi.NewInternal("failed to record outcome feedback: " + err.Error()).Render(w)
		return
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"status":  "recorded",
		"message": "feedback recorded successfully",
	})
}
