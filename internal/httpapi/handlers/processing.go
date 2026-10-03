package handlers

import (
	"encoding/json"
	"net/http"
	"strings"
	"time"

	"github.com/sih26234/food-waste/internal/httpapi"
	"github.com/sih26234/food-waste/internal/services"
)

// ProcessingHandler holds dependencies for processing metrics endpoints.
type ProcessingHandler struct {
	procSvc *services.ProcessingService
}

// NewProcessingHandler constructs a ProcessingHandler.
func NewProcessingHandler(procSvc *services.ProcessingService) *ProcessingHandler {
	return &ProcessingHandler{procSvc: procSvc}
}

// ProcessingMetricsRequest is the body for POST /processing/metrics.
type ProcessingMetricsRequest struct {
	Date            string  `json:"date"`
	KitchenID       string  `json:"kitchen_id,omitempty"`
	RawMaterialKg   float64 `json:"raw_material_kg"`
	OutputKg        float64 `json:"output_kg"`
	WasteKg         float64 `json:"waste_kg"`
	RuntimeMinutes  float64 `json:"runtime_minutes"`
	DowntimeMinutes float64 `json:"downtime_minutes"`
	EnergyKWh       float64 `json:"energy_kwh"`
}

// PostMetrics handles POST /processing/metrics.
func (h *ProcessingHandler) PostMetrics(w http.ResponseWriter, r *http.Request) {
	claims, ok := requireClaims(w, r)
	if !ok {
		return
	}
	if !requireRole(w, claims, "KITCHEN", "ADMIN") {
		return
	}

	var req ProcessingMetricsRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		httpapi.NewValidation("invalid request body", err.Error()).Render(w)
		return
	}

	kid := strings.TrimSpace(req.KitchenID)
	if kid == "" {
		kid = claims.KitchenID
	}
	if kid == "" {
		httpapi.NewValidation("kitchen_id is required", nil).Render(w)
		return
	}

	date := time.Now().UTC()
	if req.Date != "" {
		if t, err := time.Parse("2006-01-02", req.Date); err == nil {
			date = t.UTC()
		}
	}

	res, err := h.procSvc.RecordMetrics(r.Context(), services.ProcessingInput{
		KitchenID:       kid,
		Date:            date,
		PeriodStart:     date,
		PeriodEnd:       date,
		RawMaterialKg:   req.RawMaterialKg,
		OutputKg:        req.OutputKg,
		WasteKg:         req.WasteKg,
		RuntimeMinutes:  req.RuntimeMinutes,
		DowntimeMinutes: req.DowntimeMinutes,
		EnergyKWh:       req.EnergyKWh,
	})
	if err != nil {
		httpapi.NewValidation("failed to record processing metrics", err.Error()).Render(w)
		return
	}

	writeJSON(w, http.StatusCreated, res)
}

// Analytics handles GET /processing/analytics.
func (h *ProcessingHandler) Analytics(w http.ResponseWriter, r *http.Request) {
	claims, ok := requireClaims(w, r)
	if !ok {
		return
	}

	q := r.URL.Query()
	kid := q.Get("kitchen_id")
	if kid == "" {
		kid = claims.KitchenID
	}
	if kid == "" {
		httpapi.NewValidation("kitchen_id is required", nil).Render(w)
		return
	}

	from := time.Now().UTC().AddDate(0, 0, -30)
	to := time.Now().UTC()
	if fromStr := q.Get("from"); fromStr != "" {
		if t, err := time.Parse("2006-01-02", fromStr); err == nil {
			from = t.UTC()
		}
	}
	if toStr := q.Get("to"); toStr != "" {
		if t, err := time.Parse("2006-01-02", toStr); err == nil {
			to = t.UTC()
		}
	}

	summary, err := h.procSvc.GetAnalytics(r.Context(), kid, from, to)
	if err != nil {
		httpapi.NewInternal("failed to fetch processing analytics: " + err.Error()).Render(w)
		return
	}

	writeJSON(w, http.StatusOK, summary)
}
