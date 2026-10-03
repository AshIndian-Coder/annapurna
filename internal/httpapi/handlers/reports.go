package handlers

import (
	"net/http"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/sih26234/food-waste/internal/httpapi"
	"github.com/sih26234/food-waste/internal/services"
)

// ReportsHandler holds dependencies for ESG report endpoints.
type ReportsHandler struct {
	reportSvc *services.ReportService
}

// NewReportsHandler constructs a ReportsHandler.
func NewReportsHandler(reportSvc *services.ReportService) *ReportsHandler {
	return &ReportsHandler{reportSvc: reportSvc}
}

// RegisterRoutes mounts report routes onto r.
func (h *ReportsHandler) RegisterRoutes(r chi.Router) {
	r.Get("/reports/esg", h.ESG)
	r.Get("/reports/jobs/{id}", h.JobStatus)
}

// ESG handles GET /reports/esg.
func (h *ReportsHandler) ESG(w http.ResponseWriter, r *http.Request) {
	claims, ok := requireClaims(w, r)
	if !ok {
		return
	}

	q := r.URL.Query()
	kitchenID := q.Get("kitchen_id")
	if kitchenID == "" && claims.Role == "KITCHEN" {
		kitchenID = claims.KitchenID
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

	report, err := h.reportSvc.GenerateESG(r.Context(), kitchenID, from, to)
	if err != nil {
		httpapi.NewInternal("failed to generate ESG report: " + err.Error()).Render(w)
		return
	}

	writeJSON(w, http.StatusOK, report)
}

// JobStatus handles GET /reports/jobs/{id}.
func (h *ReportsHandler) JobStatus(w http.ResponseWriter, r *http.Request) {
	if _, ok := requireClaims(w, r); !ok {
		return
	}

	id := strings.TrimSpace(chi.URLParam(r, "id"))
	if id == "" {
		httpapi.NewValidation("job id is required", nil).Render(w)
		return
	}

	status, err := h.reportSvc.GetJobStatus(r.Context(), id)
	if err != nil {
		httpapi.NewInternal("failed to fetch job status: " + err.Error()).Render(w)
		return
	}

	writeJSON(w, http.StatusOK, status)
}
