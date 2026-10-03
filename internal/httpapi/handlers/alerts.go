package handlers

import (
	"encoding/json"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/sih26234/food-waste/internal/httpapi"
	"github.com/sih26234/food-waste/internal/services"
)

// AlertsHandler holds dependencies for alert endpoints.
type AlertsHandler struct {
	alertSvc *services.AlertService
}

// NewAlertsHandler constructs an AlertsHandler.
func NewAlertsHandler(alertSvc *services.AlertService) *AlertsHandler {
	return &AlertsHandler{alertSvc: alertSvc}
}

// RegisterRoutes mounts alert routes onto r.
func (h *AlertsHandler) RegisterRoutes(r chi.Router) {
	r.Get("/alerts", h.ListAlerts)
	r.Post("/alerts/{id}/ack", h.Acknowledge)
}

// ListAlerts handles GET /alerts.
func (h *AlertsHandler) ListAlerts(w http.ResponseWriter, r *http.Request) {
	claims, ok := requireClaims(w, r)
	if !ok {
		return
	}

	q := r.URL.Query()
	kitchenID := q.Get("kitchen_id")
	if kitchenID == "" && claims.Role == "KITCHEN" {
		kitchenID = claims.KitchenID
	}
	severity := q.Get("severity")
	unackedOnly, _ := strconv.ParseBool(q.Get("unacked_only"))
	page, _ := strconv.Atoi(q.Get("page"))
	if page < 1 {
		page = 1
	}
	limit, _ := strconv.Atoi(q.Get("limit"))
	if limit < 1 || limit > 200 {
		limit = 50
	}
	offset := (page - 1) * limit

	alerts, total, err := h.alertSvc.List(r.Context(), kitchenID, severity, unackedOnly, limit, offset)
	if err != nil {
		httpapi.NewInternal("failed to list alerts: " + err.Error()).Render(w)
		return
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"alerts": alerts,
		"total":  total,
		"page":   page,
		"limit":  limit,
	})
}

// Acknowledge handles POST /alerts/{id}/ack.
func (h *AlertsHandler) Acknowledge(w http.ResponseWriter, r *http.Request) {
	claims, ok := requireClaims(w, r)
	if !ok {
		return
	}

	alertID := strings.TrimSpace(chi.URLParam(r, "id"))
	if alertID == "" {
		httpapi.NewValidation("alert id is required", nil).Render(w)
		return
	}

	alert, err := h.alertSvc.Ack(r.Context(), alertID, claims.Subject)
	if err != nil {
		renderServiceError(w, err)
		return
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"alert_id": alert.ID,
		"acked_at": time.Now().UTC(),
		"status":   "ACKED",
	})
}
