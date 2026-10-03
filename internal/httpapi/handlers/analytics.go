package handlers

import (
	"net/http"

	"github.com/go-chi/chi/v5"
	"github.com/sih26234/food-waste/internal/httpapi"
	"github.com/sih26234/food-waste/internal/services"
)

// AnalyticsHandler holds dependencies for impact analytics endpoints.
type AnalyticsHandler struct {
	analyticsSvc *services.AnalyticsService
}

// NewAnalyticsHandler constructs an AnalyticsHandler.
func NewAnalyticsHandler(analyticsSvc *services.AnalyticsService) *AnalyticsHandler {
	return &AnalyticsHandler{analyticsSvc: analyticsSvc}
}

// RegisterRoutes mounts analytics routes onto r.
func (h *AnalyticsHandler) RegisterRoutes(r chi.Router) {
	r.Get("/analytics/impact", h.Impact)
}

// Impact handles GET /analytics/impact.
func (h *AnalyticsHandler) Impact(w http.ResponseWriter, r *http.Request) {
	claims, ok := requireClaims(w, r)
	if !ok {
		return
	}

	kitchenID := r.URL.Query().Get("kitchen_id")
	if kitchenID == "" && claims.Role == "KITCHEN" {
		kitchenID = claims.KitchenID
	}

	impact, err := h.analyticsSvc.GetImpact(r.Context(), kitchenID)
	if err != nil {
		httpapi.NewInternal("failed to fetch impact analytics: " + err.Error()).Render(w)
		return
	}

	writeJSON(w, http.StatusOK, impact)
}
