package handlers

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/go-chi/chi/v5"
	"github.com/sih26234/food-waste/internal/httpapi"
	"github.com/sih26234/food-waste/internal/services"
)

// RoutingHandler holds dependencies for route optimisation endpoints.
type RoutingHandler struct {
	routeSvc *services.RoutingService
}

// NewRoutingHandler constructs a RoutingHandler.
func NewRoutingHandler(routeSvc *services.RoutingService) *RoutingHandler {
	return &RoutingHandler{routeSvc: routeSvc}
}

// RegisterRoutes mounts routing routes onto r.
func (h *RoutingHandler) RegisterRoutes(r chi.Router) {
	r.Post("/route", h.Optimise)
}

// RouteRequest is the body for POST /route.
type RouteRequest struct {
	BatchID      string   `json:"batch_id"`
	RecipientIDs []string `json:"recipient_ids"`
	VehicleID    string   `json:"vehicle_id,omitempty"`
	DriverID     string   `json:"driver_id,omitempty"`
}

// Optimise handles POST /route.
func (h *RoutingHandler) Optimise(w http.ResponseWriter, r *http.Request) {
	claims, ok := requireClaims(w, r)
	if !ok {
		return
	}
	if !requireRole(w, claims, "KITCHEN", "LOGISTICS", "ADMIN") {
		return
	}

	var req RouteRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		httpapi.NewValidation("invalid request body", err.Error()).Render(w)
		return
	}

	batchID := strings.TrimSpace(req.BatchID)
	if batchID == "" {
		httpapi.NewValidation("batch_id is required", nil).Render(w)
		return
	}

	driverID := req.DriverID
	if driverID == "" {
		driverID = req.VehicleID
	}
	if driverID == "" && claims.Role == "LOGISTICS" {
		driverID = claims.Subject
	}

	plan, err := h.routeSvc.PlanRoute(r.Context(), batchID, req.RecipientIDs, driverID)
	if err != nil {
		httpapi.NewInternal("failed to plan route: " + err.Error()).Render(w)
		return
	}

	writeJSON(w, http.StatusOK, plan)
}
