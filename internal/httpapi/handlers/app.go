package handlers

import (
	"encoding/json"
	"net/http"
	"os"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/sih26234/food-waste/internal/httpapi"
)

// AppHandler holds dependencies for mobile app config endpoints.
type AppHandler struct {
	manifestPath string
}

// NewAppHandler constructs an AppHandler.
func NewAppHandler(manifestPath string) *AppHandler {
	if manifestPath == "" {
		manifestPath = "static/bundles/cv/cv-v1/manifest.json"
	}
	return &AppHandler{manifestPath: manifestPath}
}

// RegisterRoutes mounts app routes onto r.
func (h *AppHandler) RegisterRoutes(r chi.Router) {
	r.Get("/app/config", h.Config)
	r.Get("/app/model-bundle", h.ModelBundle)
}

// Config handles GET /app/config.
func (h *AppHandler) Config(w http.ResponseWriter, r *http.Request) {
	var modelBundle any
	if data, err := os.ReadFile(h.manifestPath); err == nil {
		_ = json.Unmarshal(data, &modelBundle)
	}

	resp := map[string]any{
		"min_supported_version": "1.0.0",
		"force_update":           false,
		"feature_flags": map[string]bool{
			"offline_sync":        true,
			"iot_telemetry":       true,
			"ai_demand_forecast":  true,
			"cv_quality_analysis": true,
		},
		"model_bundle": modelBundle,
		"display": map[string]any{
			"carbon_factor_source": "WRI/FAO 2026",
			"danger_zone_band_c":   []int{5, 60},
		},
		"fetched_at": time.Now().UTC(),
	}

	writeJSON(w, http.StatusOK, resp)
}

// ModelBundle handles GET /app/model-bundle.
func (h *AppHandler) ModelBundle(w http.ResponseWriter, r *http.Request) {
	data, err := os.ReadFile(h.manifestPath)
	if err != nil {
		httpapi.NewNotFound("model manifest not found").Render(w)
		return
	}

	var manifest map[string]any
	if err := json.Unmarshal(data, &manifest); err != nil {
		httpapi.NewInternal("failed to parse manifest").Render(w)
		return
	}

	writeJSON(w, http.StatusOK, manifest)
}
