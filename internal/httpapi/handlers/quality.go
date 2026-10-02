package handlers

import (
	"encoding/json"
	"net/http"
	"strconv"

	"github.com/sih26234/food-waste/internal/httpapi"
	"github.com/sih26234/food-waste/internal/services"
)

type QualityHandler struct {
	quality *services.QualityService
}

func NewQualityHandler(q *services.QualityService) *QualityHandler {
	return &QualityHandler{quality: q}
}

func (h *QualityHandler) Check(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseMultipartForm(8 << 20); err != nil {
		httpapi.NewValidation("failed to parse form", err.Error()).Render(w)
		return
	}

	batchID := r.FormValue("batch_id")
	if batchID == "" {
		httpapi.NewValidation("batch_id is required", nil).Render(w)
		return
	}

	var tempC *float64
	if tempStr := r.FormValue("temperature_c"); tempStr != "" {
		if val, err := strconv.ParseFloat(tempStr, 64); err == nil {
			tempC = &val
		}
	}

	var preview *json.RawMessage
	if prevStr := r.FormValue("device_preview"); prevStr != "" {
		raw := json.RawMessage(prevStr)
		preview = &raw
	}

	res, err := h.quality.CheckQuality(r.Context(), batchID, "user-1", "/storage/uploads/temp.jpg", tempC, preview)
	if err != nil {
		httpapi.NewInternal(err.Error()).Render(w)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"batch_id": batchID,
		"visual": map[string]any{
			"status":        res.VisualStatus,
			"risk_level":    res.RiskLevel,
			"confidence":    res.Confidence,
			"reason":        res.Reason,
			"model_version": "cv-v1",
		},
		"safety_decision": map[string]any{
			"status":               res.SafetyDecision,
			"danger_zone_minutes": res.DangerZoneMinutes,
			"model_version":        "fusion-v1",
		},
		"requires_human_approval": true,
	})
}
