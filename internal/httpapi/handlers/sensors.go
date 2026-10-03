package handlers

import (
	"encoding/json"
	"net/http"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/sih26234/food-waste/internal/httpapi"
	"github.com/sih26234/food-waste/internal/services"
)

// SensorsHandler holds dependencies for IoT sensor endpoints.
type SensorsHandler struct {
	sensorSvc *services.SensorService
}

// NewSensorsHandler constructs a SensorsHandler.
func NewSensorsHandler(sensorSvc *services.SensorService) *SensorsHandler {
	return &SensorsHandler{sensorSvc: sensorSvc}
}

// RegisterRoutes mounts sensor routes onto r.
func (h *SensorsHandler) RegisterRoutes(r chi.Router) {
	r.Post("/sensors/readings", h.PostReadings)
	r.Get("/sensors/latest", h.Latest)
	r.Get("/sensors/history", h.History)
}

// SensorReadingInput represents one IoT reading payload.
type SensorReadingInput struct {
	SensorID  string    `json:"sensor_id"`
	DeviceID  string    `json:"device_id,omitempty"`
	KitchenID string    `json:"kitchen_id,omitempty"`
	BatchID   string    `json:"batch_id,omitempty"`
	TempC     *float64  `json:"temp_c,omitempty"`
	Temp      *float64  `json:"temperature_c,omitempty"`
	Humidity  *float64  `json:"humidity_pct,omitempty"`
	EnergyKWh *float64  `json:"energy_kwh,omitempty"`
	ReadingAt time.Time `json:"reading_at,omitempty"`
	Timestamp time.Time `json:"timestamp,omitempty"`
}

// PostReadingsRequest is the body for POST /sensors/readings.
type PostReadingsRequest struct {
	Readings []SensorReadingInput `json:"readings"`
}

// PostReadings handles POST /sensors/readings.
func (h *SensorsHandler) PostReadings(w http.ResponseWriter, r *http.Request) {
	claims, ok := requireClaims(w, r)
	if !ok {
		return
	}
	if !requireRole(w, claims, "SYSTEM", "ADMIN", "KITCHEN") {
		return
	}

	var req PostReadingsRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		// Single reading fallback
		var single SensorReadingInput
		if err2 := json.NewDecoder(r.Body).Decode(&single); err2 == nil && single.SensorID != "" {
			req.Readings = []SensorReadingInput{single}
		} else {
			httpapi.NewValidation("invalid request body", err.Error()).Render(w)
			return
		}
	}

	if len(req.Readings) == 0 {
		httpapi.NewValidation("readings cannot be empty", nil).Render(w)
		return
	}

	var domainReadings []services.SensorReading
	for _, raw := range req.Readings {
		kid := raw.KitchenID
		if kid == "" {
			kid = claims.KitchenID
		}
		temp := raw.Temp
		if temp == nil {
			temp = raw.TempC
		}
		ts := raw.Timestamp
		if ts.IsZero() {
			ts = raw.ReadingAt
		}
		if ts.IsZero() {
			ts = time.Now().UTC()
		}

		domainReadings = append(domainReadings, services.SensorReading{
			SensorID:  raw.SensorID,
			KitchenID: kid,
			BatchID:   raw.BatchID,
			Timestamp: ts,
			Temp:      temp,
			Humidity:  raw.Humidity,
			EnergyKWh: raw.EnergyKWh,
		})
	}

	err := h.sensorSvc.Ingest(r.Context(), domainReadings)
	if err != nil {
		httpapi.NewInternal("failed to ingest readings: " + err.Error()).Render(w)
		return
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"accepted": len(domainReadings),
		"status":   "success",
	})
}

// Latest handles GET /sensors/latest.
func (h *SensorsHandler) Latest(w http.ResponseWriter, r *http.Request) {
	claims, ok := requireClaims(w, r)
	if !ok {
		return
	}

	kitchenID := r.URL.Query().Get("kitchen_id")
	if kitchenID == "" {
		kitchenID = claims.KitchenID
	}

	readings, err := h.sensorSvc.GetLatest(r.Context(), kitchenID)
	if err != nil {
		httpapi.NewInternal("failed to fetch latest sensor readings: " + err.Error()).Render(w)
		return
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"readings": readings,
		"count":    len(readings),
	})
}

// History handles GET /sensors/history.
func (h *SensorsHandler) History(w http.ResponseWriter, r *http.Request) {
	claims, ok := requireClaims(w, r)
	if !ok {
		return
	}

	q := r.URL.Query()
	kitchenID := q.Get("kitchen_id")
	if kitchenID == "" {
		kitchenID = claims.KitchenID
	}
	sensorID := q.Get("sensor_id")

	from := time.Now().UTC().Add(-24 * time.Hour)
	to := time.Now().UTC()
	if fromStr := q.Get("from"); fromStr != "" {
		if t, err := time.Parse(time.RFC3339, fromStr); err == nil {
			from = t.UTC()
		}
	}
	if toStr := q.Get("to"); toStr != "" {
		if t, err := time.Parse(time.RFC3339, toStr); err == nil {
			to = t.UTC()
		}
	}

	readings, err := h.sensorSvc.GetHistory(r.Context(), kitchenID, sensorID, from, to)
	if err != nil {
		httpapi.NewInternal("failed to fetch sensor history: " + err.Error()).Render(w)
		return
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"sensor_id": sensorID,
		"from":      from,
		"to":        to,
		"readings":  readings,
		"count":     len(readings),
	})
}
