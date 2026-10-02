package handlers

import (
	"encoding/json"
	"net/http"

	"github.com/sih26234/food-waste/internal/httpapi"
	"github.com/sih26234/food-waste/internal/services"
)

type SyncHandler struct {
	sync *services.SyncService
}

func NewSyncHandler(s *services.SyncService) *SyncHandler {
	return &SyncHandler{sync: s}
}

func (h *SyncHandler) Batch(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Items []services.SyncItem `json:"items"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		httpapi.NewValidation("invalid request", err.Error()).Render(w)
		return
	}

	result, err := h.sync.ProcessBatch(r.Context(), "user-1", "KITCHEN", "kitchen-1", req.Items)
	if err != nil {
		httpapi.NewInternal(err.Error()).Render(w)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(result)
}
