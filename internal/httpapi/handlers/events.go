package handlers

import (
	"fmt"
	"net/http"
	"time"

	"github.com/sih26234/food-waste/internal/sse"
)

type EventsHandler struct {
	hub *sse.Hub
}

func NewEventsHandler(hub *sse.Hub) *EventsHandler {
	return &EventsHandler{hub: hub}
}

func (h *EventsHandler) Stream(w http.ResponseWriter, r *http.Request) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming unsupported", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	kitchenID := r.URL.Query().Get("kitchen_id")
	if kitchenID == "" {
		kitchenID = "default"
	}

	ticker := time.NewTicker(15 * time.Second)
	defer ticker.Stop()

	fmt.Fprintf(w, "event: connected\ndata: {\"status\":\"connected\",\"kitchen_id\":\"%s\"}\n\n", kitchenID)
	flusher.Flush()

	for {
		select {
		case <-r.Context().Done():
			return
		case <-ticker.C:
			fmt.Fprintf(w, "event: ping\ndata: {\"ts\":%d}\n\n", time.Now().Unix())
			flusher.Flush()
		}
	}
}
