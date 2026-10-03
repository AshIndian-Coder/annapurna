package handlers

import (
	"math"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/sih26234/food-waste/internal/httpapi"
)

// RecipientsHandler holds dependencies for recipient listing endpoints.
type RecipientsHandler struct {
	pool *pgxpool.Pool
}

// NewRecipientsHandler constructs a RecipientsHandler.
func NewRecipientsHandler(pool *pgxpool.Pool) *RecipientsHandler {
	return &RecipientsHandler{pool: pool}
}

// RecipientItem is one recipient in the list.
type RecipientItem struct {
	RecipientID  string    `json:"recipient_id"`
	Name         string    `json:"name"`
	Type         string    `json:"type"` // ngo | shelter | food_bank | community_kitchen
	ContactName  string    `json:"contact_name,omitempty"`
	ContactPhone string    `json:"contact_phone,omitempty"`
	Address      string    `json:"address,omitempty"`
	Lat          float64   `json:"lat"`
	Lng          float64   `json:"lng"`
	CapacityKg   float64   `json:"capacity_kg"`
	AcceptsTypes []string  `json:"accepts_types"`
	Active       bool      `json:"active"`
	DistanceKm   *float64  `json:"distance_km,omitempty"`
	RegisteredAt time.Time `json:"registered_at"`
}

// ListRecipientsResponse is returned by GET /recipients.
type ListRecipientsResponse struct {
	Recipients []RecipientItem `json:"recipients"`
	Total      int             `json:"total"`
	Page       int             `json:"page"`
	Limit      int             `json:"limit"`
}

func haversineRecipients(lat1, lon1, lat2, lon2 float64) float64 {
	const R = 6371.0
	dLat := (lat2 - lat1) * (math.Pi / 180.0)
	dLon := (lon2 - lon1) * (math.Pi / 180.0)
	rLat1 := lat1 * (math.Pi / 180.0)
	rLat2 := lat2 * (math.Pi / 180.0)
	a := math.Sin(dLat/2)*math.Sin(dLat/2) +
		math.Cos(rLat1)*math.Cos(rLat2)*math.Sin(dLon/2)*math.Sin(dLon/2)
	c := 2 * math.Atan2(math.Sqrt(a), math.Sqrt(1-a))
	return R * c
}

// ListRecipients handles GET /recipients.
func (h *RecipientsHandler) ListRecipients(w http.ResponseWriter, r *http.Request) {
	if _, ok := requireClaims(w, r); !ok {
		return
	}

	q := r.URL.Query()
	rtype := strings.TrimSpace(q.Get("type"))
	activeOnlyStr := q.Get("active")
	latStr := q.Get("lat")
	lngStr := q.Get("lng")
	radiusKmStr := q.Get("radius_km")

	page, _ := strconv.Atoi(q.Get("page"))
	if page < 1 {
		page = 1
	}
	limit, _ := strconv.Atoi(q.Get("limit"))
	if limit < 1 || limit > 200 {
		limit = 50
	}

	activeOnly := true
	if activeOnlyStr != "" {
		if val, err := strconv.ParseBool(activeOnlyStr); err == nil {
			activeOnly = val
		}
	}

	var hasOrigin bool
	var originLat, originLng, radiusKm float64
	if latStr != "" && lngStr != "" {
		oLat, err1 := strconv.ParseFloat(latStr, 64)
		oLng, err2 := strconv.ParseFloat(lngStr, 64)
		if err1 == nil && err2 == nil {
			hasOrigin = true
			originLat = oLat
			originLng = oLng
		}
	}
	if radiusKmStr != "" {
		if rKm, err := strconv.ParseFloat(radiusKmStr, 64); err == nil && rKm > 0 {
			radiusKm = rKm
		}
	}

	query := `SELECT id, name, type, capacity_kg, latitude, longitude, accepts_categories, active, created_at
	          FROM recipients
	          WHERE ($1 = '' OR UPPER(type) = UPPER($1))
	            AND ($2::bool IS NULL OR active = $2)`

	rows, err := h.pool.Query(r.Context(), query, rtype, activeOnly)
	if err != nil {
		httpapi.NewInternal("failed to query recipients: " + err.Error()).Render(w)
		return
	}
	defer rows.Close()

	var allItems []RecipientItem
	for rows.Next() {
		var item RecipientItem
		var accepts []string
		if err := rows.Scan(
			&item.RecipientID,
			&item.Name,
			&item.Type,
			&item.CapacityKg,
			&item.Lat,
			&item.Lng,
			&accepts,
			&item.Active,
			&item.RegisteredAt,
		); err != nil {
			continue
		}
		item.AcceptsTypes = accepts

		if hasOrigin {
			dist := math.Round(haversineRecipients(originLat, originLng, item.Lat, item.Lng)*10) / 10
			if radiusKm > 0 && dist > radiusKm {
				continue
			}
			item.DistanceKm = &dist
		}
		allItems = append(allItems, item)
	}

	if hasOrigin {
		sort.Slice(allItems, func(i, j int) bool {
			if allItems[i].DistanceKm == nil || allItems[j].DistanceKm == nil {
				return false
			}
			return *allItems[i].DistanceKm < *allItems[j].DistanceKm
		})
	}

	total := len(allItems)
	start := (page - 1) * limit
	if start > total {
		start = total
	}
	end := start + limit
	if end > total {
		end = total
	}

	pagedItems := allItems[start:end]

	writeJSON(w, http.StatusOK, ListRecipientsResponse{
		Recipients: pagedItems,
		Total:      total,
		Page:       page,
		Limit:      limit,
	})
}
