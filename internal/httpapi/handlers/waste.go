package handlers

import (
	"encoding/json"
	"net/http"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/sih26234/food-waste/internal/httpapi"
	"github.com/sih26234/food-waste/internal/store/queries"
)

// WasteHandler holds dependencies for waste endpoints.
type WasteHandler struct {
	pool *pgxpool.Pool
}

// NewWasteHandler constructs a WasteHandler.
func NewWasteHandler(pool *pgxpool.Pool) *WasteHandler {
	return &WasteHandler{pool: pool}
}

// LogWasteRequest is the body for POST /waste.
type LogWasteRequest struct {
	KitchenID       string     `json:"kitchen_id"`
	MealID          string     `json:"meal_id,omitempty"`
	MealType        string     `json:"meal_type,omitempty"`
	ItemName        string     `json:"item_name"`
	FoodType        string     `json:"food_type,omitempty"`
	QuantityKg      float64    `json:"quantity_kg"`
	Cause           string     `json:"cause"`
	WasteType       string     `json:"waste_type,omitempty"`
	DiversionStream string     `json:"diversion_stream,omitempty"`
	BatchID         string     `json:"batch_id,omitempty"`
	Notes           string     `json:"notes,omitempty"`
	RecordedAt      *time.Time `json:"recorded_at,omitempty"`
	ClientEventID   *string    `json:"client_event_id,omitempty"`
}

// LogWasteResponse is returned by POST /waste.
type LogWasteResponse struct {
	WasteID    string    `json:"waste_id"`
	KitchenID  string    `json:"kitchen_id"`
	QuantityKg float64   `json:"quantity_kg"`
	CreatedAt  time.Time `json:"created_at"`
}

// WasteAnalyticsResponse is returned by GET /waste/analytics.
type WasteAnalyticsResponse struct {
	Period          string             `json:"period"`
	TotalKg         float64            `json:"total_kg"`
	ByMealType      map[string]float64 `json:"by_meal_type"`
	ByCause         map[string]float64 `json:"by_cause"`
	ByDiversion     map[string]float64 `json:"by_diversion_stream"`
	DailyTrend      []DailyWastePoint  `json:"daily_trend"`
	TopItems        []WasteItemSummary `json:"top_items"`
	ReductionVsLast float64            `json:"reduction_vs_last_period_pct"`
}

// DailyWastePoint is a single point in the daily trend series.
type DailyWastePoint struct {
	Date   string  `json:"date"`
	Kg     float64 `json:"kg"`
	Events int     `json:"events"`
}

// WasteItemSummary summarises waste for a single food item.
type WasteItemSummary struct {
	ItemName   string  `json:"item_name"`
	TotalKg    float64 `json:"total_kg"`
	EventCount int     `json:"event_count"`
}

// LogWaste handles POST /waste.
func (h *WasteHandler) LogWaste(w http.ResponseWriter, r *http.Request) {
	claims, ok := requireClaims(w, r)
	if !ok {
		return
	}
	if !requireRole(w, claims, "KITCHEN", "ADMIN") {
		return
	}

	var req LogWasteRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		httpapi.NewValidation("invalid request body", err.Error()).Render(w)
		return
	}

	kitchenIDStr := strings.TrimSpace(req.KitchenID)
	if kitchenIDStr == "" {
		kitchenIDStr = claims.KitchenID
	}
	if kitchenIDStr == "" {
		httpapi.NewValidation("kitchen_id is required", nil).Render(w)
		return
	}

	kitchenUUID, err := uuid.Parse(kitchenIDStr)
	if err != nil {
		httpapi.NewValidation("invalid kitchen_id UUID", err.Error()).Render(w)
		return
	}

	foodType := strings.TrimSpace(req.ItemName)
	if foodType == "" {
		foodType = strings.TrimSpace(req.FoodType)
	}
	if foodType == "" || req.QuantityKg <= 0 {
		httpapi.NewValidation("item_name and quantity_kg > 0 are required", nil).Render(w)
		return
	}

	recordedAt := time.Now().UTC()
	if req.RecordedAt != nil && !req.RecordedAt.IsZero() {
		recordedAt = req.RecordedAt.UTC()
	}

	var mealUUID *uuid.UUID
	if req.MealID != "" {
		if id, err := uuid.Parse(req.MealID); err == nil {
			mealUUID = &id
		}
	}

	wasteType := req.WasteType
	if wasteType == "" {
		wasteType = req.MealType
	}
	if wasteType == "" {
		wasteType = "other"
	}

	cause := req.Cause
	if cause == "" {
		cause = "OTHER"
	}

	wasteID := uuid.New()
	created, err := queries.CreateWaste(r.Context(), h.pool, queries.CreateWasteParams{
		ID:            wasteID,
		MealID:        mealUUID,
		KitchenID:     kitchenUUID,
		FoodType:      foodType,
		QuantityKg:    req.QuantityKg,
		Cause:         cause,
		WasteType:     wasteType,
		Note:          req.Notes,
		RecordedAt:    recordedAt,
		ClientEventID: req.ClientEventID,
	})
	if err != nil {
		httpapi.NewInternal("failed to log waste: " + err.Error()).Render(w)
		return
	}

	writeJSON(w, http.StatusCreated, LogWasteResponse{
		WasteID:    created.ID.String(),
		KitchenID:  created.KitchenID.String(),
		QuantityKg: created.QuantityKg,
		CreatedAt:  created.CreatedAt,
	})
}

// Analytics handles GET /waste/analytics.
func (h *WasteHandler) Analytics(w http.ResponseWriter, r *http.Request) {
	claims, ok := requireClaims(w, r)
	if !ok {
		return
	}

	q := r.URL.Query()
	kitchenIDStr := q.Get("kitchen_id")
	if kitchenIDStr == "" {
		kitchenIDStr = claims.KitchenID
	}
	if kitchenIDStr == "" {
		httpapi.NewValidation("kitchen_id is required", nil).Render(w)
		return
	}

	kitchenUUID, err := uuid.Parse(kitchenIDStr)
	if err != nil {
		httpapi.NewValidation("invalid kitchen_id UUID", err.Error()).Render(w)
		return
	}

	fromStr := q.Get("from")
	toStr := q.Get("to")
	period := q.Get("period")
	if period == "" {
		period = "day"
	}

	from := time.Now().UTC().AddDate(0, 0, -30)
	to := time.Now().UTC()
	if fromStr != "" {
		if t, err := time.Parse("2006-01-02", fromStr); err == nil {
			from = t.UTC()
		}
	}
	if toStr != "" {
		if t, err := time.Parse("2006-01-02", toStr); err == nil {
			to = t.UTC().Add(24*time.Hour - time.Nanosecond)
		}
	}

	// 1. Aggregates by cause & food_type
	analyticsRows, err := queries.WasteAnalytics(r.Context(), h.pool, kitchenUUID, from, to)
	if err != nil {
		httpapi.NewInternal("failed to fetch waste analytics: " + err.Error()).Render(w)
		return
	}

	var totalKg float64
	byCause := make(map[string]float64)
	byMealType := make(map[string]float64)
	topItemsMap := make(map[string]*WasteItemSummary)

	for _, row := range analyticsRows {
		totalKg += row.TotalKg
		if row.Cause != "" {
			byCause[row.Cause] += row.TotalKg
		}
		if row.WasteType != "" {
			byMealType[row.WasteType] += row.TotalKg
		}
		if row.FoodType != "" {
			if existing, ok := topItemsMap[row.FoodType]; ok {
				existing.TotalKg += row.TotalKg
				existing.EventCount += int(row.RecordCount)
			} else {
				topItemsMap[row.FoodType] = &WasteItemSummary{
					ItemName:   row.FoodType,
					TotalKg:    row.TotalKg,
					EventCount: int(row.RecordCount),
				}
			}
		}
	}

	// 2. Daily trend from v_daily_waste
	trendRows, err := h.pool.Query(r.Context(),
		`SELECT date::text, SUM(waste_kg), SUM(records)
		 FROM v_daily_waste
		 WHERE kitchen_id = $1 AND date BETWEEN $2::date AND $3::date
		 GROUP BY date
		 ORDER BY date ASC`,
		kitchenUUID, from.Format("2006-01-02"), to.Format("2006-01-02"),
	)
	dailyTrend := make([]DailyWastePoint, 0)
	if err == nil {
		defer trendRows.Close()
		for trendRows.Next() {
			var p DailyWastePoint
			var kg float64
			var cnt int64
			if err := trendRows.Scan(&p.Date, &kg, &cnt); err == nil {
				p.Kg = kg
				p.Events = int(cnt)
				dailyTrend = append(dailyTrend, p)
			}
		}
	}

	topItems := make([]WasteItemSummary, 0, len(topItemsMap))
	for _, it := range topItemsMap {
		topItems = append(topItems, *it)
	}

	resp := WasteAnalyticsResponse{
		Period:          period,
		TotalKg:         totalKg,
		ByMealType:      byMealType,
		ByCause:         byCause,
		ByDiversion:     map[string]float64{},
		DailyTrend:      dailyTrend,
		TopItems:        topItems,
		ReductionVsLast: 0,
	}

	writeJSON(w, http.StatusOK, resp)
}
