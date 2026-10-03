package queries

import (
	"context"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
)

// ────────────────────────────────────────────────────────────────────────────
// Domain types
// ────────────────────────────────────────────────────────────────────────────

// Waste mirrors the waste table.
type Waste struct {
	ID            uuid.UUID
	MealID        *uuid.UUID
	KitchenID     uuid.UUID
	FoodType      string
	QuantityKg    float64
	Cause         string
	WasteType     string // preparation | plate | spoilage | other
	Note          string
	RecordedAt    time.Time
	ClientEventID *string
	CreatedAt     time.Time
	UpdatedAt     time.Time
}

// WasteAnalyticsRow is the result of the waste analytics aggregation query.
type WasteAnalyticsRow struct {
	FoodType    string
	WasteType   string
	Cause       string
	TotalKg     float64
	RecordCount int64
}

// CreateWasteParams holds insert fields.
type CreateWasteParams struct {
	ID            uuid.UUID
	MealID        *uuid.UUID
	KitchenID     uuid.UUID
	FoodType      string
	QuantityKg    float64
	Cause         string
	WasteType     string
	Note          string
	RecordedAt    time.Time
	ClientEventID *string
}

// ────────────────────────────────────────────────────────────────────────────
// Queries
// ────────────────────────────────────────────────────────────────────────────

const createWasteSQL = `
INSERT INTO waste
    (id, meal_id, kitchen_id, food_type, quantity_kg, cause, waste_type, note, recorded_at, client_event_id, created_at, updated_at)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, NOW(), NOW())
ON CONFLICT (client_event_id) DO UPDATE SET updated_at = NOW()
RETURNING id, meal_id, kitchen_id, COALESCE(food_type, ''), quantity_kg, COALESCE(cause, ''), COALESCE(waste_type, ''), COALESCE(note, ''), COALESCE(recorded_at, created_at), client_event_id, created_at, updated_at`

// CreateWaste inserts a new waste record.
func CreateWaste(ctx context.Context, db DBTX, p CreateWasteParams) (Waste, error) {
	row := db.QueryRow(ctx, createWasteSQL,
		p.ID, p.MealID, p.KitchenID, p.FoodType,
		p.QuantityKg, p.Cause, p.WasteType, p.Note, p.RecordedAt, p.ClientEventID,
	)
	return scanWaste(row)
}

const listWasteByKitchenSQL = `
SELECT id, meal_id, kitchen_id, COALESCE(food_type, ''), quantity_kg, COALESCE(cause, ''), COALESCE(waste_type, ''), COALESCE(note, ''), COALESCE(recorded_at, created_at), client_event_id, created_at, updated_at
FROM waste
WHERE kitchen_id = $1
  AND ($2::uuid IS NULL OR id > $2)
ORDER BY id ASC
LIMIT $3`

// ListWasteByKitchen returns cursor-paginated waste records for a kitchen.
func ListWasteByKitchen(ctx context.Context, db DBTX, kitchenID, afterID uuid.UUID, limit int) ([]Waste, error) {
	var afterArg interface{} = afterID
	if afterID == uuid.Nil {
		afterArg = nil
	}
	rows, err := db.Query(ctx, listWasteByKitchenSQL, kitchenID, afterArg, limit)
	if err != nil {
		return nil, fmt.Errorf("queries.ListWasteByKitchen: %w", err)
	}
	defer rows.Close()
	return collectWaste(rows)
}

const wasteAnalyticsSQL = `
SELECT COALESCE(food_type, ''), COALESCE(waste_type, ''), COALESCE(cause, ''),
       SUM(quantity_kg)  AS total_kg,
       COUNT(*)          AS record_count
FROM waste
WHERE kitchen_id = $1
  AND recorded_at BETWEEN $2 AND $3
GROUP BY food_type, waste_type, cause
ORDER BY food_type, waste_type`

// WasteAnalytics aggregates waste by food_type and waste_type for a kitchen
// within a time window.
func WasteAnalytics(ctx context.Context, db DBTX, kitchenID uuid.UUID, from, to time.Time) ([]WasteAnalyticsRow, error) {
	rows, err := db.Query(ctx, wasteAnalyticsSQL, kitchenID, from, to)
	if err != nil {
		return nil, fmt.Errorf("queries.WasteAnalytics: %w", err)
	}
	defer rows.Close()

	var results []WasteAnalyticsRow
	for rows.Next() {
		var r WasteAnalyticsRow
		if err := rows.Scan(&r.FoodType, &r.WasteType, &r.Cause, &r.TotalKg, &r.RecordCount); err != nil {
			return nil, fmt.Errorf("queries.WasteAnalytics scan: %w", err)
		}
		results = append(results, r)
	}
	return results, rows.Err()
}

// ────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────

func scanWaste(row pgx.Row) (Waste, error) {
	var w Waste
	err := row.Scan(
		&w.ID, &w.MealID, &w.KitchenID, &w.FoodType,
		&w.QuantityKg, &w.Cause, &w.WasteType, &w.Note,
		&w.RecordedAt, &w.ClientEventID, &w.CreatedAt, &w.UpdatedAt,
	)
	if err != nil {
		return Waste{}, fmt.Errorf("queries.scanWaste: %w", err)
	}
	return w, nil
}

func collectWaste(rows pgx.Rows) ([]Waste, error) {
	var ws []Waste
	for rows.Next() {
		var w Waste
		if err := rows.Scan(
			&w.ID, &w.MealID, &w.KitchenID, &w.FoodType,
			&w.QuantityKg, &w.Cause, &w.WasteType, &w.Note,
			&w.RecordedAt, &w.ClientEventID, &w.CreatedAt, &w.UpdatedAt,
		); err != nil {
			return nil, fmt.Errorf("queries.collectWaste: %w", err)
		}
		ws = append(ws, w)
	}
	return ws, rows.Err()
}
