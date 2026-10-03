package services

import (
	"context"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
	"go.uber.org/zap"

	"github.com/sih26234/food-waste/internal/processing"
)

// ProcessingService manages kitchen processing metrics.
type ProcessingService struct {
	pool *pgxpool.Pool
	log  *zap.Logger
}

// NewProcessingService constructs a ProcessingService.
func NewProcessingService(pool *pgxpool.Pool, log *zap.Logger) *ProcessingService {
	return &ProcessingService{
		pool: pool,
		log:  log,
	}
}

// ProcessingInput holds input data for POST /processing/metrics.
type ProcessingInput struct {
	KitchenID       string    `json:"kitchen_id"`
	Date            time.Time `json:"date"`
	PeriodStart     time.Time `json:"period_start"`
	PeriodEnd       time.Time `json:"period_end"`
	RawMaterialKg   float64   `json:"raw_material_kg"`
	OutputKg        float64   `json:"output_kg"`
	WasteKg         float64   `json:"waste_kg"`
	RuntimeMinutes  float64   `json:"runtime_minutes"`
	DowntimeMinutes float64   `json:"downtime_minutes"`
	EnergyKWh       float64   `json:"energy_kwh"`
}

// ProcessingOutput is returned after computing and storing processing KPIs.
type ProcessingOutput struct {
	ID                   string   `json:"id"`
	KitchenID            string   `json:"kitchen_id"`
	MaterialLossPct      float64  `json:"material_loss_pct"`
	ProductionEfficiency float64  `json:"production_efficiency"`
	DowntimePct          float64  `json:"downtime_pct"`
	EnergyPerKg          float64  `json:"energy_per_kg"`
	EfficiencyScore      float64  `json:"efficiency_score"`
	Anomalies            []string `json:"anomalies"`
}

// ProcessingAnalyticsSummary is returned for GET /processing/analytics.
type ProcessingAnalyticsSummary struct {
	KitchenID            string  `json:"kitchen_id"`
	Period               string  `json:"period"`
	AvgEfficiencyScore   float64 `json:"avg_efficiency_score"`
	AvgMaterialLossPct   float64 `json:"avg_material_loss_pct"`
	AvgDowntimePct       float64 `json:"avg_downtime_pct"`
	TotalRawMaterialKg   float64 `json:"total_raw_material_kg"`
	TotalOutputKg        float64 `json:"total_output_kg"`
	TotalWasteKg         float64 `json:"total_waste_kg"`
	TotalEnergyKWh       float64 `json:"total_energy_kwh"`
	RecordCount          int     `json:"record_count"`
}

// RecordMetrics computes KPIs and persists them into the processing_metrics table.
func (s *ProcessingService) RecordMetrics(ctx context.Context, in ProcessingInput) (*ProcessingOutput, error) {
	kUUID, err := uuid.Parse(in.KitchenID)
	if err != nil {
		return nil, fmt.Errorf("invalid kitchen_id UUID: %w", err)
	}

	if in.PeriodStart.IsZero() {
		in.PeriodStart = in.Date
	}
	if in.PeriodEnd.IsZero() {
		in.PeriodEnd = in.Date
	}
	if in.PeriodEnd.Before(in.PeriodStart) {
		in.PeriodEnd = in.PeriodStart
	}

	kpis, err := processing.Compute(
		in.RawMaterialKg, in.OutputKg, in.WasteKg,
		in.DowntimeMinutes, in.RuntimeMinutes, in.EnergyKWh,
	)
	if err != nil {
		return nil, fmt.Errorf("compute metrics: %w", err)
	}

	prodEfficiency := 0.0
	if in.RawMaterialKg > 0 {
		prodEfficiency = in.OutputKg / in.RawMaterialKg
	}

	id := uuid.New()
	query := `
		INSERT INTO processing_metrics (
			id, kitchen_id, date, period_start, period_end,
			raw_material_kg, output_kg, waste_kg,
			downtime_min, runtime_min, energy_kwh,
			material_loss_pct, downtime_pct, energy_per_kg, efficiency_score,
			created_at, updated_at
		) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, NOW(), NOW())
		ON CONFLICT (kitchen_id, period_start, period_end) DO UPDATE
		SET raw_material_kg   = EXCLUDED.raw_material_kg,
		    output_kg         = EXCLUDED.output_kg,
		    waste_kg          = EXCLUDED.waste_kg,
		    downtime_min      = EXCLUDED.downtime_min,
		    runtime_min       = EXCLUDED.runtime_min,
		    energy_kwh        = EXCLUDED.energy_kwh,
		    material_loss_pct = EXCLUDED.material_loss_pct,
		    downtime_pct      = EXCLUDED.downtime_pct,
		    energy_per_kg     = EXCLUDED.energy_per_kg,
		    efficiency_score  = EXCLUDED.efficiency_score,
		    updated_at        = NOW()
		RETURNING id`

	err = s.pool.QueryRow(ctx, query,
		id, kUUID, in.Date, in.PeriodStart, in.PeriodEnd,
		in.RawMaterialKg, in.OutputKg, in.WasteKg,
		in.DowntimeMinutes, in.RuntimeMinutes, in.EnergyKWh,
		kpis.MaterialLossPct, kpis.DowntimePct, kpis.EnergyPerKg, kpis.EfficiencyScore,
	).Scan(&id)
	if err != nil {
		return nil, fmt.Errorf("insert processing_metrics: %w", err)
	}

	return &ProcessingOutput{
		ID:                   id.String(),
		KitchenID:            in.KitchenID,
		MaterialLossPct:      kpis.MaterialLossPct,
		ProductionEfficiency: prodEfficiency,
		DowntimePct:          kpis.DowntimePct,
		EnergyPerKg:          kpis.EnergyPerKg,
		EfficiencyScore:      kpis.EfficiencyScore,
		Anomalies:            []string{},
	}, nil
}

// GetAnalytics computes aggregate processing statistics for a kitchen.
func (s *ProcessingService) GetAnalytics(ctx context.Context, kitchenID string, from, to time.Time) (*ProcessingAnalyticsSummary, error) {
	kUUID, err := uuid.Parse(kitchenID)
	if err != nil {
		return nil, fmt.Errorf("invalid kitchen_id UUID: %w", err)
	}

	query := `
		SELECT COALESCE(AVG(efficiency_score), 0),
		       COALESCE(AVG(material_loss_pct), 0),
		       COALESCE(AVG(downtime_pct), 0),
		       COALESCE(SUM(raw_material_kg), 0),
		       COALESCE(SUM(output_kg), 0),
		       COALESCE(SUM(waste_kg), 0),
		       COALESCE(SUM(energy_kwh), 0),
		       COUNT(*)
		FROM processing_metrics
		WHERE kitchen_id = $1 AND date BETWEEN $2::date AND $3::date`

	var summary ProcessingAnalyticsSummary
	summary.KitchenID = kitchenID
	summary.Period = fmt.Sprintf("%s to %s", from.Format("2006-01-02"), to.Format("2006-01-02"))

	err = s.pool.QueryRow(ctx, query, kUUID, from, to).Scan(
		&summary.AvgEfficiencyScore,
		&summary.AvgMaterialLossPct,
		&summary.AvgDowntimePct,
		&summary.TotalRawMaterialKg,
		&summary.TotalOutputKg,
		&summary.TotalWasteKg,
		&summary.TotalEnergyKWh,
		&summary.RecordCount,
	)
	if err != nil {
		return nil, fmt.Errorf("query processing analytics: %w", err)
	}

	return &summary, nil
}
