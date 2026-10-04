package services

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/redis/go-redis/v9"
	"go.uber.org/zap"
)

const (
	carbonFactor   = 2.5 // kg CO2 per kg food waste avoided
	mealsPerKg     = 0.4 // kg food per meal equivalent
	impactCacheTTL = 30 * time.Second
)

// ImpactMetrics holds computed environmental and social impact figures.
type ImpactMetrics struct {
	KitchenID                 string    `json:"kitchen_id"`
	ComputedAt                time.Time `json:"computed_at"`
	WasteAvoidedKg            float64   `json:"waste_avoided_kg"`
	RedistributedKg           float64   `json:"redistributed_kg"`
	DivertedKg                float64   `json:"diverted_kg"`
	EstimatedCarbonSavedKg    float64   `json:"estimated_carbon_saved_kg"`
	CarbonFactorUsed          float64   `json:"carbon_factor_used"`
	CarbonFactorSource        string    `json:"carbon_factor_source"`
	MealsEquivalent           int       `json:"meals_equivalent"`
	SuccessfulRedistributions int       `json:"successful_redistributions"`
	Cached                    bool      `json:"cached"`
}

// AnalyticsService computes environmental and social impact from SQL views.
type AnalyticsService struct {
	pool *pgxpool.Pool
	rdb  *redis.Client
	log  *zap.Logger
}

// NewAnalyticsService constructs an AnalyticsService.
func NewAnalyticsService(pool *pgxpool.Pool, rdb *redis.Client, log *zap.Logger) *AnalyticsService {
	return &AnalyticsService{pool: pool, rdb: rdb, log: log}
}

// GetImpact computes impact metrics for a kitchen, caching results in Redis for 30s.
func (s *AnalyticsService) GetImpact(ctx context.Context, kitchenID string) (*ImpactMetrics, error) {
	cacheKey := fmt.Sprintf("cache:impact:%s", kitchenID)
	if s.rdb != nil {
		if raw, err := s.rdb.Get(ctx, cacheKey).Bytes(); err == nil {
			var m ImpactMetrics
			if json.Unmarshal(raw, &m) == nil {
				m.Cached = true
				return &m, nil
			}
		}
	}

	delivered, diverted, count, err := s.fetchImpactTotals(ctx, kitchenID)
	if err != nil {
		return nil, fmt.Errorf("fetch impact totals: %w", err)
	}

	wasteAvoided := delivered + diverted
	impact := &ImpactMetrics{
		KitchenID:                 kitchenID,
		ComputedAt:                time.Now().UTC(),
		WasteAvoidedKg:            wasteAvoided,
		RedistributedKg:           delivered,
		DivertedKg:                diverted,
		EstimatedCarbonSavedKg:    wasteAvoided * carbonFactor,
		CarbonFactorUsed:          carbonFactor,
		CarbonFactorSource:        "WRI/FAO (2.5 kg CO2e per kg)",
		MealsEquivalent:           int(wasteAvoided / mealsPerKg),
		SuccessfulRedistributions: count,
		Cached:                    false,
	}

	if s.rdb != nil {
		if raw, err := json.Marshal(impact); err == nil {
			_ = s.rdb.Set(ctx, cacheKey, raw, impactCacheTTL).Err()
		}
	}

	return impact, nil
}

// fetchImpactTotals queries surplus_batches and diversions for real values.
func (s *AnalyticsService) fetchImpactTotals(ctx context.Context, kitchenID string) (delivered, diverted float64, count int, err error) {
	// Delivered batches
	delQuery := `SELECT COALESCE(SUM(quantity_kg), 0), COUNT(*)
	            FROM surplus_batches
	            WHERE upper(status) = 'DELIVERED'`
	var delArgs []any
	if kitchenID != "" {
		delQuery += ` AND kitchen_id = $1`
		delArgs = append(delArgs, kitchenID)
	}
	err = s.pool.QueryRow(ctx, delQuery, delArgs...).Scan(&delivered, &count)
	if err != nil {
		return 0, 0, 0, fmt.Errorf("query delivered: %w", err)
	}

	// Diverted batches
	divQuery := `SELECT COALESCE(SUM(quantity_kg), 0)
	            FROM surplus_batches
	            WHERE upper(status) = 'DIVERTED'`
	var divArgs []any
	if kitchenID != "" {
		divQuery += ` AND kitchen_id = $1`
		divArgs = append(divArgs, kitchenID)
	}
	err = s.pool.QueryRow(ctx, divQuery, divArgs...).Scan(&diverted)
	if err != nil {
		return 0, 0, 0, fmt.Errorf("query diverted: %w", err)
	}

	return delivered, diverted, count, nil
}
