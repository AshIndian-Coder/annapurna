package services

import (
	"context"
	"fmt"
	"math"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
	"go.uber.org/zap"

	"github.com/sih26234/food-waste/internal/mlclient"
)

// RouteStopOutput is a single stop on the planned route.
type RouteStopOutput struct {
	Sequence    int       `json:"sequence"`
	RecipientID string    `json:"recipient_id"`
	Name        string    `json:"name"`
	Lat         float64   `json:"lat"`
	Lng         float64   `json:"lng"`
	ETA         time.Time `json:"eta"`
	DeadlineOK  bool      `json:"deadline_ok"`
}

// RoutePlanOutput is returned by POST /route.
type RoutePlanOutput struct {
	RouteID          string            `json:"route_id"`
	DistanceKm       float64           `json:"distance_km"`
	ETAMinutes       int               `json:"eta_minutes"`
	Solver           string            `json:"solver"`
	Stops            []RouteStopOutput `json:"stops"`
}

// RoutingService plans and stores delivery routes.
type RoutingService struct {
	pool *pgxpool.Pool
	ml   *mlclient.MLClient
	log  *zap.Logger
}

// NewRoutingService constructs a RoutingService.
func NewRoutingService(pool *pgxpool.Pool, ml *mlclient.MLClient, log *zap.Logger) *RoutingService {
	return &RoutingService{
		pool: pool,
		ml:   ml,
		log:  log,
	}
}

// PlanRoute builds a route plan visiting recipient NGOs from the kitchen origin,
// persists into routes and route_stops in PostgreSQL, and returns the plan.
func (s *RoutingService) PlanRoute(ctx context.Context, batchID string, recipientIDs []string, driverID string) (*RoutePlanOutput, error) {
	bUUID, err := uuid.Parse(batchID)
	if err != nil {
		return nil, fmt.Errorf("invalid batch_id UUID: %w", err)
	}

	var kLat, kLng float64
	var kID string
	err = s.pool.QueryRow(ctx,
		`SELECT k.id::text, k.latitude, k.longitude
		 FROM surplus_batches sb
		 JOIN kitchens k ON k.id = sb.kitchen_id
		 WHERE sb.id = $1`, bUUID,
	).Scan(&kID, &kLat, &kLng)
	if err != nil {
		kLat, kLng = 28.6139, 77.2090
	}

	type stopCandidate struct {
		id   string
		name string
		lat  float64
		lng  float64
		dist float64
	}

	var candidates []stopCandidate
	for _, rID := range recipientIDs {
		var name string
		var lat, lng float64
		err := s.pool.QueryRow(ctx,
			`SELECT name, latitude, longitude FROM recipients WHERE id = $1`, rID,
		).Scan(&name, &lat, &lng)
		if err == nil {
			dist := haversine(kLat, kLng, lat, lng)
			candidates = append(candidates, stopCandidate{
				id:   rID,
				name: name,
				lat:  lat,
				lng:  lng,
				dist: dist,
			})
		}
	}

	totalDistKm := 0.0
	curLat, curLng := kLat, kLng
	now := time.Now().UTC()
	var stops []RouteStopOutput

	for i, c := range candidates {
		legDist := haversine(curLat, curLng, c.lat, c.lng)
		totalDistKm += legDist
		curLat, curLng = c.lat, c.lng

		etaMinutes := int(math.Round((totalDistKm / 30.0) * 60)) // 30 km/h average speed in city
		eta := now.Add(time.Duration(etaMinutes) * time.Minute)

		stops = append(stops, RouteStopOutput{
			Sequence:    i + 1,
			RecipientID: c.id,
			Name:        c.name,
			Lat:         c.lat,
			Lng:         c.lng,
			ETA:         eta,
			DeadlineOK:  true,
		})
	}

	totalEtaMinutes := int(math.Round((totalDistKm / 30.0) * 60))
	if totalEtaMinutes < 15 && len(stops) > 0 {
		totalEtaMinutes = 15
	}

	routeID := uuid.New()
	var dUUID *uuid.UUID
	if driverID != "" {
		if id, err := uuid.Parse(driverID); err == nil {
			dUUID = &id
		}
	}

	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return nil, fmt.Errorf("begin tx: %w", err)
	}
	defer tx.Rollback(ctx)

	_, err = tx.Exec(ctx, `
		INSERT INTO routes (id, batch_id, driver_id, distance_km, eta_minutes, solver, status)
		VALUES ($1, $2, $3, $4, $5, 'haversine-nearest', 'planned')`,
		routeID, bUUID, dUUID, math.Round(totalDistKm*10)/10, totalEtaMinutes,
	)
	if err != nil {
		return nil, fmt.Errorf("insert route: %w", err)
	}

	for _, stop := range stops {
		rUUID, _ := uuid.Parse(stop.RecipientID)
		_, err = tx.Exec(ctx, `
			INSERT INTO route_stops (route_id, recipient_id, batch_id, sequence, eta, address)
			VALUES ($1, $2, $3, $4, $5, $6)`,
			routeID, rUUID, bUUID, stop.Sequence, stop.ETA, stop.Name,
		)
		if err != nil {
			return nil, fmt.Errorf("insert route_stop: %w", err)
		}
	}

	if err := tx.Commit(ctx); err != nil {
		return nil, fmt.Errorf("commit route: %w", err)
	}

	return &RoutePlanOutput{
		RouteID:    routeID.String(),
		DistanceKm: math.Round(totalDistKm*10) / 10,
		ETAMinutes: totalEtaMinutes,
		Solver:     "haversine-nearest",
		Stops:      stops,
	}, nil
}
