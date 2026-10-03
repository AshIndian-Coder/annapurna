package services

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"sort"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/sih26234/food-waste/internal/pushx"
	"github.com/sih26234/food-waste/internal/qrchain"
)

type ScoreBreakdown struct {
	Need           float64 `json:"need"`
	Capacity       float64 `json:"capacity"`
	Distance       float64 `json:"distance"`
	WindowOverlap  float64 `json:"window_overlap"`
	ShelfLifeSlack float64 `json:"shelf_life_slack"`
	Fairness       float64 `json:"fairness"`
}

type MatchResult struct {
	MatchID        string         `json:"match_id"`
	RecipientID    string         `json:"recipient_id"`
	RecipientName  string         `json:"name"`
	Type           string         `json:"type"`
	CapacityKg     float64        `json:"capacity_kg"`
	DistanceKm     float64        `json:"distance_km"`
	PickupWindow   string         `json:"pickup_window"`
	Score          float64        `json:"score"`
	ScoreBreakdown ScoreBreakdown `json:"score_breakdown"`
	Reasons        []string       `json:"reasons"`
}

type MatchingService struct {
	pool    *pgxpool.Pool
	qrchain *qrchain.Service
	push    *pushx.PushService
}

func NewMatchingService(pool *pgxpool.Pool, chain *qrchain.Service, push *pushx.PushService) *MatchingService {
	return &MatchingService{pool: pool, qrchain: chain, push: push}
}

func haversine(lat1, lon1, lat2, lon2 float64) float64 {
	const R = 6371.0 // Earth radius in km
	dLat := (lat2 - lat1) * (math.Pi / 180.0)
	dLon := (lon2 - lon1) * (math.Pi / 180.0)

	rLat1 := lat1 * (math.Pi / 180.0)
	rLat2 := lat2 * (math.Pi / 180.0)

	a := math.Sin(dLat/2)*math.Sin(dLat/2) +
		math.Cos(rLat1)*math.Cos(rLat2)*math.Sin(dLon/2)*math.Sin(dLon/2)
	c := 2 * math.Atan2(math.Sqrt(a), math.Sqrt(1-a))
	return R * c
}

func (s *MatchingService) Match(ctx context.Context, batchID string, maxResults int, kitchenID string) ([]MatchResult, error) {
	if maxResults <= 0 {
		maxResults = 5
	}

	var batchQty float64
	var batchCat string
	var batchExpiry, batchPrepared time.Time
	var batchKitchenID string

	err := s.pool.QueryRow(ctx,
		`SELECT quantity_kg, food_category, expiry_at, prepared_at, kitchen_id 
		 FROM surplus_batches WHERE id = $1`, batchID,
	).Scan(&batchQty, &batchCat, &batchExpiry, &batchPrepared, &batchKitchenID)

	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, &NotFoundError{Resource: "surplus_batch", ID: batchID}
		}
		return nil, fmt.Errorf("fetch batch: %w", err)
	}

	if kitchenID == "" {
		kitchenID = batchKitchenID
	}

	var kLat, kLng float64
	err = s.pool.QueryRow(ctx, `SELECT latitude, longitude FROM kitchens WHERE id = $1`, kitchenID).Scan(&kLat, &kLng)
	if err != nil {
		kLat, kLng = 28.6139, 77.2090
	}

	rows, err := s.pool.Query(ctx,
		`SELECT id, name, type, capacity_kg, latitude, longitude,
		        to_char(pickup_window_start, 'HH24:MI'), to_char(pickup_window_end, 'HH24:MI'),
		        accepts_categories, last_received_at
		 FROM recipients WHERE active = true`)
	if err != nil {
		return nil, fmt.Errorf("query recipients: %w", err)
	}
	defer rows.Close()

	var candidates []MatchResult
	now := time.Now().UTC()

	for rows.Next() {
		var id, name, rType, winStart, winEnd string
		var capKg, lat, lng float64
		var categories []string
		var lastRecv *time.Time

		if err := rows.Scan(&id, &name, &rType, &capKg, &lat, &lng, &winStart, &winEnd, &categories, &lastRecv); err != nil {
			continue
		}

		if len(categories) > 0 {
			categoryMatch := false
			for _, cat := range categories {
				if cat == batchCat || cat == "ALL" {
					categoryMatch = true
					break
				}
			}
			if !categoryMatch {
				continue
			}
		}

		if capKg < math.Min(batchQty, 5.0) {
			continue
		}

		distKm := haversine(kLat, kLng, lat, lng)
		if math.IsNaN(distKm) || distKm < 0 {
			distKm = 1.0
		}

		needScore := 20.0
		if rType == "FOOD_BANK" || rType == "SHELTER" {
			needScore = 20.0
		} else if rType == "COMMUNITY_KITCHEN" {
			needScore = 16.0
		} else {
			needScore = 12.0
		}

		capRatio := capKg / math.Max(batchQty, 1.0)
		if capRatio > 1.0 {
			capRatio = 1.0
		}
		capScore := capRatio * 20.0

		distScore := 25.0 * math.Exp(-0.1*distKm)
		if distScore > 25.0 {
			distScore = 25.0
		}

		winScore := 15.0

		shelfSlackMinutes := batchExpiry.Sub(now).Minutes()
		shelfScore := 10.0
		if shelfSlackMinutes < 60 {
			shelfScore = 2.0
		} else if shelfSlackMinutes < 180 {
			shelfScore = 6.0
		}

		fairnessScore := 10.0
		if lastRecv != nil && time.Since(*lastRecv) < 48*time.Hour {
			hoursSince := time.Since(*lastRecv).Hours()
			fairnessScore = (hoursSince / 48.0) * 10.0
		}

		totalScore := math.Round((needScore+capScore+distScore+winScore+shelfScore+fairnessScore)*10) / 10

		reasons := []string{}
		if distKm <= 5.0 {
			reasons = append(reasons, fmt.Sprintf("Nearby (%.1f km)", distKm))
		}
		if capRatio >= 0.8 {
			reasons = append(reasons, "High capacity match")
		}
		if fairnessScore >= 8.0 {
			reasons = append(reasons, "Distribution fairness priority")
		}
		if len(reasons) == 0 {
			reasons = append(reasons, "Eligible recipient")
		}

		bd := ScoreBreakdown{
			Need:           math.Round(needScore*10) / 10,
			Capacity:       math.Round(capScore*10) / 10,
			Distance:       math.Round(distScore*10) / 10,
			WindowOverlap:  math.Round(winScore*10) / 10,
			ShelfLifeSlack: math.Round(shelfScore*10) / 10,
			Fairness:       math.Round(fairnessScore*10) / 10,
		}

		candidates = append(candidates, MatchResult{
			MatchID:        uuid.NewString(),
			RecipientID:    id,
			RecipientName:  name,
			Type:           rType,
			CapacityKg:     capKg,
			DistanceKm:     math.Round(distKm*10) / 10,
			PickupWindow:   fmt.Sprintf("%s-%s", winStart, winEnd),
			Score:          totalScore,
			ScoreBreakdown: bd,
			Reasons:        reasons,
		})
	}

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].Score > candidates[j].Score
	})

	if len(candidates) > maxResults {
		candidates = candidates[:maxResults]
	}

	for _, c := range candidates {
		bdBytes, _ := json.Marshal(c.ScoreBreakdown)
		_, _ = s.pool.Exec(ctx,
			`INSERT INTO matches (id, batch_id, recipient_id, score, score_breakdown, reasons, status)
			 VALUES ($1, $2, $3, $4, $5, $6, 'OFFERED')
			 ON CONFLICT (batch_id, recipient_id) DO UPDATE SET score = EXCLUDED.score`,
			c.MatchID, batchID, c.RecipientID, c.Score, bdBytes, c.Reasons,
		)
	}

	return candidates, nil
}

func (s *MatchingService) Respond(ctx context.Context, matchID, decision string) error {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("begin tx: %w", err)
	}
	defer tx.Rollback(ctx)

	var batchID, currentStatus string
	err = tx.QueryRow(ctx,
		`SELECT batch_id, status FROM matches WHERE id = $1 FOR UPDATE`, matchID,
	).Scan(&batchID, &currentStatus)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return &NotFoundError{Resource: "match", ID: matchID}
		}
		return fmt.Errorf("fetch match: %w", err)
	}

	now := time.Now().UTC()
	_, err = tx.Exec(ctx,
		`UPDATE matches SET status = $1, responded_at = $2 WHERE id = $3`,
		decision, now, matchID,
	)
	if err != nil {
		return fmt.Errorf("update match: %w", err)
	}

	if decision == "ACCEPTED" {
		_, err = tx.Exec(ctx,
			`UPDATE surplus_batches SET status = 'MATCHED', updated_at = $1 WHERE id = $2`,
			now, batchID,
		)
		if err != nil {
			return fmt.Errorf("set batch matched: %w", err)
		}

		_, _ = tx.Exec(ctx,
			`UPDATE matches SET status = 'EXPIRED', responded_at = $1 
			 WHERE batch_id = $2 AND id != $3 AND status = 'OFFERED'`,
			now, batchID, matchID,
		)
	}

	return tx.Commit(ctx)
}
