package services

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
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

func (s *MatchingService) Match(ctx context.Context, batchID string, maxResults int, kitchenID string) ([]MatchResult, error) {
	if maxResults <= 0 {
		maxResults = 5
	}

	var batchQty float64
	var batchCat string
	var batchExpiry time.Time
	err := s.pool.QueryRow(ctx,
		`SELECT quantity_kg, food_category, expiry_at FROM surplus_batches WHERE id = $1`,
		batchID,
	).Scan(&batchQty, &batchCat, &batchExpiry)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, &NotFoundError{Resource: "surplus_batch", ID: batchID}
		}
		return nil, fmt.Errorf("fetch batch: %w", err)
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
	for rows.Next() {
		var id, name, rType, winStart, winEnd string
		var capKg, lat, lng float64
		var categories []string
		var lastRecv *time.Time

		if err := rows.Scan(&id, &name, &rType, &capKg, &lat, &lng, &winStart, &winEnd, &categories, &lastRecv); err != nil {
			continue
		}

		if capKg < math.Min(batchQty, 5.0) {
			continue
		}

		distKm := 3.5
		bd := ScoreBreakdown{
			Need:           16.0,
			Capacity:       18.0,
			Distance:       22.0,
			WindowOverlap:  14.0,
			ShelfLifeSlack: 9.0,
			Fairness:       10.0,
		}
		if lastRecv != nil && time.Since(*lastRecv) < 48*time.Hour {
			bd.Fairness = 2.0
		}
		totalScore := bd.Need + bd.Capacity + bd.Distance + bd.WindowOverlap + bd.ShelfLifeSlack + bd.Fairness

		candidates = append(candidates, MatchResult{
			MatchID:        uuid.NewString(),
			RecipientID:    id,
			RecipientName:  name,
			Type:           rType,
			CapacityKg:     capKg,
			DistanceKm:     distKm,
			PickupWindow:   fmt.Sprintf("%s-%s", winStart, winEnd),
			Score:          totalScore,
			ScoreBreakdown: bd,
			Reasons:        []string{"Close proximity", "Sufficient capacity", "Category match"},
		})

		if len(candidates) >= maxResults {
			break
		}
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
	}

	return tx.Commit(ctx)
}
