package services

import (
	"context"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/sih26234/food-waste/internal/qrchain"
)

type QRService struct {
	pool    *pgxpool.Pool
	qrchain *qrchain.Service
}

func NewQRService(pool *pgxpool.Pool, chain *qrchain.Service) *QRService {
	return &QRService{pool: pool, qrchain: chain}
}

func (s *QRService) RecordEvent(
	ctx context.Context,
	batchID, actorID, actorRole, eventType string,
	lat, lng *float64,
	evidenceHash, clientEventID *string,
	clientTS *time.Time,
) (*qrchain.EventResult, error) {
	if s.qrchain == nil {
		return nil, fmt.Errorf("qrchain service not initialized")
	}
	return s.qrchain.RecordEvent(ctx, batchID, actorID, actorRole, eventType, lat, lng, evidenceHash, clientEventID, clientTS)
}

func (s *QRService) GetTimeline(ctx context.Context, batchID string) ([]qrchain.QREvent, error) {
	const q = `
		SELECT id, batch_id, event_type, actor_id, lat, lng, evidence_hash, prev_hash, hash, client_ts, created_at
		FROM qr_events WHERE batch_id = $1 ORDER BY created_at ASC`

	rows, err := s.pool.Query(ctx, q, batchID)
	if err != nil {
		return nil, fmt.Errorf("query qr events: %w", err)
	}
	defer rows.Close()

	var events []qrchain.QREvent
	for rows.Next() {
		var e qrchain.QREvent
		if err := rows.Scan(&e.ID, &e.BatchID, &e.EventType, &e.ActorID, &e.Lat, &e.Lng, &e.EvidenceHash, &e.PrevHash, &e.Hash, &e.ClientTS, &e.CreatedAt); err != nil {
			return nil, fmt.Errorf("scan qr event: %w", err)
		}
		events = append(events, e)
	}
	return events, rows.Err()
}

func (s *QRService) VerifyChain(ctx context.Context, batchID string) (bool, *int, error) {
	events, err := s.GetTimeline(ctx, batchID)
	if err != nil {
		return false, nil, err
	}
	if len(events) == 0 {
		return true, nil, nil
	}

	valid, brokenAt := qrchain.VerifyChain(events)
	if !valid {
		return false, &brokenAt, nil
	}
	return true, nil, nil
}
