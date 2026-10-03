package services

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/redis/go-redis/v9"
	"go.uber.org/zap"

	"github.com/sih26234/food-waste/internal/asynqx"
)

const sseAlertChannel = "sse:alerts"

// AlertDomain mirrors the alerts table.
type AlertDomain struct {
	ID             string     `json:"id"`
	KitchenID      string     `json:"kitchen_id"`
	AlertType      string     `json:"alert_type"`
	Severity       string     `json:"severity"`
	Message        string     `json:"message"`
	IsAcked        bool       `json:"is_acked"`
	AcknowledgedBy *string    `json:"acknowledged_by,omitempty"`
	AcknowledgedAt *time.Time `json:"acknowledged_at,omitempty"`
	CreatedAt      time.Time  `json:"created_at"`
	UpdatedAt      time.Time  `json:"updated_at"`
}

// CreateAlertReq holds params to create an alert.
type CreateAlertReq struct {
	KitchenID string `json:"kitchen_id"`
	AlertType string `json:"alert_type"`
	Severity  string `json:"severity"`
	Message   string `json:"message"`
	DedupeKey string `json:"dedupe_key,omitempty"`
}

// AlertService manages the lifecycle of system alerts.
type AlertService struct {
	pool        *pgxpool.Pool
	rdb         *redis.Client
	asynqClient *asynqx.Client
	log         *zap.Logger
}

// NewAlertService constructs an AlertService.
func NewAlertService(pool *pgxpool.Pool, rdb *redis.Client, asynqClient *asynqx.Client, log *zap.Logger) *AlertService {
	return &AlertService{
		pool:        pool,
		rdb:         rdb,
		asynqClient: asynqClient,
		log:         log,
	}
}

// Create persists a new alert, publishes to SSE, and enqueues a push task for WARN/CRITICAL.
func (s *AlertService) Create(ctx context.Context, req CreateAlertReq) (*AlertDomain, error) {
	now := time.Now().UTC()
	alertID := uuid.NewString()

	alertType := req.AlertType
	if alertType == "" {
		alertType = "TEMP_EXCURSION"
	}
	severity := req.Severity
	if severity == "" {
		severity = "WARN"
	}

	alert := &AlertDomain{
		ID:        alertID,
		KitchenID: req.KitchenID,
		AlertType: alertType,
		Severity:  severity,
		Message:   req.Message,
		IsAcked:   false,
		CreatedAt: now,
		UpdatedAt: now,
	}

	var kitchenUUID *uuid.UUID
	if req.KitchenID != "" {
		if kID, err := uuid.Parse(req.KitchenID); err == nil {
			kitchenUUID = &kID
		}
	}

	_, err := s.pool.Exec(ctx, `
		INSERT INTO alerts (id, kitchen_id, alert_type, type, severity, message, is_acked, dedupe_key, created_at, updated_at)
		VALUES ($1, $2, $3, $3, $4, $5, false, $6, $7, $7)`,
		alertID, kitchenUUID, alertType, severity, req.Message, req.DedupeKey, now,
	)
	if err != nil {
		return nil, fmt.Errorf("insert alert: %w", err)
	}

	s.publishSSE(ctx, alert)

	return alert, nil
}

// List returns alerts scoped to a kitchen, with optional filtering.
func (s *AlertService) List(ctx context.Context, kitchenID, severity string, unackedOnly bool, limit, offset int) ([]AlertDomain, int, error) {
	if limit <= 0 {
		limit = 50
	}

	query := `SELECT id, COALESCE(kitchen_id::text, ''), COALESCE(alert_type, COALESCE(type, 'TEMP_EXCURSION')),
	                 severity, message, is_acked, COALESCE(acknowledged_by::text, acked_by::text),
	                 COALESCE(acknowledged_at, acked_at), created_at, updated_at
	          FROM alerts WHERE 1=1`
	countQuery := `SELECT COUNT(*) FROM alerts WHERE 1=1`
	var args []any

	if kitchenID != "" {
		args = append(args, kitchenID)
		query += fmt.Sprintf(" AND kitchen_id = $%d", len(args))
		countQuery += fmt.Sprintf(" AND kitchen_id = $%d", len(args))
	}
	if severity != "" {
		args = append(args, severity)
		query += fmt.Sprintf(" AND severity = $%d", len(args))
		countQuery += fmt.Sprintf(" AND severity = $%d", len(args))
	}
	if unackedOnly {
		query += " AND is_acked = false"
		countQuery += " AND is_acked = false"
	}

	var total int
	_ = s.pool.QueryRow(ctx, countQuery, args...).Scan(&total)

	args = append(args, limit, offset)
	query += fmt.Sprintf(" ORDER BY created_at DESC LIMIT $%d OFFSET $%d", len(args)-1, len(args))

	rows, err := s.pool.Query(ctx, query, args...)
	if err != nil {
		return nil, 0, err
	}
	defer rows.Close()

	var alerts []AlertDomain
	for rows.Next() {
		var a AlertDomain
		var ackedBy *string
		var ackedAt *time.Time
		if err := rows.Scan(
			&a.ID, &a.KitchenID, &a.AlertType,
			&a.Severity, &a.Message, &a.IsAcked,
			&ackedBy, &ackedAt,
			&a.CreatedAt, &a.UpdatedAt,
		); err != nil {
			return nil, 0, err
		}
		a.AcknowledgedBy = ackedBy
		a.AcknowledgedAt = ackedAt
		alerts = append(alerts, a)
	}
	return alerts, total, rows.Err()
}

// Ack marks an alert as acknowledged.
func (s *AlertService) Ack(ctx context.Context, alertID, userID string) (*AlertDomain, error) {
	now := time.Now().UTC()
	var ackerUUID *uuid.UUID
	if userID != "" {
		if uID, err := uuid.Parse(userID); err == nil {
			ackerUUID = &uID
		}
	}

	res, err := s.pool.Exec(ctx, `
		UPDATE alerts
		SET is_acked = true, acknowledged_by = $1, acked_by = $1, acknowledged_at = $2, acked_at = $2, updated_at = $2
		WHERE id = $3`,
		ackerUUID, now, alertID,
	)
	if err != nil {
		return nil, fmt.Errorf("ack alert: %w", err)
	}
	if res.RowsAffected() == 0 {
		return nil, &NotFoundError{Resource: "alert", ID: alertID}
	}

	var a AlertDomain
	var ackedBy *string
	var ackedAt *time.Time
	err = s.pool.QueryRow(ctx, `
		SELECT id, COALESCE(kitchen_id::text, ''), COALESCE(alert_type, 'TEMP_EXCURSION'),
		       severity, message, is_acked, COALESCE(acknowledged_by::text, acked_by::text),
		       COALESCE(acknowledged_at, acked_at), created_at, updated_at
		FROM alerts WHERE id = $1`, alertID,
	).Scan(
		&a.ID, &a.KitchenID, &a.AlertType,
		&a.Severity, &a.Message, &a.IsAcked,
		&ackedBy, &ackedAt,
		&a.CreatedAt, &a.UpdatedAt,
	)
	if err != nil {
		return nil, err
	}
	a.AcknowledgedBy = ackedBy
	a.AcknowledgedAt = ackedAt
	return &a, nil
}

// publishSSE emits the alert to the kitchen-scoped SSE Redis pub/sub channel.
func (s *AlertService) publishSSE(ctx context.Context, alert *AlertDomain) {
	if s.rdb == nil {
		return
	}
	raw, err := json.Marshal(alert)
	if err != nil {
		s.log.Warn("failed to marshal alert for SSE", zap.Error(err))
		return
	}
	channel := fmt.Sprintf("%s:%s", sseAlertChannel, alert.KitchenID)
	if err := s.rdb.Publish(ctx, channel, raw).Err(); err != nil {
		s.log.Warn("failed to publish SSE alert", zap.Error(err))
	}
}
