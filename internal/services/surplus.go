package services

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/sih26234/food-waste/internal/qrchain"
)

type SurplusStatus string

const (
	StatusPendingSafety SurplusStatus = "PENDING_SAFETY"
	StatusAvailable     SurplusStatus = "AVAILABLE"
	StatusHold          SurplusStatus = "HOLD"
	StatusDiverted      SurplusStatus = "DIVERTED"
	StatusExpired       SurplusStatus = "EXPIRED"
	StatusMatched       SurplusStatus = "MATCHED"
	StatusInTransit     SurplusStatus = "IN_TRANSIT"
	StatusDelivered     SurplusStatus = "DELIVERED"
)

var allowedTransitions = map[SurplusStatus]map[SurplusStatus]bool{
	StatusPendingSafety: {
		StatusAvailable: true,
		StatusHold:      true,
		StatusDiverted:  true,
		StatusExpired:   true,
	},
	StatusHold: {
		StatusAvailable: true,
		StatusDiverted:  true,
		StatusExpired:   true,
	},
	StatusAvailable: {
		StatusMatched:  true,
		StatusDiverted: true,
		StatusExpired:  true,
	},
	StatusMatched: {
		StatusInTransit: true,
		StatusAvailable: true,
		StatusExpired:   true,
	},
	StatusInTransit: {
		StatusDelivered: true,
	},
}

type CreateSurplusRequest struct {
	FoodName          string    `json:"food_name"`
	FoodCategory      string    `json:"food_category"`
	QuantityKg        float64   `json:"quantity_kg"`
	PreparedAt        time.Time `json:"prepared_at"`
	ExpiryAt          time.Time `json:"expiry_at"`
	MealID            *string   `json:"meal_id,omitempty"`
	ClientEventID     *string   `json:"client_event_id,omitempty"`
}

type SurplusBatch struct {
	ID           string        `json:"id"`
	BatchCode    string        `json:"batch_code"`
	MealID       *string       `json:"meal_id,omitempty"`
	KitchenID    string        `json:"kitchen_id"`
	FoodName     string        `json:"food_name"`
	FoodCategory string        `json:"food_category"`
	QuantityKg   float64       `json:"quantity_kg"`
	PreparedAt   time.Time     `json:"prepared_at"`
	ExpiryAt     time.Time     `json:"expiry_at"`
	SafetyStatus string        `json:"safety_status"`
	Status       SurplusStatus `json:"status"`
	ApprovedBy   *string       `json:"approved_by,omitempty"`
	ApprovedAt   *time.Time    `json:"approved_at,omitempty"`
	CreatedAt    time.Time     `json:"created_at"`
	UpdatedAt    time.Time     `json:"updated_at"`
}

type ListSurplusParams struct {
	KitchenID string
	Status    *SurplusStatus
	Cursor    *string
	Limit     int
}

type ListSurplusResult struct {
	Items      []*SurplusBatch `json:"items"`
	NextCursor *string         `json:"next_cursor,omitempty"`
}

type SurplusService struct {
	pool    *pgxpool.Pool
	qrchain *qrchain.Service
}

func NewSurplusService(pool *pgxpool.Pool, chain *qrchain.Service) *SurplusService {
	return &SurplusService{pool: pool, qrchain: chain}
}

func (s *SurplusService) CreateSurplus(ctx context.Context, kitchenID, userID string, req CreateSurplusRequest) (*SurplusBatch, error) {
	if kitchenID == "" || userID == "" {
		return nil, fmt.Errorf("kitchenID and userID are required")
	}
	if req.QuantityKg <= 0 {
		return nil, fmt.Errorf("quantity_kg must be positive")
	}
	if req.ExpiryAt.IsZero() || req.ExpiryAt.Before(time.Now()) {
		return nil, fmt.Errorf("expiry_at must be in the future")
	}

	now := time.Now().UTC()
	batchID := uuid.NewString()
	batchCode := fmt.Sprintf("B-%s", uuid.NewString()[:8])

	batch := &SurplusBatch{
		ID:           batchID,
		BatchCode:    batchCode,
		MealID:       req.MealID,
		KitchenID:    kitchenID,
		FoodName:     req.FoodName,
		FoodCategory: req.FoodCategory,
		QuantityKg:   req.QuantityKg,
		PreparedAt:   req.PreparedAt,
		ExpiryAt:     req.ExpiryAt,
		SafetyStatus: "PENDING",
		Status:       StatusPendingSafety,
		CreatedAt:    now,
		UpdatedAt:    now,
	}

	const insertSQL = `
		INSERT INTO surplus_batches
			(id, batch_code, meal_id, kitchen_id, food_name, food_category,
			 quantity_kg, prepared_at, expiry_at, safety_status, status,
			 created_at, updated_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)`

	_, err := s.pool.Exec(ctx, insertSQL,
		batch.ID, batch.BatchCode, batch.MealID, batch.KitchenID,
		batch.FoodName, batch.FoodCategory, batch.QuantityKg,
		batch.PreparedAt, batch.ExpiryAt, batch.SafetyStatus,
		string(batch.Status), batch.CreatedAt, batch.UpdatedAt,
	)
	if err != nil {
		return nil, fmt.Errorf("insert surplus batch: %w", err)
	}

	if s.qrchain != nil {
		_, _ = s.qrchain.RecordEvent(ctx, batch.ID, userID, "KITCHEN", "CREATED", nil, nil, nil, req.ClientEventID, nil)
	}

	return batch, nil
}

func (s *SurplusService) Transition(ctx context.Context, batchID string, newStatus SurplusStatus) error {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("begin tx: %w", err)
	}
	defer tx.Rollback(ctx)

	var current SurplusStatus
	err = tx.QueryRow(ctx,
		`SELECT status FROM surplus_batches WHERE id = $1 FOR UPDATE`, batchID,
	).Scan(&current)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return &NotFoundError{Resource: "surplus_batch", ID: batchID}
		}
		return fmt.Errorf("fetch batch status: %w", err)
	}

	if !allowedTransitions[current][newStatus] {
		return &ConflictError{
			Code:    "INVALID_STATE_TRANSITION",
			Message: fmt.Sprintf("cannot transition from %s to %s", current, newStatus),
		}
	}

	_, err = tx.Exec(ctx,
		`UPDATE surplus_batches SET status=$1, updated_at=$2 WHERE id=$3`,
		string(newStatus), time.Now().UTC(), batchID,
	)
	if err != nil {
		return fmt.Errorf("update surplus status: %w", err)
	}

	return tx.Commit(ctx)
}

func (s *SurplusService) GetSurplus(ctx context.Context, batchID string) (*SurplusBatch, error) {
	const q = `
		SELECT id, batch_code, meal_id, kitchen_id, food_name, food_category,
		       quantity_kg, prepared_at, expiry_at, safety_status, status,
		       approved_by, approved_at, created_at, updated_at
		FROM surplus_batches WHERE id = $1`

	b := &SurplusBatch{}
	err := s.pool.QueryRow(ctx, q, batchID).Scan(
		&b.ID, &b.BatchCode, &b.MealID, &b.KitchenID, &b.FoodName, &b.FoodCategory,
		&b.QuantityKg, &b.PreparedAt, &b.ExpiryAt, &b.SafetyStatus, &b.Status,
		&b.ApprovedBy, &b.ApprovedAt, &b.CreatedAt, &b.UpdatedAt,
	)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, &NotFoundError{Resource: "surplus_batch", ID: batchID}
		}
		return nil, fmt.Errorf("get surplus batch: %w", err)
	}
	return b, nil
}

func (s *SurplusService) ListSurplus(ctx context.Context, params ListSurplusParams) (*ListSurplusResult, error) {
	if params.Limit <= 0 || params.Limit > 100 {
		params.Limit = 20
	}

	args := []interface{}{}
	n := 1
	cond := "WHERE 1=1"

	if params.KitchenID != "" {
		cond += fmt.Sprintf(" AND kitchen_id = $%d", n)
		args = append(args, params.KitchenID)
		n++
	}
	if params.Status != nil {
		cond += fmt.Sprintf(" AND status = $%d", n)
		args = append(args, string(*params.Status))
		n++
	}
	if params.Cursor != nil {
		cond += fmt.Sprintf(" AND id < $%d", n)
		args = append(args, *params.Cursor)
		n++
	}

	args = append(args, params.Limit+1)
	query := fmt.Sprintf(`
		SELECT id, batch_code, meal_id, kitchen_id, food_name, food_category,
		       quantity_kg, prepared_at, expiry_at, safety_status, status,
		       approved_by, approved_at, created_at, updated_at
		FROM surplus_batches %s
		ORDER BY created_at DESC, id DESC
		LIMIT $%d`, cond, n)

	rows, err := s.pool.Query(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("list surplus batches: %w", err)
	}
	defer rows.Close()

	var items []*SurplusBatch
	for rows.Next() {
		b := &SurplusBatch{}
		if err := rows.Scan(
			&b.ID, &b.BatchCode, &b.MealID, &b.KitchenID, &b.FoodName, &b.FoodCategory,
			&b.QuantityKg, &b.PreparedAt, &b.ExpiryAt, &b.SafetyStatus, &b.Status,
			&b.ApprovedBy, &b.ApprovedAt, &b.CreatedAt, &b.UpdatedAt,
		); err != nil {
			return nil, fmt.Errorf("scan surplus batch: %w", err)
		}
		items = append(items, b)
	}

	result := &ListSurplusResult{}
	if len(items) > params.Limit {
		last := items[params.Limit-1]
		cursor := last.ID
		result.NextCursor = &cursor
		result.Items = items[:params.Limit]
	} else {
		result.Items = items
	}
	return result, nil
}

func (s *SurplusService) ExpireBatches(ctx context.Context) (int, error) {
	now := time.Now().UTC()
	const query = `
		UPDATE surplus_batches
		SET status = 'EXPIRED', updated_at = $5
		WHERE status IN ($1, $2, $3, $4)
		  AND expiry_at < $5`

	res, err := s.pool.Exec(ctx, query,
		string(StatusAvailable), string(StatusPendingSafety),
		string(StatusHold), string(StatusMatched), now,
	)
	if err != nil {
		return 0, fmt.Errorf("expire batches: %w", err)
	}
	return int(res.RowsAffected()), nil
}
