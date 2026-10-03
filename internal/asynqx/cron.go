package asynqx

import (
	"context"
	"time"

	"github.com/hibiken/asynq"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/redis/go-redis/v9"
	"github.com/sih26234/food-waste/internal/services"
)

func HandleExpirySweepCron(surplus *services.SurplusService, rdb *redis.Client) asynq.HandlerFunc {
	return func(ctx context.Context, t *asynq.Task) error {
		if rdb != nil {
			acquired, err := rdb.SetNX(ctx, "lock:expiry_sweep", "locked", 55*time.Second).Result()
			if err != nil || !acquired {
				return nil // Another worker already sweeping
			}
			defer rdb.Del(ctx, "lock:expiry_sweep")
		}

		if surplus != nil {
			_, err := surplus.ExpireBatches(ctx)
			return err
		}
		return nil
	}
}

func HandleOfferExpiryCron(pool *pgxpool.Pool) asynq.HandlerFunc {
	return func(ctx context.Context, t *asynq.Task) error {
		if pool == nil {
			return nil
		}
		now := time.Now().UTC()
		_, err := pool.Exec(ctx,
			`UPDATE matches SET status = 'EXPIRED', responded_at = $1 
			 WHERE status = 'OFFERED' AND created_at < $2`,
			now, now.Add(-30*time.Minute),
		)
		return err
	}
}
