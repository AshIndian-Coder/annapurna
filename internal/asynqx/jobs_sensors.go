package asynqx

import (
	"context"
	"time"

	"github.com/hibiken/asynq"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/redis/go-redis/v9"
)

func HandleSensorBulkIngest(pool *pgxpool.Pool, rdb *redis.Client) asynq.HandlerFunc {
	return func(ctx context.Context, t *asynq.Task) error {
		if rdb == nil || pool == nil {
			return nil
		}

		entries, err := rdb.XReadGroup(ctx, &redis.XReadGroupArgs{
			Group:    "g_db",
			Consumer: "c_worker_1",
			Streams:  []string{"stream:sensor", ">"},
			Count:    500,
			Block:    1 * time.Second,
		}).Result()

		if err != nil || len(entries) == 0 {
			return nil
		}

		return nil
	}
}
