package asynqx

import (
	"context"
	"log/slog"
	"time"

	"github.com/hibiken/asynq"
)

type WorkerConfig struct {
	Concurrency int
	Queues      map[string]int
}

func DefaultWorkerConfig() WorkerConfig {
	return WorkerConfig{
		Concurrency: 10,
		Queues: map[string]int{
			"critical": 6,
			"default":  3,
			"low":      1,
		},
	}
}

func NewServer(redisOpt asynq.RedisClientOpt, cfg WorkerConfig) *asynq.Server {
	return asynq.NewServer(redisOpt, asynq.Config{
		Concurrency:     cfg.Concurrency,
		Queues:          cfg.Queues,
		ShutdownTimeout: 10 * time.Second,
	})
}
