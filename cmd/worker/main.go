package main

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/hibiken/asynq"
	"github.com/sih26234/food-waste/internal/asynqx"
	"github.com/sih26234/food-waste/internal/config"
	"github.com/sih26234/food-waste/internal/store"
)

func main() {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))
	slog.SetDefault(logger)

	cfg := config.Load()

	pool, err := store.NewPool(cfg.DatabaseURL)
	if err != nil {
		logger.Warn("worker: postgres connection failed", "error", err)
	} else {
		defer pool.Close()
	}

	redisOpt := asynq.RedisClientOpt{
		Addr: "localhost:6379",
		DB:   1,
	}

	srv := asynq.NewServer(redisOpt, asynq.Config{
		Concurrency: 10,
		Queues: map[string]int{
			"critical": 6,
			"default":  3,
			"low":      1,
		},
		ShutdownTimeout: 10 * time.Second,
	})

	mux := asynq.NewServeMux()
	mux.HandleFunc(asynqx.TypeTaskExpirySweep, func(ctx context.Context, t *asynq.Task) error {
		logger.Info("executing expiry sweep")
		return nil
	})
	mux.HandleFunc(asynqx.TypeTaskOfferExpiry, func(ctx context.Context, t *asynq.Task) error {
		logger.Info("executing offer expiry check")
		return nil
	})

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		logger.Info("Asynq worker running...")
		if err := srv.Run(mux); err != nil {
			logger.Error("worker error", "error", err)
		}
	}()

	<-quit
	logger.Info("shutting down worker...")
	srv.Shutdown()
	logger.Info("worker stopped")
}
