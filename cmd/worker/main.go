package main

import (
	"context"
	"log/slog"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/hibiken/asynq"
	"github.com/sih26234/food-waste/internal/asynqx"
	"github.com/sih26234/food-waste/internal/config"
	"github.com/sih26234/food-waste/internal/qrchain"
	"github.com/sih26234/food-waste/internal/services"
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

	chainSvc := qrchain.NewService(pool)
	surplusSvc := services.NewSurplusService(pool, chainSvc)

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
		if surplusSvc != nil {
			n, err := surplusSvc.ExpireBatches(ctx)
			if err != nil {
				logger.Error("expiry sweep error", "error", err)
				return err
			}
			logger.Info("expiry sweep completed", "expired_batches", n)
		}
		return nil
	})

	mux.HandleFunc(asynqx.TypeTaskOfferExpiry, func(ctx context.Context, t *asynq.Task) error {
		if pool != nil {
			now := time.Now().UTC()
			res, err := pool.Exec(ctx,
				`UPDATE matches SET status = 'EXPIRED', responded_at = $1 
				 WHERE status = 'OFFERED' AND created_at < $2`,
				now, now.Add(-30*time.Minute),
			)
			if err != nil {
				logger.Error("offer expiry error", "error", err)
				return err
			}
			logger.Info("offer expiry check completed", "expired_offers", res.RowsAffected())
		}
		return nil
	})

	scheduler := asynq.NewScheduler(redisOpt, &asynq.SchedulerOpts{})
	if _, err := scheduler.Register("*/1 * * * *", asynq.NewTask(asynqx.TypeTaskExpirySweep, nil)); err != nil {
		logger.Error("failed to register expiry_sweep cron", "error", err)
	}
	if _, err := scheduler.Register("*/30 * * * *", asynq.NewTask(asynqx.TypeTaskOfferExpiry, nil)); err != nil {
		logger.Error("failed to register offer_expiry cron", "error", err)
	}

	if err := scheduler.Start(); err != nil {
		logger.Error("scheduler start error", "error", err)
	}
	defer scheduler.Shutdown()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		logger.Info("Asynq worker running with registered crons...")
		if err := srv.Run(mux); err != nil {
			logger.Error("worker error", "error", err)
		}
	}()

	<-quit
	logger.Info("shutting down worker...")
	srv.Shutdown()
	logger.Info("worker stopped")
}
