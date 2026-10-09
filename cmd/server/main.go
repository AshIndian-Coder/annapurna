package main

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/go-chi/chi/v5"
	chiMiddleware "github.com/go-chi/chi/v5/middleware"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/redis/go-redis/v9"

	"github.com/sih26234/food-waste/internal/config"
	"github.com/sih26234/food-waste/internal/httpapi/handlers"
	mw "github.com/sih26234/food-waste/internal/httpapi/middleware"
	"github.com/sih26234/food-waste/internal/metrics"
	"github.com/sih26234/food-waste/internal/mlclient"
	"github.com/sih26234/food-waste/internal/pushx"
	"github.com/sih26234/food-waste/internal/qrchain"
	"github.com/sih26234/food-waste/internal/redisx"
	"github.com/sih26234/food-waste/internal/services"
	"github.com/sih26234/food-waste/internal/sse"
	"github.com/sih26234/food-waste/internal/store"
)

func main() {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))
	slog.SetDefault(logger)

	cfg := config.Load()

	ctx := context.Background()

	pool, err := store.NewPool(cfg.DatabaseURL)
	if err != nil {
		logger.Warn("could not connect to postgres (running in offline/mock mode)", "error", err)
	} else {
		defer pool.Close()
		logger.Info("connected to postgres")
	}

	rdb, err := redisx.NewClient(cfg.RedisURL)
	if err != nil {
		logger.Warn("could not connect to redis (degraded mode active)", "error", err)
	} else {
		defer rdb.Close()
		logger.Info("connected to redis")
	}

	metrics.Init(rdb)

	mlClient := mlclient.NewMLClient(
		cfg.MLServingURL,
		cfg.MLServingToken,
		cfg.FeatureMockML,
		cfg.FeatureMockCV,
		cfg.FeatureMockRouting,
	)

	pushSvc := pushx.NewPushService(nil, rdb)
	qrChainSvc := qrchain.NewService(pool)

	surplusSvc := services.NewSurplusService(pool, qrChainSvc)
	qualitySvc := services.NewQualityService(pool, rdb, mlClient)
	approveSvc := services.NewApproveService(pool, qrChainSvc)
	matchSvc := services.NewMatchingService(pool, qrChainSvc, pushSvc)
	qrSvc := services.NewQRService(pool, qrChainSvc)
	syncSvc := services.NewSyncService(pool, rdb, surplusSvc, qualitySvc, qrSvc)

	sseHub := sse.NewHub(rdb)

	hAuth := handlers.NewAuthHandler(pool, rdb, cfg)
	hSurplus := handlers.NewSurplusHandler(surplusSvc, approveSvc)
	hQuality := handlers.NewQualityHandler(qualitySvc)
	hSync := handlers.NewSyncHandler(syncSvc)
	hHealth := handlers.NewHealthHandler(pool, rdb)
	hEvents := handlers.NewEventsHandler(sseHub)

	r := chi.NewRouter()

	r.Use(mw.RequestID)
	r.Use(mw.Recoverer(logger))
	r.Use(chiMiddleware.RealIP)

	r.Get("/health", hHealth.Health)
	r.Get("/ready", hHealth.Ready)
	r.Handle("/metrics", promhttp.Handler())

	r.Route("/api/v1", func(r chi.Router) {
		r.Post("/auth/login", hAuth.Login)
		r.Post("/auth/refresh", hAuth.Refresh)
		r.Post("/auth/logout", hAuth.Logout)
		r.Post("/auth/logout-all", hAuth.LogoutAll)
		r.Get("/auth/me", hAuth.Me)

		r.Post("/surplus", hSurplus.Create)
		r.Get("/surplus", hSurplus.List)
		r.Get("/surplus/{id}", hSurplus.Get)
		r.Post("/surplus/{id}/approve", hSurplus.Approve)
		r.Post("/surplus/{id}/divert", hSurplus.Divert)

		r.Post("/quality/check", hQuality.Check)

		r.Post("/sync/batch", hSync.Batch)

		r.Get("/events/stream", hEvents.Stream)
	})

	srv := &http.Server{
		Addr:         fmt.Sprintf(":%s", cfg.Port),
		Handler:      r,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 30 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		logger.Info("Go backend server starting", "port", cfg.Port)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Error("server error", "error", err)
			os.Exit(1)
		}
	}()

	<-quit
	logger.Info("shutdown signal received, draining...")

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := srv.Shutdown(shutdownCtx); err != nil {
		logger.Error("graceful shutdown failed", "error", err)
	}
	logger.Info("server stopped gracefully")
}
