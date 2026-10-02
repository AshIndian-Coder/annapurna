package main

import (
	"bufio"
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/go-chi/chi/v5"
	chiMiddleware "github.com/go-chi/chi/v5/middleware"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"go.uber.org/zap"

	"github.com/sih26234/food-waste/internal/config"
	"github.com/sih26234/food-waste/internal/httpapi/handlers"
	mw "github.com/sih26234/food-waste/internal/httpapi/middleware"
	_ "github.com/sih26234/food-waste/internal/metrics"
	"github.com/sih26234/food-waste/internal/mlclient"
	"github.com/sih26234/food-waste/internal/pushx"
	"github.com/sih26234/food-waste/internal/qrchain"
	"github.com/sih26234/food-waste/internal/redisx"
	"github.com/sih26234/food-waste/internal/services"
	"github.com/sih26234/food-waste/internal/sse"
	"github.com/sih26234/food-waste/internal/store"
)

// loadDotEnv reads a .env file and sets any key=value pairs as environment
// variables, skipping lines that are blank or start with '#'. Existing env
// vars are never overwritten so real OS env always takes precedence.
func loadDotEnv(path string) {
	f, err := os.Open(path)
	if err != nil {
		return // .env is optional; silently ignore if absent
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.SplitN(line, "=", 2)
		if len(parts) != 2 {
			continue
		}
		key := strings.TrimSpace(parts[0])
		val := strings.TrimSpace(parts[1])
		if os.Getenv(key) == "" { // don't override existing env vars
			_ = os.Setenv(key, val)
		}
	}
}

func main() {
	loadDotEnv(".env") // load .env before anything else

	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))
	slog.SetDefault(logger)

	cfg := config.Load()

	ctx := context.Background()

	pool, err := store.NewPool(ctx, cfg.DatabaseURL)
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

	mlClient := mlclient.NewMLClient(
		cfg.MLServingURL,
		cfg.MLServingToken,
		cfg.FeatureMockML,
		cfg.FeatureMockCV,
		cfg.FeatureMockRouting,
	)

	pushSvc, _ := pushx.NewPushService(ctx, "", nil, nil, true, zap.NewNop())
	qrChainSvc := qrchain.NewService(pool)

	surplusSvc := services.NewSurplusService(pool, qrChainSvc)
	qualitySvc := services.NewQualityService(pool, rdb, mlClient)
	approveSvc := services.NewApproveService(pool, qrChainSvc)
	matchSvc := services.NewMatchingService(pool, qrChainSvc, pushSvc)
	_ = matchSvc
	qrSvc := services.NewQRService(pool, qrChainSvc)
	syncSvc := services.NewSyncService(pool, rdb, surplusSvc, qualitySvc, qrSvc)

	sseHub := sse.NewHub(ctx, rdb)

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
		Addr:         fmt.Sprintf(":%d", cfg.Port),
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
