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

func main() {
	config.LoadDotEnv(".env") // developer convenience; real env vars win

	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))
	slog.SetDefault(logger)

	cfg := config.Load()

	// Uploads are written with UUID filenames (never client-supplied paths).
	if err := os.MkdirAll(cfg.UploadsDir, 0o755); err != nil {
		logger.Error("could not create uploads directory", "dir", cfg.UploadsDir, "error", err)
	}

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
		logger.Info("connected to redis DB0")
	}

	// DB1 queue client: stream:sensor XADD, asynq queues — separate from cache DB0.
	queueRdb, err := redisx.NewQueueClient(cfg.RedisURL)
	if err != nil {
		logger.Warn("could not connect to redis DB1 queue (stream writes disabled)", "error", err)
	} else {
		defer queueRdb.Close()
		logger.Info("connected to redis DB1 (queue)")
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
	qrSvc := services.NewQRService(pool, qrChainSvc)
	syncSvc := services.NewSyncService(pool, rdb, surplusSvc, qualitySvc, qrSvc)

	alertSvc := services.NewAlertService(pool, rdb, nil, zap.NewNop())
	sensorSvc := services.NewSensorService(pool, rdb, queueRdb, alertSvc, zap.NewNop())
	procSvc := services.NewProcessingService(pool, zap.NewNop())
	analyticsSvc := services.NewAnalyticsService(pool, rdb, zap.NewNop())
	predSvc := services.NewPredictionService(pool, rdb, mlClient, zap.NewNop())
	routingSvc := services.NewRoutingService(pool, mlClient, zap.NewNop())
	reportSvc := services.NewReportService(pool, rdb, analyticsSvc, zap.NewNop())

	sseHub := sse.NewHub(ctx, rdb)

	hAuth := handlers.NewAuthHandler(pool, rdb, cfg)
	hSurplus := handlers.NewSurplusHandler(surplusSvc, approveSvc)
	hQuality := handlers.NewQualityHandler(qualitySvc, cfg.MaxUploadMB, cfg.UploadsDir)
	hSync := handlers.NewSyncHandler(syncSvc)
	hQR := handlers.NewQRHandler(qrSvc)
	hHealth := handlers.NewHealthHandler(pool, rdb)
	hEvents := handlers.NewEventsHandler(sseHub)

	hMatch := handlers.NewMatchingHandler(matchSvc)
	hWaste := handlers.NewWasteHandler(pool)
	hRecipients := handlers.NewRecipientsHandler(pool)
	hSensors := handlers.NewSensorsHandler(sensorSvc)
	hAlerts := handlers.NewAlertsHandler(alertSvc)
	hProcessing := handlers.NewProcessingHandler(procSvc)
	hAnalytics := handlers.NewAnalyticsHandler(analyticsSvc)
	hPrediction := handlers.NewPredictionHandler(predSvc)
	hRouting := handlers.NewRoutingHandler(routingSvc)
	hReports := handlers.NewReportsHandler(reportSvc)
	hDevices := handlers.NewDevicesHandler(pool)
	hAdmin := handlers.NewAdminHandler(pool)
	hApp := handlers.NewAppHandler("static/bundles/cv/cv-v1/manifest.json")

	r := chi.NewRouter()

	r.Use(mw.RequestID)
	r.Use(mw.Recoverer(logger))
	r.Use(chiMiddleware.RealIP)

	r.Get("/health", hHealth.Health)
	r.Get("/ready", hHealth.Ready)
	r.Handle("/metrics", promhttp.Handler())

	r.Route("/api/v1", func(r chi.Router) {
		// 1-3. Authentication
		r.Post("/auth/login", hAuth.Login)
		r.Post("/auth/refresh", hAuth.Refresh)
		r.Post("/auth/logout", hAuth.Logout)
		r.Post("/auth/logout-all", hAuth.LogoutAll)
		r.Get("/auth/me", hAuth.Me)

		// 4-6. Kitchen ops
		r.Get("/kitchen/overview", hAdmin.KitchenOverview)
		r.Post("/kitchen/attendance", hAdmin.PostAttendance)
		r.Post("/kitchen/menu", hAdmin.PostMenu)

		// 7-9. Predictions & feedback
		r.Post("/predict-demand", hPrediction.PredictDemand)
		r.Post("/planning/what-if", hPrediction.WhatIf)
		r.Post("/feedback/outcome", hPrediction.RecordOutcome)

		// 10-11. Waste
		r.Post("/waste", hWaste.LogWaste)
		r.Get("/waste/analytics", hWaste.Analytics)

		// 12-14. Surplus batches
		r.Post("/surplus", hSurplus.Create)
		r.Get("/surplus", hSurplus.List)
		r.Get("/surplus/{id}", hSurplus.Get)
		r.Post("/surplus/{id}/approve", hSurplus.Approve)
		r.Post("/surplus/{id}/divert", hSurplus.Divert)

		// 15. Quality check
		r.Post("/quality/check", hQuality.Check)

		// 18-19. Matching
		r.Post("/match", hMatch.CreateMatch)
		r.Post("/matches/{id}/respond", hMatch.Respond)

		// 20. Recipients
		r.Get("/recipients", hRecipients.ListRecipients)

		// 21. Route planning
		r.Post("/route", hRouting.Optimise)

		// 22-24. Append-only QR hash chain
		r.Get("/qr/{batchID}", hQR.Get)
		r.Post("/qr/{batchID}/event", hQR.RecordEvent)
		r.Get("/qr/{batchID}/verify", hQR.Verify)

		// 25-26. Sensors
		r.Post("/sensors/readings", hSensors.PostReadings)
		r.Get("/sensors/latest", hSensors.Latest)
		r.Get("/sensors/history", hSensors.History)

		// 27. Alerts
		r.Get("/alerts", hAlerts.ListAlerts)
		r.Post("/alerts/{id}/ack", hAlerts.Acknowledge)

		// 28-29. Processing metrics
		r.Post("/processing/metrics", hProcessing.PostMetrics)
		r.Get("/processing/analytics", hProcessing.Analytics)

		// 30. Impact analytics
		r.Get("/analytics/impact", hAnalytics.Impact)

		// 31. Reports
		r.Get("/reports/esg", hReports.ESG)
		r.Get("/reports/jobs/{id}", hReports.JobStatus)

		// 32. Real-time events
		r.Get("/events/stream", hEvents.Stream)

		// 33. Admin
		r.Get("/admin/users", hAdmin.ListUsers)
		r.Post("/admin/users", hAdmin.CreateUser)
		r.Patch("/admin/users/{id}", hAdmin.UpdateUser)
		r.Get("/admin/models", hAdmin.ListModels)
		r.Get("/admin/audit-log", hAdmin.AuditLog)
		r.Post("/admin/reseed", hAdmin.Reseed)

		// 37. Devices
		r.Post("/devices", hDevices.Register)
		r.Delete("/devices/{id}", hDevices.Deregister)

		// 38. Offline sync batch replay
		r.Post("/sync/batch", hSync.Batch)

		// 39-40. App configuration & bundles
		r.Get("/app/config", hApp.Config)
		r.Get("/app/model-bundle", hApp.ModelBundle)
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
