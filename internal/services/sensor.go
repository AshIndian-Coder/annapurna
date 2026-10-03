package services

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/redis/go-redis/v9"
	"go.uber.org/zap"
)

const (
	sensorStreamKey  = "stream:sensor"
	sensorLatestKey  = "sensor:latest"
	sensorPubChannel = "chan:kitchen"
	alertDedupeTTL   = 10 * time.Minute
	staleSensorAge   = 24 * time.Hour
)

// SensorReading is a single measurement from a kitchen sensor.
type SensorReading struct {
	SensorID  string            `json:"sensor_id"`
	KitchenID string            `json:"kitchen_id"`
	BatchID   string            `json:"batch_id,omitempty"`
	Timestamp time.Time         `json:"timestamp"`
	Temp      *float64          `json:"temperature_c,omitempty"`
	Humidity  *float64          `json:"humidity_pct,omitempty"`
	EnergyKWh *float64          `json:"energy_kwh,omitempty"`
	Extra     map[string]string `json:"extra,omitempty"`
}

// SensorService manages sensor data ingestion and retrieval.
type SensorService struct {
	pool         *pgxpool.Pool
	rdb          *redis.Client
	alertService *AlertService
	log          *zap.Logger
}

// NewSensorService constructs a SensorService.
func NewSensorService(pool *pgxpool.Pool, rdb *redis.Client, alerts *AlertService, log *zap.Logger) *SensorService {
	return &SensorService{
		pool:         pool,
		rdb:          rdb,
		alertService: alerts,
		log:          log,
	}
}

// Ingest validates and persists a batch of sensor readings.
func (s *SensorService) Ingest(ctx context.Context, readings []SensorReading) error {
	var firstErr error
	validCount := 0
	for _, r := range readings {
		if err := s.validateReading(r); err != nil {
			s.log.Warn("invalid sensor reading, skipping",
				zap.String("sensor_id", r.SensorID), zap.Error(err))
			if firstErr == nil {
				firstErr = fmt.Errorf("sensor %s: %w", r.SensorID, err)
			}
			continue
		}
		validCount++

		// 1. Insert into PostgreSQL sensor_readings table
		var kUUID, bUUID *uuid.UUID
		if r.KitchenID != "" {
			if id, err := uuid.Parse(r.KitchenID); err == nil {
				kUUID = &id
			}
		}
		if r.BatchID != "" {
			if id, err := uuid.Parse(r.BatchID); err == nil {
				bUUID = &id
			}
		}

		_, err := s.pool.Exec(ctx, `
			INSERT INTO sensor_readings (sensor_id, kitchen_id, batch_id, temperature_c, humidity_pct, energy_kwh, recorded_at, created_at)
			VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())`,
			r.SensorID, kUUID, bUUID, r.Temp, r.Humidity, r.EnergyKWh, r.Timestamp,
		)
		if err != nil {
			s.log.Error("failed to persist sensor reading to postgres", zap.Error(err))
		}

		// 2. Redis updates
		if s.rdb != nil {
			if err := s.xaddStream(ctx, r); err != nil {
				s.log.Error("XADD failed", zap.String("sensor_id", r.SensorID), zap.Error(err))
			}
			if err := s.hsetLatest(ctx, r); err != nil {
				s.log.Error("HSET latest failed", zap.String("sensor_id", r.SensorID), zap.Error(err))
			}
			s.publishKitchen(ctx, r)
		}

		s.evaluateThresholds(ctx, r)
	}

	if validCount == 0 && len(readings) > 0 {
		return firstErr
	}
	return nil
}

func (s *SensorService) validateReading(r SensorReading) error {
	if r.SensorID == "" {
		return fmt.Errorf("sensor_id is required")
	}
	if r.KitchenID == "" {
		return fmt.Errorf("kitchen_id is required")
	}
	if time.Since(r.Timestamp) > staleSensorAge {
		return fmt.Errorf("stale reading: timestamp %s is older than 24h", r.Timestamp)
	}
	if r.Temp != nil && (*r.Temp < -30 || *r.Temp > 120) {
		return fmt.Errorf("temperature out of range [-30, 120]: %.2f", *r.Temp)
	}
	if r.Humidity != nil && (*r.Humidity < 0 || *r.Humidity > 100) {
		return fmt.Errorf("humidity out of range [0, 100]: %.2f", *r.Humidity)
	}
	if r.EnergyKWh != nil && *r.EnergyKWh < 0 {
		return fmt.Errorf("energy_kwh must be >= 0: %.4f", *r.EnergyKWh)
	}
	return nil
}

func (s *SensorService) xaddStream(ctx context.Context, r SensorReading) error {
	fields := map[string]interface{}{
		"sensor_id":  r.SensorID,
		"kitchen_id": r.KitchenID,
		"timestamp":  r.Timestamp.UnixMilli(),
	}
	if r.Temp != nil {
		fields["temperature_c"] = strconv.FormatFloat(*r.Temp, 'f', 4, 64)
	}
	if r.Humidity != nil {
		fields["humidity_pct"] = strconv.FormatFloat(*r.Humidity, 'f', 4, 64)
	}
	if r.EnergyKWh != nil {
		fields["energy_kwh"] = strconv.FormatFloat(*r.EnergyKWh, 'f', 6, 64)
	}
	return s.rdb.XAdd(ctx, &redis.XAddArgs{
		Stream: sensorStreamKey,
		Values: fields,
	}).Err()
}

func (s *SensorService) hsetLatest(ctx context.Context, r SensorReading) error {
	raw, err := json.Marshal(r)
	if err != nil {
		return err
	}
	return s.rdb.HSet(ctx, sensorLatestKey, r.SensorID, raw).Err()
}

func (s *SensorService) publishKitchen(ctx context.Context, r SensorReading) {
	raw, err := json.Marshal(r)
	if err != nil {
		s.log.Warn("failed to marshal reading for publish", zap.Error(err))
		return
	}
	channel := fmt.Sprintf("%s:%s", sensorPubChannel, r.KitchenID)
	if err := s.rdb.Publish(ctx, channel, raw).Err(); err != nil {
		s.log.Warn("failed to publish sensor reading", zap.Error(err))
	}
}

func (s *SensorService) evaluateThresholds(ctx context.Context, r SensorReading) {
	if s.alertService == nil {
		return
	}

	type rule struct {
		field     string
		alertType string
		value     float64
		warnLo    *float64
		warnHi    *float64
		critLo    *float64
		critHi    *float64
	}

	fp := func(v float64) *float64 { return &v }

	var rules []rule
	if r.Temp != nil {
		rules = append(rules, rule{
			field:     "temperature_c",
			alertType: "TEMP_EXCURSION",
			value:     *r.Temp,
			warnLo:    fp(0),
			warnHi:    fp(60),
			critLo:    fp(-10),
			critHi:    fp(85),
		})
	}
	if r.Humidity != nil {
		rules = append(rules, rule{
			field:     "humidity_pct",
			alertType: "HUMIDITY_HIGH",
			value:     *r.Humidity,
			warnHi:    fp(80),
			critHi:    fp(95),
		})
	}
	if r.EnergyKWh != nil {
		rules = append(rules, rule{
			field:     "energy_kwh",
			alertType: "ENERGY_SPIKE",
			value:     *r.EnergyKWh,
			warnHi:    fp(500),
			critHi:    fp(1000),
		})
	}

	for _, rl := range rules {
		severity := ""
		switch {
		case rl.critLo != nil && rl.value < *rl.critLo:
			severity = "CRITICAL"
		case rl.critHi != nil && rl.value > *rl.critHi:
			severity = "CRITICAL"
		case rl.warnLo != nil && rl.value < *rl.warnLo:
			severity = "WARN"
		case rl.warnHi != nil && rl.value > *rl.warnHi:
			severity = "WARN"
		}
		if severity == "" {
			continue
		}

		if s.rdb != nil {
			dedupeKey := fmt.Sprintf("alert:dedupe:%s:%s:%s", r.KitchenID, r.SensorID, rl.field)
			exists, _ := s.rdb.Exists(ctx, dedupeKey).Result()
			if exists > 0 {
				continue
			}
			_ = s.rdb.Set(ctx, dedupeKey, "1", alertDedupeTTL).Err()
		}

		msg := fmt.Sprintf("Sensor %s: %s=%.2f exceeded %s threshold", r.SensorID, rl.field, rl.value, severity)
		_, err := s.alertService.Create(ctx, CreateAlertReq{
			KitchenID: r.KitchenID,
			AlertType: rl.alertType,
			Severity:  severity,
			Message:   msg,
			DedupeKey: fmt.Sprintf("%s:%s", r.SensorID, rl.field),
		})
		if err != nil {
			s.log.Error("failed to create sensor alert", zap.Error(err))
		}
	}
}

// GetLatest returns the most recent reading for sensors in a kitchen.
func (s *SensorService) GetLatest(ctx context.Context, kitchenID string) ([]SensorReading, error) {
	// First check Redis
	if s.rdb != nil {
		all, err := s.rdb.HGetAll(ctx, sensorLatestKey).Result()
		if err == nil && len(all) > 0 {
			var readings []SensorReading
			for _, raw := range all {
				var r SensorReading
				if err := json.Unmarshal([]byte(raw), &r); err == nil && (kitchenID == "" || r.KitchenID == kitchenID) {
					readings = append(readings, r)
				}
			}
			if len(readings) > 0 {
				return readings, nil
			}
		}
	}

	// Fallback to PostgreSQL
	query := `SELECT DISTINCT ON (sensor_id) sensor_id, COALESCE(kitchen_id::text, ''),
	                 temperature_c, humidity_pct, energy_kwh, recorded_at
	          FROM sensor_readings
	          WHERE ($1 = '' OR kitchen_id::text = $1)
	          ORDER BY sensor_id, recorded_at DESC`
	rows, err := s.pool.Query(ctx, query, kitchenID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var readings []SensorReading
	for rows.Next() {
		var r SensorReading
		if err := rows.Scan(&r.SensorID, &r.KitchenID, &r.Temp, &r.Humidity, &r.EnergyKWh, &r.Timestamp); err == nil {
			readings = append(readings, r)
		}
	}
	return readings, nil
}

// GetHistory retrieves sensor readings in the time window.
func (s *SensorService) GetHistory(ctx context.Context, kitchenID, sensorID string, from, to time.Time) ([]SensorReading, error) {
	query := `SELECT sensor_id, COALESCE(kitchen_id::text, ''), temperature_c, humidity_pct, energy_kwh, recorded_at
	          FROM sensor_readings
	          WHERE recorded_at BETWEEN $1 AND $2
	            AND ($3 = '' OR kitchen_id::text = $3)
	            AND ($4 = '' OR sensor_id = $4)
	          ORDER BY recorded_at ASC LIMIT 1000`
	rows, err := s.pool.Query(ctx, query, from, to, kitchenID, sensorID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var readings []SensorReading
	for rows.Next() {
		var r SensorReading
		if err := rows.Scan(&r.SensorID, &r.KitchenID, &r.Temp, &r.Humidity, &r.EnergyKWh, &r.Timestamp); err == nil {
			readings = append(readings, r)
		}
	}
	return readings, nil
}
