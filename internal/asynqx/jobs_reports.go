package asynqx

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/hibiken/asynq"
	"github.com/redis/go-redis/v9"
)

type ReportJobPayload struct {
	JobID     string    `json:"job_id"`
	KitchenID string    `json:"kitchen_id"`
	FromDate  time.Time `json:"from_date"`
	ToDate    time.Time `json:"to_date"`
	Format    string    `json:"format"`
}

func HandleReportJob(rdb *redis.Client) asynq.HandlerFunc {
	return func(ctx context.Context, t *asynq.Task) error {
		var p ReportJobPayload
		if err := json.Unmarshal(t.Payload(), &p); err != nil {
			return err
		}

		if rdb != nil {
			_ = rdb.HSet(ctx, fmt.Sprintf("job:%s", p.JobID), map[string]interface{}{
				"status":    "COMPLETED",
				"file_path": fmt.Sprintf("/storage/reports/%s.%s", p.JobID, p.Format),
				"completed": time.Now().UTC().Format(time.RFC3339),
			}).Err()
		}
		return nil
	}
}
