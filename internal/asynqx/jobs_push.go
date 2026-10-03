package asynqx

import (
	"context"
	"encoding/json"

	"github.com/hibiken/asynq"
	"github.com/sih26234/food-waste/internal/pushx"
)

type PushJobPayload struct {
	UserIDs   []string          `json:"user_ids"`
	EventType string            `json:"event_type"`
	Title     string            `json:"title"`
	Body      string            `json:"body"`
	Data      map[string]string `json:"data"`
}

func HandlePushJob(pushSvc *pushx.PushService) asynq.HandlerFunc {
	return func(ctx context.Context, t *asynq.Task) error {
		var p PushJobPayload
		if err := json.Unmarshal(t.Payload(), &p); err != nil {
			return err
		}
		if pushSvc != nil {
			_ = pushSvc.Send(ctx, p.UserIDs, p.EventType, p.Title, p.Body, p.Data)
		}
		return nil
	}
}
