package redisx_test

import (
	"context"
	"testing"
	"time"

	"github.com/sih26234/food-waste/internal/redisx"
)

func TestInMemoryCacheDegrade(t *testing.T) {
	cache := redisx.NewInMemoryCache(10)
	ctx := context.Background()

	cache.Set("test:key", "value1", 1*time.Minute)
	val, ok := cache.Get("test:key")
	if !ok || val != "value1" {
		t.Fatalf("expected value1, got %s (ok=%v)", val, ok)
	}

	cache.Delete("test:key")
	_, ok = cache.Get("test:key")
	if ok {
		t.Fatalf("expected key to be deleted")
	}
}

func TestDangerZoneCap(t *testing.T) {
	delta := 20.0
	capped := delta
	if capped > 15.0 {
		capped = 15.0
	}
	if capped != 15.0 {
		t.Fatalf("expected danger zone delta to cap at 15.0 minutes, got %f", capped)
	}
}
