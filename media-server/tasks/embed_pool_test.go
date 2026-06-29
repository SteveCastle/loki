package tasks

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/stevecastle/shrike/appconfig"
)

func TestRunWithTimeout(t *testing.T) {
	ctx := context.Background()

	// Fast fn finishes before the deadline.
	v, err, timedOut := runWithTimeout(ctx, 200*time.Millisecond, func() (int, error) {
		return 42, nil
	})
	if timedOut || err != nil || v != 42 {
		t.Errorf("fast: got (%d, %v, timedOut=%v), want (42, nil, false)", v, err, timedOut)
	}

	// Slow fn exceeds the deadline → timedOut.
	_, _, timedOut = runWithTimeout(ctx, 20*time.Millisecond, func() (int, error) {
		time.Sleep(300 * time.Millisecond)
		return 1, nil
	})
	if !timedOut {
		t.Error("slow: expected timedOut=true")
	}

	// Error is propagated (not a timeout).
	sentinel := errors.New("boom")
	_, err, timedOut = runWithTimeout(ctx, 200*time.Millisecond, func() (int, error) {
		return 0, sentinel
	})
	if timedOut || !errors.Is(err, sentinel) {
		t.Errorf("error: got (%v, timedOut=%v), want (boom, false)", err, timedOut)
	}

	// d <= 0 disables the timeout.
	v, _, timedOut = runWithTimeout(ctx, 0, func() (int, error) { return 7, nil })
	if timedOut || v != 7 {
		t.Errorf("disabled: got (%d, timedOut=%v), want (7, false)", v, timedOut)
	}

	// Cancelled context returns promptly without a timeout flag.
	cctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err, timedOut = runWithTimeout(cctx, time.Second, func() (int, error) {
		time.Sleep(time.Second)
		return 0, nil
	})
	if timedOut || err == nil {
		t.Errorf("cancelled: got (%v, timedOut=%v), want (ctx err, false)", err, timedOut)
	}
}

func TestOnnxFileTimeout(t *testing.T) {
	prev := appconfig.Get()
	t.Cleanup(func() { appconfig.Set(prev) })

	for _, c := range []struct {
		secs int
		want time.Duration
	}{
		{120, 120 * time.Second},
		{5, 5 * time.Second},
		{0, 0},   // unset → disabled at use time
		{-1, 0},  // negative → disabled
	} {
		cfg := prev
		cfg.OnnxFileTimeoutSeconds = c.secs
		appconfig.Set(cfg)
		if got := OnnxFileTimeout(); got != c.want {
			t.Errorf("OnnxFileTimeout(secs=%d) = %v, want %v", c.secs, got, c.want)
		}
	}
}
