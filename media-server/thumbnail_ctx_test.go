package main

import (
	"context"
	"errors"
	"path/filepath"
	"testing"
	"time"
)

// With every ffmpeg slot busy, an abandoned request must stop waiting
// instead of queueing a generation nobody will see.
func TestThumbnailThrottleHonoursCancelledRequest(t *testing.T) {
	for i := 0; i < cap(thumbSem); i++ {
		thumbSem <- struct{}{}
	}
	defer func() {
		for i := 0; i < cap(thumbSem); i++ {
			<-thumbSem
		}
	}()

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	start := time.Now()
	_, err := generateThumbnailThrottled(ctx, filepath.Join(t.TempDir(), "x.jpg"), t.TempDir(), "thumbnail_path_600", 0)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("err = %v, want deadline exceeded", err)
	}
	if time.Since(start) > 2*time.Second {
		t.Fatal("waited far past the request deadline")
	}
	inflightMu.Lock()
	n := len(inflight)
	inflightMu.Unlock()
	if n != 0 {
		t.Fatalf("abandoned request left %d in-flight entries", n)
	}
}

// A request waiting on someone else's in-flight generation stops too.
func TestThumbnailInflightWaitHonoursCancel(t *testing.T) {
	media := filepath.Join(t.TempDir(), "y.jpg")
	key := thumbKey(media, "thumbnail_path_600", 0)
	busy := make(chan struct{})
	inflightMu.Lock()
	inflight[key] = busy
	inflightMu.Unlock()
	defer func() {
		inflightMu.Lock()
		delete(inflight, key)
		inflightMu.Unlock()
		close(busy)
	}()

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	if _, err := generateThumbnailThrottled(ctx, media, t.TempDir(), "thumbnail_path_600", 0); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("err = %v, want deadline exceeded", err)
	}
}
