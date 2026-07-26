package main

import (
	"io"
	"os"
	"strings"
	"testing"
	"time"
)

// captureStderr runs fn with os.Stderr redirected to a pipe and returns what
// was written. The progress bar writes straight to the file descriptor, so
// this is the only way to see it.
func captureStderr(t *testing.T, fn func()) string {
	t.Helper()
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	orig := os.Stderr
	os.Stderr = w
	done := make(chan string, 1)
	go func() {
		b, _ := io.ReadAll(r)
		done <- string(b)
	}()
	fn()
	os.Stderr = orig
	w.Close()
	out := <-done
	r.Close()
	return out
}

// The startup bars are labeled, so the embedding and face passes are
// distinguishable as they scroll past. A pipe is not a TTY, which also
// exercises the plain-text fallback used when logs are redirected to a file
// or a service manager.
func TestIndexProgressLabelsEachIndex(t *testing.T) {
	for _, label := range []string{"embedding index", "face index"} {
		out := captureStderr(t, func() {
			report := indexProgressFn(label)
			report(0, 1000)
			for done := 100; done <= 1000; done += 100 {
				report(done, 1000)
			}
		})
		if !strings.Contains(out, "Building "+label+":") {
			t.Fatalf("%q bar is unlabeled: %q", label, out)
		}
		if !strings.Contains(out, "100% (1,000/1,000)") {
			t.Fatalf("%q bar never completed: %q", label, out)
		}
	}
}

// The drawn bar (the TTY path, which the plain-text fallback above never
// reaches) carries the label, the percentage and an ETA mid-flight.
func TestDrawIndexBarRendersLabelAndProgress(t *testing.T) {
	out := captureStderr(t, func() {
		drawIndexBar("face index", 250, 1000, 32, time.Now().Add(-4*time.Second))
	})
	for _, want := range []string{"Building face index", " 25%", "250/1,000", "ETA "} {
		if !strings.Contains(out, want) {
			t.Fatalf("bar frame missing %q: %q", want, out)
		}
	}
	if !strings.HasPrefix(out, "\r\x1b[2K") {
		t.Fatalf("frame must redraw in place: %q", out)
	}
}

// An empty index must stay silent rather than leaving a stranded 0% bar or a
// bare newline in the boot log — the common case for a library with no faces
// scanned yet.
func TestIndexProgressSilentWhenNothingToIndex(t *testing.T) {
	out := captureStderr(t, func() {
		indexProgressFn("face index")(0, 0)
	})
	if out != "" {
		t.Fatalf("empty index wrote %q, want nothing", out)
	}
}
