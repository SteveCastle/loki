package tasks

import "sync/atomic"

// Progress kinds reported to the notifier. Each corresponds to one coverage
// counter on the library stats snapshot (stats_api.go in package main): the
// notifier fires once per item that gains that piece of metadata.
const (
	ProgressDescription = "description"
	ProgressTranscript  = "transcript"
	ProgressHash        = "hash"
	ProgressSize        = "size"
	ProgressDimensions  = "dimensions"
	ProgressTags        = "tags"
	ProgressEmbedding   = "embedding"
)

// progressNotifier holds a func(kind string, n int). atomic.Value keeps the
// hot notifyProgress path lock-free — tasks call it once per completed item.
var progressNotifier atomic.Value

// SetProgressNotifier registers the callback invoked each time a running task
// finishes one item's worth of metadata (one description written, one file
// hashed, one embedding stored, ...). Wired once at startup by the stats API
// so coverage counters advance live; when unset (tests, lokictl) notifications
// are dropped.
func SetProgressNotifier(fn func(kind string, n int)) {
	progressNotifier.Store(fn)
}

func notifyProgress(kind string, n int) {
	if fn, ok := progressNotifier.Load().(func(string, int)); ok && fn != nil {
		fn(kind, n)
	}
}
