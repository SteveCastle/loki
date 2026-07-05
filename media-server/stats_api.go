package main

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"sync"
	"time"

	"github.com/stevecastle/shrike/stream"
	"github.com/stevecastle/shrike/tasks"
)

// -----------------------------------------------------------------------------
// Library stats API (shared across all platform mains).
//
//   GET /api/stats — media totals plus per-metadata-type completion counts.
//
// Powers the home page's metadata-coverage cards. Transcript coverage is
// counted against videos only (the transcript task skips everything else), and
// embedding coverage is counted against the active embedding model.
//
// On multi-million-item libraries a full recount takes tens of seconds (the
// tag-coverage count alone walks a media_tag_by_category index with 10M+
// rows), so the handler never computes inline: it serves the last snapshot
// and refreshes it in a single-flight background goroutine. The first request
// after boot answers {"ready": false} while the initial count runs; the page
// retries until a snapshot exists.
//
// Real-time updates: running tasks report each completed item through
// tasks.SetProgressNotifier → applyStatsDelta. Deltas accumulate on top of the
// snapshot, the merged view is served by the handler, and a coalescing
// broadcaster pushes it over SSE as a "stats" event (at most once per second)
// so the home page's progress bars advance live while jobs run. Deltas are an
// optimistic overlay — the next full recount reconciles any drift (overwrite
// re-runs, items deleted mid-job) and resets them.
// -----------------------------------------------------------------------------

// videoExtCase is a SQL boolean expression matching media rows whose path has
// a video extension the transcript task can process (see
// tasks/metadata_ops.go). SUBSTR comparison is used instead of LIKE — it is
// several times cheaper per row, which matters when the expression runs
// against every row of a large media table.
const videoExtCase = `(LOWER(SUBSTR(path, -4)) IN ('.mp4', '.mov', '.avi', '.mkv', '.wmv') OR LOWER(SUBSTR(path, -5)) = '.webm')`

// statsBroadcastInterval coalesces per-item progress notifications into at
// most one SSE "stats" event per tick — tasks can complete hundreds of items
// per second and each SSE event fans out to every connected client.
const statsBroadcastInterval = time.Second

type statsModelInfo struct {
	ID   string `json:"id"`
	Name string `json:"name"`
}

type statsAPIResponse struct {
	Ready       bool  `json:"ready"`
	GeneratedAt int64 `json:"generatedAt"` // unix seconds the snapshot was computed

	TotalMedia  int `json:"totalMedia"`
	TotalVideos int `json:"totalVideos"`
	TotalImages int `json:"totalImages"`

	WithDescription      int `json:"withDescription"`
	WithHash             int `json:"withHash"`
	WithSize             int `json:"withSize"`
	WithTags             int `json:"withTags"`
	WithDimensions       int `json:"withDimensions"`
	VideosWithTranscript int `json:"videosWithTranscript"`
	WithEmbedding        int `json:"withEmbedding"`

	AutotagModel statsModelInfo `json:"autotagModel"`
	EmbedModel   statsModelInfo `json:"embedModel"`
}

// addProgress folds one task progress delta into the counters. Kinds are the
// tasks.Progress* constants.
func (s *statsAPIResponse) addProgress(kind string, n int) {
	switch kind {
	case tasks.ProgressDescription:
		s.WithDescription += n
	case tasks.ProgressTranscript:
		s.VideosWithTranscript += n
	case tasks.ProgressHash:
		s.WithHash += n
	case tasks.ProgressSize:
		s.WithSize += n
	case tasks.ProgressDimensions:
		s.WithDimensions += n
	case tasks.ProgressTags:
		s.WithTags += n
	case tasks.ProgressEmbedding:
		s.WithEmbedding += n
	}
}

// clamp keeps the optimistic delta overlay from pushing a coverage counter
// past its denominator (e.g. an --overwrite re-run notifies for items the
// snapshot already counted).
func (s *statsAPIResponse) clamp() {
	capTo := func(v *int, max int) {
		if *v > max {
			*v = max
		}
	}
	capTo(&s.WithDescription, s.TotalMedia)
	capTo(&s.WithHash, s.TotalMedia)
	capTo(&s.WithSize, s.TotalMedia)
	capTo(&s.WithTags, s.TotalMedia)
	capTo(&s.WithDimensions, s.TotalMedia)
	capTo(&s.WithEmbedding, s.TotalMedia)
	capTo(&s.VideosWithTranscript, s.TotalVideos)
}

var libStats struct {
	mu        sync.Mutex
	snapshot  *statsAPIResponse
	deltas    map[string]int // progress accumulated since the snapshot
	dirty     bool           // deltas or snapshot changed since last broadcast
	computeMs int64
	computing bool
}

// applyStatsDelta is the tasks.SetProgressNotifier callback: one completed
// item of one metadata kind.
func applyStatsDelta(kind string, n int) {
	libStats.mu.Lock()
	if libStats.deltas == nil {
		libStats.deltas = map[string]int{}
	}
	libStats.deltas[kind] += n
	libStats.dirty = true
	libStats.mu.Unlock()
}

// mergedStatsLocked returns the snapshot with pending deltas applied. Caller
// must hold libStats.mu. Returns nil while no snapshot exists yet.
func mergedStatsLocked() *statsAPIResponse {
	if libStats.snapshot == nil {
		return nil
	}
	merged := *libStats.snapshot
	for kind, n := range libStats.deltas {
		merged.addProgress(kind, n)
	}
	merged.clamp()
	return &merged
}

// libStatsTTL scales the refresh interval with how expensive the last count
// was: small libraries recount every 15s, huge ones back off so polling the
// endpoint doesn't keep the database permanently busy.
func libStatsTTL(computeMs int64) time.Duration {
	ttl := time.Duration(computeMs) * time.Millisecond * 4
	if ttl < 15*time.Second {
		ttl = 15 * time.Second
	}
	if ttl > 5*time.Minute {
		ttl = 5 * time.Minute
	}
	return ttl
}

var statsRealtimeOnce sync.Once

// initStatsRealtime wires task progress notifications into the delta overlay
// and starts the coalescing SSE broadcaster. Called once, from the first
// statsAPIHandler construction (each platform main constructs it exactly once
// at route registration).
func initStatsRealtime() {
	tasks.SetProgressNotifier(applyStatsDelta)
	go func() {
		ticker := time.NewTicker(statsBroadcastInterval)
		defer ticker.Stop()
		for range ticker.C {
			libStats.mu.Lock()
			if !libStats.dirty {
				libStats.mu.Unlock()
				continue
			}
			merged := mergedStatsLocked()
			libStats.dirty = false
			libStats.mu.Unlock()
			if merged == nil {
				continue // progress before the first snapshot; nothing to show yet
			}
			payload, err := json.Marshal(merged)
			if err != nil {
				continue
			}
			stream.Broadcast(stream.Message{Type: "stats", Msg: string(payload)})
		}
	}()
}

func statsAPIHandler(deps *Dependencies) http.HandlerFunc {
	statsRealtimeOnce.Do(initStatsRealtime)
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			httpError(w, "use GET", http.StatusMethodNotAllowed)
			return
		}

		libStats.mu.Lock()
		merged := mergedStatsLocked()
		stale := libStats.snapshot == nil ||
			time.Since(time.Unix(libStats.snapshot.GeneratedAt, 0)) > libStatsTTL(libStats.computeMs)
		if stale && !libStats.computing {
			libStats.computing = true
			go computeLibraryStats(deps)
		}
		libStats.mu.Unlock()

		if merged == nil {
			writeJSON(w, map[string]any{"ready": false})
			return
		}
		writeJSON(w, merged)
	}
}

// computeLibraryStats runs the full recount and installs the snapshot. Always
// called with libStats.computing already set; clears it on exit.
func computeLibraryStats(deps *Dependencies) {
	started := time.Now()
	defer func() {
		libStats.mu.Lock()
		libStats.computing = false
		libStats.mu.Unlock()
	}()

	// Deltas that existed when the recount started are (mostly) included in
	// what the recount will read, so they are subtracted once the snapshot
	// installs. Progress reported DURING the recount stays in the overlay.
	libStats.mu.Lock()
	preDeltas := make(map[string]int, len(libStats.deltas))
	for k, v := range libStats.deltas {
		preDeltas[k] = v
	}
	libStats.mu.Unlock()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	embedModel := tasks.ActiveEmbedModel()
	taggerModel := tasks.ActiveTaggerModel()

	data := statsAPIResponse{Ready: true}

	// One pass over media for everything derivable from the row itself.
	err := deps.DB.QueryRowContext(ctx, `
        SELECT
            COUNT(*),
            SUM(CASE WHEN `+videoExtCase+` THEN 1 ELSE 0 END),
            SUM(CASE WHEN description IS NOT NULL AND TRIM(description) <> '' THEN 1 ELSE 0 END),
            SUM(CASE WHEN hash IS NOT NULL AND TRIM(hash) <> '' THEN 1 ELSE 0 END),
            SUM(CASE WHEN size IS NOT NULL THEN 1 ELSE 0 END),
            SUM(CASE WHEN width IS NOT NULL AND height IS NOT NULL THEN 1 ELSE 0 END),
            SUM(CASE WHEN `+videoExtCase+` AND transcript IS NOT NULL AND TRIM(transcript) <> '' THEN 1 ELSE 0 END)
        FROM media
    `).Scan(
		&data.TotalMedia,
		&data.TotalVideos,
		&data.WithDescription,
		&data.WithHash,
		&data.WithSize,
		&data.WithDimensions,
		&data.VideosWithTranscript,
	)
	if err != nil {
		log.Printf("stats: media scan failed: %v", err)
		return
	}

	// Tag coverage: distinct tagged paths that still exist in media. The join
	// filters orphaned tag rows left behind by out-of-band deletions.
	err = deps.DB.QueryRowContext(ctx, `
        SELECT COUNT(*)
        FROM (SELECT DISTINCT media_path FROM media_tag_by_category) t
        JOIN media m ON m.path = t.media_path
    `).Scan(&data.WithTags)
	if err != nil {
		log.Printf("stats: tag coverage count failed: %v", err)
		return
	}

	// Embedding coverage for the active model, ignoring orphaned vectors.
	err = deps.DB.QueryRowContext(ctx, `
        SELECT COUNT(*)
        FROM media_embedding e
        JOIN media m ON m.path = e.media_path
        WHERE e.model = ?
    `, embedModel.ID).Scan(&data.WithEmbedding)
	if err != nil {
		log.Printf("stats: embedding coverage count failed: %v", err)
		return
	}

	data.TotalImages = data.TotalMedia - data.TotalVideos
	data.AutotagModel = statsModelInfo{ID: taggerModel.ID, Name: taggerModel.DisplayName}
	data.EmbedModel = statsModelInfo{ID: embedModel.ID, Name: embedModel.DisplayName}
	data.GeneratedAt = time.Now().Unix()

	libStats.mu.Lock()
	libStats.snapshot = &data
	// Retire the deltas the recount has absorbed; keep progress that arrived
	// while it ran.
	for k, v := range preDeltas {
		if remaining := libStats.deltas[k] - v; remaining > 0 {
			libStats.deltas[k] = remaining
		} else {
			delete(libStats.deltas, k)
		}
	}
	libStats.computeMs = time.Since(started).Milliseconds()
	libStats.dirty = true // broadcast the fresh totals on the next tick
	libStats.mu.Unlock()
	log.Printf("stats: snapshot computed in %s (%d items)", time.Since(started).Round(time.Millisecond), data.TotalMedia)
}
