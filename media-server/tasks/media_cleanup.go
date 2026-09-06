package tasks

import (
	"fmt"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/media"
)

// The cleanup task forgets media whose files no longer exist — see
// media.CleanupLibrary for the phases and the safety guards. This file is
// the job-queue face of it: option parsing, continual progress, pause and
// cancel, and a readable log.

var cleanupOptions = []TaskOption{
	{Name: "dry-run", Label: "Dry Run", Type: "bool",
		Description: "Report what would be removed without deleting anything"},
	{Name: "dir", Label: "Scope Directory", Type: "string",
		Description: "Only check media under this directory (recursive). Leave empty for the whole library."},
	{Name: "max-missing-percent", Label: "Max Missing %", Type: "number", Default: media.DefaultCleanupMaxMissingPercent,
		Description: "Per-volume guard: when more than this share of a volume's items are missing, the volume is skipped and reported instead of purged (a disconnected, moved, or renamed drive looks exactly like mass deletion). 100 disables the guard."},
	{Name: "skip-orphans", Label: "Skip Dangling Rows", Type: "bool",
		Description: "Skip the final sweep of tag, embedding, face, and battle rows whose media path is no longer in the library"},
}

// cleanupLogEvery is how often a phase in progress writes a status line.
// Every line is persisted with the job, so this is a clock, not a count.
const cleanupLogEvery = 10 * time.Second

func cleanUpFn(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	ctx := j.Ctx
	tokens := dirTaskTokens(j)
	opts := ParseOptions(&jobqueue.Job{Arguments: tokens}, cleanupOptions)
	dryRun, _ := opts["dry-run"].(bool)
	skipOrphans, _ := opts["skip-orphans"].(bool)
	maxPct, _ := opts["max-missing-percent"].(float64)
	scopeDir, _ := opts["dir"].(string)
	scopeDir = strings.TrimSpace(scopeDir)

	// A bare positional token is the scope directory (values of string and
	// number options are skipped so they can't be mistaken for it).
	if scopeDir == "" {
		valueOpts := map[string]bool{}
		for _, o := range cleanupOptions {
			if o.Type != "bool" {
				valueOpts[o.Name] = true
			}
		}
		for i := 0; i < len(tokens); i++ {
			tok := tokens[i]
			if strings.HasPrefix(tok, "--") {
				if name, _, hasEq := strings.Cut(strings.TrimPrefix(tok, "--"), "="); !hasEq && valueOpts[name] {
					i++
				}
				continue
			}
			if !strings.HasPrefix(tok, "-") {
				scopeDir = tok
				break
			}
		}
	}
	if scopeDir != "" && !filepath.IsAbs(scopeDir) {
		q.PushJobStdout(j.ID, fmt.Sprintf("Error: scope directory must be an absolute path, got %q", scopeDir))
		q.ErrorJob(j.ID)
		return fmt.Errorf("scope directory must be absolute")
	}
	if maxPct < 0 || maxPct > 100 {
		q.PushJobStdout(j.ID, fmt.Sprintf("Error: --max-missing-percent must be between 0 and 100, got %v", maxPct))
		q.ErrorJob(j.ID)
		return fmt.Errorf("invalid max-missing-percent")
	}

	log := func(line string) { q.PushJobStdout(j.ID, line) }
	if dryRun {
		log("Dry run: nothing will be deleted")
	}
	if scopeDir != "" {
		log(fmt.Sprintf("Scope: %s (and subdirectories)", filepath.Clean(scopeDir)))
	}
	log(fmt.Sprintf("Phase 1/4: scanning the library for files that no longer exist (missing-ratio guard at %.0f%% per volume)", maxPct))

	// Progress: the job's done/total tracks the current phase; a status line
	// lands on every phase change and at most every cleanupLogEvery inside a
	// phase. SetJobProgress throttles its own broadcast, so it is called on
	// every report.
	var lastPhase media.CleanupPhase
	lastLine := time.Now()
	phaseStart := time.Now()
	statusLine := func(p media.CleanupProgress) string {
		switch p.Phase {
		case media.CleanupPhaseScan:
			return fmt.Sprintf("Scanned %d/%d — %d missing, %d skipped (volume unavailable)",
				p.Done, p.Total, p.Missing, p.SkippedUnavailable)
		case media.CleanupPhaseVerify:
			return fmt.Sprintf("Verified %d/%d candidates — %d present again, %d held back by the guard",
				p.Done, p.Total, p.Recovered, p.SkippedGuard)
		case media.CleanupPhaseRemove:
			verb := "Removed"
			if dryRun {
				verb = "Would remove"
			}
			return fmt.Sprintf("%s %d/%d", verb, p.Done, p.Total)
		default:
			return fmt.Sprintf("Swept %d dangling path(s) from sidecar tables", p.OrphanPaths)
		}
	}
	onProgress := func(p media.CleanupProgress) {
		_ = q.SetJobProgress(j.ID, int(p.Done), int(p.Total))
		if p.Phase != lastPhase {
			if lastPhase != "" {
				log(fmt.Sprintf("Phase done in %s: %s", time.Since(phaseStart).Round(time.Millisecond), statusLine(media.CleanupProgress{
					Phase: lastPhase, Done: p.Done, Total: p.Total, Missing: p.Missing, SkippedUnavailable: p.SkippedUnavailable,
					Recovered: p.Recovered, SkippedGuard: p.SkippedGuard, OrphanPaths: p.OrphanPaths})))
			}
			switch p.Phase {
			case media.CleanupPhaseVerify:
				log(fmt.Sprintf("Phase 2/4: re-checking %d missing candidate(s) and applying the guards", p.Total))
			case media.CleanupPhaseRemove:
				if dryRun {
					log(fmt.Sprintf("Phase 3/4: %d item(s) would be removed with every tag, embedding, face, and battle row that names them", p.Total))
				} else {
					log(fmt.Sprintf("Phase 3/4: removing %d item(s) with every tag, embedding, face, and battle row that names them", p.Total))
				}
			case media.CleanupPhaseOrphans:
				log("Phase 4/4: sweeping sidecar tables for rows whose media path is no longer in the library")
			}
			lastPhase = p.Phase
			phaseStart = time.Now()
			lastLine = time.Now()
			return
		}
		if time.Since(lastLine) >= cleanupLogEvery {
			lastLine = time.Now()
			log(statusLine(p))
		}
	}

	res, err := media.CleanupLibrary(ctx, q.Db, media.CleanupOptions{
		DryRun:            dryRun,
		Scope:             scopeDir,
		MaxMissingPercent: maxPct,
		SkipOrphans:       skipOrphans,
		Progress:          onProgress,
		Log:               log,
		Interrupt: func() error {
			if q.PauseRequested(j.ID) {
				return jobqueue.ErrPaused
			}
			return nil
		},
	})
	removedSoFar := int64(0)
	if res != nil {
		removedSoFar = res.MediaRemoved
	}
	if err != nil {
		switch {
		case err == jobqueue.ErrPaused:
			log(fmt.Sprintf("Paused after removing %d item(s) — resume to start the scan again (removals are durable)", removedSoFar))
			return err
		case ctx.Err() != nil:
			log(fmt.Sprintf("Task was canceled after removing %d item(s) — removals are durable", removedSoFar))
			_ = q.CancelJob(j.ID)
			return err
		default:
			log(fmt.Sprintf("Error during cleanup (%d item(s) removed before it): %v", removedSoFar, err))
			q.ErrorJob(j.ID)
			return err
		}
	}

	// Summary.
	verb := "Removed"
	if dryRun {
		verb = "Would remove"
	}
	log(fmt.Sprintf("Scanned %d item(s): %d missing, %d present again on re-check, %d skipped on unavailable volumes, %d held back by the guard",
		res.MediaScanned, res.MissingFound, res.Recovered, res.SkippedUnavailable, res.SkippedByGuard))
	if dryRun {
		log(fmt.Sprintf("%s %d media item(s) and every row naming them", verb, res.MediaRemoved))
		if !skipOrphans {
			log(fmt.Sprintf("Would sweep %d dangling path(s) from sidecar tables (%s)", res.OrphanPaths, formatByTable(res.OrphanPathsByTable)))
		}
	} else {
		log(fmt.Sprintf("%s %d media item(s); rows removed with them: %d tags, %d embeddings, %d faces (+%d curation assertions), %d face-scan markers, %d battle-log entries",
			verb, res.MediaRemoved, res.TagsRemoved, res.EmbeddingsRemoved, res.FacesRemoved, res.FaceAssertionsRemoved, res.FaceScansRemoved, res.BattlesRemoved))
		if !skipOrphans {
			log(fmt.Sprintf("Swept %d dangling path(s) from sidecar tables (%s)", res.OrphanPaths, formatByTable(res.OrphanPathsByTable)))
		}
	}
	for _, r := range res.GuardedRoots {
		log(fmt.Sprintf("WARNING: %s was NOT cleaned — %d of %d items missing (%.0f%%). If the volume is gone for good, rerun with --max-missing-percent 100; if it moved, use `media move` instead.",
			r.Root, r.Missing, r.Scanned, float64(r.Missing)*100/float64(r.Scanned)))
	}
	if res.SkippedUnavailable > 0 {
		log(fmt.Sprintf("WARNING: skipped %d missing item(s) because their volume(s) are offline: %s — reconnect and run cleanup again to process them",
			res.SkippedUnavailable, strings.Join(res.UnavailableRoots, ", ")))
	}
	if len(res.Errors) > 0 {
		log(fmt.Sprintf("Note: %d error(s) occurred during cleanup", len(res.Errors)))
	}
	if res.MediaRemoved == 0 && res.OrphanPaths == 0 && res.SkippedUnavailable == 0 && res.SkippedByGuard == 0 {
		log("No missing media found — the library is clean")
	}
	q.CompleteJob(j.ID)
	return nil
}

// formatByTable renders per-table counts compactly, in a stable order.
func formatByTable(m map[string]int64) string {
	if len(m) == 0 {
		return "none"
	}
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, fmt.Sprintf("%s: %d", k, m[k]))
	}
	return strings.Join(parts, ", ")
}
