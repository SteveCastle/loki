package tasks

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"strings"

	"github.com/stevecastle/shrike/deps/models"
	"github.com/stevecastle/shrike/jobqueue"
)

// --- On-demand model installs ---
//
// Every feature backed by a downloadable model installs it the first time a
// job needs it, instead of failing with "install it from Dependencies":
// EnsureModels is the one path for that (jobs), and startModelInstalls the
// non-blocking one for interactive requests that can't wait for gigabytes.
// Both go through the Dependencies page's tracker, so a job's download shows
// there as downloading and the page never starts a parallel one; the
// per-model install lock plus a re-check under it make concurrent ensures of
// one model download it once.

// modelInstalled reports whether a model dependency is completely installed.
// A variable so tests can declare models present without files on disk.
var modelInstalled = models.IsInstalled

// EnsureModels makes sure every listed model dependency is installed,
// downloading what is missing (only the missing or unverified files; an
// interrupted download resumes) with progress lines through logf. Installed
// models cost a few stats. It blocks for the download and returns when ctx
// ends.
func EnsureModels(ctx context.Context, logf func(string), ids ...string) error {
	if logf == nil {
		logf = func(string) {}
	}
	if ctx == nil {
		ctx = context.Background()
	}
	for _, id := range ids {
		if modelInstalled(id) {
			continue
		}
		m, ok := models.Lookup(id)
		if !ok {
			return fmt.Errorf("unknown dependency %q", id)
		}
		if !models.Available(id) {
			return fmt.Errorf("%s is not available on %s", modelLabel(m), runtime.GOOS)
		}
		if models.Tracker.Active(id) {
			logf(fmt.Sprintf("Waiting for the %s download already in progress (Dependencies page)…", modelLabel(m)))
		} else {
			logf(fmt.Sprintf("Downloading %s, %s — first use only…", modelLabel(m), formatModelSize(m.EffectiveSizeBytes())))
		}
		if err := installAIDep(ctx, func(format string, a ...any) { logf(fmt.Sprintf(format, a...)) }, id); err != nil {
			if ctx.Err() != nil {
				return ctx.Err()
			}
			return fmt.Errorf("could not install %s: %w (retrying resumes where it stopped; it can also be installed from Dependencies)", modelLabel(m), err)
		}
	}
	return nil
}

// jobLogf is EnsureModels' logger for a job: lines go to the job log.
func jobLogf(q *jobqueue.Queue, jobID string) func(string) {
	return func(s string) { q.PushJobStdout(jobID, s) }
}

// missingModels returns the ids that are not installed yet.
func missingModels(ids ...string) []string {
	var out []string
	for _, id := range ids {
		if !modelInstalled(id) {
			out = append(out, id)
		}
	}
	return out
}

// startModelInstall is how interactive paths start a background install; a
// variable so tests never touch the network.
var startModelInstall = func(id string) { _, _ = models.Tracker.StartInstall(id) }

// startModelInstalls kicks off background installs of the missing ids (each
// joins an install already in flight) and returns the user-facing error for
// the request that found them missing — interactive requests never wait for
// a download. Returns nil when nothing is missing.
func startModelInstalls(what string, ids ...string) error {
	missing := missingModels(ids...)
	if len(missing) == 0 {
		return nil
	}
	var size int64
	for _, id := range missing {
		m, ok := models.Lookup(id)
		if !ok || !models.Available(id) {
			label := id
			if ok {
				label = modelLabel(m)
			}
			return fmt.Errorf("%s isn't available on this platform (%s)", label, runtime.GOOS)
		}
		size += m.EffectiveSizeBytes()
		startModelInstall(id)
	}
	return &modelsDownloadingError{what: what, size: size}
}

// modelsDownloadingError is returned by a request whose models are being
// downloaded in the background.
type modelsDownloadingError struct {
	what string
	size int64
}

func (e *modelsDownloadingError) Error() string {
	return fmt.Sprintf("Downloading the %s (%s) — first use only. Try again in a few minutes; progress is on the Dependencies page.", e.what, formatModelSize(e.size))
}

// IsModelsDownloading reports whether err means "the model is downloading
// in the background; try again later" (for HTTP handlers picking a status).
func IsModelsDownloading(err error) bool {
	var d *modelsDownloadingError
	return errors.As(err, &d)
}

// modelLabel names a dependency for a job log: "the Auto-tagging model (WD
// EVA02 Large Tagger v3)", or a tool's own name.
func modelLabel(m models.Model) string {
	kind := "model"
	if m.EffectiveCategory() == "tool" {
		kind = "engine"
	}
	if m.Feature != "" && !strings.EqualFold(m.Feature, m.Name) {
		return fmt.Sprintf("the %s %s (%s)", m.Feature, kind, m.Name)
	}
	return m.Name
}

func formatModelSize(n int64) string {
	switch {
	case n >= 1e9:
		return fmt.Sprintf("%.2f GB", float64(n)/1e9)
	case n >= 1e6:
		return fmt.Sprintf("%.0f MB", float64(n)/1e6)
	case n > 0:
		return fmt.Sprintf("%.0f KB", float64(n)/1e3)
	}
	return "size unknown"
}

// reinstallServerHint is the remedy for a missing bundled binary (ffmpeg,
// ffprobe, exiftool, onnxruntime, embed, onnxtag): they ship with the server
// and can't be installed from the Dependencies page.
const reinstallServerHint = "it ships with Lowkey Media Server — reinstall or update the server to restore it"

// bundledMissing is the error for a missing bundled binary.
func bundledMissing(name string) error {
	return fmt.Errorf("the bundled %s binary is missing; %s", name, reinstallServerHint)
}
