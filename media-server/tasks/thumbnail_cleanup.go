package tasks

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/stevecastle/shrike/appconfig"
	"github.com/stevecastle/shrike/jobqueue"
)

// Cleaning up orphaned thumbnails.
//
// Thumbnails live next to the database in thumbnail_path_100/600/1200
// directories, named by sha256 of the media path (thumbnail.go's
// getThumbnailPath, which the Electron app's image-processing-worker.js
// mirrors exactly): the hex digest alone for images, digest + ".mp4" for
// videos and animated formats, and — when a video frame other than the
// default is wanted — sha256(path + timestamp) where the timestamp is
// formatted like JavaScript's Number.toString(). Nothing deletes those files
// when a media row goes away, so over time the cache accumulates thumbnails
// of items the library no longer knows.
//
// This task rebuilds the set of names the library could still ask for —
// sha256(path) for every media row, sha256(path + ts) for every tagged
// timestamp on a live row, plus any file the media/tag tables reference by
// literal path (tag previews and DB-recorded thumbnails) — and deletes cache
// files answering to no name in that set. A deleted thumbnail that turns out
// to be wanted again is regenerated on demand, so the worst case of an
// overly-aggressive pass is a one-time re-render, never data loss. Files
// whose names don't look like ours (not a 64-char hex stem) are left alone.

var thumbnailCleanupOptions = []TaskOption{
	{Name: "dry-run", Label: "Dry Run", Type: "bool",
		Description: "Report which thumbnails would be deleted without removing anything"},
}

// thumbCacheDirs are the cache directories getThumbnailPath writes into,
// one per size the apps request.
var thumbCacheDirs = []string{"thumbnail_path_100", "thumbnail_path_600", "thumbnail_path_1200"}

// thumbCleanupMinAge protects freshly-written files: a thumbnail generated
// for a row inserted after this task snapshotted the media table would look
// orphaned, so anything newer than this is skipped and caught on a later run.
const thumbCleanupMinAge = 15 * time.Minute

// thumbCleanupPreviewMax caps how many individual deletions a dry run prints.
const thumbCleanupPreviewMax = 40

// thumbHashHex must produce byte-identical output to createHash in
// thumbnail.go and createHash in the Electron app's
// image-processing-worker.js: lowercase hex of sha256.
func thumbHashHex(input string) string {
	h := sha256.Sum256([]byte(input))
	return fmt.Sprintf("%x", h)
}

// thumbTimeStampKey formats a timestamp the way both generators feed it into
// the hash: JavaScript Number.toString() semantics (no trailing zeros, no
// exponent for normal values), matching formatTimeStamp in thumbnail.go.
func thumbTimeStampKey(ts float64) string {
	return strconv.FormatFloat(ts, 'f', -1, 64)
}

// thumbKeepSet is everything the cleanup must not delete: hash stems the
// library can still derive, and literal file paths the database references.
type thumbKeepSet struct {
	stems map[string]struct{} // lowercase 64-hex digests
	paths map[string]struct{} // slash-normalized, lowercased absolute paths
}

func (k *thumbKeepSet) addStemsFor(input string) {
	// The hash is exact-string sensitive but the two apps may spell the same
	// file with different separators; keeping both spellings costs two set
	// entries and prevents deleting a thumbnail that is still valid under the
	// other spelling.
	k.stems[thumbHashHex(input)] = struct{}{}
	if alt := strings.ReplaceAll(input, `\`, "/"); alt != input {
		k.stems[thumbHashHex(alt)] = struct{}{}
	}
	if alt := strings.ReplaceAll(input, "/", `\`); alt != input {
		k.stems[thumbHashHex(alt)] = struct{}{}
	}
}

func (k *thumbKeepSet) addPath(p string) {
	if p = strings.TrimSpace(p); p == "" || strings.HasPrefix(p, "s3://") {
		return
	}
	k.paths[strings.ToLower(strings.ReplaceAll(p, `\`, "/"))] = struct{}{}
}

// loadThumbKeepSet builds the keep set from the database:
//  1. sha256(path) for every media row — the default thumbnail of each item.
//  2. sha256(path + timestamp) for every tagged timestamp on a live media
//     row — tag-at-time thumbnails both apps generate for videos.
//  3. Every literal path stored in media.thumbnail_path_600/1200 and
//     tag.thumbnail_path_600 — DB-recorded thumbnails and tag previews,
//     which may predate the current naming scheme.
func loadThumbKeepSet(ctx context.Context, db *sql.DB) (*thumbKeepSet, error) {
	keep := &thumbKeepSet{stems: map[string]struct{}{}, paths: map[string]struct{}{}}

	rows, err := db.QueryContext(ctx, `SELECT "path" FROM media`)
	if err != nil {
		return nil, fmt.Errorf("loading media paths: %w", err)
	}
	for rows.Next() {
		var p string
		if err := rows.Scan(&p); err != nil {
			rows.Close()
			return nil, err
		}
		keep.addStemsFor(p)
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return nil, err
	}

	// Timestamped variants: only for rows the media table still has — a
	// timestamp on an orphaned tag row names a thumbnail of a deleted item,
	// which is exactly what this task exists to remove.
	tsRows, err := db.QueryContext(ctx, `
		SELECT DISTINCT mtbc.media_path, mtbc.time_stamp
		FROM media_tag_by_category mtbc
		WHERE mtbc.time_stamp > 0
		  AND EXISTS (SELECT 1 FROM media m WHERE m."path" = mtbc.media_path)`)
	if err != nil {
		return nil, fmt.Errorf("loading tagged timestamps: %w", err)
	}
	for tsRows.Next() {
		var p string
		var ts float64
		if err := tsRows.Scan(&p, &ts); err != nil {
			tsRows.Close()
			return nil, err
		}
		suffix := thumbTimeStampKey(ts)
		keep.addStemsFor(p + suffix)
		if alt := strings.ReplaceAll(p, `\`, "/"); alt != p {
			keep.addStemsFor(alt + suffix)
		}
	}
	tsRows.Close()
	if err := tsRows.Err(); err != nil {
		return nil, err
	}

	// Literal references. Column set differs between library generations
	// (the viewer's schema and the server's have drifted before), so a pair
	// whose query fails is skipped rather than failing the sweep — a missing
	// column can't be referencing anything.
	for _, ref := range []struct{ table, column string }{
		{"media", "thumbnail_path_600"},
		{"media", "thumbnail_path_1200"},
		{"media", "thumbnail_path_100"},
		{"tag", "thumbnail_path_600"},
	} {
		refRows, err := db.QueryContext(ctx, fmt.Sprintf(
			`SELECT DISTINCT %s FROM %s WHERE %s IS NOT NULL AND %s != ''`,
			ref.column, ref.table, ref.column, ref.column))
		if err != nil {
			continue
		}
		for refRows.Next() {
			var p sql.NullString
			if err := refRows.Scan(&p); err != nil {
				break
			}
			if p.Valid {
				keep.addPath(p.String)
			}
		}
		refRows.Close()
	}

	return keep, nil
}

// thumbCacheFile is one candidate file found in a cache directory.
type thumbCacheFile struct {
	Path    string
	Stem    string // lowercase name minus extension
	Size    int64
	ModTime time.Time
}

// isHex64 reports whether s is exactly 64 lowercase hex characters — the
// shape of every filename the thumbnail generators produce.
func isHex64(s string) bool {
	if len(s) != 64 {
		return false
	}
	for _, c := range s {
		if (c < '0' || c > '9') && (c < 'a' || c > 'f') {
			return false
		}
	}
	return true
}

// listThumbCacheFiles walks the cache directories under baseDir and returns
// the files whose names match the generators' naming scheme. Files that
// don't (README drops, foreign caches, whatever) are counted but never
// candidates for deletion. Missing directories are fine — a library that
// never rendered a size simply doesn't have that folder.
func listThumbCacheFiles(baseDir string) (files []thumbCacheFile, unrecognized int, err error) {
	for _, dir := range thumbCacheDirs {
		entries, rerr := os.ReadDir(filepath.Join(baseDir, dir))
		if rerr != nil {
			if os.IsNotExist(rerr) {
				continue
			}
			return nil, unrecognized, fmt.Errorf("reading %s: %w", dir, rerr)
		}
		for _, e := range entries {
			if e.IsDir() {
				continue
			}
			name := strings.ToLower(e.Name())
			ext := filepath.Ext(name)
			stem := strings.TrimSuffix(name, ext)
			switch ext {
			case "", ".mp4", ".png", ".jpg", ".jpeg", ".webp":
			default:
				unrecognized++
				continue
			}
			if !isHex64(stem) {
				unrecognized++
				continue
			}
			info, ierr := e.Info()
			if ierr != nil {
				continue
			}
			files = append(files, thumbCacheFile{
				Path:    filepath.Join(baseDir, dir, e.Name()),
				Stem:    stem,
				Size:    info.Size(),
				ModTime: info.ModTime(),
			})
		}
	}
	return files, unrecognized, nil
}

// keeps reports whether the keep set protects f.
func (k *thumbKeepSet) keeps(f thumbCacheFile) bool {
	if _, ok := k.stems[f.Stem]; ok {
		return true
	}
	_, ok := k.paths[strings.ToLower(strings.ReplaceAll(f.Path, `\`, "/"))]
	return ok
}

func thumbnailCleanupTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	ctx := j.Ctx
	opts := ParseOptions(&jobqueue.Job{Arguments: dirTaskTokens(j)}, thumbnailCleanupOptions)
	dryRun, _ := opts["dry-run"].(bool)

	dbPath := appconfig.Get().DBPath
	if dbPath == "" {
		q.PushJobStdout(j.ID, "Error: no database path configured — cannot locate the thumbnail cache")
		q.ErrorJob(j.ID)
		return fmt.Errorf("no database path configured")
	}
	baseDir := filepath.Dir(dbPath)
	q.PushJobStdout(j.ID, fmt.Sprintf("Scanning thumbnail cache under %s", baseDir))

	keep, err := loadThumbKeepSet(ctx, q.Db)
	if err != nil {
		if ctx.Err() != nil {
			q.PushJobStdout(j.ID, "Task was canceled")
			_ = q.CancelJob(j.ID)
			return err
		}
		q.PushJobStdout(j.ID, fmt.Sprintf("Error building the set of valid thumbnails: %v", err))
		q.ErrorJob(j.ID)
		return err
	}

	files, unrecognized, err := listThumbCacheFiles(baseDir)
	if err != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("Error scanning thumbnail directories: %v", err))
		q.ErrorJob(j.ID)
		return err
	}
	q.PushJobStdout(j.ID, fmt.Sprintf(
		"Found %d thumbnail file(s) in the cache (%d unrelated file(s) left alone)",
		len(files), unrecognized))
	if len(files) == 0 {
		q.PushJobStdout(j.ID, "Nothing to clean")
		q.CompleteJob(j.ID)
		return nil
	}

	cutoff := time.Now().Add(-thumbCleanupMinAge)
	var removed, failed, tooNew int
	var freed int64
	var previewShown int
	_ = q.SetJobProgress(j.ID, 0, len(files))
	for i, f := range files {
		select {
		case <-ctx.Done():
			q.PushJobStdout(j.ID, fmt.Sprintf("Task was canceled after removing %d thumbnail(s)", removed))
			_ = q.CancelJob(j.ID)
			return ctx.Err()
		default:
		}
		if q.PauseRequested(j.ID) {
			q.PushJobStdout(j.ID, fmt.Sprintf("Paused at %d/%d — resume to continue", i, len(files)))
			return jobqueue.ErrPaused
		}
		_ = q.SetJobProgress(j.ID, i, len(files))

		if keep.keeps(f) {
			continue
		}
		if f.ModTime.After(cutoff) {
			// Could be a thumbnail for a row added since our snapshot —
			// leave it for the next run rather than racing the generator.
			tooNew++
			continue
		}

		if dryRun {
			removed++
			freed += f.Size
			if previewShown < thumbCleanupPreviewMax {
				q.PushJobStdout(j.ID, "Would delete: "+f.Path)
				previewShown++
			} else if previewShown == thumbCleanupPreviewMax {
				q.PushJobStdout(j.ID, "... (further deletions not listed)")
				previewShown++
			}
			continue
		}
		if err := os.Remove(f.Path); err != nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("Warning: failed to delete %s: %v", f.Path, err))
			failed++
			continue
		}
		removed++
		freed += f.Size
		if removed%500 == 0 {
			q.PushJobStdout(j.ID, fmt.Sprintf("Progress: %d orphaned thumbnail(s) removed so far", removed))
		}
	}
	_ = q.SetJobProgress(j.ID, len(files), len(files))

	verb := "Removed"
	if dryRun {
		verb = "Would remove"
	}
	q.PushJobStdout(j.ID, fmt.Sprintf(
		"%s %d orphaned thumbnail(s), reclaiming %.1f MB (%d kept, %d too new to judge, %d failed)",
		verb, removed, float64(freed)/(1024*1024), len(files)-removed-failed-tooNew, tooNew, failed))
	if dryRun {
		q.PushJobStdout(j.ID, "Dry run: nothing was deleted")
	}

	q.CompleteJob(j.ID)
	return nil
}
