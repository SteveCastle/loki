package tasks

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
	"unicode"
	"unicode/utf8"

	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/media"
)

// Splitting an oversized directory.
//
// A folder with tens of thousands of images takes minutes to open in Explorer
// (and is unpleasant in any file manager). This task fans one flat directory
// out into subfolders — alphabetically by leading character, or by the file's
// modification date — and keeps the library in agreement by re-pointing every
// database reference through media.MovePath, which covers ALL path-bearing
// tables (media, tags, embeddings, faces, face scans, battle log).
//
// The two halves must not drift: a file moved on disk whose row still names the
// old path loses its tags, its similarity vector, and its faces. So each file
// is renamed, then its rows are rewritten, and a failure on the database side
// rolls the filesystem move back rather than leaving the pair inconsistent.

var splitDirOptions = []TaskOption{
	{Name: "target", Label: "Target Directory", Type: "string", Required: true,
		Description: "Directory whose files will be split into subfolders"},
	{Name: "mode", Label: "Organize By", Type: "enum", Choices: []string{"alpha", "date"}, Default: "alpha",
		Description: "alpha: bucket by the first character of the filename. date: bucket by file modification time"},
	{Name: "granularity", Label: "Date Granularity", Type: "enum", Choices: []string{"year", "month", "day"}, Default: "month",
		Description: "Date mode only: folder per year (2026), month (2026-08), or day (2026-08-08)"},
	{Name: "max-per-dir", Label: "Max Files Per Subfolder", Type: "number", Default: 0.0,
		Description: "Cap each subfolder at this many files, overflowing into numbered siblings (A_1, A_2). 0 = no cap"},
	{Name: "scope", Label: "Files To Move", Type: "enum", Choices: []string{"all", "library"}, Default: "all",
		Description: "all: every file in the directory. library: only files the media library knows about, plus their sidecars — leaves unrelated files (installers, archives, stray downloads) where they are"},
	{Name: "keep-recent", Label: "Keep Recent Periods In Place", Type: "number", Default: 0.0,
		Description: "Date mode only: leave the N most recent periods where they are, so the directory stays a working folder and only settled files get filed away. 1 with month granularity keeps the current month. 0 files everything"},
	{Name: "prune-missing", Label: "Prune Missing Files", Type: "bool",
		Description: "Delete library rows for files in this directory that no longer exist on disk (tags, embeddings, faces, battle log and all)"},
	{Name: "dry-run", Label: "Dry Run", Type: "bool",
		Description: "Report the planned layout without moving anything"},
}

// splitPlanPreviewMax caps how many individual moves a dry run prints — the
// per-bucket summary is the useful part; the sample is just a sanity check.
const splitPlanPreviewMax = 40

// splitFile is one file in the source directory and where it is headed.
type splitFile struct {
	Name    string    // base name as it appears on disk
	ModTime time.Time // for date bucketing and within-bucket ordering
	Bucket  string    // bucket key before any overflow numbering
	Folder  string    // final subfolder name
	// Follows is the name of the file this one is a sidecar of, if any. A
	// sidecar inherits its primary's folder so the pair is never separated.
	Follows string
}

func splitDirTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	ctx := j.Ctx

	tokens := splitDirTokens(j)
	opts := ParseOptions(&jobqueue.Job{Arguments: tokens}, splitDirOptions)
	targetDir, _ := opts["target"].(string)
	mode, _ := opts["mode"].(string)
	granularity, _ := opts["granularity"].(string)
	maxPerDirF, _ := opts["max-per-dir"].(float64)
	keepRecentF, _ := opts["keep-recent"].(float64)
	keepRecent := int(keepRecentF)
	scope, _ := opts["scope"].(string)
	pruneMissing, _ := opts["prune-missing"].(bool)
	dryRun, _ := opts["dry-run"].(bool)
	maxPerDir := int(maxPerDirF)

	// Same shape as the move task: a bare positional token is the directory.
	if targetDir == "" {
		for _, tok := range tokens {
			if !strings.HasPrefix(tok, "-") {
				targetDir = tok
				break
			}
		}
	}
	if targetDir == "" {
		q.PushJobStdout(j.ID, "Error: no target directory specified")
		q.ErrorJob(j.ID)
		return fmt.Errorf("no target directory specified")
	}
	switch mode {
	case "alpha", "date":
	default:
		q.PushJobStdout(j.ID, fmt.Sprintf("Error: unknown mode %q (expected alpha or date)", mode))
		q.ErrorJob(j.ID)
		return fmt.Errorf("unknown mode %q", mode)
	}
	switch granularity {
	case "year", "month", "day":
	default:
		q.PushJobStdout(j.ID, fmt.Sprintf("Error: unknown granularity %q (expected year, month, or day)", granularity))
		q.ErrorJob(j.ID)
		return fmt.Errorf("unknown granularity %q", granularity)
	}
	switch scope {
	case "all", "library":
	default:
		q.PushJobStdout(j.ID, fmt.Sprintf("Error: unknown scope %q (expected all or library)", scope))
		q.ErrorJob(j.ID)
		return fmt.Errorf("unknown scope %q", scope)
	}
	if maxPerDir < 0 {
		maxPerDir = 0
	}
	if keepRecent < 0 {
		keepRecent = 0
	}
	if keepRecent > 0 && mode != "date" {
		// There is no "recent" in alphabetical order, and silently ignoring the
		// flag would archive the working files it was meant to protect.
		q.PushJobStdout(j.ID, "Error: --keep-recent needs --mode date (alphabetical buckets have no recency)")
		q.ErrorJob(j.ID)
		return fmt.Errorf("--keep-recent requires mode=date")
	}

	absTarget, err := filepath.Abs(targetDir)
	if err != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("Error resolving target directory: %v", err))
		q.ErrorJob(j.ID)
		return err
	}
	absTarget = filepath.Clean(filepath.FromSlash(absTarget))
	info, err := os.Stat(absTarget)
	if err != nil || !info.IsDir() {
		q.PushJobStdout(j.ID, fmt.Sprintf("Error: not a directory: %s", absTarget))
		q.ErrorJob(j.ID)
		return fmt.Errorf("not a directory: %s", absTarget)
	}

	files, err := listSplitCandidates(absTarget)
	if err != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("Error reading directory: %v", err))
		q.ErrorJob(j.ID)
		return err
	}
	if len(files) == 0 {
		q.PushJobStdout(j.ID, fmt.Sprintf("No files directly in %s — nothing to split", absTarget))
		q.CompleteJob(j.ID)
		return nil
	}

	q.PushJobStdout(j.ID, fmt.Sprintf("Splitting %s", absTarget))
	q.PushJobStdout(j.ID, fmt.Sprintf("Files directly in the directory: %d  mode=%s  granularity=%s  maxPerDir=%d  scope=%s",
		len(files), mode, granularity, maxPerDir, scope))

	// One indexed sweep instead of a lookup per file: the library's stored
	// spelling of a path (separator style and letter case) need not match what
	// the directory listing reports, and a miss here would move the file while
	// silently leaving every row behind.
	stored, err := storedPathsUnder(ctx, q.Db, absTarget)
	if err != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("Error loading library paths under target: %v", err))
		q.ErrorJob(j.ID)
		return err
	}

	if pruneMissing {
		// Sidecar tables can name a path the media table doesn't — an item
		// whose media row was deleted without its embedding, say. Those rows
		// are invisible to a media-only sweep, so they are collected too.
		strays, err := rootPathsOutsideMedia(ctx, q.Db, absTarget, stored)
		if err != nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("Warning: could not scan for rows outside the media table: %v", err))
		}
		orphans := rootPathsMissingFromDisk(absTarget, files, stored, strays)
		if len(orphans) == 0 {
			q.PushJobStdout(j.ID, "No library rows point at files missing from this directory")
		} else if dryRun {
			q.PushJobStdout(j.ID, fmt.Sprintf("Would prune %d library row(s) whose file is gone (dry run)", len(orphans)))
			for _, p := range orphans[:min(len(orphans), splitPlanPreviewMax)] {
				q.PushJobStdout(j.ID, "  would prune: "+p)
			}
		} else {
			mediaRows, err := pruneMissingItems(ctx, q.Db, orphans)
			if err != nil {
				// A failed prune is not a reason to abandon the split — the
				// stale rows were already stale.
				q.PushJobStdout(j.ID, fmt.Sprintf("Warning: pruning missing items failed: %v", err))
			}
			// Paths, not media rows: a stray embedding or battle entry whose
			// media row is already gone still counts as cleaned up, and
			// reporting only media rows would say "0" while deleting real rows.
			q.PushJobStdout(j.ID, fmt.Sprintf(
				"Pruned %d path(s) whose file no longer exists (%d had a media row; the rest were stray tag/embedding/face/battle references)",
				len(orphans), mediaRows))
			for _, p := range orphans {
				stored.Forget(p)
			}
		}
	}

	if scope == "library" {
		kept, dropped := filterToLibrary(files, absTarget, stored)
		q.PushJobStdout(j.ID, fmt.Sprintf(
			"Scope 'library': %d files participate, %d left in place (no library row and not a sidecar of one)",
			len(kept), dropped))
		files = kept
		if len(files) == 0 {
			q.PushJobStdout(j.ID, "Nothing to split")
			q.CompleteJob(j.ID)
			return nil
		}
	}

	assignSplitBuckets(files, mode, granularity)

	if keepRecent > 0 {
		cutoff := recentBucketCutoff(time.Now(), granularity, keepRecent)
		kept, held := partitionRecent(files, cutoff)
		q.PushJobStdout(j.ID, fmt.Sprintf(
			"Keeping the %d most recent %s period(s) in place: %d file(s) at or after %s stay in the directory",
			keepRecent, granularity, held, cutoff))
		files = kept
		if len(files) == 0 {
			q.PushJobStdout(j.ID, "Nothing older than the working window — nothing to file away")
			q.CompleteJob(j.ID)
			return nil
		}
	}

	// Subfolders that already exist count against the cap, so pausing and
	// resuming (or re-running to sweep newly-added files) keeps honoring it
	// instead of refilling a folder that is already at its limit.
	existing, err := existingSubdirCounts(absTarget)
	if err != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("Warning: could not survey existing subfolders: %v", err))
		existing = map[string]int{}
	}
	assignSplitFolders(files, maxPerDir, existing)

	for _, line := range splitPlanSummary(files) {
		q.PushJobStdout(j.ID, line)
	}

	q.PushJobStdout(j.ID, fmt.Sprintf("Library rows under this directory: %d", stored.Len()))

	if dryRun {
		shown := 0
		for _, f := range files {
			if shown >= splitPlanPreviewMax {
				q.PushJobStdout(j.ID, fmt.Sprintf("... and %d more", len(files)-shown))
				break
			}
			q.PushJobStdout(j.ID, fmt.Sprintf("Would move: %s -> %s", f.Name, filepath.Join(f.Folder, f.Name)))
			shown++
		}
		q.PushJobStdout(j.ID, "Dry run: nothing was moved and no rows were changed")
		q.CompleteJob(j.ID)
		return nil
	}

	_ = q.SetJobProgress(j.ID, 0, len(files))
	var moved, updated, unlinked, skipped int
	for i, f := range files {
		select {
		case <-ctx.Done():
			q.PushJobStdout(j.ID, "Task was canceled")
			_ = q.CancelJob(j.ID)
			return ctx.Err()
		default:
		}
		if q.PauseRequested(j.ID) {
			q.PushJobStdout(j.ID, fmt.Sprintf("Paused at %d/%d - resume to continue", i, len(files)))
			return jobqueue.ErrPaused
		}
		_ = q.SetJobProgress(j.ID, i, len(files))

		srcPath := filepath.Join(absTarget, f.Name)
		destDir := filepath.Join(absTarget, f.Folder)
		destPath := filepath.Join(destDir, f.Name)

		if _, err := os.Stat(destPath); err == nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("Warning: destination already exists, skipping: %s", destPath))
			skipped++
			continue
		}
		if err := os.MkdirAll(destDir, 0755); err != nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("Warning: failed to create %s: %v", destDir, err))
			skipped++
			continue
		}
		if err := os.Rename(srcPath, destPath); err != nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("Warning: failed to move %s: %v", f.Name, err))
			skipped++
			continue
		}
		moved++

		// The library's own spelling of this path, so the rewrite matches the
		// rows that actually exist.
		dbFrom, inLibrary := stored.Lookup(srcPath)
		if !inLibrary {
			unlinked++
			q.RegisterOutputFile(j.ID, destPath)
			continue
		}
		dbTo := siblingUnder(dbFrom, f.Folder)
		res, err := media.MovePath(ctx, q.Db, dbFrom, dbTo, media.MoveOptions{})
		if err != nil {
			// Leaving the file in its new home with the library still naming
			// the old one is the failure this task exists to prevent, so put
			// the file back and report it.
			var conflict *media.MoveConflictError
			if errors.As(err, &conflict) {
				q.PushJobStdout(j.ID, fmt.Sprintf("Warning: %s already in the library at the destination, reverting move of %s", conflict.Error(), f.Name))
			} else {
				q.PushJobStdout(j.ID, fmt.Sprintf("Warning: database update failed for %s: %v — reverting move", f.Name, err))
			}
			if rerr := os.Rename(destPath, srcPath); rerr != nil {
				q.PushJobStdout(j.ID, fmt.Sprintf("Error: could not revert %s -> %s: %v (file and library are now out of sync)", destPath, srcPath, rerr))
			}
			moved--
			skipped++
			continue
		}
		updated += int(res.Items)
		// Derived in-memory state is keyed by path: without this the vector
		// index keeps returning the old path and the face index's path→keys
		// map misses on the next eviction.
		if res.Items > 0 {
			IndexRenamePath(q.Db, dbFrom, dbTo)
			FaceIndexRenamePath(dbFrom, dbTo)
		}
		q.RegisterOutputFile(j.ID, destPath)

		if (i+1)%250 == 0 {
			q.PushJobStdout(j.ID, fmt.Sprintf("Progress: %d/%d files processed", i+1, len(files)))
		}
	}

	_ = q.SetJobProgress(j.ID, len(files), len(files))
	q.PushJobStdout(j.ID, fmt.Sprintf(
		"Split complete: %d files moved, %d library items re-pointed, %d not in the library, %d skipped",
		moved, updated, unlinked, skipped))

	select {
	case <-ctx.Done():
		q.PushJobStdout(j.ID, "Task was canceled")
		q.ErrorJob(j.ID)
		return ctx.Err()
	default:
	}
	q.CompleteJob(j.ID)
	return nil
}

// splitDirTokens is the job's option tokens, arguments and input together.
//
// /create splits a typed command line at the LAST token and stores that one as
// the job's Input — a convention that fits per-item tasks, whose final token is
// the query or path list. This task's final token is just as likely to be
// "--mode=date" or the value of a preceding flag, so both halves are read back
// as one list. Input lines are taken whole (never re-split on spaces): /create
// has already handled quoting, and a workflow hands paths down one per line.
func splitDirTokens(j *jobqueue.Job) []string {
	tokens := append([]string{}, j.Arguments...)
	for line := range strings.SplitSeq(j.Input, "\n") {
		if line = strings.TrimSpace(line); line != "" {
			tokens = append(tokens, line)
		}
	}
	return tokens
}

// listSplitCandidates returns the regular files sitting DIRECTLY in dir. The
// task never recurses: subfolders are where files are going, and descending
// into them would re-shuffle a previous run's output.
func listSplitCandidates(dir string) ([]splitFile, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	names := make(map[string]bool, len(entries))
	for _, e := range entries {
		if !e.IsDir() {
			names[e.Name()] = true
		}
	}
	out := make([]splitFile, 0, len(names))
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		info, err := e.Info()
		if err != nil {
			continue
		}
		if !info.Mode().IsRegular() {
			continue
		}
		out = append(out, splitFile{Name: e.Name(), ModTime: info.ModTime()})
	}
	resolveSidecars(out, names)
	sort.Slice(out, func(i, j int) bool {
		return strings.ToLower(out[i].Name) < strings.ToLower(out[j].Name)
	})
	return out, nil
}

// sidecarExts are the extensions that describe another file rather than being
// content in their own right: transcripts, subtitles, and the metadata blobs
// downloaders write beside what they fetched.
var sidecarExts = map[string]bool{
	".json": true, ".vtt": true, ".srt": true, ".ass": true, ".ssa": true,
	".txt": true, ".nfo": true, ".description": true, ".info": true,
}

// resolveSidecars points each sidecar at the file it describes, so the two are
// never split up. Two spellings are in the wild:
//
//	photo.jpg.json  — the full name plus an extension (gallery-dl, yt-dlp)
//	video.srt       — the same stem, a different extension (subtitle tools)
//
// The second form is only applied to known sidecar extensions: "a.jpg" and
// "a.png" share a stem too, and neither describes the other.
func resolveSidecars(files []splitFile, names map[string]bool) {
	stems := map[string][]string{}
	for name := range names {
		if ext := strings.ToLower(filepath.Ext(name)); !sidecarExts[ext] {
			stem := strings.TrimSuffix(name, filepath.Ext(name))
			stems[stem] = append(stems[stem], name)
		}
	}
	for i := range files {
		name := files[i].Name
		stem := strings.TrimSuffix(name, filepath.Ext(name))
		if stem == name {
			continue
		}
		if names[stem] {
			files[i].Follows = stem
			continue
		}
		// Exactly one candidate, or the pairing is a guess — and a guess that
		// moves a subtitle away from its video is no better than leaving it.
		if sidecarExts[strings.ToLower(filepath.Ext(name))] && len(stems[stem]) == 1 {
			files[i].Follows = stems[stem][0]
		}
	}
}

// rootPathsMissingFromDisk returns the library's paths that sit DIRECTLY in dir
// and have no file behind them any more.
//
// The scan is deliberately narrow. An unmounted volume stats exactly like a
// deleted file, and purging a library because a drive was unplugged is not
// recoverable — so this only ever considers the one directory the task just
// listed successfully, and never its subdirectories. A name that matches only
// case-insensitively still counts as present: refusing to prune costs nothing,
// and pruning a file that exists costs everything.
func rootPathsMissingFromDisk(dir string, onDisk []splitFile, stored *storedPaths, strays []string) []string {
	present := make(map[string]bool, len(onDisk)*2)
	for _, f := range onDisk {
		present[f.Name] = true
		present[strings.ToLower(f.Name)] = true
	}
	prefix := slashKey(dir) + "/"
	seen := map[string]bool{}
	var missing []string
	consider := func(path string) {
		key := slashKey(path)
		rest, under := strings.CutPrefix(key, prefix)
		if !under || rest == "" || strings.Contains(rest, "/") {
			return // in a subdirectory, or not under dir at all
		}
		if present[rest] || present[strings.ToLower(rest)] || seen[key] {
			return
		}
		seen[key] = true
		missing = append(missing, path)
	}
	for _, path := range stored.exact {
		consider(path)
	}
	for _, path := range strays {
		consider(path)
	}
	sort.Strings(missing)
	return missing
}

// sidecarPathColumns are the path-bearing columns OUTSIDE the media table. It
// mirrors media.movablePathColumns; a table missing here is a reference the
// prune can't see.
var sidecarPathColumns = []struct{ Table, Column string }{
	{"media_tag_by_category", "media_path"},
	{"media_embedding", "media_path"},
	{"face", "media_path"},
	{"face_scan", "media_path"},
	{"battle", "winner_path"},
	{"battle", "loser_path"},
}

// rootPathsOutsideMedia finds paths under dir that a sidecar table references
// but the media table does not. These are rows the library can never reach —
// every query joins through media — so if their file is gone too, nothing will
// ever notice them again except a sweep like this one.
func rootPathsOutsideMedia(ctx context.Context, db *sql.DB, dir string, stored *storedPaths) ([]string, error) {
	if db == nil {
		return nil, fmt.Errorf("database connection not available")
	}
	n := len([]rune(dir)) + 1
	want := slashKey(dir) + "/"
	seen := map[string]bool{}
	var out []string
	for _, pc := range sidecarPathColumns {
		ok, err := tableExists(ctx, db, pc.Table)
		if err != nil {
			return out, err
		}
		if !ok {
			continue // viewer-only library: no face/embedding tables
		}
		rows, err := db.QueryContext(ctx, fmt.Sprintf(
			`SELECT DISTINCT %s FROM %s WHERE lower(replace(substr(%s, 1, ?), '\', '/')) = lower(?)`,
			pc.Column, pc.Table, pc.Column), n, want)
		if err != nil {
			return out, fmt.Errorf("scanning %s.%s: %w", pc.Table, pc.Column, err)
		}
		for rows.Next() {
			var p string
			if err := rows.Scan(&p); err != nil {
				rows.Close()
				return out, err
			}
			if _, isMedia := stored.Lookup(p); isMedia || seen[slashKey(p)] {
				continue
			}
			seen[slashKey(p)] = true
			out = append(out, p)
		}
		rows.Close()
		if err := rows.Err(); err != nil {
			return out, err
		}
	}
	return out, nil
}

func tableExists(ctx context.Context, db *sql.DB, name string) (bool, error) {
	var one int
	err := db.QueryRowContext(ctx,
		`SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?`, name).Scan(&one)
	if err == sql.ErrNoRows {
		return false, nil
	}
	return err == nil, err
}

// pruneMissingItems erases every trace of paths whose files are gone: the
// curation assertions keyed by their face ids first (those outlive the faces
// otherwise), then tags, embeddings, faces, scan markers and the media row,
// then the battle-log entries that would keep a deleted item in the Elo
// history. Returns how many media rows were removed, even on a partial failure.
func pruneMissingItems(ctx context.Context, db *sql.DB, paths []string) (int64, error) {
	if len(paths) == 0 {
		return 0, nil
	}
	for _, p := range paths {
		if err := media.DeleteFacesForMedia(db, p); err != nil {
			return 0, fmt.Errorf("clearing faces for %s: %w", p, err)
		}
	}
	// RemoveItemsFromDB batches and commits per chunk, and its removal hook
	// evicts the paths from the live vector and face indexes.
	res, err := media.RemoveItemsFromDB(ctx, db, paths)
	var removed int64
	if res != nil {
		removed = res.MediaItemsRemoved
	}
	if err != nil {
		return removed, err
	}
	const batch = 500
	for i := 0; i < len(paths); i += batch {
		chunk := paths[i:min(i+batch, len(paths))]
		placeholders := strings.TrimSuffix(strings.Repeat("?,", len(chunk)), ",")
		args := make([]any, 0, len(chunk)*2)
		for _, p := range chunk {
			args = append(args, p)
		}
		for _, p := range chunk {
			args = append(args, p)
		}
		if _, err := db.ExecContext(ctx, fmt.Sprintf(
			`DELETE FROM battle WHERE winner_path IN (%s) OR loser_path IN (%s)`,
			placeholders, placeholders), args...); err != nil {
			return removed, fmt.Errorf("clearing battle log: %w", err)
		}
	}
	media.InvalidateRandomSampleCache()
	return removed, nil
}

// filterToLibrary keeps only the files the media library has a row for, plus
// the sidecars of those files. It is how a directory holding both a library and
// unrelated clutter (installers, archives, an old database dump) gets split
// without dragging the clutter along.
func filterToLibrary(files []splitFile, dir string, stored *storedPaths) (kept []splitFile, dropped int) {
	inLibrary := make(map[string]bool, len(files))
	for _, f := range files {
		if _, ok := stored.Lookup(filepath.Join(dir, f.Name)); ok {
			inLibrary[f.Name] = true
		}
	}
	kept = make([]splitFile, 0, len(files))
	for _, f := range files {
		if inLibrary[f.Name] || (f.Follows != "" && inLibrary[f.Follows]) {
			kept = append(kept, f)
			continue
		}
		dropped++
	}
	return kept, dropped
}

// assignSplitBuckets fills in each file's bucket key. Sidecars are resolved
// last so they land wherever their primary did.
func assignSplitBuckets(files []splitFile, mode, granularity string) {
	byName := make(map[string]int, len(files))
	for i := range files {
		byName[files[i].Name] = i
	}
	for i := range files {
		if mode == "date" {
			files[i].Bucket = dateBucket(files[i].ModTime, granularity)
		} else {
			files[i].Bucket = alphaBucket(files[i].Name)
		}
	}
	for i := range files {
		if files[i].Follows == "" {
			continue
		}
		if p, ok := byName[files[i].Follows]; ok {
			files[i].Bucket = files[p].Bucket
		}
	}
}

// alphaBucket names the folder a filename belongs in: its first letter
// upper-cased (so "apple.jpg" and "Apple.jpg" share one folder), "0-9" for a
// leading digit, and "#" for anything else.
func alphaBucket(name string) string {
	r, _ := utf8.DecodeRuneInString(name)
	switch {
	case unicode.IsLetter(r):
		return string(unicode.ToUpper(r))
	case unicode.IsDigit(r):
		return "0-9"
	}
	return "#"
}

// recentBucketCutoff returns the oldest bucket key that still counts as "the
// working window": keep=1 is the period containing now, keep=2 adds the one
// before it, and so on.
//
// Recency is measured against the CLOCK, not against the newest file present.
// The directory is a working folder — if nothing has been saved for three
// months, those three-month-old files have settled and belong in the archive.
// Anchoring to the newest file instead would keep them in place forever purely
// because nothing newer arrived.
func recentBucketCutoff(now time.Time, granularity string, keep int) string {
	if keep < 1 {
		keep = 1
	}
	back := keep - 1
	switch granularity {
	case "year":
		return dateBucket(now.AddDate(-back, 0, 0), granularity)
	case "day":
		return dateBucket(now.AddDate(0, 0, -back), granularity)
	default:
		// Step a month at a time: AddDate(0,-n,0) from the 31st lands on a
		// short month's overflow (March 31 - 1 month = March 3), which would
		// silently skip February.
		t := now
		for range back {
			t = t.AddDate(0, 0, -t.Day()) // last day of the previous month
		}
		return dateBucket(t, granularity)
	}
}

// partitionRecent splits files into the ones to file away and a count of those
// left in place. Bucket keys are zero-padded and sort lexicographically, so a
// string compare orders them by date — and a file dated in the future (clock
// skew, or a deliberately post-dated mtime) compares above the cutoff and
// stays put, which is the right answer for a working directory.
func partitionRecent(files []splitFile, cutoff string) (toFile []splitFile, held int) {
	toFile = make([]splitFile, 0, len(files))
	for _, f := range files {
		if f.Bucket >= cutoff {
			held++
			continue
		}
		toFile = append(toFile, f)
	}
	return toFile, held
}

// dateBucket names the folder for a modification time. The formats sort
// lexicographically, so the subfolders read in chronological order.
func dateBucket(t time.Time, granularity string) string {
	switch granularity {
	case "year":
		return t.Format("2006")
	case "day":
		return t.Format("2006-01-02")
	default:
		return t.Format("2006-01")
	}
}

// assignSplitFolders turns bucket keys into concrete folder names. Without a
// cap the folder IS the bucket. With one, every folder is numbered ("A_1",
// "A_2") — uniform naming means a resumed or repeated run can tell how full
// each folder already is and keep filling from there instead of overflowing a
// folder that has no number to grow from.
func assignSplitFolders(files []splitFile, maxPerDir int, existing map[string]int) {
	if maxPerDir <= 0 {
		for i := range files {
			files[i].Folder = files[i].Bucket
		}
		return
	}
	// Per bucket: which numbered folder is being filled, and how much room is
	// left in it after whatever a previous run already put there.
	type cursor struct {
		n    int
		room int
	}
	cursors := map[string]*cursor{}
	for i := range files {
		b := files[i].Bucket
		c, ok := cursors[b]
		if !ok {
			c = &cursor{n: 1}
			c.room = maxPerDir - existing[splitFolderName(b, 1)]
			cursors[b] = c
		}
		for c.room <= 0 {
			c.n++
			c.room = maxPerDir - existing[splitFolderName(b, c.n)]
		}
		files[i].Folder = splitFolderName(b, c.n)
		c.room--
	}
}

func splitFolderName(bucket string, n int) string {
	return fmt.Sprintf("%s_%d", bucket, n)
}

// existingSubdirCounts reports how many files each immediate subfolder of dir
// already holds.
func existingSubdirCounts(dir string) (map[string]int, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	counts := map[string]int{}
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		sub, err := os.ReadDir(filepath.Join(dir, e.Name()))
		if err != nil {
			continue
		}
		n := 0
		for _, s := range sub {
			if !s.IsDir() {
				n++
			}
		}
		counts[e.Name()] = n
	}
	return counts, nil
}

// splitPlanSummary renders the folder→count breakdown a human reads before
// deciding the split is what they wanted.
func splitPlanSummary(files []splitFile) []string {
	counts := map[string]int{}
	for _, f := range files {
		counts[f.Folder]++
	}
	folders := make([]string, 0, len(counts))
	for name := range counts {
		folders = append(folders, name)
	}
	sort.Strings(folders)
	out := make([]string, 0, len(folders)+1)
	out = append(out, fmt.Sprintf("Plan: %d subfolders", len(folders)))
	for _, name := range folders {
		out = append(out, fmt.Sprintf("  %s  %d files", name, counts[name]))
	}
	return out
}

// slashKey normalizes only the separator style, which is a spelling choice
// rather than an identity: "C:\pics\a.jpg" and "C:/pics/a.jpg" are the same
// file on every platform this runs on.
func slashKey(p string) string {
	return strings.ReplaceAll(p, `\`, "/")
}

// storedPaths indexes the library's own spelling of every path under a
// directory. media.MovePath matches rows by exact string, so handing it the
// directory listing's spelling of a path the library stores differently would
// move the file and update nothing.
type storedPaths struct {
	// exact is keyed on separator-normalized, case-preserved path — the only
	// key that is safe on a case-sensitive filesystem.
	exact map[string]string
	// folded is keyed on the lower-cased form, for the Windows case where the
	// stored path and the on-disk listing disagree about letter case. Keys that
	// two different stored paths share are removed rather than guessed at.
	folded map[string]string
}

// Lookup returns the library's spelling of an on-disk path, if it has one.
func (s *storedPaths) Lookup(p string) (string, bool) {
	k := slashKey(p)
	if stored, ok := s.exact[k]; ok {
		return stored, true
	}
	stored, ok := s.folded[strings.ToLower(k)]
	return stored, ok
}

func (s *storedPaths) Len() int { return len(s.exact) }

// Forget drops a path that no longer has a row, so a later lookup can't hand
// the mover a target that was just deleted.
func (s *storedPaths) Forget(stored string) {
	k := slashKey(stored)
	delete(s.exact, k)
	delete(s.folded, strings.ToLower(k))
}

// storedPathsUnder loads every media row under dir in one indexed sweep.
func storedPathsUnder(ctx context.Context, db *sql.DB, dir string) (*storedPaths, error) {
	if db == nil {
		return nil, fmt.Errorf("database connection not available")
	}
	// Segment-aligned, so "/a/foo" never drags "/a/foobar" in. Separators are
	// folded to "/" on both sides because the library's spelling of a path need
	// not match the caller's, and lower() widens the sweep to case variants;
	// which of those actually correspond is settled per-file by Lookup.
	n := len([]rune(dir)) + 1
	rows, err := db.QueryContext(ctx, `
		SELECT "path" FROM media
		WHERE lower(replace(substr("path", 1, ?), '\', '/')) = lower(?)`,
		n, slashKey(dir)+"/",
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := &storedPaths{exact: map[string]string{}, folded: map[string]string{}}
	ambiguous := map[string]bool{}
	for rows.Next() {
		var p string
		if err := rows.Scan(&p); err != nil {
			return nil, err
		}
		k := slashKey(p)
		out.exact[k] = p
		fold := strings.ToLower(k)
		if prev, seen := out.folded[fold]; seen && prev != p {
			ambiguous[fold] = true
		}
		out.folded[fold] = p
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	for fold := range ambiguous {
		delete(out.folded, fold)
	}
	return out, nil
}

// siblingUnder rewrites a stored path to sit in a subfolder of its own
// directory, preserving the separator style the library recorded.
func siblingUnder(stored, folder string) string {
	sep := "/"
	if strings.LastIndex(stored, `\`) > strings.LastIndex(stored, "/") {
		sep = `\`
	}
	i := strings.LastIndexAny(stored, `/\`)
	if i < 0 {
		return folder + sep + stored
	}
	return stored[:i] + sep + folder + sep + stored[i+1:]
}
