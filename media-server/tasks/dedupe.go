package tasks

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"fmt"
	"hash/crc32"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/media"
)

// Deduplicating a directory, a query, or an explicit selection.
//
// Downloads, imports, and manual copies leave a library holding the same bytes
// under several names. This task deletes zero-byte files and finds EXACT
// duplicates with the cheapest escalation that stays certain: files are
// grouped by size (free — no bytes read), size collisions are subdivided by a
// CRC32 of the first 64KB (one small read), and only files that still collide
// get a full SHA-256. Files whose full hashes match are byte-identical
// duplicates.
//
// Input follows the bulk-task contract: a directory (--target, optionally
// --recursive), a library search query (--query / --query64 — the only form
// that can address more media than any one folder), or a newline-separated
// path list (the palette's discrete selection).
//
// Each duplicate group is then collapsed through media.MergeInto — the same
// merge behind the viewer's Merge action — so the kept file gains every tag,
// per-model embedding, and transcript its duplicates had, and the duplicates
// are deleted from disk with every database reference (tags, media row,
// embeddings, faces, battle log) erased. The kept path is the one the library
// already has a row for, so metadata always consolidates onto a path queries
// can reach.

var dedupeOptions = []TaskOption{
	{Name: "target", Label: "Target Directory", Type: "string",
		Description: "Directory to scan for exact duplicate files. Omit to pass a search query (--query/--query64) or a newline-separated path list instead"},
	{Name: "recursive", Label: "Recursive", Type: "bool",
		Description: "Directory mode only: scan subdirectories too. Duplicates are matched across the whole tree"},
	{Name: "dry-run", Label: "Dry Run", Type: "bool",
		Description: "Report zero-byte files and duplicate groups without deleting or merging anything"},
}

// dedupePreviewMax caps how many individual lines a dry run prints per
// section; the totals are the useful part.
const dedupePreviewMax = 40

// dedupeHeadSize is how much of each size-colliding file the cheap
// subdivision hash reads. 64KB separates almost everything that is not a true
// duplicate (headers, timestamps, and encoder metadata live up front) while
// costing a single small read per candidate.
const dedupeHeadSize = 64 * 1024

type dedupeFile struct {
	path string
	size int64
}

func dedupeTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	ctx := j.Ctx

	tokens := dirTaskTokens(j)
	opts := ParseOptions(&jobqueue.Job{Arguments: tokens}, dedupeOptions)
	targetDir, _ := opts["target"].(string)
	recursive, _ := opts["recursive"].(bool)
	dryRun, _ := opts["dry-run"].(bool)

	// Like the other bulk tasks, a search query addresses media the palette's
	// directory scan can't — the whole library view, a tag, a filter stack.
	queryStr, hasQuery := extractQueryFromJob(j)

	// With --target and --query absent, the bare positional tokens are either
	// the directory to scan (split-dir/move convention) or an explicit path
	// list (the palette's discrete selection, one path per line). A single
	// token that stats as a directory is a directory; anything else is paths.
	var positional []string
	if targetDir == "" && !hasQuery {
		for _, tok := range tokens {
			if !strings.HasPrefix(tok, "-") {
				positional = append(positional, tok)
			}
		}
		if len(positional) == 1 {
			if st, err := os.Stat(positional[0]); err == nil && st.IsDir() {
				targetDir = positional[0]
				positional = nil
			}
		}
	}

	var files []dedupeFile
	var stored *storedPaths
	switch {
	case targetDir != "":
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
		files, err = listDedupeFiles(absTarget, recursive)
		if err != nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("Error reading directory: %v", err))
			q.ErrorJob(j.ID)
			return err
		}
		q.PushJobStdout(j.ID, fmt.Sprintf("Deduplicating %s (recursive=%v)", absTarget, recursive))
		q.PushJobStdout(j.ID, fmt.Sprintf("Files scanned: %d", len(files)))

		// The library's own spelling of every path under the target
		// (separator style and letter case need not match the directory
		// listing), so merges and prunes hit the rows that actually exist.
		stored, err = storedPathsUnder(ctx, q.Db, absTarget)
		if err != nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("Error loading library paths under target: %v", err))
			q.ErrorJob(j.ID)
			return err
		}

	case hasQuery:
		paths, err := getMediaPathsByQueryFast(q.Db, queryStr)
		if err != nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("Error resolving query: %v", err))
			q.ErrorJob(j.ID)
			return err
		}
		q.PushJobStdout(j.ID, fmt.Sprintf("Deduplicating query %q: %d item(s)", queryStr, len(paths)))
		files = statDedupeCandidates(q, j, paths)
		stored, err = storedPathsFor(ctx, q.Db, paths)
		if err != nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("Error loading library paths: %v", err))
			q.ErrorJob(j.ID)
			return err
		}

	case len(positional) > 0:
		// The explicit-list shape every per-item task accepts: local paths
		// absolutized, s3:// identities kept verbatim, non-media dropped.
		paths := make([]string, 0, len(positional))
		for _, p := range positional {
			if !strings.HasPrefix(p, "s3://") {
				if abs, err := filepath.Abs(p); err == nil {
					p = filepath.FromSlash(abs)
				}
			}
			paths = append(paths, p)
		}
		paths = filterMediaPaths(paths)
		q.PushJobStdout(j.ID, fmt.Sprintf("Deduplicating %d listed path(s)", len(paths)))
		files = statDedupeCandidates(q, j, paths)
		var err error
		stored, err = storedPathsFor(ctx, q.Db, paths)
		if err != nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("Error loading library paths: %v", err))
			q.ErrorJob(j.ID)
			return err
		}

	default:
		q.PushJobStdout(j.ID, "Error: no target directory, search query, or path list specified")
		q.ErrorJob(j.ID)
		return fmt.Errorf("no target directory, search query, or path list specified")
	}

	// Zero-byte files first: they are damage, not duplicates — nothing to
	// merge, just a file to remove and rows to prune.
	var zero, nonEmpty []dedupeFile
	for _, f := range files {
		if f.size == 0 {
			zero = append(zero, f)
		} else {
			nonEmpty = append(nonEmpty, f)
		}
	}
	if len(zero) > 0 {
		if dryRun {
			q.PushJobStdout(j.ID, fmt.Sprintf("Would delete %d zero-byte file(s) (dry run)", len(zero)))
			for i, f := range zero {
				if i >= dedupePreviewMax {
					q.PushJobStdout(j.ID, fmt.Sprintf("  ... and %d more", len(zero)-i))
					break
				}
				q.PushJobStdout(j.ID, "  would delete: "+f.path)
			}
		} else {
			var prune []string
			removed := 0
			for _, f := range zero {
				if err := os.Remove(f.path); err != nil {
					q.PushJobStdout(j.ID, fmt.Sprintf("Warning: failed to delete zero-byte file %s: %v", f.path, err))
					continue
				}
				removed++
				if dbPath, ok := stored.Lookup(f.path); ok {
					prune = append(prune, dbPath)
					stored.Forget(dbPath)
				}
			}
			if len(prune) > 0 {
				if _, err := pruneMissingItems(ctx, q.Db, prune); err != nil {
					q.PushJobStdout(j.ID, fmt.Sprintf("Warning: pruning rows for deleted zero-byte files failed: %v", err))
				}
			}
			q.PushJobStdout(j.ID, fmt.Sprintf("Deleted %d zero-byte file(s), pruned %d library row(s)", removed, len(prune)))
		}
	}

	// Sidecars (transcripts, subtitles, downloader metadata) are excluded
	// from duplicate matching: two byte-identical .vtt files can describe two
	// DIFFERENT videos, and deleting one silently breaks the other video's
	// transcript. They still participate in zero-byte cleanup above, and a
	// merged-away media file's own sidecar is removed by the merge.
	candidates := make([]dedupeFile, 0, len(nonEmpty))
	sidecarsSkipped := 0
	for _, f := range nonEmpty {
		if sidecarExts[strings.ToLower(filepath.Ext(f.path))] {
			sidecarsSkipped++
			continue
		}
		candidates = append(candidates, f)
	}
	if sidecarsSkipped > 0 {
		q.PushJobStdout(j.ID, fmt.Sprintf("Sidecar files excluded from duplicate matching: %d", sidecarsSkipped))
	}

	groups, err := findDuplicateGroups(ctx, q, j, candidates)
	if err != nil {
		if err == jobqueue.ErrPaused {
			return err
		}
		q.PushJobStdout(j.ID, "Task was canceled")
		_ = q.CancelJob(j.ID)
		return err
	}
	if len(groups) == 0 {
		q.PushJobStdout(j.ID, "No exact duplicates found")
		q.CompleteJob(j.ID)
		return nil
	}
	dupCount := 0
	for _, g := range groups {
		dupCount += len(g) - 1
	}
	q.PushJobStdout(j.ID, fmt.Sprintf("Duplicate groups: %d (%d redundant file(s))", len(groups), dupCount))

	if dryRun {
		for i, g := range groups {
			if i >= dedupePreviewMax {
				q.PushJobStdout(j.ID, fmt.Sprintf("... and %d more group(s)", len(groups)-i))
				break
			}
			keeper, dupes := pickDedupeKeeper(g, stored)
			q.PushJobStdout(j.ID, fmt.Sprintf("Would keep %s and merge+delete: %s", keeper, strings.Join(dupes, ", ")))
		}
		q.PushJobStdout(j.ID, "Dry run: nothing was deleted and no rows were changed")
		q.CompleteJob(j.ID)
		return nil
	}

	_ = q.SetJobProgress(j.ID, 0, len(groups))
	var merged, deleted int
	var tagsGained, embGained, facesRemoved int64
	var failed []string
	for i, g := range groups {
		select {
		case <-ctx.Done():
			q.PushJobStdout(j.ID, "Task was canceled")
			_ = q.CancelJob(j.ID)
			return ctx.Err()
		default:
		}
		if q.PauseRequested(j.ID) {
			q.PushJobStdout(j.ID, fmt.Sprintf("Paused at group %d/%d - resume to continue", i, len(groups)))
			return jobqueue.ErrPaused
		}
		_ = q.SetJobProgress(j.ID, i, len(groups))

		keeper, dupes := pickDedupeKeeper(g, stored)
		res, err := media.MergeInto(ctx, q.Db, keeper, dupes)
		if err != nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("Warning: merge into %s failed: %v", keeper, err))
			failed = append(failed, dupes...)
			continue
		}
		merged++
		deleted += len(res.Deleted)
		tagsGained += res.Tags
		embGained += res.Embeddings
		facesRemoved += res.FacesRemoved
		failed = append(failed, res.Failed...)
		for _, p := range res.Deleted {
			stored.Forget(p)
		}
		q.PushJobStdout(j.ID, fmt.Sprintf("Kept %s: deleted %d duplicate(s), gained %d tag(s), %d embedding(s)",
			keeper, len(res.Deleted), res.Tags, res.Embeddings))
	}
	_ = q.SetJobProgress(j.ID, len(groups), len(groups))

	if facesRemoved > 0 {
		// The duplicates' faces were in someone's group — open People views
		// are now showing stale counts.
		broadcastPeopleUpdated([]string{})
	}
	for _, p := range failed {
		q.PushJobStdout(j.ID, "Warning: could not delete/merge: "+p)
	}
	q.PushJobStdout(j.ID, fmt.Sprintf(
		"Dedupe complete: %d group(s) merged, %d duplicate file(s) deleted, %d tag(s) and %d embedding(s) consolidated, %d failure(s)",
		merged, deleted, tagsGained, embGained, len(failed)))

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

// listDedupeFiles returns the regular files in dir — directly, or the whole
// tree when recursive. Unreadable entries are skipped rather than failing the
// scan; symlinked directories are not followed.
func listDedupeFiles(dir string, recursive bool) ([]dedupeFile, error) {
	var out []dedupeFile
	if !recursive {
		entries, err := os.ReadDir(dir)
		if err != nil {
			return nil, err
		}
		for _, e := range entries {
			info, err := e.Info()
			if err != nil || !info.Mode().IsRegular() {
				continue
			}
			out = append(out, dedupeFile{path: filepath.Join(dir, e.Name()), size: info.Size()})
		}
		return out, nil
	}
	err := filepath.WalkDir(dir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			if path == dir {
				return err
			}
			return nil // an unreadable subtree shouldn't abort the whole scan
		}
		if d.IsDir() {
			return nil
		}
		info, err := d.Info()
		if err != nil || !info.Mode().IsRegular() {
			return nil
		}
		out = append(out, dedupeFile{path: path, size: info.Size()})
		return nil
	})
	if err != nil {
		return nil, err
	}
	return out, nil
}

// statDedupeCandidates turns a resolved query/path-list input into concrete
// on-disk files. Duplicate detection needs the bytes, so s3:// objects and
// paths whose file is missing can't participate — they are counted and left
// alone (their rows stay; a missing file here may be an unmounted volume).
func statDedupeCandidates(q *jobqueue.Queue, j *jobqueue.Job, paths []string) []dedupeFile {
	var out []dedupeFile
	remote, missing := 0, 0
	for _, p := range paths {
		if strings.HasPrefix(p, "s3://") {
			remote++
			continue
		}
		st, err := os.Stat(p)
		if err != nil || !st.Mode().IsRegular() {
			missing++
			continue
		}
		out = append(out, dedupeFile{path: p, size: st.Size()})
	}
	if remote > 0 {
		q.PushJobStdout(j.ID, fmt.Sprintf("Skipping %d s3:// item(s) — remote objects are not scanned for duplicates", remote))
	}
	if missing > 0 {
		q.PushJobStdout(j.ID, fmt.Sprintf("Skipping %d item(s) whose file is not on disk", missing))
	}
	q.PushJobStdout(j.ID, fmt.Sprintf("Files to scan: %d", len(out)))
	return out
}

// storedPathsFor indexes the library's spelling of an explicit path set — the
// query/path-list counterpart of storedPathsUnder. Query results ARE the
// stored spelling already (they came from the media table); a client-supplied
// selection is confirmed against the table so keeper selection knows which
// duplicates actually have rows.
func storedPathsFor(ctx context.Context, db *sql.DB, paths []string) (*storedPaths, error) {
	if db == nil {
		return nil, fmt.Errorf("database connection not available")
	}
	out := &storedPaths{exact: map[string]string{}, folded: map[string]string{}}
	ambiguous := map[string]bool{}
	const batch = 500
	for i := 0; i < len(paths); i += batch {
		chunk := paths[i:min(i+batch, len(paths))]
		placeholders := strings.TrimSuffix(strings.Repeat("?,", len(chunk)), ",")
		args := make([]any, len(chunk))
		for k, p := range chunk {
			args[k] = p
		}
		rows, err := db.QueryContext(ctx, fmt.Sprintf(
			`SELECT "path" FROM media WHERE "path" IN (%s)`, placeholders), args...)
		if err != nil {
			return nil, err
		}
		for rows.Next() {
			var p string
			if err := rows.Scan(&p); err != nil {
				rows.Close()
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
		rows.Close()
		if err := rows.Err(); err != nil {
			return nil, err
		}
	}
	for fold := range ambiguous {
		delete(out.folded, fold)
	}
	return out, nil
}

// findDuplicateGroups returns groups of byte-identical files, cheapest test
// first: size buckets cost nothing, a 64KB head CRC32 splits most size
// collisions after one small read, and a full SHA-256 settles the rest. Only
// files that survive every stage — same size, same head, same full hash — are
// grouped. Progress counts candidate files as their head hash lands.
func findDuplicateGroups(ctx context.Context, q *jobqueue.Queue, j *jobqueue.Job, files []dedupeFile) ([][]string, error) {
	bySize := map[int64][]dedupeFile{}
	for _, f := range files {
		bySize[f.size] = append(bySize[f.size], f)
	}
	total := 0
	for _, g := range bySize {
		if len(g) > 1 {
			total += len(g)
		}
	}
	if total == 0 {
		return nil, nil
	}
	q.PushJobStdout(j.ID, fmt.Sprintf("Size collisions to hash: %d file(s)", total))
	_ = q.SetJobProgress(j.ID, 0, total)

	// Deterministic order (by size) so pause/resume and repeated runs report
	// groups the same way.
	sizes := make([]int64, 0, len(bySize))
	for s, g := range bySize {
		if len(g) > 1 {
			sizes = append(sizes, s)
		}
	}
	sort.Slice(sizes, func(a, b int) bool { return sizes[a] < sizes[b] })

	var groups [][]string
	done := 0
	for _, size := range sizes {
		byHead := map[uint32][]dedupeFile{}
		for _, f := range bySize[size] {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			default:
			}
			if q.PauseRequested(j.ID) {
				q.PushJobStdout(j.ID, fmt.Sprintf("Paused while hashing (%d/%d) - resume to continue", done, total))
				return nil, jobqueue.ErrPaused
			}
			head, err := dedupeHeadCRC(f.path)
			if err != nil {
				q.PushJobStdout(j.ID, fmt.Sprintf("Warning: could not read %s: %v", f.path, err))
				done++
				continue
			}
			byHead[head] = append(byHead[head], f)
			done++
			_ = q.SetJobProgress(j.ID, done, total)
		}
		for _, headGroup := range byHead {
			if len(headGroup) < 2 {
				continue
			}
			byFull := map[string][]string{}
			for _, f := range headGroup {
				select {
				case <-ctx.Done():
					return nil, ctx.Err()
				default:
				}
				sum, err := dedupeFullSHA256(f.path)
				if err != nil {
					q.PushJobStdout(j.ID, fmt.Sprintf("Warning: could not hash %s: %v", f.path, err))
					continue
				}
				byFull[sum] = append(byFull[sum], f.path)
			}
			for _, g := range byFull {
				if len(g) > 1 {
					sort.Strings(g)
					groups = append(groups, g)
				}
			}
		}
	}
	sort.Slice(groups, func(a, b int) bool { return groups[a][0] < groups[b][0] })
	return groups, nil
}

// crc32c is hardware-accelerated on every platform this runs on.
var crc32c = crc32.MakeTable(crc32.Castagnoli)

func dedupeHeadCRC(path string) (uint32, error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, err
	}
	defer f.Close()
	h := crc32.New(crc32c)
	if _, err := io.Copy(h, io.LimitReader(f, dedupeHeadSize)); err != nil {
		return 0, err
	}
	return h.Sum32(), nil
}

func dedupeFullSHA256(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// pickDedupeKeeper chooses which file in a duplicate group survives: a path
// the library has a row for beats one it doesn't (so tags and embeddings
// consolidate onto a path queries can reach), then the shortest path, then
// lexicographic order for determinism. Both the keeper and the dupes come
// back in the library's own spelling where one exists, because the merge
// matches rows by exact string.
func pickDedupeKeeper(group []string, stored *storedPaths) (string, []string) {
	type cand struct {
		path  string
		inLib bool
	}
	cands := make([]cand, 0, len(group))
	for _, p := range group {
		if dbPath, ok := stored.Lookup(p); ok {
			cands = append(cands, cand{path: dbPath, inLib: true})
		} else {
			cands = append(cands, cand{path: p, inLib: false})
		}
	}
	sort.Slice(cands, func(a, b int) bool {
		if cands[a].inLib != cands[b].inLib {
			return cands[a].inLib
		}
		if len(cands[a].path) != len(cands[b].path) {
			return len(cands[a].path) < len(cands[b].path)
		}
		return cands[a].path < cands[b].path
	})
	rest := make([]string, 0, len(cands)-1)
	for _, c := range cands[1:] {
		rest = append(rest, c.path)
	}
	return cands[0].path, rest
}
