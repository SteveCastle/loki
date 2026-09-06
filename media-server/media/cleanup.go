package media

import (
	"context"
	"database/sql"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

// Library cleanup: forgetting media that no longer exists.
//
// A library of millions of rows accumulates entries whose files were deleted,
// moved, or lost with a drive. Those rows keep showing up in queries,
// similarity results, swipe pools, and people groups. CleanupLibrary removes
// them — and every row in any other table that names them — in four phases:
//
//  1. Scan: walk the media table by primary key in chunks and stat every
//     path (local: a bounded worker pool; remote: the storage layer). Nothing
//     is deleted. Missing paths are collected as candidates, grouped by the
//     volume/root they live on.
//  2. Verify: apply the safety guards, then stat every candidate AGAIN. A
//     file that reappears (a share that hiccupped, a drive that reconnected)
//     is dropped from the candidates. Only what is missing twice, minutes
//     apart, moves on.
//  3. Remove: RemoveItemsFromDBStream over the verified candidates, in
//     independently committed batches — media row, tags, embeddings, faces
//     and their curation assertions, scan markers, battle-log rows — with the
//     removal hook evicting each batch from the live vector and face indexes.
//  4. Orphans: sweep every sidecar table for rows whose media path has no
//     media row at all (left behind by older deleters that only removed the
//     media row, or by the viewer deleting through a different code path),
//     and remove those references too.
//
// Guards — the reason the scan and the removal are separate passes:
//
//   - Unavailable roots. A path on an unmounted drive, an unreachable share,
//     or a bucket that is no longer configured stats exactly like a deleted
//     file. Missing paths whose root is unavailable are never candidates.
//     Roots are the configured storage root containing the path when there
//     is one (a bind mount that is present but empty is caught by the next
//     guard), else the drive/UNC volume, else the remote bucket.
//   - Missing-ratio cap. If more than MaxMissingPercent of a root's scanned
//     items are missing (and the sample is not tiny), the root is skipped
//     and reported instead of purged: a drive remounted under a new letter,
//     a folder renamed, or a mount that came up empty look exactly like mass
//     deletion, and the fix for those is `media move`, not forgetting the
//     metadata. The cap can be raised to 100 for a deliberate purge.
//   - Re-verification, described above, for intermittent connectivity.
//
// Progress is reported per chunk in every phase so a run over millions of
// rows never looks hung, and each removal batch is its own transaction, so
// cancelling or pausing keeps everything committed so far.

// CleanupPhase names the stage a progress report comes from.
type CleanupPhase string

const (
	CleanupPhaseScan    CleanupPhase = "scan"
	CleanupPhaseVerify  CleanupPhase = "verify"
	CleanupPhaseRemove  CleanupPhase = "remove"
	CleanupPhaseOrphans CleanupPhase = "orphans"
)

const (
	// DefaultCleanupMaxMissingPercent is the per-root guard threshold.
	DefaultCleanupMaxMissingPercent = 50.0
	// DefaultCleanupMinGuardSample is how many items a root must have before
	// the ratio guard applies — a root of three files, two deleted, is not a
	// disconnected drive.
	DefaultCleanupMinGuardSample = 20
	defaultCleanupScanBatch      = 5000
	defaultCleanupRemoveChunk    = 5000
)

// CleanupOptions tunes a CleanupLibrary run.
type CleanupOptions struct {
	// DryRun reports what would be removed without deleting anything.
	DryRun bool
	// Scope restricts the run to media paths under this directory
	// (recursive, both separator spellings). Empty means the whole library.
	Scope string
	// MaxMissingPercent is the per-root guard; <= 0 uses the default, 100
	// disables the guard.
	MaxMissingPercent float64
	// MinGuardSample is the minimum scanned items for the guard to apply;
	// <= 0 uses the default.
	MinGuardSample int
	// SkipOrphans skips phase 4.
	SkipOrphans bool
	// ScanBatch is rows per database read; <= 0 uses the default.
	ScanBatch int
	// Progress is called after every chunk of every phase, on the calling
	// goroutine. May be nil.
	Progress func(CleanupProgress)
	// Log receives human-readable notes (guard decisions, warnings). May be nil.
	Log func(string)
	// Interrupt is polled between chunks; a non-nil error stops the run
	// with that error (the job queue's pause signal, typically). May be nil.
	Interrupt func() error

	// beforeVerify runs between the scan and verify phases (tests use it to
	// bring a file back).
	beforeVerify func()
}

// CleanupProgress is one progress report. Counters are running totals for
// the whole run; Done/Total are for the current phase (Total is 0 while
// unknown, as in the orphan sweep).
type CleanupProgress struct {
	Phase              CleanupPhase
	Done, Total        int64
	Scanned            int64
	Missing            int64
	Recovered          int64
	SkippedUnavailable int64
	SkippedGuard       int64
	Removed            int64
	OrphanPaths        int64
}

// CleanupRoot is one availability group seen during the scan.
type CleanupRoot struct {
	Root      string
	Scanned   int64
	Missing   int64
	Available bool
	Guarded   bool
}

// CleanupResult is what a run did (or, in a dry run, would do).
type CleanupResult struct {
	DryRun bool

	MediaScanned int64
	MissingFound int64
	// Recovered counts candidates that were present again on verification.
	Recovered int64

	// Rows removed per table, phases 3 and 4 combined. In a dry run
	// MediaRemoved is the number of verified candidates and the rest are 0.
	MediaRemoved          int64
	TagsRemoved           int64
	EmbeddingsRemoved     int64
	FacesRemoved          int64
	FaceAssertionsRemoved int64
	FaceScansRemoved      int64
	BattlesRemoved        int64

	// OrphanPaths is how many distinct dangling paths phase 4 swept (in a dry
	// run: counted per table, so a path dangling in two tables counts twice).
	OrphanPaths        int64
	OrphanPathsByTable map[string]int64

	SkippedUnavailable int64
	UnavailableRoots   []string
	SkippedByGuard     int64
	GuardedRoots       []CleanupRoot
	Roots              []CleanupRoot
	Errors             []error
}

// Root resolution hooks, wired at startup from the storage registry. Without
// them a path's root is its drive/UNC volume (or nothing on unix paths) and
// every remote bucket counts as available.
var (
	cleanupHookMu       sync.RWMutex
	cleanupRootResolver func(path string) (root string, ok bool)
	remoteRootChecker   func(root string) bool
)

// SetCleanupRootResolver installs the function that maps a local path to the
// configured storage root containing it. Returning ok=false falls back to
// the volume root.
func SetCleanupRootResolver(fn func(path string) (string, bool)) {
	cleanupHookMu.Lock()
	cleanupRootResolver = fn
	cleanupHookMu.Unlock()
}

// SetRemoteRootChecker installs the function that reports whether a remote
// root ("s3://bucket/") is currently configured and reachable enough to be
// trusted about missing objects.
func SetRemoteRootChecker(fn func(root string) bool) {
	cleanupHookMu.Lock()
	remoteRootChecker = fn
	cleanupHookMu.Unlock()
}

// cleanupRoot returns the availability group a path belongs to.
func cleanupRoot(p string) string {
	if IsRemotePath(p) {
		rest := strings.TrimPrefix(p, "s3://")
		if i := strings.IndexByte(rest, '/'); i >= 0 {
			rest = rest[:i]
		}
		return "s3://" + rest + "/"
	}
	cleanupHookMu.RLock()
	fn := cleanupRootResolver
	cleanupHookMu.RUnlock()
	if fn != nil {
		if r, ok := fn(p); ok && r != "" {
			return r
		}
	}
	return volumeRoot(p)
}

// cleanupRootAvailable reports whether a root can be trusted to answer
// "missing": a local root must stat, a remote root must still be configured.
// The empty root (paths with no volume) is always trusted — the ratio guard
// covers it.
func cleanupRootAvailable(root string) bool {
	if root == "" {
		return true
	}
	if IsRemotePath(root) {
		cleanupHookMu.RLock()
		fn := remoteRootChecker
		cleanupHookMu.RUnlock()
		if fn == nil {
			return true
		}
		return fn(root)
	}
	_, err := os.Stat(root)
	return err == nil
}

// pathRange is one keyset-walkable slice of a path-keyed column: paths in
// [lo, hi) — or everything when both bounds are empty.
type pathRange struct {
	lo, hi string
}

// cleanupRanges returns the ranges a scope covers: one open range for the
// whole library, else one half-open prefix range per separator spelling of
// the scope directory.
func cleanupRanges(scope string) []pathRange {
	scope = strings.TrimSpace(scope)
	if scope == "" {
		return []pathRange{{}}
	}
	base := filepath.Clean(scope) + string(filepath.Separator)
	seen := map[string]bool{}
	var out []pathRange
	for _, p := range []string{base, strings.ReplaceAll(base, `\`, "/"), strings.ReplaceAll(base, "/", `\`)} {
		if seen[p] {
			continue
		}
		seen[p] = true
		out = append(out, pathRange{lo: p, hi: p + "\U0010FFFF"})
	}
	return out
}

// where builds the range predicate for a keyset step: after the last path
// seen (exclusive), or from the range's lower bound (inclusive) on the first
// step; bounded above when the range is.
func (r pathRange) where(col, after string, first bool) (string, []any) {
	var conds []string
	var args []any
	switch {
	case first && r.lo != "":
		conds = append(conds, col+" >= ?")
		args = append(args, r.lo)
	case !first:
		conds = append(conds, col+" > ?")
		args = append(args, after)
	}
	if r.hi != "" {
		conds = append(conds, col+" < ?")
		args = append(args, r.hi)
	}
	if len(conds) == 0 {
		return "1=1", nil
	}
	return strings.Join(conds, " AND "), args
}

// existingTables returns the names of the tables in the database.
func existingTables(ctx context.Context, db *sql.DB) (map[string]bool, error) {
	rows, err := db.QueryContext(ctx, `SELECT name FROM sqlite_master WHERE type = 'table'`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := map[string]bool{}
	for rows.Next() {
		var n string
		if err := rows.Scan(&n); err != nil {
			return nil, err
		}
		out[n] = true
	}
	return out, rows.Err()
}

// accumulate folds one removal result into the cleanup totals.
func (res *CleanupResult) accumulate(r *RemovalResult) {
	if r == nil {
		return
	}
	res.MediaRemoved += r.MediaItemsRemoved
	res.TagsRemoved += r.TagsRemoved
	res.EmbeddingsRemoved += r.EmbeddingsRemoved
	res.FacesRemoved += r.FacesRemoved
	res.FaceAssertionsRemoved += r.FaceAssertionsRemoved
	res.FaceScansRemoved += r.FaceScansRemoved
	res.BattlesRemoved += r.BattlesRemoved
	res.Errors = append(res.Errors, r.Errors...)
}

// CleanupLibrary runs the four phases described at the top of this file.
// On interruption or cancellation it returns the partial result with the
// error; everything already removed stays removed.
func CleanupLibrary(ctx context.Context, db *sql.DB, opts CleanupOptions) (*CleanupResult, error) {
	res := &CleanupResult{DryRun: opts.DryRun, OrphanPathsByTable: map[string]int64{}}
	if db == nil {
		return res, fmt.Errorf("database connection not available")
	}
	if err := ctx.Err(); err != nil {
		return res, err
	}

	maxPct := opts.MaxMissingPercent
	if maxPct <= 0 {
		maxPct = DefaultCleanupMaxMissingPercent
	}
	minSample := opts.MinGuardSample
	if minSample <= 0 {
		minSample = DefaultCleanupMinGuardSample
	}
	scanBatch := opts.ScanBatch
	if scanBatch <= 0 {
		scanBatch = defaultCleanupScanBatch
	}
	logf := opts.Log
	if logf == nil {
		logf = func(string) {}
	}
	progress := opts.Progress
	if progress == nil {
		progress = func(CleanupProgress) {}
	}
	interrupt := func() error {
		if err := ctx.Err(); err != nil {
			return err
		}
		if opts.Interrupt != nil {
			return opts.Interrupt()
		}
		return nil
	}

	ranges := cleanupRanges(opts.Scope)
	prog := CleanupProgress{Phase: CleanupPhaseScan}
	for _, r := range ranges {
		where, args := r.where(`"path"`, "", true)
		var n int64
		if err := db.QueryRowContext(ctx, `SELECT COUNT(*) FROM media WHERE `+where, args...).Scan(&n); err != nil {
			return res, fmt.Errorf("counting media: %w", err)
		}
		prog.Total += n
	}
	progress(prog)

	// ---- Phase 1: scan ------------------------------------------------------
	roots := map[string]*CleanupRoot{}
	var rootOrder []string
	rootFor := func(p string) *CleanupRoot {
		key := cleanupRoot(p)
		rs := roots[key]
		if rs == nil {
			rs = &CleanupRoot{Root: key, Available: cleanupRootAvailable(key)}
			roots[key] = rs
			rootOrder = append(rootOrder, key)
			if !rs.Available {
				logf(fmt.Sprintf("Root %s is not available; its missing items will be left alone", displayRoot(key)))
			}
		}
		return rs
	}
	var candidates []string
	for _, r := range ranges {
		after, first := "", true
		for {
			if err := interrupt(); err != nil {
				return res, err
			}
			where, args := r.where(`"path"`, after, first)
			rows, err := db.QueryContext(ctx,
				`SELECT "path" FROM media WHERE `+where+` ORDER BY "path" LIMIT ?`,
				append(args, scanBatch)...)
			if err != nil {
				return res, fmt.Errorf("scanning media: %w", err)
			}
			var batch []string
			for rows.Next() {
				var p string
				if err := rows.Scan(&p); err != nil {
					rows.Close()
					return res, err
				}
				batch = append(batch, p)
			}
			rows.Close()
			if err := rows.Err(); err != nil {
				return res, err
			}
			if len(batch) == 0 {
				break
			}
			after, first = batch[len(batch)-1], false

			exists := CheckFilesExistConcurrent(batch)
			for _, p := range batch {
				prog.Scanned++
				rs := rootFor(p)
				rs.Scanned++
				if exists[p] {
					continue
				}
				rs.Missing++
				prog.Missing++
				if !rs.Available {
					prog.SkippedUnavailable++
					continue
				}
				candidates = append(candidates, p)
			}
			prog.Done = prog.Scanned
			progress(prog)
			if len(batch) < scanBatch {
				break
			}
		}
	}
	res.MediaScanned = prog.Scanned
	res.MissingFound = prog.Missing

	// ---- Phase 2: guards + verify ------------------------------------------
	if opts.beforeVerify != nil {
		opts.beforeVerify()
	}
	for _, key := range rootOrder {
		rs := roots[key]
		if !rs.Available {
			continue
		}
		if rs.Scanned >= int64(minSample) && float64(rs.Missing)*100 > maxPct*float64(rs.Scanned) {
			rs.Guarded = true
			logf(fmt.Sprintf(
				"GUARD: %d of %d items under %s are missing (%.0f%%, cap %.0f%%) — this looks like a disconnected, moved, or renamed volume, not deletions. Its items are left alone; raise the cap to purge them deliberately.",
				rs.Missing, rs.Scanned, displayRoot(key), float64(rs.Missing)*100/float64(rs.Scanned), maxPct))
		}
	}

	prog.Phase = CleanupPhaseVerify
	prog.Done, prog.Total = 0, int64(len(candidates))
	progress(prog)
	verified := make([]string, 0, len(candidates))
	for i := 0; i < len(candidates); i += scanBatch {
		if err := interrupt(); err != nil {
			return res, err
		}
		end := i + scanBatch
		if end > len(candidates) {
			end = len(candidates)
		}
		chunk := candidates[i:end]
		exists := CheckFilesExistConcurrent(chunk)
		for _, p := range chunk {
			rs := roots[cleanupRoot(p)]
			if rs != nil && rs.Guarded {
				prog.SkippedGuard++
				continue
			}
			if exists[p] {
				prog.Recovered++
				continue
			}
			verified = append(verified, p)
		}
		prog.Done = int64(end)
		progress(prog)
	}
	candidates = nil

	// A root that vanished during the run (drive unplugged mid-scan) must not
	// have its now-unverifiable items removed.
	vanished := false
	for _, key := range rootOrder {
		rs := roots[key]
		if rs.Available && !rs.Guarded && !cleanupRootAvailable(key) {
			rs.Available = false
			vanished = true
			logf(fmt.Sprintf("Root %s became unavailable during the run; its missing items are left alone", displayRoot(key)))
		}
	}
	if vanished {
		kept := verified[:0]
		for _, p := range verified {
			if rs := roots[cleanupRoot(p)]; rs != nil && !rs.Available {
				prog.SkippedUnavailable++
				continue
			}
			kept = append(kept, p)
		}
		verified = kept
	}
	for _, key := range rootOrder {
		rs := roots[key]
		res.Roots = append(res.Roots, *rs)
		if !rs.Available {
			res.UnavailableRoots = append(res.UnavailableRoots, key)
		}
		if rs.Guarded {
			res.GuardedRoots = append(res.GuardedRoots, *rs)
		}
	}
	res.Recovered = prog.Recovered
	res.SkippedUnavailable = prog.SkippedUnavailable
	res.SkippedByGuard = prog.SkippedGuard

	// ---- Phase 3: remove ----------------------------------------------------
	prog.Phase = CleanupPhaseRemove
	prog.Done, prog.Total = 0, int64(len(verified))
	progress(prog)
	if opts.DryRun {
		res.MediaRemoved = int64(len(verified))
		prog.Removed = res.MediaRemoved
		prog.Done = prog.Total
		progress(prog)
	} else {
		for i := 0; i < len(verified); i += defaultCleanupRemoveChunk {
			if err := interrupt(); err != nil {
				return res, err
			}
			end := i + defaultCleanupRemoveChunk
			if end > len(verified) {
				end = len(verified)
			}
			base := res.MediaRemoved
			r, err := RemoveItemsFromDBStream(ctx, db, verified[i:end], func(b RemovalBatch) {
				prog.Done = int64(i + b.Done)
				prog.Removed = base + b.MediaItemsRemoved
				progress(prog)
			})
			res.accumulate(r)
			if err != nil {
				return res, err
			}
			prog.Removed = res.MediaRemoved
		}
	}
	verified = nil

	// ---- Phase 4: dangling sidecar rows ------------------------------------
	if opts.SkipOrphans {
		return res, nil
	}
	prog.Phase = CleanupPhaseOrphans
	prog.Done, prog.Total = 0, 0
	progress(prog)
	tables, err := existingTables(ctx, db)
	if err != nil {
		return res, err
	}
	for _, pc := range movablePathColumns {
		if pc.Table == "media" || !tables[pc.Table] {
			continue
		}
		key := pc.Table + "." + pc.Column
		dangling := fmt.Sprintf(`NOT EXISTS (SELECT 1 FROM media m WHERE m."path" = %s.%s)`, pc.Table, pc.quoted)
		for _, r := range ranges {
			if opts.DryRun {
				where, args := r.where(pc.quoted, "", true)
				var n int64
				if err := db.QueryRowContext(ctx, fmt.Sprintf(
					`SELECT COUNT(DISTINCT %s) FROM %s WHERE %s AND %s`, pc.quoted, pc.Table, where, dangling), args...,
				).Scan(&n); err != nil {
					return res, fmt.Errorf("counting dangling %s: %w", key, err)
				}
				res.OrphanPathsByTable[key] += n
				res.OrphanPaths += n
				prog.OrphanPaths = res.OrphanPaths
				prog.Done = prog.OrphanPaths
				progress(prog)
				continue
			}
			after, first := "", true
			for {
				if err := interrupt(); err != nil {
					return res, err
				}
				where, args := r.where(pc.quoted, after, first)
				rows, err := db.QueryContext(ctx, fmt.Sprintf(
					`SELECT DISTINCT %s FROM %s WHERE %s AND %s ORDER BY %s LIMIT ?`,
					pc.quoted, pc.Table, where, dangling, pc.quoted), append(args, scanBatch)...)
				if err != nil {
					return res, fmt.Errorf("scanning dangling %s: %w", key, err)
				}
				var batch []string
				for rows.Next() {
					var p sql.NullString
					if err := rows.Scan(&p); err != nil {
						rows.Close()
						return res, err
					}
					if p.Valid {
						batch = append(batch, p.String)
					}
				}
				rows.Close()
				if err := rows.Err(); err != nil {
					return res, err
				}
				if len(batch) == 0 {
					break
				}
				after, first = batch[len(batch)-1], false
				rr, err := RemoveItemsFromDBStream(ctx, db, batch, nil)
				res.accumulate(rr)
				if err != nil {
					return res, err
				}
				res.OrphanPathsByTable[key] += int64(len(batch))
				res.OrphanPaths += int64(len(batch))
				prog.OrphanPaths = res.OrphanPaths
				prog.Done = prog.OrphanPaths
				progress(prog)
				if len(batch) < scanBatch {
					break
				}
			}
		}
	}
	return res, nil
}

// displayRoot names a root in log lines.
func displayRoot(root string) string {
	if root == "" {
		return "(paths without a volume)"
	}
	return root
}

// StreamingCleanupNonExistentItems is the pre-phase API: scan and remove,
// no orphan sweep, progress as (missing found, removed so far). Kept for
// callers and tests written against it; new code should use CleanupLibrary.
func StreamingCleanupNonExistentItems(ctx context.Context, db *sql.DB, progressCallback func(found, removed int)) (*RemovalResult, error) {
	res, err := CleanupLibrary(ctx, db, CleanupOptions{
		SkipOrphans: true,
		Progress: func(p CleanupProgress) {
			if progressCallback != nil && p.Phase == CleanupPhaseRemove && p.Done > 0 {
				progressCallback(int(p.Missing), int(p.Removed))
			}
		},
	})
	out := &RemovalResult{}
	if res != nil {
		out.MediaItemsRemoved = res.MediaRemoved
		out.TagsRemoved = res.TagsRemoved
		out.EmbeddingsRemoved = res.EmbeddingsRemoved
		out.FacesRemoved = res.FacesRemoved
		out.FaceAssertionsRemoved = res.FaceAssertionsRemoved
		out.FaceScansRemoved = res.FaceScansRemoved
		out.BattlesRemoved = res.BattlesRemoved
		out.SkippedUnavailable = res.SkippedUnavailable
		out.UnavailableRoots = res.UnavailableRoots
		out.Errors = res.Errors
	}
	return out, err
}
