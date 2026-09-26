package tasks

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/stevecastle/shrike/embedvec"
	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/media"
	"github.com/stevecastle/shrike/stream"
)

// Finding duplicate CANDIDATES for review — not deduplicating.
//
// The "dedupe" task finds byte-identical files and merges + deletes them on
// its own. This task is the other half of the story: it walks the visual
// embedding vectors, groups items whose similarity to a group anchor clears
// a threshold (default 100%, i.e. visually identical), and records each
// group as a duplicate_group row. Nothing is touched on disk or in the media
// tables — the user reviews the groups in the Duplicates panel and merges,
// dismisses, or trims each one by hand.
//
// Input follows the bulk-task contract of dedupe/split-dir/move: a directory
// (--target, optionally --recursive), a library search query (--query /
// --query64), or a newline-separated path list (the palette's discrete
// selection). With no input at all it runs over EVERY stored vector of the
// active embedding model — the whole library.
//
// Results stream: a group is written (and broadcast to open Duplicates
// panels) the moment it is found, so a long run over a big library yields
// reviewable groups within seconds and can be paused or canceled at any
// point without losing what it already found. Runs are incremental: items
// already in a group (pending or dismissed) are never regrouped, new items
// that match an existing group's anchor join that group, and only items
// nobody has grouped yet can seed a new group — so the task can be re-run
// (or scheduled) forever to keep discovering fresh clusters as the library
// and its embeddings grow.
//
// Candidate pairs come from locality-sensitive hashing (random hyperplanes
// over mean-centred vectors), sized from the threshold so the expected recall
// at exactly the threshold is ~98%; every candidate is verified with the exact
// cosine before it is grouped, so precision is exact and only recall is
// probabilistic. Hyperplanes are re-drawn on every run, which is what makes
// repeated runs converge on the pairs a single pass might miss.

var findDuplicatesOptions = []TaskOption{
	{Name: "threshold", Label: "Similarity Threshold (%)", Type: "number", Default: 100.0,
		Description: "Items whose visual similarity to a group's anchor is at least this percentage are grouped as duplicate candidates. 100 = visually identical (exact copies, re-encodes, resizes); lower to catch crops and edits"},
	{Name: "target", Label: "Target Directory", Type: "string",
		Description: "Directory whose media to scan. Omit to pass a search query (--query/--query64), a newline-separated path list, or nothing at all to scan the whole library"},
	{Name: "recursive", Label: "Recursive", Type: "bool",
		Description: "Directory mode only: scan subdirectories too"},
	{Name: "reset", Label: "Reset Pending Groups", Type: "bool",
		Description: "Forget every pending (not yet reviewed) group before scanning, so groups are rebuilt from scratch at this threshold. Dismissed groups are kept"},
	{Name: "model", Label: "Embedding Model", Type: "string",
		Description: "Embedding model whose vectors to compare (default: the active model)"},
}

// findDuplicatesRecallTarget is the expected recall at exactly the threshold
// the LSH parameters are sized for; pairs well above the threshold are found
// with higher probability still.
const findDuplicatesRecallTarget = 0.98

// findDuplicatesMaxTables bounds the LSH cost at low thresholds; a run that
// hits it reports its (lower) expected recall and simply finds the rest on
// the next run.
const findDuplicatesMaxTables = 64

// findDuplicatesLogGroups is how many groups are announced one per line
// before the log switches to periodic totals.
const findDuplicatesLogGroups = 100

const (
	findDuplicatesBroadcastInterval = 1500 * time.Millisecond
	findDuplicatesLogInterval       = 3 * time.Second
)

// lshSeed draws the hyperplane seed for a run. Package-level so tests can
// pin it; production draws from the clock so consecutive runs use different
// hash families (see the file comment).
var lshSeed = func() int64 { return time.Now().UnixNano() }

// duplicateThresholdScore maps the option's percentage to the cosine floor.
// A tiny slack absorbs float32 accumulation noise so "100%" still matches
// vectors that are identical up to rounding (an identical file embeds to
// the same vector, but re-encodes land a few 1e-5 apart).
func duplicateThresholdScore(pct float64) float32 {
	if pct > 100 {
		pct = 100
	}
	if pct < 1 {
		pct = 1
	}
	return float32(pct/100) - 0.0005
}

// lshParams sizes the hash family: enough bits that a bucket holds ~4 items
// on average (so verification stays cheap), then enough tables that a pair
// at exactly the threshold collides in at least one with probability
// findDuplicatesRecallTarget.
func lshParams(n int, minScore float32) (bits, tables int, recall float64) {
	bits = 6
	if n > 4 {
		bits = int(math.Ceil(math.Log2(float64(n) / 4)))
	}
	bits = max(6, min(bits, 28))
	t := float64(minScore)
	if t > 0.9999 {
		t = 0.9999
	}
	if t < -1 {
		t = -1
	}
	// Probability a random hyperplane separates two vectors at angle θ is θ/π.
	p := math.Acos(t) / math.Pi
	per := math.Pow(1-p, float64(bits))
	if per >= 0.999 {
		tables = 1
	} else {
		tables = int(math.Ceil(math.Log(1-findDuplicatesRecallTarget) / math.Log(1-per)))
	}
	tables = max(1, min(tables, findDuplicatesMaxTables))
	recall = 1 - math.Pow(1-per, float64(tables))
	return bits, tables, recall
}

// dupItem is one vector in the scan.
type dupItem struct {
	path string
	vec  []float32
	// group is the id of the group the item already belongs to (0 = none).
	group int64
	// anchor marks the item as an existing group's anchor: unassigned items
	// that match it join that group instead of seeding a new one.
	anchor bool
	// joinOnly items were loaded solely as join targets (anchors outside the
	// requested scope); they never seed groups.
	joinOnly bool
}

// findDuplicatesStats is what the run reports.
type findDuplicatesStats struct {
	Groups  int // new groups created
	Joined  int // items added to pre-existing groups
	Grouped int // items placed (new groups' members + joined)
}

func findDuplicatesTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	ctx := j.Ctx
	db := q.Db
	if db == nil {
		q.PushJobStdout(j.ID, "Error: database connection not available")
		q.ErrorJob(j.ID)
		return fmt.Errorf("database connection not available")
	}

	tokens := dirTaskTokens(j)
	opts := ParseOptions(&jobqueue.Job{Arguments: tokens}, findDuplicatesOptions)
	pct, _ := opts["threshold"].(float64)
	if pct == 0 {
		pct = 100
	}
	targetDir, _ := opts["target"].(string)
	recursive, _ := opts["recursive"].(bool)
	reset, _ := opts["reset"].(bool)
	minScore := duplicateThresholdScore(pct)

	model := ActiveEmbedModel().ID
	if m, _ := opts["model"].(string); strings.TrimSpace(m) != "" {
		model = strings.TrimSpace(m)
	} else if m, ok := embedModelOverrideFromJob(&jobqueue.Job{Arguments: tokens}); ok {
		model = m
	}

	// Scope resolution — the dedupe task's contract, plus "nothing = all".
	queryStr, hasQuery := extractQueryFromJob(j)
	var positional []string
	if targetDir == "" && !hasQuery {
		skipNext := false
		for i, tok := range tokens {
			if skipNext {
				skipNext = false
				continue
			}
			if strings.HasPrefix(tok, "-") {
				// Value-taking flags written as "--flag value" consume the
				// next token; "--flag=value" is self-contained.
				if !strings.Contains(tok, "=") && i+1 < len(tokens) {
					name := strings.TrimPrefix(tok, "--")
					for _, o := range findDuplicatesOptions {
						if o.Name == name && o.Type != "bool" {
							skipNext = true
						}
					}
				}
				continue
			}
			positional = append(positional, tok)
		}
		if len(positional) == 1 {
			if st, err := os.Stat(positional[0]); err == nil && st.IsDir() {
				targetDir = positional[0]
				positional = nil
			}
		}
	}

	var scope []string // nil = whole library
	scopeLabel := "the whole library"
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
		stored, err := storedPathsUnder(ctx, db, absTarget)
		if err != nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("Error loading library paths under target: %v", err))
			q.ErrorJob(j.ID)
			return err
		}
		scope = make([]string, 0, stored.Len())
		for _, p := range stored.exact {
			if recursive || slashKey(filepath.Dir(p)) == slashKey(absTarget) {
				scope = append(scope, p)
			}
		}
		scope = filterMediaPaths(scope)
		scopeLabel = fmt.Sprintf("%s (recursive=%v)", absTarget, recursive)
	case hasQuery:
		paths, err := getMediaPathsByQueryFast(db, queryStr)
		if err != nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("Error resolving query: %v", err))
			q.ErrorJob(j.ID)
			return err
		}
		scope = paths
		scopeLabel = fmt.Sprintf("query %q", queryStr)
	case len(positional) > 0:
		paths := make([]string, 0, len(positional))
		for _, p := range positional {
			if !strings.HasPrefix(p, "s3://") {
				if abs, err := filepath.Abs(p); err == nil {
					p = filepath.FromSlash(abs)
				}
			}
			paths = append(paths, p)
		}
		scope = filterMediaPaths(paths)
		scopeLabel = fmt.Sprintf("%d listed path(s)", len(scope))
	}
	if scope != nil && len(scope) == 0 {
		q.PushJobStdout(j.ID, "No media items in scope; nothing to compare")
		q.CompleteJob(j.ID)
		return nil
	}

	q.PushJobStdout(j.ID, fmt.Sprintf("Finding duplicate candidates in %s at ≥ %.1f%% similarity (model %s)", scopeLabel, pct, model))
	q.PushJobStdout(j.ID, "Review mode: groups are recorded for the Duplicates panel; nothing is merged or deleted")

	if reset {
		n, err := media.DeleteDuplicateGroups(db, media.DuplicateStatusPending)
		if err != nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("Error resetting pending groups: %v", err))
			q.ErrorJob(j.ID)
			return err
		}
		q.PushJobStdout(j.ID, fmt.Sprintf("Reset: forgot %d pending group(s); dismissed groups kept", n))
		if n > 0 {
			broadcastDuplicatesUpdated(0, 0)
		}
	}
	// Repair anything media removal outside the server left behind (a viewer
	// delete, a manual DB edit) so the assignments below are honest.
	if err := media.PruneDuplicateGroups(db); err != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("Warning: could not prune stale groups: %v", err))
	}

	items, err := loadDuplicateItems(ctx, db, model, scope)
	if err != nil {
		if ctx.Err() != nil {
			q.PushJobStdout(j.ID, "Task was canceled")
			_ = q.CancelJob(j.ID)
			return ctx.Err()
		}
		q.PushJobStdout(j.ID, fmt.Sprintf("Error loading embeddings: %v", err))
		q.ErrorJob(j.ID)
		return err
	}
	if len(items) == 0 {
		q.PushJobStdout(j.ID, "No embeddings found for the selected items. Run the Visual Embedding task first")
		q.CompleteJob(j.ID)
		return nil
	}

	candidates, grouped, anchors := 0, 0, 0
	for _, it := range items {
		switch {
		case it.joinOnly:
			anchors++
		case it.group != 0:
			grouped++
			if it.anchor {
				anchors++
			}
		default:
			candidates++
		}
	}
	q.PushJobStdout(j.ID, fmt.Sprintf("Vectors loaded: %d — %d already in a group (%d group anchor(s)), %d to place",
		len(items), grouped, anchors, candidates))
	if candidates == 0 {
		q.PushJobStdout(j.ID, "Every item in scope is already in a group; nothing new to find")
		q.CompleteJob(j.ID)
		return nil
	}

	bits, tables, recall := lshParams(len(items), minScore)
	q.PushJobStdout(j.ID, fmt.Sprintf("Hashing: %d table(s) × %d bit(s); expected recall at exactly the threshold ≈ %.1f%% (higher above it; re-run to pick up misses)",
		tables, bits, recall*100))

	stats, err := scanDuplicates(ctx, q, j, items, model, float64(pct)/100, minScore, bits, tables, candidates)
	if err != nil {
		if err == jobqueue.ErrPaused {
			return err
		}
		if ctx.Err() != nil {
			broadcastDuplicatesUpdated(stats.Groups, stats.Grouped)
			q.PushJobStdout(j.ID, fmt.Sprintf("Canceled — %d group(s) found so far are saved; run again to continue", stats.Groups))
			_ = q.CancelJob(j.ID)
			return err
		}
		q.PushJobStdout(j.ID, fmt.Sprintf("Error: %v", err))
		q.ErrorJob(j.ID)
		return err
	}
	broadcastDuplicatesUpdated(stats.Groups, stats.Grouped)
	q.PushJobStdout(j.ID, fmt.Sprintf(
		"Done: %d new group(s), %d item(s) joined existing groups, %d item(s) placed in total. Review them in the Duplicates panel",
		stats.Groups, stats.Joined, stats.Grouped))
	q.CompleteJob(j.ID)
	return nil
}

// loadDuplicateItems loads the vectors to compare — every stored vector of
// the model when scope is nil, else the scope's — normalized, sorted by path
// for deterministic grouping, and annotated with existing group membership.
// Anchors of existing groups that fall outside the scope are loaded as
// join-only targets so new items can still land in the right group.
func loadDuplicateItems(ctx context.Context, db *sql.DB, model string, scope []string) ([]dupItem, error) {
	byPath, anchors, err := media.DuplicateAssignments(db)
	if err != nil {
		return nil, err
	}
	anchorPaths := make(map[string]bool, len(anchors))
	for _, p := range anchors {
		anchorPaths[p] = true
	}

	var items []dupItem
	add := func(path string, vec []float32, joinOnly bool) {
		if len(vec) == 0 {
			return
		}
		it := dupItem{path: path, vec: embedvec.Normalize(vec), joinOnly: joinOnly}
		if gid, ok := byPath[path]; ok {
			it.group = gid
			it.anchor = anchorPaths[path]
		}
		if joinOnly && !it.anchor {
			return // the anchor was re-pointed since we listed it
		}
		items = append(items, it)
	}

	if scope == nil {
		all, err := media.LoadAllEmbeddings(db, model)
		if err != nil {
			return nil, err
		}
		for _, e := range all {
			if ctx.Err() != nil {
				return nil, ctx.Err()
			}
			add(e.Path, e.Vec, false)
		}
	} else {
		vecs, err := media.GetEmbeddingsForPaths(db, model, scope)
		if err != nil {
			return nil, err
		}
		inScope := make(map[string]bool, len(scope))
		for _, p := range scope {
			inScope[p] = true
			add(p, vecs[p], false)
		}
		var extra []string
		for _, p := range anchors {
			if !inScope[p] {
				extra = append(extra, p)
			}
		}
		if len(extra) > 0 {
			avecs, err := media.GetEmbeddingsForPaths(db, model, extra)
			if err != nil {
				return nil, err
			}
			for _, p := range extra {
				add(p, avecs[p], true)
			}
		}
	}
	sort.Slice(items, func(a, b int) bool { return items[a].path < items[b].path })
	return items, nil
}

// lshTable is one random-hyperplane hash family over mean-centred vectors.
type lshTable struct {
	planes  [][]float32 // bits × dim
	offsets []float32   // mu · plane, so centring costs one subtraction
}

func newLSHTable(rng *rand.Rand, bits, dim int, mu []float32) lshTable {
	t := lshTable{planes: make([][]float32, bits), offsets: make([]float32, bits)}
	for k := range t.planes {
		h := make([]float32, dim)
		for d := range h {
			h[d] = float32(rng.NormFloat64())
		}
		t.planes[k] = h
		t.offsets[k] = dotf(mu, h)
	}
	return t
}

func (t lshTable) hash(v []float32) uint32 {
	var h uint32
	for k, plane := range t.planes {
		if dotf(v, plane)-t.offsets[k] > 0 {
			h |= 1 << uint(k)
		}
	}
	return h
}

// parallelRange runs fn over [0,n) split into contiguous per-CPU chunks.
func parallelRange(n int, fn func(lo, hi int)) {
	workers := runtime.NumCPU()
	if workers > n {
		workers = n
	}
	if workers <= 1 {
		fn(0, n)
		return
	}
	chunk := (n + workers - 1) / workers
	var wg sync.WaitGroup
	for lo := 0; lo < n; lo += chunk {
		hi := min(lo+chunk, n)
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			fn(lo, hi)
		}(lo, hi)
	}
	wg.Wait()
}

// scanDuplicates hashes every item into `tables` LSH tables and walks each
// bucket: unassigned items that match an existing group's anchor join it;
// the rest are star-clustered (first unassigned item anchors, every other
// unassigned item at or above minScore joins, the group is written at once).
// Groups are persisted and broadcast as they form.
func scanDuplicates(ctx context.Context, q *jobqueue.Queue, j *jobqueue.Job, items []dupItem,
	model string, threshold float64, minScore float32, bits, tables, candidates int) (findDuplicatesStats, error) {

	var stats findDuplicatesStats
	n := len(items)
	dim := len(items[0].vec)

	// Mean of the stored vectors: hashing centred vectors spreads the
	// buckets (raw image embeddings share a strong common direction that
	// would land most items on the same side of most hyperplanes).
	mu := make([]float32, dim)
	for _, it := range items {
		for d, x := range it.vec {
			mu[d] += x
		}
	}
	for d := range mu {
		mu[d] /= float32(n)
	}

	rng := rand.New(rand.NewSource(lshSeed()))
	families := make([]lshTable, tables)
	for t := range families {
		families[t] = newLSHTable(rng, bits, dim, mu)
	}
	hashes := make([][]uint32, tables) // [table][item]
	for t := range hashes {
		hashes[t] = make([]uint32, n)
	}
	parallelRange(n, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			for t := range families {
				hashes[t][i] = families[t].hash(items[i].vec)
			}
		}
	})

	// anchorGroup maps an item index that anchors a group (pre-existing or
	// created during this run) to that group's id.
	anchorGroup := map[int]int64{}
	for i := range items {
		if items[i].anchor {
			anchorGroup[i] = items[i].group
		}
	}

	total := tables * n
	done := 0
	_ = q.SetJobProgress(j.ID, 0, total)
	var lastLog, lastBroadcast time.Time
	var lastBroadcastGrouped int
	progress := func(force bool) {
		now := time.Now()
		if force || now.Sub(lastLog) >= findDuplicatesLogInterval {
			lastLog = now
			q.PushJobStdout(j.ID, fmt.Sprintf("  %d/%d hashed items walked — %d new group(s), %d joined, %d/%d candidates placed",
				done, total, stats.Groups, stats.Joined, stats.Grouped, candidates))
		}
		if stats.Grouped != lastBroadcastGrouped && (force || now.Sub(lastBroadcast) >= findDuplicatesBroadcastInterval) {
			lastBroadcast, lastBroadcastGrouped = now, stats.Grouped
			broadcastDuplicatesUpdated(stats.Groups, stats.Grouped)
		}
	}

	for t := 0; t < tables; t++ {
		buckets := map[uint32][]int{}
		for i := 0; i < n; i++ {
			h := hashes[t][i]
			buckets[h] = append(buckets[h], i)
		}
		keys := make([]uint32, 0, len(buckets))
		for k, b := range buckets {
			if len(b) >= 2 {
				keys = append(keys, k)
			} else {
				done += len(b)
			}
		}
		sort.Slice(keys, func(a, b int) bool { return keys[a] < keys[b] })

		for _, k := range keys {
			bucket := buckets[k]
			select {
			case <-ctx.Done():
				return stats, ctx.Err()
			default:
			}
			if q.PauseRequested(j.ID) {
				progress(true)
				q.PushJobStdout(j.ID, "Paused — every group found so far is saved; resume to continue from here")
				return stats, jobqueue.ErrPaused
			}

			// 1. Join existing groups through their anchors.
			for _, a := range bucket {
				gid, isAnchor := anchorGroup[a]
				if !isAnchor {
					continue
				}
				var joins []media.DuplicateMember
				var joinIdx []int
				for _, u := range bucket {
					if items[u].group != 0 || items[u].joinOnly {
						continue
					}
					if s := dotf(items[a].vec, items[u].vec); s >= minScore {
						joins = append(joins, media.DuplicateMember{Path: items[u].path, Score: float64(s)})
						joinIdx = append(joinIdx, u)
					}
				}
				if len(joins) == 0 {
					continue
				}
				added, err := media.AddDuplicateMembers(q.Db, gid, joins)
				if err != nil {
					return stats, fmt.Errorf("join group %d: %w", gid, err)
				}
				for _, u := range joinIdx {
					items[u].group = gid
				}
				stats.Joined += added
				stats.Grouped += added
				if stats.Groups+stats.Joined <= findDuplicatesLogGroups {
					q.PushJobStdout(j.ID, fmt.Sprintf("  +%d item(s) joined group #%d (%s)", added, gid, filepath.Base(items[a].path)))
				}
			}

			// 2. Star-cluster what is still unassigned.
			for ai, a := range bucket {
				if items[a].group != 0 || items[a].joinOnly {
					continue
				}
				members := []media.DuplicateMember{{Path: items[a].path, Score: 1}}
				var memberIdx []int
				lowest := float32(1)
				for _, u := range bucket[ai+1:] {
					if items[u].group != 0 || items[u].joinOnly {
						continue
					}
					if s := dotf(items[a].vec, items[u].vec); s >= minScore {
						members = append(members, media.DuplicateMember{Path: items[u].path, Score: float64(s)})
						memberIdx = append(memberIdx, u)
						if s < lowest {
							lowest = s
						}
					}
				}
				if len(members) < 2 {
					continue
				}
				gid, err := media.CreateDuplicateGroup(q.Db, model, threshold, items[a].path, members)
				if err != nil {
					return stats, fmt.Errorf("create group: %w", err)
				}
				if gid == 0 {
					// Raced with a concurrent grouping of the same paths (another
					// run); treat those items as taken for the rest of this pass.
					items[a].group = -1
					for _, u := range memberIdx {
						items[u].group = -1
					}
					continue
				}
				items[a].group, items[a].anchor = gid, true
				anchorGroup[a] = gid
				for _, u := range memberIdx {
					items[u].group = gid
				}
				stats.Groups++
				stats.Grouped += len(members)
				if stats.Groups <= findDuplicatesLogGroups {
					q.PushJobStdout(j.ID, fmt.Sprintf("  Group #%d: %d items at ≥ %.1f%% — %s",
						gid, len(members), float64(lowest)*100, filepath.Base(items[a].path)))
				} else if stats.Groups == findDuplicatesLogGroups+1 {
					q.PushJobStdout(j.ID, "  (further groups are counted in the periodic totals below)")
				}
			}

			done += len(bucket)
			_ = q.SetJobProgress(j.ID, min(done, total), total)
			progress(false)
		}
		_ = q.SetJobProgress(j.ID, min((t+1)*n, total), total)
	}
	done = total
	_ = q.SetJobProgress(j.ID, total, total)
	progress(true)
	return stats, nil
}

// broadcastDuplicatesUpdated tells open Duplicates panels that groups
// changed. Counts are informational (the panel refetches).
func broadcastDuplicatesUpdated(groups, items int) {
	payload, err := json.Marshal(map[string]any{"groups": groups, "items": items})
	if err != nil {
		return
	}
	stream.Broadcast(stream.Message{Type: "duplicates-updated", Msg: string(payload)})
}
