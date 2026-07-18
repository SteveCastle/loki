package tasks

// itemops.go — the unified per-item task engine.
//
// An ItemOp is one unit of per-file work (generate a description, compute a
// hash, store an embedding, ...). Every op runs through the same runner
// (runItemOps), which gives all of them one consistent contract:
//
//   - Input resolution: a job can carry a search query (--query/--query64) or
//     an explicit path list; both resolve through resolveJobItems.
//   - Progress: the runner reports done/total via Queue.SetJobProgress as soon
//     as the input resolves and after every item.
//   - Overwrite: the shared --overwrite flag means "reprocess items that
//     already have this op's output". Without it, existing output is skipped.
//   - Durability: each item's DB writes are committed as that item finishes,
//     so cancelling or pausing a job never loses completed work.
//   - Composition: multiple ops can run as one job (the "process" task or the
//     legacy "metadata" alias), applied together to each file — one pass over
//     the items instead of one job per op.

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/stevecastle/shrike/jobqueue"
)

// localizeItem returns a readable local file for a media path. Local paths
// pass through untouched; s3:// items are downloaded to a temp file (with
// the original extension — workers sniff type by extension) that cleanup
// removes. The library path stays the DB key throughout; only reads use
// the returned path.
func localizeItem(ctx context.Context, path string) (localPath string, cleanup func(), err error) {
	noop := func() {}
	if !strings.HasPrefix(path, "s3://") {
		return path, noop, nil
	}
	if storageReg == nil {
		return "", noop, fmt.Errorf("no storage registry configured")
	}
	backend := storageReg.BackendFor(path)
	if backend == nil {
		return "", noop, fmt.Errorf("no storage backend for %s", path)
	}
	r, err := backend.Download(ctx, path)
	if err != nil {
		return "", noop, fmt.Errorf("download: %w", err)
	}
	defer r.Close()
	tmp, err := os.CreateTemp("", "loki-op-src-*"+filepath.Ext(path))
	if err != nil {
		return "", noop, fmt.Errorf("temp file: %w", err)
	}
	if _, err := io.Copy(tmp, r); err != nil {
		tmp.Close()
		os.Remove(tmp.Name())
		return "", noop, fmt.Errorf("download copy: %w", err)
	}
	tmp.Close()
	name := tmp.Name()
	return name, func() { os.Remove(name) }, nil
}

// ItemRun is the per-job context handed to each op's Prepare.
type ItemRun struct {
	Job       *jobqueue.Job
	Queue     *jobqueue.Queue
	Opts      map[string]any // this op's options, unprefixed keys
	Overwrite bool
	FromQuery bool
	Items     []string // all resolved items (for prepare-time prefetch)
	Workers   int      // runner-level parallelism this job will use
	// Background marks scheduler-initiated jobs (--scheduled): their worker
	// subprocesses run at below-normal OS priority so foreground work always
	// wins the machine.
	Background bool
}

// ItemCommit is the durable half of one op applied to one item: Commit is
// executed by the runner's single writer goroutine (serialized DB access),
// Detail is an optional human line for the job log.
type ItemCommit struct {
	Commit func() error
	Detail string
}

// ItemProcessor is a prepared op, ready to process items. Process may be
// called from multiple goroutines concurrently (up to ItemRun.Workers).
type ItemProcessor struct {
	// SkipExisting reports whether the item already has this op's output.
	// Consulted only when --overwrite is off. nil = never skip.
	SkipExisting func(path string) (bool, error)
	// Process computes the op for one item and returns the commit to apply.
	// A nil ItemCommit with nil error means "nothing to do" (counted as a skip).
	// path is the item's library identity (DB key — may be an s3:// path);
	// localPath is a readable file on local disk holding the item's bytes
	// (identical to path for local media, a temp download for s3://).
	Process func(ctx context.Context, path, localPath string) (*ItemCommit, error)
	// Close releases prepare-time resources (worker pools). May be nil.
	Close func()
	// Finalize runs once after the whole run completes SUCCESSFULLY (all items
	// committed — not on pause, cancel, or error), before Close. It lets an op
	// queue follow-up work that depends on the full run, e.g. faces queuing a
	// clustering pass once new faces are stored. May be nil.
	Finalize func() error
}

// ItemOp describes one registered per-item operation.
type ItemOp struct {
	ID      string
	Name    string
	Options []TaskOption
	// Concurrency is the op's preferred max parallel Process calls
	// (nil or <1 = serial). A combined job runs at the minimum across its ops.
	Concurrency func() int
	// Applies filters by file kind (extension). nil = any media file.
	Applies func(path string) bool
	Prepare func(run *ItemRun) (*ItemProcessor, error)
}

var (
	itemOps      = make(map[string]ItemOp)
	itemOpsOrder []string
)

// RegisterItemOp adds an op to the registry (registration order is preserved
// for option listings and the combined task's op choices).
func RegisterItemOp(op ItemOp) {
	if _, exists := itemOps[op.ID]; !exists {
		itemOpsOrder = append(itemOpsOrder, op.ID)
	}
	itemOps[op.ID] = op
}

// ItemOpIDs returns all registered op IDs in registration order.
func ItemOpIDs() []string {
	out := make([]string, len(itemOpsOrder))
	copy(out, itemOpsOrder)
	return out
}

// sharedItemOptions are appended to every item-op task's option list so the
// flags behave identically across all of them.
var sharedItemOptions = []TaskOption{
	{Name: "overwrite", Label: "Overwrite Existing", Type: "bool", Description: "Reprocess items that already have this output (default: skip them)"},
	{Name: "workers", Label: "Worker Override", Type: "number", Description: "Override the number of parallel workers (0 = automatic)"},
}

// itemOpTaskOptions builds the option list for a standalone single-op task:
// the op's own options followed by the shared overwrite/workers flags.
func itemOpTaskOptions(opID string) []TaskOption {
	op := itemOps[opID]
	out := append([]TaskOption{}, op.Options...)
	return append(out, sharedItemOptions...)
}

// prefixedOpOptions returns an op's options renamed to "<opID>-<name>" for
// use in the combined "process" task, where different ops may share option
// names (e.g. describe --model vs embed --model).
func prefixedOpOptions(op ItemOp) []TaskOption {
	out := make([]TaskOption, len(op.Options))
	for i, o := range op.Options {
		o.Name = op.ID + "-" + o.Name
		o.Label = op.Name + ": " + o.Label
		out[i] = o
	}
	return out
}

// opOptionValues parses one op's option values from the job. When prefixed,
// flags are read as --<opID>-<name> and returned under their unprefixed keys.
func opOptionValues(j *jobqueue.Job, op ItemOp, prefixed bool) map[string]any {
	if !prefixed {
		return ParseOptions(j, op.Options)
	}
	vals := ParseOptions(j, prefixedOpOptions(op))
	out := make(map[string]any, len(vals))
	for k, v := range vals {
		out[strings.TrimPrefix(k, op.ID+"-")] = v
	}
	return out
}

// makeItemOpTaskFn returns a TaskFn that runs the given ops through the
// unified runner with unprefixed options (standalone task form).
func makeItemOpTaskFn(opIDs ...string) TaskFn {
	return func(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
		return runItemOps(j, q, opIDs, false)
	}
}

// itemEnvelope carries one finished item from the compute workers to the
// single committer goroutine.
type itemEnvelope struct {
	path      string
	commits   []ItemCommit
	errs      []string
	attempted bool // at least one op computed something (vs all skipped)
}

// runItemOps is the unified runner: resolve items, prepare ops, fan out
// compute across workers, serialize commits, and report progress.
func runItemOps(j *jobqueue.Job, q *jobqueue.Queue, opIDs []string, prefixed bool) error {
	ctx := j.Ctx

	// Resolve the requested ops.
	var ops []ItemOp
	for _, id := range opIDs {
		op, ok := itemOps[id]
		if !ok {
			q.PushJobStdout(j.ID, fmt.Sprintf("Unknown operation %q - valid operations: %s", id, strings.Join(ItemOpIDs(), ", ")))
			continue
		}
		ops = append(ops, op)
	}
	if len(ops) == 0 {
		q.PushJobStdout(j.ID, "No valid operations selected")
		q.ErrorJob(j.ID)
		return fmt.Errorf("no valid operations in %v", opIDs)
	}

	shared := ParseOptions(j, sharedItemOptions)
	overwrite, _ := shared["overwrite"].(bool)

	// Resolve the input to a concrete item list (query or path list).
	res, err := resolveJobItems(j, q)
	if err != nil {
		q.PushJobStdout(j.ID, "Failed to resolve input: "+err.Error())
		q.ErrorJob(j.ID)
		return err
	}
	if res.FromQuery {
		q.PushJobStdout(j.ID, fmt.Sprintf("Query: %s", res.Query))
	}
	items := res.Paths
	// Publish the resolved item list to the path→job index — this is the
	// moment query jobs (and dependency-fed inputs) become path-queryable.
	q.SetJobItems(j.ID, items)
	total := len(items)
	opNames := make([]string, len(ops))
	for i, op := range ops {
		opNames[i] = op.ID
	}
	q.PushJobStdout(j.ID, fmt.Sprintf("Operations: %s | overwrite=%t | %d item(s) to process", strings.Join(opNames, ", "), overwrite, total))
	if total == 0 {
		q.PushJobStdout(j.ID, "No files to process")
		q.CompleteJob(j.ID)
		return nil
	}

	// The total is known the moment the input resolves; report it immediately.
	// A resumed overwrite run continues from its persisted progress offset:
	// skip-existing can't identify finished items when overwriting, so the
	// runner relies on deterministic item order (same query, same list) and
	// skips the first ProgressDone items. Non-overwrite runs resume naturally
	// by skipping items that already have output.
	startAt := 0
	if overwrite && j.ProgressDone > 0 && j.ProgressDone < total {
		startAt = j.ProgressDone
		q.PushJobStdout(j.ID, fmt.Sprintf("Resuming overwrite run at item %d/%d", startAt+1, total))
	}
	_ = q.SetJobProgress(j.ID, startAt, total)

	// Runner parallelism: the minimum across the selected ops' preferences,
	// so an op that must run serially (LLM calls) is never hammered by a
	// faster co-op. --workers overrides.
	workers := 0
	for _, op := range ops {
		c := 1
		if op.Concurrency != nil {
			if v := op.Concurrency(); v > 0 {
				c = v
			}
		}
		if workers == 0 || c < workers {
			workers = c
		}
	}
	if w, ok := shared["workers"].(float64); ok && int(w) > 0 {
		workers = int(w)
	}
	if workers > total-startAt {
		workers = total - startAt
	}
	if workers < 1 {
		workers = 1
	}

	// Prepare every op up front (load models, start worker pools, prefetch).
	// A prepare failure fails the whole job: silently dropping one of the
	// requested ops would break the "these ops ran on these items" contract.
	background := false
	for _, a := range j.Arguments {
		if a == "--scheduled" {
			background = true
			break
		}
	}
	if background {
		q.PushJobStdout(j.ID, "Scheduled run: workers execute at background OS priority")
	}

	run := &ItemRun{
		Job:        j,
		Queue:      q,
		Overwrite:  overwrite,
		FromQuery:  res.FromQuery,
		Items:      items,
		Workers:    workers,
		Background: background,
	}
	procs := make([]*ItemProcessor, len(ops))
	closeProcs := func() {
		for _, p := range procs {
			if p != nil && p.Close != nil {
				p.Close()
			}
		}
	}
	for i, op := range ops {
		opRun := *run
		opRun.Opts = opOptionValues(j, op, prefixed)
		p, perr := op.Prepare(&opRun)
		if perr != nil {
			closeProcs()
			q.PushJobStdout(j.ID, fmt.Sprintf("%s: %v", op.ID, perr))
			q.ErrorJob(j.ID)
			return fmt.Errorf("prepare %s: %w", op.ID, perr)
		}
		procs[i] = p
	}
	defer closeProcs()

	if workers > 1 {
		q.PushJobStdout(j.ID, fmt.Sprintf("Processing with %d parallel worker(s)", workers))
	}

	// Per-item stdout detail is only useful for small jobs; for large ones the
	// progress events carry the signal and per-line stdout persistence is
	// too expensive.
	verbose := total <= 50

	feed := make(chan string)
	commitCh := make(chan itemEnvelope, workers*2)
	pausedRequested := false

	// Feeder: stops at cancel or pause request. Pausing between items is what
	// makes pause lossless — in-flight items still drain through the committer.
	go func() {
		defer close(feed)
		for _, path := range items[startAt:] {
			if ctx.Err() != nil {
				return
			}
			if q.PauseRequested(j.ID) {
				pausedRequested = true
				return
			}
			select {
			case feed <- path:
			case <-ctx.Done():
				return
			}
		}
	}()

	// Compute workers: apply every op to each item, gather commits.
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for path := range feed {
				if ctx.Err() != nil {
					return
				}
				env := itemEnvelope{path: path}
				// Input-list items may be arbitrary paths: require existence
				// (disk stat for local, backend HEAD for s3://) and library
				// membership. Query items came from the DB and are trusted
				// (stat-ing millions of rows on a network drive stalls).
				if !res.FromQuery {
					if strings.HasPrefix(path, "s3://") {
						missing := storageReg == nil
						if !missing {
							b := storageReg.BackendFor(path)
							if b == nil {
								missing = true
							} else if ok, herr := b.Exists(ctx, path); herr == nil && !ok {
								missing = true
							}
						}
						if missing {
							env.errs = append(env.errs, "not found in storage")
							commitCh <- env
							continue
						}
					} else if _, statErr := os.Stat(path); os.IsNotExist(statErr) {
						env.errs = append(env.errs, "not found on disk")
						commitCh <- env
						continue
					}
					if exists, dbErr := fileExistsInDatabase(q.Db, path); dbErr == nil && !exists {
						env.errs = append(env.errs, "not in library")
						commitCh <- env
						continue
					}
				}
				// s3:// items are downloaded to a temp file lazily — only when
				// an op actually runs (all-skipped items cost no bandwidth).
				localPath, localCleanup, localized := path, func() {}, !strings.HasPrefix(path, "s3://")
				ensureLocal := func() error {
					if localized {
						return nil
					}
					lp, cl, lerr := localizeItem(ctx, path)
					if lerr != nil {
						return lerr
					}
					localPath, localCleanup, localized = lp, cl, true
					return nil
				}
				for i, op := range ops {
					if ctx.Err() != nil {
						break
					}
					if op.Applies != nil && !op.Applies(path) {
						continue
					}
					if !overwrite && procs[i].SkipExisting != nil {
						if has, herr := procs[i].SkipExisting(path); herr == nil && has {
							continue
						}
					}
					if lerr := ensureLocal(); lerr != nil {
						env.errs = append(env.errs, fmt.Sprintf("%s: fetch: %v", op.ID, lerr))
						continue
					}
					result, perr := procs[i].Process(ctx, path, localPath)
					if perr != nil {
						if ctx.Err() != nil {
							break
						}
						env.errs = append(env.errs, fmt.Sprintf("%s: %v", op.ID, perr))
						continue
					}
					env.attempted = true
					if result != nil {
						env.commits = append(env.commits, *result)
					}
				}
				localCleanup()
				commitCh <- env
			}
		}()
	}

	// Committer: the single writer. All DB mutations flow through here, one
	// item at a time, so every finished item is durable before the next
	// write starts and progress advances exactly once per item.
	var processed, skipped, failed int
	done := startAt
	committerDone := make(chan struct{})
	go func() {
		defer close(committerDone)
		for env := range commitCh {
			wrote := false
			for _, c := range env.commits {
				if c.Commit != nil {
					if cerr := c.Commit(); cerr != nil {
						env.errs = append(env.errs, "write failed: "+cerr.Error())
						continue
					}
				}
				wrote = true
				if verbose && c.Detail != "" {
					q.PushJobStdout(j.ID, fmt.Sprintf("  %s: %s", filepath.Base(env.path), c.Detail))
				}
			}
			for _, e := range env.errs {
				q.PushJobStdout(j.ID, fmt.Sprintf("  %s: %s", filepath.Base(env.path), e))
			}
			switch {
			case wrote:
				processed++
				q.RegisterOutputFile(j.ID, env.path)
			case len(env.errs) > 0:
				failed++
			default:
				skipped++
			}
			done++
			_ = q.SetJobProgress(j.ID, done, total)
			if !verbose && (processed+skipped+failed)%250 == 0 {
				q.PushJobStdout(j.ID, fmt.Sprintf("Progress: %d processed, %d skipped, %d failed (of %d)", processed, skipped, failed, total))
			}
		}
	}()

	wg.Wait()
	close(commitCh)
	<-committerDone

	if ctx.Err() != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("Canceled at %d/%d - completed work is saved", done, total))
		_ = q.CancelJob(j.ID)
		return ctx.Err()
	}
	if pausedRequested {
		q.PushJobStdout(j.ID, fmt.Sprintf("Paused at %d/%d - resume to continue", done, total))
		return jobqueue.ErrPaused
	}

	q.PushJobStdout(j.ID, fmt.Sprintf("Completed: %d processed, %d skipped, %d failed", processed, skipped, failed))
	q.CompleteJob(j.ID)

	// Success-path only: give each op a chance to queue follow-up work now that
	// the whole run committed. Reached solely here — the pause/cancel/error
	// returns above skip it, so a resumed run fires Finalize when it truly
	// completes. A Finalize error is logged, never fatal (the run succeeded).
	for _, p := range procs {
		if p != nil && p.Finalize != nil {
			if ferr := p.Finalize(); ferr != nil {
				q.PushJobStdout(j.ID, "finalize: "+ferr.Error())
			}
		}
	}
	return nil
}

// extAppliesFn returns an Applies filter matching a set of lowercase
// extensions (with leading dot).
func extAppliesFn(exts ...string) func(string) bool {
	set := make(map[string]struct{}, len(exts))
	for _, e := range exts {
		set[e] = struct{}{}
	}
	return func(path string) bool {
		_, ok := set[strings.ToLower(filepath.Ext(path))]
		return ok
	}
}

// -----------------------------------------------------------------------------
// The combined "process" task: any set of ops as one atomic per-file pass.
// -----------------------------------------------------------------------------

// processTaskOptions builds the option list for the combined task: the op
// selector, the shared flags, then every op's own options under prefixed
// names (--describe-model, --embed-model, ...) so same-named options from
// different ops don't collide. Must be called after registerBuiltinItemOps.
func processTaskOptions() []TaskOption {
	ids := ItemOpIDs()
	opts := []TaskOption{
		{Name: "ops", Label: "Operations", Type: "multi-enum", Choices: ids, Default: "hash,dimensions", Required: true,
			Description: "Comma-separated operations to apply to each file, together, in one pass"},
	}
	opts = append(opts, sharedItemOptions...)
	for _, id := range ids {
		opts = append(opts, prefixedOpOptions(itemOps[id])...)
	}
	return opts
}

// processTask runs the selected ops as one combined pass: each file is
// visited once and every op is applied to it before moving on, with the
// item's writes committed together.
func processTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	vals := ParseOptions(j, []TaskOption{processTaskOptions()[0]})
	raw, _ := vals["ops"].(string)
	var opIDs []string
	for _, id := range strings.Split(raw, ",") {
		if id = strings.ToLower(strings.TrimSpace(id)); id != "" {
			opIDs = append(opIDs, id)
		}
	}
	return runItemOps(j, q, opIDs, true)
}
