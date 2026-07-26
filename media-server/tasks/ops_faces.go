package tasks

// ops_faces.go — face detection + identity embeddings as an ItemOp, so face
// scanning composes with every other operation in a single per-file pass
// (the "process" task) under the unified query/overwrite/progress/pause
// contract.
//
// The photo/anime domain routing that used to make faces a standalone
// pipeline is folded into Process: each item's whole-image SigLIP embedding
// (stored, or embedded on the fly) picks the recognizer, and a lazily-started
// per-recognizer worker pool does the scan. Routing anchors and pools are
// resolved once in Prepare.

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/stevecastle/shrike/deps"
	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/media"
	"github.com/stevecastle/shrike/stream"
)

// defaultClusterEvery is how many newly stored faces trigger an incremental
// in-scan clustering pass, so People form and grow WHILE a scan runs and the
// UI updates continually instead of only after a separate faces-cluster job.
// Each pass stalls the committer for its duration (loading every assigned
// seed and scoring the batch against them), so the cadence trades UI
// liveness against scan throughput — 500 keeps the pass cost a small
// fraction of the scan work.
const defaultClusterEvery = 500

func registerFacesItemOp() {
	RegisterItemOp(ItemOp{
		ID:   "faces",
		Name: "Faces (ONNX)",
		Options: []TaskOption{
			{Name: "model", Label: "Face Model", Type: "string", Description: "Pin scanning to one face model ID (default: automatic photo/anime routing)"},
			{Name: "cluster-every", Label: "Cluster Every N Faces", Type: "number", Default: float64(defaultClusterEvery), Description: "Run an incremental people-clustering pass after this many newly scanned faces so People appear while the scan runs (0 = only queue one clustering job at the end)"},
		},
		Concurrency: func() int {
			workers, _ := ResolveFaceResources()
			return workers
		},
		Prepare: prepareFacesOp,
	})
}

// facesOpState holds the per-job routing + pool machinery shared by
// concurrent Process calls.
type facesOpState struct {
	q        *jobQueueRef
	embedBin string
	timeout  time.Duration

	// Routing. anchors == nil means routing is off (pinned or unavailable)
	// and pinnedModel is used for everything.
	anchors     *anchorCache
	pinnedModel FaceModel
	embedModel  EmbedModel // SigLIP model whose vectors drive routing

	// candidateIDs are the models whose face_scan markers count as "already
	// scanned" (both routing targets, or just the pinned model).
	candidateIDs []string

	// Lazily-started per-recognizer pools.
	mu    sync.Mutex
	pools map[string]*servePool
	// Lazily-started SigLIP pool for routing embeds (items with no stored
	// whole-image vector), plus its sticky start failure so a broken embed
	// setup degrades to the active model once instead of retrying per item.
	routePool    *servePool
	routePoolErr error

	// newFaces counts face rows stored this run (across all items/workers). A
	// positive count at Finalize queues a clustering pass so the fresh faces
	// form/join people without a manual Cluster press.
	newFaces atomic.Int64

	// Incremental clustering bookkeeping. Touched ONLY from the runner's
	// single committer goroutine (inside Commit closures) and from Finalize
	// (which runs after the committer drains), so no locking is needed.
	clusterEvery int                // faces per in-scan pass; 0 = end-of-scan job only
	sinceCluster int                // faces stored since the last in-scan pass
	dirtyBatches map[string][]int64 // model ID -> face ids stored since its last pass
	runBatches   map[string][]int64 // model ID -> EVERY face id stored this run (final pass)
}

// noteFacesScanned records freshly stored face ids for a model and reports
// whether an in-scan clustering pass is due. When due it returns each dirty
// model's accumulated batch and resets the counters (the caller runs the
// pass). Passing the ids — not just a count — is what lets the incremental
// pass restrict itself to the new faces instead of re-scoring the whole
// unassigned backlog.
func (st *facesOpState) noteFacesScanned(modelID string, ids []int64) (due bool, batches map[string][]int64) {
	if len(ids) == 0 {
		return false, nil
	}
	if st.dirtyBatches == nil {
		st.dirtyBatches = map[string][]int64{}
	}
	if st.runBatches == nil {
		st.runBatches = map[string][]int64{}
	}
	st.dirtyBatches[modelID] = append(st.dirtyBatches[modelID], ids...)
	st.runBatches[modelID] = append(st.runBatches[modelID], ids...)
	st.sinceCluster += len(ids)
	if st.clusterEvery <= 0 || st.sinceCluster < st.clusterEvery {
		return false, nil
	}
	return true, st.takeDirtyBatches()
}

// takeDirtyBatches returns the per-model batches of faces stored since each
// model's last pass and resets the counters.
func (st *facesOpState) takeDirtyBatches() map[string][]int64 {
	batches := st.dirtyBatches
	st.dirtyBatches = map[string][]int64{}
	st.sinceCluster = 0
	return batches
}

// runClusterPass runs one clustering pass per batched model, inline in the
// scan job (a separate faces-cluster job could not run anyway: it shares the
// faces + local-compute buckets with the scan). EVERY pass is RESTRICTED to
// its batch of newly stored faces — the pass cost tracks the batch, not the
// library, so scans don't slow down as the unassigned backlog grows. Mid-scan
// passes use the strict incremental params (high-precision preview) over the
// faces since the last pass; the final pass at scan end uses the normal
// defaults over ALL faces stored this run, settling this run's borderline
// cases exactly once. It used to be unrestricted, "exactly like the old
// scan-then-cluster pipeline" — which meant every scan, however small,
// re-scored the whole ungrouped backlog (tens of thousands of junk faces that
// never join anything) against every seed: minutes of every-core compute to
// add 600 items. The backlog already had its full-data chance in its own
// runs' final passes; re-litigating it is the explicit faces-cluster job
// ("Group new faces" in People), not a per-scan tax. Neither pass reprocesses
// already-assigned faces — in-scan rebuilds (reset + full regroup) were tried
// and dropped: re-litigating every unnamed group on a cadence stalled long
// scans far too much. Logs the outcome and broadcasts "people-updated" so
// open People views refresh.
func (st *facesOpState) runClusterPass(batches map[string][]int64, final bool) {
	db := st.q.run.Queue.Db
	ctx := st.q.run.Job.Ctx
	label := "incremental (strict)"
	if final {
		label = "final"
	}
	models := make([]string, 0, len(batches))
	for id, ids := range batches {
		if ctx.Err() != nil {
			st.q.log(fmt.Sprintf("  clustering %s canceled", label))
			break
		}
		m, known := FaceModelByID(id)
		if !known {
			continue
		}
		models = append(models, id)
		params := incrementalClusterParams(m)
		if final {
			params = defaultClusterParams(m)
		}
		only := make(map[int64]bool, len(ids))
		for _, fid := range ids {
			only[fid] = true
		}
		params.onlyFaceIDs = only
		stats, err := clusterFaces(ctx, db, m, params)
		if err != nil {
			if ctx.Err() != nil {
				st.q.log(fmt.Sprintf("  clustering %s (%s) canceled", label, id))
				break
			}
			st.q.log(fmt.Sprintf("  clustering %s (%s) failed: %v", label, id, err))
			continue
		}
		st.q.log(fmt.Sprintf("  clustering %s (%s): +%d people, %d faces joined existing, %d newly grouped",
			label, id, stats.NewPeople, stats.JoinedExisting, stats.NewlyClustered))
	}
	broadcastPeopleUpdated(models)
}

// broadcastPeopleUpdated tells live UIs (People grid) that person groupings
// changed and they should refetch.
func broadcastPeopleUpdated(modelIDs []string) {
	payload, err := json.Marshal(map[string]any{"models": modelIDs})
	if err != nil {
		return
	}
	stream.Broadcast(stream.Message{Type: "people-updated", Msg: string(payload)})
}

// jobQueueRef bundles the queue+job identifiers the op needs for logging.
type jobQueueRef struct {
	run *ItemRun
}

func (r *jobQueueRef) log(line string) {
	r.run.Queue.PushJobStdout(r.run.Job.ID, line)
}

func prepareFacesOp(run *ItemRun) (*ItemProcessor, error) {
	q, j := run.Queue, run.Job
	db := q.Db

	embedBin := deps.BundledOrEmpty("embed")
	if embedBin == "" {
		return nil, fmt.Errorf("embed binary not installed; install it from Dependencies")
	}

	st := &facesOpState{
		q:            &jobQueueRef{run: run},
		embedBin:     embedBin,
		timeout:      OnnxFileTimeout(),
		pools:        map[string]*servePool{},
		clusterEvery: defaultClusterEvery,
	}
	if v, ok := run.Opts["cluster-every"].(float64); ok {
		st.clusterEvery = int(v)
	}
	if st.clusterEvery > 0 {
		q.PushJobStdout(j.ID, fmt.Sprintf("Incremental clustering: people update every %d new faces", st.clusterEvery))
	}

	// Resolve routing vs a pinned model. An explicit model option pins
	// everything (background migration, same contract as embed); otherwise
	// routing classifies each item photo-vs-anime, degrading to the active
	// model when the router is unavailable.
	if id, _ := run.Opts["model"].(string); id != "" {
		m, known := FaceModelByID(id)
		if !known {
			m = ActiveFaceModel()
			q.PushJobStdout(j.ID, fmt.Sprintf("Unknown face model %q; using active model %q", id, m.ID))
		}
		st.pinnedModel = m
		st.candidateIDs = []string{m.ID}
	} else if FaceRoutingEnabled() {
		anchors, err := domainAnchors(j.Ctx)
		if err != nil {
			q.PushJobStdout(j.ID, "Domain routing unavailable ("+err.Error()+"); scanning everything with the active model")
			st.pinnedModel = ActiveFaceModel()
			st.candidateIDs = []string{st.pinnedModel.ID}
		} else {
			st.anchors = anchors
			st.embedModel = TextSearchModel()
			photo := faceModelForDomain("photo")
			anime := faceModelForDomain("anime")
			st.candidateIDs = []string{photo.ID}
			if anime.ID != photo.ID {
				st.candidateIDs = append(st.candidateIDs, anime.ID)
			}
			q.PushJobStdout(j.ID, fmt.Sprintf("Face routing: photo=%s anime=%s", photo.ID, anime.ID))
		}
	} else {
		st.pinnedModel = ActiveFaceModel()
		st.candidateIDs = []string{st.pinnedModel.ID}
	}

	return &ItemProcessor{
		SkipExisting: func(path string) (bool, error) {
			// "Already scanned" means a face_scan marker under ANY routing
			// candidate — matching the marker the scan itself would write.
			scanned, err := media.FaceScansForPaths(db, st.candidateIDs, []string{path})
			if err != nil {
				return false, err
			}
			return scanned[path], nil
		},
		Process: func(ctx context.Context, path, localPath string) (*ItemCommit, error) {
			return st.processOne(ctx, run, path, localPath)
		},
		Finalize: func() error {
			if st.clusterEvery > 0 {
				// Incremental mode: the in-scan passes were deliberately
				// strict AND batch-restricted, so finish with one pass at the
				// normal defaults over every face stored THIS RUN — the
				// strict passes' leftovers get their borderline cases settled
				// with complete seed data. Restricted to the run's faces on
				// purpose: the pre-existing ungrouped backlog is not
				// re-litigated per scan (see runClusterPass), so this cost
				// tracks the scan, not the library. No follow-up job needed.
				st.takeDirtyBatches() // counters are superseded by the run-wide pass
				if len(st.runBatches) > 0 {
					newFaces := 0
					for _, ids := range st.runBatches {
						newFaces += len(ids)
					}
					q.PushJobStdout(j.ID, fmt.Sprintf("Final clustering pass over this run's %d new face(s)", newFaces))
					st.runClusterPass(st.runBatches, true)
					if n, err := CountUngroupedFaces(q.Db); err == nil && n > 0 {
						q.PushJobStdout(j.ID, fmt.Sprintf(
							"%d ungrouped face(s) remain (this run's no-joins plus earlier scans); use Group new faces in People to re-litigate them", n))
					}
				}
				return nil
			}
			if id, queued := maybeQueueFaceClustering(q, st.newFaces.Load()); queued {
				q.PushJobStdout(j.ID, fmt.Sprintf("Stored %d new face(s) — queued clustering pass (job %s)", st.newFaces.Load(), id))
			}
			return nil
		},
		Close: func() {
			st.mu.Lock()
			defer st.mu.Unlock()
			for _, p := range st.pools {
				p.close()
			}
			if st.routePool != nil {
				st.routePool.close()
			}
		},
	}, nil
}

// maybeQueueFaceClustering queues a full clustering pass after a face scan that
// stored new faces, so freshly-scanned faces form or join people without the
// user pressing Cluster. Gated on newFaces > 0, and deduped against an
// already-pending faces-cluster job — that pending pass will pick up these
// faces when it runs, so at most one is ever queued at a time. A cluster job
// never scans faces, so it can't retrigger this. Returns the queued job ID and
// whether one was created.
func maybeQueueFaceClustering(q *jobqueue.Queue, newFaces int64) (string, bool) {
	if newFaces <= 0 {
		return "", false
	}
	for _, j := range q.GetJobs() {
		if j.Command == "faces-cluster" && j.State == jobqueue.StatePending {
			return "", false
		}
	}
	id, err := q.AddJob("", "faces-cluster", nil, "", nil)
	if err != nil {
		return "", false
	}
	return id, true
}

// modelFor routes one item to its recognizer: the pinned model when routing
// is off, otherwise the SigLIP-classified domain model (stored vector when
// present, embedded on the fly — which persists the vector — otherwise).
// Classification failure falls back to the active model.
func (st *facesOpState) modelFor(ctx context.Context, run *ItemRun, path, localPath string) FaceModel {
	if st.anchors == nil {
		return st.pinnedModel
	}
	if vec, ok, err := media.GetEmbedding(run.Queue.Db, path, st.embedModel.ID); err == nil && ok {
		return routeVec(vec, st.anchors)
	}
	if fresh, err := st.routeEmbed(ctx, run, path, localPath); err == nil {
		return routeVec(fresh, st.anchors)
	}
	return ActiveFaceModel()
}

// routingPool lazily starts (and caches) the SigLIP serve pool that embeds
// items with no stored whole-image vector at routing time. The old path spawned
// a ONE-SHOT embed subprocess per item, reloading the model every time — on a
// fresh library (where the embed op's writes for the same item commit only
// after faces computes) that was a continuous stream of full model loads
// across every worker for the whole run. A start failure is sticky: routing
// degrades to the active model for the rest of the run instead of paying the
// failed spawn per item.
func (st *facesOpState) routingPool(ctx context.Context, run *ItemRun) (*servePool, error) {
	st.mu.Lock()
	defer st.mu.Unlock()
	if st.routePool != nil || st.routePoolErr != nil {
		return st.routePool, st.routePoolErr
	}
	imageModel, err := deps.ModelPath(st.embedModel.ID, st.embedModel.ImageModelFile)
	if err != nil || imageModel == "" {
		st.routePoolErr = fmt.Errorf("%s image model not installed", st.embedModel.DisplayName)
		return nil, st.routePoolErr
	}
	_, threads := ResolveEmbedResources()
	ortLib, provider := resolveONNXRuntime(EmbedProviderFromConfig())
	st.q.log(fmt.Sprintf("Routing embeds: %s pool, %d worker(s), provider=%s", st.embedModel.ID, run.Workers, provider))
	pool, perr := newServePool(ctx, run.Workers, st.embedBin, buildServeArgs(imageModel, ortLib, st.embedModel, provider, threads), run.Background)
	if perr != nil {
		st.routePoolErr = fmt.Errorf("start routing embed worker: %w", perr)
		return nil, st.routePoolErr
	}
	st.routePool = pool
	return pool, nil
}

// routeEmbed computes the whole-image routing embedding for one item through
// the persistent routing pool and persists it (same contract as the old
// on-the-fly path: the next lookup — search or a later scan — is free).
func (st *facesOpState) routeEmbed(ctx context.Context, run *ItemRun, path, localPath string) ([]float32, error) {
	pool, perr := st.routingPool(ctx, run)
	if perr != nil {
		return nil, perr
	}
	imagePath, tempFrame, ferr := extractFrameForFile(ctx, localPath, st.timeout)
	if ferr != nil {
		return nil, fmt.Errorf("frame extract: %w", ferr)
	}
	defer func() {
		if tempFrame != "" {
			_ = os.Remove(tempFrame)
		}
	}()
	w, aerr := pool.acquire(ctx)
	if aerr != nil {
		return nil, aerr
	}
	vec, err, abandoned := runWithTimeout(ctx, st.timeout, func() ([]float32, error) { return w.embed(imagePath) })
	if abandoned {
		pool.discard(w) // request still in flight — the worker is unusable
		if err != nil {
			return nil, err // cancelled mid-compute
		}
		return nil, fmt.Errorf("routing embed timed out after %s", st.timeout)
	}
	pool.release(w)
	if err != nil {
		return nil, err
	}
	db := run.Queue.Db
	if uerr := media.UpsertEmbedding(db, path, st.embedModel.ID, vec, 0); uerr == nil {
		indexAdd(st.embedModel.ID, path, vec) // index normalizes internally
	}
	return vec, nil
}

// poolFor lazily starts (and caches) the serve pool for one recognizer,
// sized to the runner's worker count so a combined job never over-spawns.
func (st *facesOpState) poolFor(ctx context.Context, run *ItemRun, m FaceModel) (*servePool, error) {
	st.mu.Lock()
	defer st.mu.Unlock()
	if p, ok := st.pools[m.ID]; ok {
		return p, nil
	}

	detectorPath, err := FaceDetectorPathFor(m)
	if err != nil {
		return nil, err
	}
	recognizerPath, err := FaceRecognizerPath(m)
	if err != nil {
		return nil, err
	}
	secondaryPath, err := FaceSecondaryPath(m)
	if err != nil {
		return nil, err
	}

	_, threads := ResolveFaceResources()
	ortLib, provider := resolveONNXRuntime(FaceProviderFromConfig())
	if FaceProviderFromConfig() == "directml" && provider != "directml" {
		st.q.log("DirectML runtime not installed; falling back to CPU. Install it from Dependencies.")
	}
	st.q.log(fmt.Sprintf("[%s] scanning with %d worker(s), %d thread(s) each, provider=%s",
		m.ID, run.Workers, threads, provider))

	args := buildFacesServeArgs(detectorPath, recognizerPath, secondaryPath, ortLib, m, provider, threads)
	pool, err := newServePool(ctx, run.Workers, st.embedBin, args, run.Background)
	if err != nil {
		return nil, fmt.Errorf("start faces worker (%s): %w", m.ID, err)
	}
	st.pools[m.ID] = pool
	return pool, nil
}

// processOne scans a single item: route → frame → worker → parse, returning
// the serialized commit (store faces + index + incremental people assignment).
func (st *facesOpState) processOne(ctx context.Context, run *ItemRun, path, localPath string) (*ItemCommit, error) {
	model := st.modelFor(ctx, run, path, localPath)
	pool, perr := st.poolFor(ctx, run, model)
	if perr != nil {
		return nil, perr
	}

	imagePath, tempFrame, ferr := extractFrameForFile(ctx, localPath, st.timeout)
	if ferr != nil {
		return nil, fmt.Errorf("frame extract: %w", ferr)
	}
	defer func() {
		if tempFrame != "" {
			_ = os.Remove(tempFrame)
		}
	}()

	w, aerr := pool.acquire(ctx)
	if aerr != nil {
		return nil, aerr
	}
	faces, err, abandoned := runWithTimeout(ctx, st.timeout, func() ([]media.NewFace, error) {
		if werr := w.writeLine(imagePath); werr != nil {
			return nil, werr
		}
		line, ok := w.readLine()
		if !ok {
			return nil, fmt.Errorf("faces worker died: %s", w.stderrString())
		}
		if msg, found := strings.CutPrefix(line, "ERR "); found {
			return nil, fmt.Errorf("%s", msg)
		}
		return parseFacesLine(line)
	})
	if abandoned {
		pool.discard(w) // request still in flight — the worker is unusable
		if err != nil {
			return nil, err // cancelled mid-compute
		}
		return nil, fmt.Errorf("timed out after %s", st.timeout)
	}
	pool.release(w)
	if err != nil {
		return nil, err
	}

	db := run.Queue.Db
	return &ItemCommit{
		Commit: func() error {
			ids, cerr := media.ReplaceFaces(db, path, model.ID, faces, time.Now().Unix())
			if cerr != nil {
				return cerr
			}
			st.newFaces.Add(int64(len(ids)))
			// One item gained a face_scan marker — advance the live coverage
			// counter (the stats snapshot recount reconciles any drift).
			notifyProgress(ProgressFaces, 1)
			faceIndexReplacePath(model.ID, path, ids, faces) // index normalizes internally
			// Fresh faces join existing people immediately when a confident
			// match exists...
			autoAssignNewFaces(db, model, ids, faces)
			// ...and every clusterEvery new faces a strict incremental pass
			// runs inline so NEW people form mid-scan too. It executes on the
			// committer goroutine — commits stall for its duration, which is
			// the intended throttle (clustering shares the machine anyway).
			if due, batches := st.noteFacesScanned(model.ID, ids); due {
				st.runClusterPass(batches, false)
			}
			return nil
		},
		Detail: fmt.Sprintf("%d face(s) [%s]", len(faces), model.ID),
	}, nil
}
