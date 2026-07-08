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
const defaultClusterEvery = 100

// defaultRebuildEvery is how many newly stored faces trigger an in-scan
// REBUILD: dissolve the anonymous groups and regroup them from scratch at
// full strength (same as the Rebuild button). The incremental passes between
// rebuilds are greedy and order-dependent — identities fragment across
// several Unknowns and borderline joins accumulate as seeds; a periodic
// rebuild re-litigates the anonymous groups against everything scanned so
// far, so those mistakes are wiped instead of compounding across a long
// scan. Named people, confirmed faces, rejections, and group bans all
// survive a rebuild. 0 disables.
const defaultRebuildEvery = 1000

func registerFacesItemOp() {
	RegisterItemOp(ItemOp{
		ID:   "faces",
		Name: "Faces (ONNX)",
		Options: []TaskOption{
			{Name: "model", Label: "Face Model", Type: "string", Description: "Pin scanning to one face model ID (default: automatic photo/anime routing)"},
			{Name: "cluster-every", Label: "Cluster Every N Faces", Type: "number", Default: float64(defaultClusterEvery), Description: "Run an incremental people-clustering pass after this many newly scanned faces so People appear while the scan runs (0 = only queue one clustering job at the end)"},
			{Name: "rebuild-every", Label: "Rebuild Groups Every N Faces", Type: "number", Default: float64(defaultRebuildEvery), Description: "Periodically dissolve the unnamed groups and regroup them from scratch during the scan, wiping incremental clustering's accumulated mistakes (0 = only the final rebuild at scan end)"},
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

	// newFaces counts face rows stored this run (across all items/workers). A
	// positive count at Finalize queues a clustering pass so the fresh faces
	// form/join people without a manual Cluster press.
	newFaces atomic.Int64

	// Incremental clustering bookkeeping. Touched ONLY from the runner's
	// single committer goroutine (inside Commit closures) and from Finalize
	// (which runs after the committer drains), so no locking is needed.
	clusterEvery  int                 // faces per in-scan pass; 0 = disabled
	rebuildEvery  int                 // faces per in-scan REBUILD (reset + full regroup); 0 = disabled
	sinceCluster  int                 // faces stored since the last in-scan pass
	sinceRebuild  int                 // faces stored since the last in-scan rebuild
	dirtyModels   map[string]int      // model ID -> faces stored since its last pass
	touchedModels map[string]struct{} // every model that stored faces this run
}

// noteFacesScanned records n freshly stored faces for a model and reports
// whether an in-scan clustering pass is due — and whether that pass should be
// a full REBUILD (reset + full-strength regroup) rather than the strict
// incremental preview. When due it returns the models with new faces and
// resets the counters (the caller runs the pass). The rebuild cadence takes
// priority: when both are due at once, one rebuild does strictly more than an
// incremental pass would.
func (st *facesOpState) noteFacesScanned(modelID string, n int) (due, rebuild bool, models []string) {
	if n <= 0 {
		return false, false, nil
	}
	if st.dirtyModels == nil {
		st.dirtyModels = map[string]int{}
	}
	if st.touchedModels == nil {
		st.touchedModels = map[string]struct{}{}
	}
	st.dirtyModels[modelID] += n
	st.touchedModels[modelID] = struct{}{}
	st.sinceCluster += n
	st.sinceRebuild += n
	if st.rebuildEvery > 0 && st.sinceRebuild >= st.rebuildEvery {
		st.sinceRebuild = 0
		return true, true, st.takeDirtyModels()
	}
	if st.clusterEvery <= 0 || st.sinceCluster < st.clusterEvery {
		return false, false, nil
	}
	return true, false, st.takeDirtyModels()
}

// takeDirtyModels returns the models with new unclustered faces and resets
// the counters.
func (st *facesOpState) takeDirtyModels() []string {
	models := make([]string, 0, len(st.dirtyModels))
	for id := range st.dirtyModels {
		models = append(models, id)
	}
	st.dirtyModels = map[string]int{}
	st.sinceCluster = 0
	return models
}

// runClusterPass runs one clustering pass for the given models, inline in
// the scan job (a separate faces-cluster job could not run anyway: it shares
// the faces + local-compute buckets with the scan). Incremental passes use
// the strict params — a high-precision live preview. Rebuild passes (the
// periodic cadence and the one at scan end) first dissolve the anonymous
// groups, then regroup at the normal full-strength defaults: because every
// rebuild starts from a clean slate, incremental mistakes (fragmented
// identities, accumulated borderline joins) are wiped instead of compounding.
// Named people, user-confirmed faces, rejections, and dissolved-group bans
// all survive a rebuild. Logs the outcome and broadcasts "people-updated" so
// open People views refresh.
func (st *facesOpState) runClusterPass(modelIDs []string, rebuild bool) {
	db := st.q.run.Queue.Db
	label := "incremental (strict)"
	if rebuild {
		label = "rebuild (reset + full)"
	}
	for _, id := range modelIDs {
		m, known := FaceModelByID(id)
		if !known {
			continue
		}
		params := incrementalClusterParams(m)
		if rebuild {
			params = defaultClusterParams(m)
			n, err := resetAutoAssignments(db, id)
			if err != nil {
				st.q.log(fmt.Sprintf("  clustering %s (%s) reset failed: %v", label, id, err))
				continue
			}
			if n > 0 {
				st.q.log(fmt.Sprintf("  clustering %s (%s): dissolved %d auto assignment(s) in unnamed groups", label, id, n))
			}
		}
		stats, err := clusterFaces(db, m, params)
		if err != nil {
			st.q.log(fmt.Sprintf("  clustering %s (%s) failed: %v", label, id, err))
			continue
		}
		st.q.log(fmt.Sprintf("  clustering %s (%s): +%d people, %d faces joined existing, %d newly grouped",
			label, id, stats.NewPeople, stats.JoinedExisting, stats.NewlyClustered))
	}
	broadcastPeopleUpdated(modelIDs)
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
		rebuildEvery: defaultRebuildEvery,
	}
	if v, ok := run.Opts["cluster-every"].(float64); ok {
		st.clusterEvery = int(v)
	}
	if v, ok := run.Opts["rebuild-every"].(float64); ok {
		st.rebuildEvery = int(v)
	}
	if st.clusterEvery > 0 {
		q.PushJobStdout(j.ID, fmt.Sprintf("Incremental clustering: people update every %d new faces", st.clusterEvery))
	}
	if st.rebuildEvery > 0 {
		q.PushJobStdout(j.ID, fmt.Sprintf("Periodic rebuild: unnamed groups regrouped from scratch every %d new faces", st.rebuildEvery))
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
		Process: func(ctx context.Context, path string) (*ItemCommit, error) {
			return st.processOne(ctx, run, path)
		},
		Finalize: func() error {
			if st.clusterEvery > 0 || st.rebuildEvery > 0 {
				// In-scan mode always ends with ONE full rebuild over every
				// model that stored faces this run: the unnamed groups are
				// dissolved and regrouped with complete data — incremental
				// fragmentation and order-dependence don't outlive the scan.
				st.takeDirtyModels() // counters are superseded by the rebuild
				if len(st.touchedModels) > 0 {
					models := make([]string, 0, len(st.touchedModels))
					for id := range st.touchedModels {
						models = append(models, id)
					}
					q.PushJobStdout(j.ID, "Final rebuild: regrouping unnamed groups with the complete scan data")
					st.runClusterPass(models, true)
				}
				return nil
			}
			if id, queued := maybeQueueFaceClustering(q, st.newFaces.Load()); queued {
				q.PushJobStdout(j.ID, fmt.Sprintf("Stored %d new face(s) — queued rebuild clustering pass (job %s)", st.newFaces.Load(), id))
			}
			return nil
		},
		Close: func() {
			st.mu.Lock()
			defer st.mu.Unlock()
			for _, p := range st.pools {
				p.close()
			}
		},
	}, nil
}

// maybeQueueFaceClustering queues a rebuild clustering pass (--reset: unnamed
// groups dissolved and regrouped from scratch) after a face scan that stored
// new faces, so freshly-scanned faces form or join people without the user
// pressing Cluster — with batch-quality groups, not incremental leftovers.
// Gated on newFaces > 0, and deduped against an already-pending faces-cluster
// job — that pending pass will pick up these faces when it runs, so at most
// one is ever queued at a time. A cluster job never scans faces, so it can't
// retrigger this. Returns the queued job ID and whether one was created.
func maybeQueueFaceClustering(q *jobqueue.Queue, newFaces int64) (string, bool) {
	if newFaces <= 0 {
		return "", false
	}
	for _, j := range q.GetJobs() {
		if j.Command == "faces-cluster" && j.State == jobqueue.StatePending {
			return "", false
		}
	}
	id, err := q.AddJob("", "faces-cluster", []string{"--reset"}, "", nil)
	if err != nil {
		return "", false
	}
	return id, true
}

// modelFor routes one item to its recognizer: the pinned model when routing
// is off, otherwise the SigLIP-classified domain model (stored vector when
// present, embedded on the fly — which persists the vector — otherwise).
// Classification failure falls back to the active model.
func (st *facesOpState) modelFor(ctx context.Context, run *ItemRun, path string) FaceModel {
	if st.anchors == nil {
		return st.pinnedModel
	}
	if vec, ok, err := media.GetEmbedding(run.Queue.Db, path, st.embedModel.ID); err == nil && ok {
		return routeVec(vec, st.anchors)
	}
	if fresh, err := ImageQueryVectorForPath(ctx, run.Queue.Db, st.embedModel, path); err == nil {
		return routeVec(fresh, st.anchors)
	}
	return ActiveFaceModel()
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
func (st *facesOpState) processOne(ctx context.Context, run *ItemRun, path string) (*ItemCommit, error) {
	model := st.modelFor(ctx, run, path)
	pool, perr := st.poolFor(ctx, run, model)
	if perr != nil {
		return nil, perr
	}

	imagePath, tempFrame, ferr := extractFrameForFile(ctx, path, st.timeout)
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
	faces, err, timedOut := runWithTimeout(ctx, st.timeout, func() ([]media.NewFace, error) {
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
	if timedOut {
		pool.discard(w)
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
			// runs inline so NEW people form mid-scan too, with a full
			// rebuild every rebuildEvery faces wiping the incremental
			// passes' accumulated mistakes. It executes on the committer
			// goroutine — commits stall for its duration, which is the
			// intended throttle (clustering shares the machine anyway).
			if due, rebuild, models := st.noteFacesScanned(model.ID, len(ids)); due {
				st.runClusterPass(models, rebuild)
			}
			return nil
		},
		Detail: fmt.Sprintf("%d face(s) [%s]", len(faces), model.ID),
	}, nil
}
