package tasks

import (
	"context"
	"database/sql"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/stevecastle/shrike/appconfig"
	"github.com/stevecastle/shrike/deps"
	"github.com/stevecastle/shrike/embedvec"
	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/media"
)

// FaceProviderFromConfig returns the configured face-task execution provider.
func FaceProviderFromConfig() string {
	return normalizeProvider(appconfig.Get().FaceProvider)
}

// ResolveFaceResources mirrors ResolveEmbedResources for the face task,
// reading the Face* config fields.
func ResolveFaceResources() (workers, threads int) {
	cfg := appconfig.Get()
	return resolveResources(cfg.FacePerformance, cfg.FaceWorkers, cfg.FaceThreadsPerWorker, FaceProviderFromConfig())
}

// faceLineJSON mirrors cmd/embed's --faces output: per-image dimensions plus
// relative-coordinate faces with base64 vectors.
type faceLineJSON struct {
	ImageW int `json:"image_w"`
	ImageH int `json:"image_h"`
	Faces  []struct {
		X         float64       `json:"x"`
		Y         float64       `json:"y"`
		W         float64       `json:"w"`
		H         float64       `json:"h"`
		Score     float64       `json:"score"`
		Landmarks [5][2]float64 `json:"landmarks"`
		Vec       string        `json:"vec"`
	} `json:"faces"`
}

// parseFacesLine decodes one worker output line into storable faces.
func parseFacesLine(line string) ([]media.NewFace, error) {
	var parsed faceLineJSON
	if err := json.Unmarshal([]byte(line), &parsed); err != nil {
		return nil, fmt.Errorf("decode faces JSON: %w", err)
	}
	out := make([]media.NewFace, 0, len(parsed.Faces))
	for _, f := range parsed.Faces {
		raw, err := base64.StdEncoding.DecodeString(f.Vec)
		if err != nil {
			return nil, fmt.Errorf("decode face vector: %w", err)
		}
		vec, err := embedvec.Decode(raw)
		if err != nil {
			return nil, err
		}
		out = append(out, media.NewFace{
			X: f.X, Y: f.Y, W: f.W, H: f.H,
			Score: f.Score,
			Vec:   vec,
		})
	}
	return out, nil
}

// faceModelArgs assembles the model-driven `embed --faces` flags shared by
// serve and one-shot invocations. When m is a fused model (Secondary set),
// the primary dim is m.Dim minus the secondary's — m.Dim is the concat total.
func faceModelArgs(m FaceModel, detectorPath, recognizerPath, secondaryPath string) []string {
	primaryDim := m.Dim
	if m.Secondary != nil {
		primaryDim = m.Dim - m.Secondary.Dim
	}
	args := []string{
		"--faces",
		"--detect-model=" + detectorPath,
		"--model=" + recognizerPath,
		fmt.Sprintf("--dim=%d", primaryDim),
		"--face-input=" + m.InputName,
		"--face-output=" + m.OutputName,
		fmt.Sprintf("--face-mean=%g,%g,%g", m.Mean[0], m.Mean[1], m.Mean[2]),
		fmt.Sprintf("--face-std=%g,%g,%g", m.Std[0], m.Std[1], m.Std[2]),
		"--face-color=" + m.ColorOrder,
	}
	if m.InputSize > 0 {
		args = append(args, fmt.Sprintf("--face-size=%d", m.InputSize))
	}
	if m.DetectorKind != "" {
		args = append(args, "--detect-kind="+m.DetectorKind)
	}
	if m.Align != "" {
		args = append(args, "--align="+m.Align)
	}
	if m.CropExpand > 0 {
		args = append(args, fmt.Sprintf("--crop-expand=%g", m.CropExpand))
	}
	if m.Weight > 0 {
		args = append(args, fmt.Sprintf("--face-weight=%g", m.Weight))
	}
	if m.Secondary != nil && secondaryPath != "" {
		s := m.Secondary
		args = append(args,
			"--face2-model="+secondaryPath,
			fmt.Sprintf("--face2-dim=%d", s.Dim),
			"--face2-input="+s.InputName,
			"--face2-output="+s.OutputName,
			fmt.Sprintf("--face2-size=%d", s.InputSize),
			fmt.Sprintf("--face2-mean=%g,%g,%g", s.Mean[0], s.Mean[1], s.Mean[2]),
			fmt.Sprintf("--face2-std=%g,%g,%g", s.Std[0], s.Std[1], s.Std[2]),
			"--face2-color="+s.ColorOrder,
		)
		if m.SecondaryWeight > 0 {
			args = append(args, fmt.Sprintf("--face2-weight=%g", m.SecondaryWeight))
		}
	}
	return args
}

// buildFacesServeArgs assembles the `embed.exe --faces --serve` arguments for
// a recognizer + provider + thread config.
func buildFacesServeArgs(detectorPath, recognizerPath, secondaryPath, ortLib string, m FaceModel, provider string, threads int) []string {
	args := append(faceModelArgs(m, detectorPath, recognizerPath, secondaryPath),
		"--serve",
		"--provider="+provider,
	)
	if threads > 0 {
		args = append(args, fmt.Sprintf("--threads=%d", threads))
	}
	if ortLib != "" {
		args = append(args, "--ort="+ortLib)
	}
	return args
}

func shouldSkipFaceScan(db *sql.DB, path, model string) bool {
	ok, err := media.HasFaceScan(db, path, model)
	return err == nil && ok
}

// facesResult is one scanned-or-skipped media item handed to the collector.
type facesResult struct {
	mediaPath string
	faces     []media.NewFace
	ok        bool
}

// facesTask scans media for faces and stores per-face identity embeddings.
// It mirrors embedTask: paths from a query or explicit input, a pool of
// persistent `embed --faces --serve` workers, and a single collector goroutine
// owning all DB writes. Already-scanned media (face_scan marker) is skipped,
// so "no faces found" isn't rescanned every run.
func facesTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	ctx := j.Ctx

	var paths []string
	fromQuery := false
	if qstr, ok := extractQueryFromJob(j); ok {
		q.PushJobStdout(j.ID, fmt.Sprintf("Using query: %s", qstr))
		mediaPaths, err := getMediaPathsByQueryFast(q.Db, qstr)
		if err != nil {
			q.PushJobStdout(j.ID, "Failed to load paths from query: "+err.Error())
			q.ErrorJob(j.ID)
			return err
		}
		paths = mediaPaths
		fromQuery = true
		q.PushJobStdout(j.ID, fmt.Sprintf("Query matched %d items", len(paths)))
	} else {
		raw := strings.TrimSpace(j.Input)
		if raw == "" {
			q.PushJobStdout(j.ID, "No image path provided")
			q.CompleteJob(j.ID)
			return nil
		}
		paths = parseInputPaths(raw)
		q.PushJobStdout(j.ID, fmt.Sprintf("Processing %d files from input", len(paths)))
	}
	if len(paths) == 0 {
		q.PushJobStdout(j.ID, "No files to process")
		q.CompleteJob(j.ID)
		return nil
	}

	embedBin := deps.BundledOrEmpty("embed")
	if embedBin == "" {
		q.PushJobStdout(j.ID, "embed binary not installed; install it from Dependencies")
		q.ErrorJob(j.ID)
		return fmt.Errorf("embed binary not installed")
	}

	// Resolve which recognizer(s) scan which paths. An explicit `--model=<id>`
	// pins everything to that model (background migration, same contract as
	// the embed task). Otherwise routing runs in STREAMING mode: paths are
	// classified in chunks and fed to per-model pools as they're routed, so
	// scanning starts within the first chunk instead of after a full-library
	// routing pass (which on 100k items would sit silent for a long time).
	var scanned, facesFound, autoAssigned, skipped int
	var scanErr error
	if id, ok := embedModelOverrideFromJob(j); ok {
		m, known := FaceModelByID(id)
		if !known {
			m = ActiveFaceModel()
			q.PushJobStdout(j.ID, fmt.Sprintf("Unknown --model %q; using active model %q", id, m.ID))
		}
		scanned, facesFound, autoAssigned, skipped, scanErr = scanPathsWithModel(ctx, j, q, paths, fromQuery, m, embedBin)
	} else {
		scanned, facesFound, autoAssigned, skipped, scanErr = runRoutedFacesScan(ctx, j, q, paths, fromQuery, embedBin)
	}
	if scanErr != nil {
		if ctx.Err() != nil {
			q.PushJobStdout(j.ID, "Task canceled")
			_ = q.CancelJob(j.ID)
			return scanErr
		}
		q.PushJobStdout(j.ID, "Face scan failed: "+scanErr.Error())
		q.ErrorJob(j.ID)
		return scanErr
	}

	q.PushJobStdout(j.ID, fmt.Sprintf("Completed: %d scanned (%d faces found, %d auto-assigned to people), %d skipped", scanned, facesFound, autoAssigned, skipped))
	q.CompleteJob(j.ID)
	return nil
}

// scanPathsWithModel resolves one recognizer's on-disk pieces and scans the
// whole path list with it.
func scanPathsWithModel(ctx context.Context, j *jobqueue.Job, q *jobqueue.Queue, paths []string, fromQuery bool, model FaceModel, embedBin string) (int, int, int, int, error) {
	q.PushJobStdout(j.ID, fmt.Sprintf("Face model: %s (dim %d, detector %s)", model.ID, model.Dim, model.DetectorKindOrDefault()))
	detectorPath, err := FaceDetectorPathFor(model)
	if err != nil {
		return 0, 0, 0, 0, err
	}
	recognizerPath, err := FaceRecognizerPath(model)
	if err != nil {
		return 0, 0, 0, 0, err
	}
	secondaryPath, err := FaceSecondaryPath(model)
	if err != nil {
		return 0, 0, 0, 0, err
	}
	return runFacesPool(ctx, j, q, paths, fromQuery, model, detectorPath, recognizerPath, secondaryPath, embedBin)
}

// startPoolForModel resolves a recognizer's on-disk pieces and starts its
// live pool.
func startPoolForModel(ctx context.Context, j *jobqueue.Job, q *jobqueue.Queue, fromQuery bool, model FaceModel, embedBin string) (*facesPoolRun, error) {
	detectorPath, err := FaceDetectorPathFor(model)
	if err != nil {
		return nil, err
	}
	recognizerPath, err := FaceRecognizerPath(model)
	if err != nil {
		return nil, err
	}
	secondaryPath, err := FaceSecondaryPath(model)
	if err != nil {
		return nil, err
	}
	return startFacesPool(ctx, j, q, fromQuery, model, detectorPath, recognizerPath, secondaryPath, embedBin)
}

// runRoutedFacesScan classifies paths photo-vs-anime in chunks and streams
// them into lazily-started per-model pools, so scanning overlaps routing:
//
//   - already-scanned items (under either candidate model) are skipped up
//     front from a batched face_scan lookup — no classification cost at all;
//   - items with a stored SigLIP embedding are classified with one cosine;
//   - items with no embedding are embedded on the fly (persisting the vector)
//     BETWEEN chunk feeds, so the pools keep scanning while stragglers embed.
//
// Progress is logged per chunk so a big library never looks stuck.
func runRoutedFacesScan(ctx context.Context, j *jobqueue.Job, q *jobqueue.Queue, paths []string, fromQuery bool, embedBin string) (int, int, int, int, error) {
	var anchors *anchorCache
	var anchErr error
	if FaceRoutingEnabled() {
		anchors, anchErr = domainAnchors(ctx)
		if anchErr != nil {
			q.PushJobStdout(j.ID, "Domain routing unavailable ("+anchErr.Error()+"); scanning everything with the active model")
		}
	}
	if anchors == nil {
		return scanPathsWithModel(ctx, j, q, paths, fromQuery, ActiveFaceModel(), embedBin)
	}

	photoModel := faceModelForDomain("photo")
	animeModel := faceModelForDomain("anime")
	candidateIDs := []string{photoModel.ID}
	if animeModel.ID != photoModel.ID {
		candidateIDs = append(candidateIDs, animeModel.ID)
	}
	embedModel := TextSearchModel()

	pools := map[string]*facesPoolRun{}
	// finishAll drains every started pool exactly once and sums their counts.
	finishAll := func() (int, int, int, int) {
		var s, f, a, sk int
		for _, p := range pools {
			ps, pf, pa, psk := p.finish()
			s += ps
			f += pf
			a += pa
			sk += psk
		}
		return s, f, a, sk
	}
	poolFor := func(m FaceModel) (*facesPoolRun, error) {
		if p, ok := pools[m.ID]; ok {
			return p, nil
		}
		p, err := startPoolForModel(ctx, j, q, fromQuery, m, embedBin)
		if err != nil {
			return nil, err
		}
		pools[m.ID] = p
		return p, nil
	}

	const chunkSize = 512
	routed, alreadyScanned, embeddedOnTheFly := 0, 0, 0
	for lo := 0; lo < len(paths) && ctx.Err() == nil; lo += chunkSize {
		hi := lo + chunkSize
		if hi > len(paths) {
			hi = len(paths)
		}
		chunk := paths[lo:hi]

		scannedSet, err := media.FaceScansForPaths(q.Db, candidateIDs, chunk)
		if err != nil {
			s, f, a, sk := finishAll()
			return s, f, a, sk, err
		}
		stored, err := media.GetEmbeddingsForPaths(q.Db, embedModel.ID, chunk)
		if err != nil {
			s, f, a, sk := finishAll()
			return s, f, a, sk, err
		}

		for _, p := range chunk {
			if ctx.Err() != nil {
				break
			}
			if scannedSet[p] {
				alreadyScanned++
				continue
			}
			m := ActiveFaceModel() // classification-failure fallback
			if vec, ok := stored[p]; ok {
				m = routeVec(vec, anchors)
			} else if fresh, err := ImageQueryVectorForPath(ctx, q.Db, embedModel, p); err == nil {
				embeddedOnTheFly++
				m = routeVec(fresh, anchors)
			}
			pool, err := poolFor(m)
			if err != nil {
				s, f, a, sk := finishAll()
				return s, f, a, sk, err
			}
			if !pool.feed(p) {
				break
			}
			routed++
		}
		q.PushJobStdout(j.ID, fmt.Sprintf(
			"Routing: %d/%d processed (%d routed, %d already scanned, %d embedded on the fly)",
			hi, len(paths), routed, alreadyScanned, embeddedOnTheFly,
		))
	}
	if embeddedOnTheFly > 0 {
		q.PushJobStdout(j.ID, fmt.Sprintf("(%d item(s) had no stored embedding — run the Embeddings task first to make routing instant)", embeddedOnTheFly))
	}

	s, f, a, sk := finishAll()
	sk += alreadyScanned
	if ctx.Err() != nil {
		return s, f, a, sk, ctx.Err()
	}
	return s, f, a, sk, nil
}

// facesPoolRun is a live, channel-fed scanning pool for one recognizer. The
// caller feeds paths as it discovers them (e.g. while routing classifies the
// library in chunks) and calls finish() to drain and collect the counts —
// scanning overlaps with whatever produces the paths instead of waiting for
// the full list.
type facesPoolRun struct {
	model FaceModel
	jobs  chan string

	// Owned by the collector goroutine; read only after finish().
	scanned, facesFound, skipped, autoAssigned int

	workerWG    sync.WaitGroup
	collectorWG sync.WaitGroup
	results     chan facesResult
	ctx         context.Context
}

// feed hands one path to the pool. Returns false when the job was canceled.
func (p *facesPoolRun) feed(path string) bool {
	select {
	case p.jobs <- path:
		return true
	case <-p.ctx.Done():
		return false
	}
}

// finish stops accepting paths, drains the workers, and returns the counts
// (scanned, facesFound, autoAssigned, skipped).
func (p *facesPoolRun) finish() (int, int, int, int) {
	close(p.jobs)
	p.workerWG.Wait()
	close(p.results)
	p.collectorWG.Wait()
	return p.scanned, p.facesFound, p.autoAssigned, p.skipped
}

// startFacesPool launches the persistent worker subprocesses and the DB
// collector for one recognizer and returns the live pool.
func startFacesPool(ctx context.Context, j *jobqueue.Job, q *jobqueue.Queue, fromQuery bool, model FaceModel, detectorPath, recognizerPath, secondaryPath, embedBin string) (*facesPoolRun, error) {
	workers, threads := ResolveFaceResources()
	ortLib, provider := resolveONNXRuntime(FaceProviderFromConfig())
	if FaceProviderFromConfig() == "directml" && provider != "directml" {
		q.PushJobStdout(j.ID, "DirectML runtime not installed; falling back to CPU. Install it from Dependencies.")
	}
	if workers < 1 {
		workers = 1
	}
	q.PushJobStdout(j.ID, fmt.Sprintf("[%s] scanning with %d worker(s), %d thread(s) each, provider=%s", model.ID, workers, threads, provider))

	baseArgs := buildFacesServeArgs(detectorPath, recognizerPath, secondaryPath, ortLib, model, provider, threads)

	pool := make([]*serveWorker, 0, workers)
	for i := 0; i < workers; i++ {
		wkr, err := startServeWorker(ctx, embedBin, baseArgs)
		if err != nil {
			for _, p := range pool {
				p.close()
			}
			return nil, fmt.Errorf("start faces worker: %w", err)
		}
		pool = append(pool, wkr)
	}

	run := &facesPoolRun{
		model:   model,
		jobs:    make(chan string),
		results: make(chan facesResult, workers*2),
		ctx:     ctx,
	}

	// Collector: single goroutine owns all DB writes.
	run.collectorWG.Add(1)
	go func() {
		defer run.collectorWG.Done()
		for r := range run.results {
			if !r.ok {
				run.skipped++
				continue
			}
			ids, err := media.ReplaceFaces(q.Db, r.mediaPath, model.ID, r.faces, time.Now().Unix())
			if err != nil {
				q.PushJobStdout(j.ID, "  Failed to store faces: "+err.Error())
				run.skipped++
				continue
			}
			faceIndexReplacePath(model.ID, r.mediaPath, ids, r.faces) // index normalizes internally
			// Incremental clustering: fresh faces join existing people when a
			// confident match exists (full grouping is the faces-cluster task).
			run.autoAssigned += autoAssignNewFaces(q.Db, model, ids, r.faces)
			q.RegisterOutputFile(j.ID, r.mediaPath)
			run.scanned++
			run.facesFound += len(r.faces)
			if (run.scanned+run.skipped)%50 == 0 {
				q.PushJobStdout(j.ID, fmt.Sprintf("[%s] progress: %d scanned (%d faces), %d skipped", model.ID, run.scanned, run.facesFound, run.skipped))
			}
		}
	}()

	// Workers: each owns one subprocess and pulls paths off the channel.
	timeout := OnnxFileTimeout()
	for _, wkr := range pool {
		run.workerWG.Add(1)
		go func(w *serveWorker) {
			defer run.workerWG.Done()
			defer func() {
				if w != nil {
					w.close()
				}
			}()
			for mediaPath := range run.jobs {
				if ctx.Err() != nil {
					return
				}
				if shouldSkipFaceScan(q.Db, mediaPath, model.ID) {
					run.results <- facesResult{ok: false}
					continue
				}
				if !fromQuery {
					if _, err := os.Stat(mediaPath); os.IsNotExist(err) {
						run.results <- facesResult{ok: false}
						continue
					}
				}
				imagePath, tempFrame, ferr := extractFrameForFile(ctx, mediaPath, timeout)
				if ferr != nil {
					q.PushJobStdout(j.ID, fmt.Sprintf("  frame extract failed/timed out (%s): %v", filepath.Base(mediaPath), ferr))
					run.results <- facesResult{ok: false}
					continue
				}
				if w == nil { // a previous restart failed; drain as skips
					if tempFrame != "" {
						_ = os.Remove(tempFrame)
					}
					run.results <- facesResult{ok: false}
					continue
				}
				faces, err, timedOut := runWithTimeout(ctx, timeout, func() ([]media.NewFace, error) {
					if werr := w.writeLine(imagePath); werr != nil {
						return nil, werr
					}
					line, ok := w.readLine()
					if !ok {
						return nil, fmt.Errorf("faces worker died: %s", w.stderrString())
					}
					if strings.HasPrefix(line, "ERR ") {
						return nil, fmt.Errorf("%s", strings.TrimPrefix(line, "ERR "))
					}
					return parseFacesLine(line)
				})
				if tempFrame != "" {
					_ = os.Remove(tempFrame)
				}
				if timedOut {
					q.PushJobStdout(j.ID, fmt.Sprintf("  timed out after %s, skipping + restarting worker: %s", timeout, filepath.Base(mediaPath)))
					w.kill()
					if nw, rerr := startServeWorker(ctx, embedBin, baseArgs); rerr == nil {
						w = nw
					} else {
						q.PushJobStdout(j.ID, "  worker restart failed; its remaining files will be skipped: "+rerr.Error())
						w = nil
					}
					run.results <- facesResult{ok: false}
					continue
				}
				if err != nil {
					if ctx.Err() != nil {
						return
					}
					q.PushJobStdout(j.ID, fmt.Sprintf("  face scan failed (%s): %v", filepath.Base(mediaPath), err))
					run.results <- facesResult{ok: false}
					continue
				}
				run.results <- facesResult{mediaPath: mediaPath, faces: faces, ok: true}
			}
		}(wkr)
	}
	return run, nil
}

// runFacesPool scans a fixed path list with one recognizer (the pinned
// --model flow). Returns (scanned, facesFound, autoAssigned, skipped).
func runFacesPool(ctx context.Context, j *jobqueue.Job, q *jobqueue.Queue, paths []string, fromQuery bool, model FaceModel, detectorPath, recognizerPath, secondaryPath, embedBin string) (int, int, int, int, error) {
	run, err := startFacesPool(ctx, j, q, fromQuery, model, detectorPath, recognizerPath, secondaryPath, embedBin)
	if err != nil {
		return 0, 0, 0, 0, err
	}
	for _, p := range paths {
		if !run.feed(p) {
			break
		}
	}
	scanned, facesFound, autoAssigned, skipped := run.finish()
	if ctx.Err() != nil {
		return scanned, facesFound, autoAssigned, skipped, ctx.Err()
	}
	return scanned, facesFound, autoAssigned, skipped, nil
}
