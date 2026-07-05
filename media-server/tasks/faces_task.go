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

	// Resolve the recognizer. An explicit `--model=<id>` in the job overrides
	// the configured active model (background migration to a new recognizer,
	// same contract as the embed task).
	model := ActiveFaceModel()
	if id, ok := embedModelOverrideFromJob(j); ok {
		if m, known := FaceModelByID(id); known {
			model = m
		} else {
			q.PushJobStdout(j.ID, fmt.Sprintf("Unknown --model %q; using active model %q", id, model.ID))
		}
	}
	q.PushJobStdout(j.ID, fmt.Sprintf("Face model: %s (dim %d), detector: YuNet", model.ID, model.Dim))

	detectorPath, err := FaceDetectorPathFor(model)
	if err != nil {
		q.PushJobStdout(j.ID, err.Error())
		q.ErrorJob(j.ID)
		return err
	}
	recognizerPath, err := FaceRecognizerPath(model)
	if err != nil {
		q.PushJobStdout(j.ID, err.Error())
		q.ErrorJob(j.ID)
		return err
	}
	secondaryPath, err := FaceSecondaryPath(model)
	if err != nil {
		q.PushJobStdout(j.ID, err.Error())
		q.ErrorJob(j.ID)
		return err
	}
	embedBin := deps.BundledOrEmpty("embed")
	if embedBin == "" {
		q.PushJobStdout(j.ID, "embed binary not installed; install it from Dependencies")
		q.ErrorJob(j.ID)
		return fmt.Errorf("embed binary not installed")
	}

	scanned, facesFound, autoAssigned, skipped, err := runFacesPool(ctx, j, q, paths, fromQuery, model, detectorPath, recognizerPath, secondaryPath, embedBin)
	if err != nil {
		if ctx.Err() != nil {
			q.PushJobStdout(j.ID, "Task canceled")
			_ = q.CancelJob(j.ID)
			return err
		}
		q.PushJobStdout(j.ID, "Face scan failed: "+err.Error())
		q.ErrorJob(j.ID)
		return err
	}

	q.PushJobStdout(j.ID, fmt.Sprintf("Completed: %d scanned (%d faces found, %d auto-assigned to people), %d skipped", scanned, facesFound, autoAssigned, skipped))
	q.CompleteJob(j.ID)
	return nil
}

// runFacesPool scans all paths using a pool of persistent worker processes,
// storing faces under model.ID. Returns (scanned, facesFound, autoAssigned,
// skipped).
func runFacesPool(ctx context.Context, j *jobqueue.Job, q *jobqueue.Queue, paths []string, fromQuery bool, model FaceModel, detectorPath, recognizerPath, secondaryPath, embedBin string) (int, int, int, int, error) {
	workers, threads := ResolveFaceResources()
	ortLib, provider := resolveONNXRuntime(FaceProviderFromConfig())
	if FaceProviderFromConfig() == "directml" && provider != "directml" {
		q.PushJobStdout(j.ID, "DirectML runtime not installed; falling back to CPU. Install it from Dependencies.")
	}
	if workers > len(paths) {
		workers = len(paths)
	}
	if workers < 1 {
		workers = 1
	}
	q.PushJobStdout(j.ID, fmt.Sprintf("Scanning faces with %d worker(s), %d thread(s) each, provider=%s", workers, threads, provider))

	baseArgs := buildFacesServeArgs(detectorPath, recognizerPath, secondaryPath, ortLib, model, provider, threads)

	pool := make([]*serveWorker, 0, workers)
	for i := 0; i < workers; i++ {
		wkr, err := startServeWorker(ctx, embedBin, baseArgs)
		if err != nil {
			for _, p := range pool {
				p.close()
			}
			return 0, 0, 0, 0, fmt.Errorf("start faces worker: %w", err)
		}
		pool = append(pool, wkr)
	}

	jobs := make(chan string)
	results := make(chan facesResult, workers*2)

	// Collector: single goroutine owns all DB writes.
	var scanned, facesFound, skipped, autoAssigned int
	var collectorWG sync.WaitGroup
	collectorWG.Add(1)
	go func() {
		defer collectorWG.Done()
		for r := range results {
			if !r.ok {
				skipped++
				continue
			}
			ids, err := media.ReplaceFaces(q.Db, r.mediaPath, model.ID, r.faces, time.Now().Unix())
			if err != nil {
				q.PushJobStdout(j.ID, "  Failed to store faces: "+err.Error())
				skipped++
				continue
			}
			faceIndexReplacePath(model.ID, r.mediaPath, ids, r.faces) // index normalizes internally
			// Incremental clustering: fresh faces join existing people when a
			// confident match exists (full grouping is the faces-cluster task).
			autoAssigned += autoAssignNewFaces(q.Db, model, ids, r.faces)
			q.RegisterOutputFile(j.ID, r.mediaPath)
			scanned++
			facesFound += len(r.faces)
			if (scanned+skipped)%50 == 0 {
				q.PushJobStdout(j.ID, fmt.Sprintf("Progress: %d scanned (%d faces), %d skipped (of %d)", scanned, facesFound, skipped, len(paths)))
			}
		}
	}()

	// Workers: each owns one subprocess and pulls paths off the channel.
	timeout := OnnxFileTimeout()
	var workerWG sync.WaitGroup
	for _, wkr := range pool {
		workerWG.Add(1)
		go func(w *serveWorker) {
			defer workerWG.Done()
			defer func() {
				if w != nil {
					w.close()
				}
			}()
			for mediaPath := range jobs {
				if ctx.Err() != nil {
					return
				}
				if shouldSkipFaceScan(q.Db, mediaPath, model.ID) {
					results <- facesResult{ok: false}
					continue
				}
				if !fromQuery {
					if _, err := os.Stat(mediaPath); os.IsNotExist(err) {
						results <- facesResult{ok: false}
						continue
					}
				}
				imagePath, tempFrame, ferr := extractFrameForFile(ctx, mediaPath, timeout)
				if ferr != nil {
					q.PushJobStdout(j.ID, fmt.Sprintf("  frame extract failed/timed out (%s): %v", filepath.Base(mediaPath), ferr))
					results <- facesResult{ok: false}
					continue
				}
				if w == nil { // a previous restart failed; drain as skips
					if tempFrame != "" {
						_ = os.Remove(tempFrame)
					}
					results <- facesResult{ok: false}
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
					results <- facesResult{ok: false}
					continue
				}
				if err != nil {
					if ctx.Err() != nil {
						return
					}
					q.PushJobStdout(j.ID, fmt.Sprintf("  face scan failed (%s): %v", filepath.Base(mediaPath), err))
					results <- facesResult{ok: false}
					continue
				}
				results <- facesResult{mediaPath: mediaPath, faces: faces, ok: true}
			}
		}(wkr)
	}

	// Feed paths (stops early on cancel).
	feedDone := make(chan struct{})
	go func() {
		defer close(feedDone)
		defer close(jobs)
		for _, p := range paths {
			if ctx.Err() != nil {
				return
			}
			jobs <- p
		}
	}()

	<-feedDone
	workerWG.Wait()
	close(results)
	collectorWG.Wait()

	if ctx.Err() != nil {
		return scanned, facesFound, autoAssigned, skipped, ctx.Err()
	}
	return scanned, facesFound, autoAssigned, skipped, nil
}
