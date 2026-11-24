package tasks

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/stevecastle/shrike/jobqueue"
)

// train-lora: a MOSS-Image (Kromoss) LoRA from the job's images with loki-tune. The task owns everything about the
// library: it resolves the selection or query to image files, optionally hands their library descriptions over as
// captions, and calls the engine with a .jsonl list (loki-tune itself only takes files). Runs live in
// <data>/loras/<name>/ (loki-tune's run folder: run.json, checkpoints/, samples/, NAME.safetensors).
//
// Pausing or cancelling asks loki-tune for a checkpoint (a STOP file in the run folder: it finishes the step, saves
// and exits) before the process is stopped; a resumed job continues from that checkpoint, as does a new job with
// the `resume` option naming the run.

var trainLoraKinds = []string{"character", "style", "object", "concept", "general"}

var trainLoraOptions = []TaskOption{
	{Name: "name", Label: "Name", Type: "string", Description: "LoRA name (file and run folder name); default: the trigger, else lora"},
	{Name: "trigger", Label: "Trigger word", Type: "string", Description: "Word put first in every caption; prompts with it call up the LoRA"},
	{Name: "kind", Label: "Kind", Type: "enum", Choices: trainLoraKinds, Default: "character", Description: "What the LoRA teaches: decides what the captions leave to the trigger (character: identity; style: the look; object: its design; concept: the pose/action; general: nothing)"},
	{Name: "class", Label: "Class", Type: "string", Description: "What the trigger is (woman, dog, mug): captions name it \"TRIGGER, a CLASS\""},
	{Name: "captions", Label: "Captions", Type: "enum", Choices: []string{"auto", "library", "none"}, Default: "auto", Description: "auto: loki-scribe writes captions for the kind; library: the items' descriptions where they have one (loki-scribe for the rest); none: the trigger alone"},
	{Name: "steps", Label: "Steps", Type: "number", Default: 0.0, Description: "Training steps (0 = about 50 epochs, 1000-3000; styles about 70, at least 2000)"},
	{Name: "rank", Label: "Rank", Type: "number", Default: 16.0, Description: "LoRA rank (8-128, multiple of 8)"},
	{Name: "lr", Label: "Learning rate", Type: "number", Default: 0.0, Description: "0 = 1e-4"},
	{Name: "resolution", Label: "Resolution", Type: "enum", Choices: []string{"512", "768", "1024"}, Default: "1024", Description: "Training size per image (about RES x RES pixels); 512 trains ~3x faster"},
	{Name: "model", Label: "Base model", Type: "string", Description: "MOSS-Image checkpoint to train against (default Kromoss)"},
	{Name: "flip", Label: "Mirror images", Type: "bool", Default: false, Description: "Also train on mirrored images (not for text or asymmetric subjects)"},
	{Name: "saveevery", Label: "Checkpoint every", Type: "number", Default: 0.0, Description: "Steps between checkpoints and previews (0 = 250)"},
	{Name: "out", Label: "Runs folder", Type: "string", Description: "Folder the run folder goes in (default: the server's data folder/loras)"},
	{Name: "resume", Label: "Resume", Type: "string", Description: "Continue a run (its name or folder, or a checkpoint folder) instead of starting one; the images are the run's"},
}

type trainLoraParams struct {
	Name, Trigger, Kind, Class, Captions, Model, Out, Resume string
	Steps, Rank, SaveEvery, Resolution                       int
	LR                                                       float64
	Flip                                                     bool
}

func trainLoraParamsFromOptions(opts map[string]any) trainLoraParams {
	p := trainLoraParams{
		Name:      strings.TrimSpace(optString(opts, "name")),
		Trigger:   strings.TrimSpace(optString(opts, "trigger")),
		Kind:      optString(opts, "kind"),
		Class:     strings.TrimSpace(optString(opts, "class")),
		Captions:  optString(opts, "captions"),
		Model:     strings.TrimSpace(optString(opts, "model")),
		Out:       strings.TrimSpace(optString(opts, "out")),
		Resume:    strings.TrimSpace(optString(opts, "resume")),
		Steps:     int(optFloat(opts, "steps")),
		Rank:      int(optFloat(opts, "rank")),
		SaveEvery: int(optFloat(opts, "saveevery")),
		LR:        optFloat(opts, "lr"),
		Flip:      optBool(opts, "flip"),
	}
	if !hasString(trainLoraKinds, p.Kind) {
		p.Kind = "character"
	}
	switch p.Captions {
	case "auto", "library", "none":
	default:
		p.Captions = "auto"
	}
	p.Resolution, _ = strconv.Atoi(optString(opts, "resolution"))
	if p.Resolution != 512 && p.Resolution != 768 {
		p.Resolution = 1024
	}
	if p.Rank <= 0 {
		p.Rank = 16
	}
	p.Rank = clampInt((p.Rank+7)/8*8, 8, 128)
	if p.Steps < 0 {
		p.Steps = 0
	}
	return p
}

func hasString(xs []string, s string) bool {
	for _, x := range xs {
		if x == s {
			return true
		}
	}
	return false
}

var loraNameUnsafe = regexp.MustCompile(`[^A-Za-z0-9._-]+`)

// loraRunName is the run's name: the given name, else the trigger, else "lora" (file-name safe).
func loraRunName(p trainLoraParams) string {
	n := p.Name
	if n == "" {
		n = p.Trigger
	}
	n = strings.Trim(loraNameUnsafe.ReplaceAllString(n, "_"), "._")
	if n == "" {
		n = "lora"
	}
	return n
}

// loraRunsDir is where runs live: the `out` option, else <data>/loras.
func loraRunsDir(p trainLoraParams) string {
	if p.Out != "" {
		return p.Out
	}
	return LoraTrainingDir()
}

// buildTrainLoraArgs is the loki-tune command line for a new run from `list` (a .jsonl of the images) into runDir,
// or, with resumeAt, continuing that run / checkpoint (training settings only: the images and model are the run's).
func buildTrainLoraArgs(p trainLoraParams, list, runDir, resumeAt, meta string) []string {
	var a []string
	if resumeAt != "" {
		a = append(a, "--resume", resumeAt)
	} else {
		a = append(a, list, "-o", runDir, "--name", filepath.Base(runDir), "--kind", p.Kind, "--rank", strconv.Itoa(p.Rank), "--resolution", strconv.Itoa(p.Resolution))
		if p.Trigger != "" {
			a = append(a, "--trigger", p.Trigger)
		}
		if p.Class != "" {
			a = append(a, "--class", p.Class)
		}
		switch p.Captions {
		case "none":
			a = append(a, "--captions", "none")
		default:
			a = append(a, "--captions", "auto")
		}
		if p.Model != "" {
			a = append(a, "--model", p.Model)
		}
		if p.Flip {
			a = append(a, "--flip")
		}
		if meta != "" {
			a = append(a, "--meta", meta)
		}
	}
	if p.Steps > 0 {
		a = append(a, "--steps", strconv.Itoa(p.Steps))
	}
	if p.LR > 0 {
		a = append(a, "--lr", strconv.FormatFloat(p.LR, 'g', -1, 64))
	}
	if p.SaveEvery > 0 {
		a = append(a, "--save-every", strconv.Itoa(p.SaveEvery))
	}
	return a
}

// loraTrainList writes the images as loki-tune's .jsonl list; with library captions, an item's description (when it
// has one) is its caption.
func loraTrainList(db *sql.DB, paths []string, libraryCaptions bool, file string) (int, error) {
	f, err := os.Create(file)
	if err != nil {
		return 0, err
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	captioned := 0
	for _, p := range paths {
		line := map[string]string{"path": p}
		if libraryCaptions && db != nil {
			var d sql.NullString
			if err := db.QueryRow(`SELECT description FROM media WHERE path = ?`, p).Scan(&d); err == nil && strings.TrimSpace(d.String) != "" {
				line["caption"] = strings.TrimSpace(d.String)
				captioned++
			}
		}
		if err := enc.Encode(line); err != nil {
			return captioned, err
		}
	}
	return captioned, nil
}

var loraTrainImageExts = map[string]bool{".png": true, ".jpg": true, ".jpeg": true, ".webp": true, ".bmp": true, ".tif": true, ".tiff": true, ".jfif": true}

// runJobMeta is the job id a run was started by (its run.json `meta.job`), so a paused job resumes its own run.
func runJobMeta(runDir string) string {
	b, err := os.ReadFile(filepath.Join(runDir, "run.json"))
	if err != nil {
		return ""
	}
	var c struct {
		Meta struct {
			Job string `json:"job"`
		} `json:"meta"`
	}
	_ = json.Unmarshal(b, &c)
	return c.Meta.Job
}

var checkpointDirRe = regexp.MustCompile(`step-(\d+)$`)

// checkpointStep: the step of a checkpoint folder (RUN/checkpoints/step-000500 -> 500), -1 when it is not one.
func checkpointStep(dir string) int {
	m := checkpointDirRe.FindStringSubmatch(filepath.Base(filepath.Clean(dir)))
	if m == nil {
		return -1
	}
	n, _ := strconv.Atoi(m[1])
	return n
}

// latestCheckpointStep: the highest resumable checkpoint (one with optimizer state) of a run, -1 when none.
func latestCheckpointStep(runDir string) int {
	best := -1
	dirs, _ := filepath.Glob(filepath.Join(runDir, "checkpoints", "step-*"))
	for _, d := range dirs {
		if _, err := os.Stat(filepath.Join(d, "state.safetensors")); err == nil {
			if s := checkpointStep(d); s > best {
				best = s
			}
		}
	}
	return best
}

// readJobMarker: the id of the job that last trained in a run folder (its .job file).
func readJobMarker(runDir string) string {
	b, err := os.ReadFile(filepath.Join(runDir, ".job"))
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(b))
}

// lokiTuneBin finds the engine: $LOKI_TUNE, else loki-tune on PATH.
var lokiTuneBin = func() (string, error) {
	if p := os.Getenv("LOKI_TUNE"); p != "" {
		return p, nil
	}
	p, err := exec.LookPath("loki-tune")
	if err != nil {
		return "", fmt.Errorf("loki-tune is not installed (put loki-tune on PATH, next to the other loki- engines)")
	}
	return p, nil
}

func trainLoraTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	ctx := j.Ctx
	const tag = "train-lora"
	defer beginGPUHeavy()()
	fail := func(msg string, err error) error {
		q.PushJobStdout(j.ID, tag+": "+msg)
		q.ErrorJob(j.ID)
		if err == nil {
			err = fmt.Errorf("%s: %s", tag, msg)
		}
		return err
	}
	p := trainLoraParamsFromOptions(ParseOptions(j, trainLoraOptions))
	bin, err := lokiTuneBin()
	if err != nil {
		return fail(err.Error(), err)
	}

	// ---- which run: a resume, the run this job started before a pause, or a new one
	var runDir, resumeAt, list string
	if p.Resume != "" {
		resumeAt = p.Resume
		if !filepath.IsAbs(resumeAt) {
			resumeAt = filepath.Join(loraRunsDir(p), resumeAt)
		}
		runDir = resumeAt
		if _, err := os.Stat(filepath.Join(resumeAt, "state.safetensors")); err == nil {
			runDir = filepath.Dir(filepath.Dir(resumeAt))
			// an earlier checkpoint than the run's last: loki-tune branches into <run>-from<step>
			if step, latest := checkpointStep(resumeAt), latestCheckpointStep(runDir); step >= 0 && step < latest {
				runDir = fmt.Sprintf("%s-from%d", runDir, step)
			}
		}
		// this job ran before and was paused: continue from where it stopped, not from the checkpoint it started at
		if readJobMarker(runDir) == j.ID && latestCheckpointStep(runDir) >= 0 {
			resumeAt = runDir
		}
	} else {
		base := filepath.Join(loraRunsDir(p), loraRunName(p))
		runDir = base
		for i := 2; ; i++ {
			if _, err := os.Stat(filepath.Join(runDir, "run.json")); err != nil {
				break
			}
			if runJobMeta(runDir) == j.ID {
				resumeAt = runDir // this job's own run, paused earlier
				break
			}
			runDir = fmt.Sprintf("%s-%d", base, i)
		}
	}
	if resumeAt == "" {
		items, err := resolveJobItems(j, q)
		if err != nil {
			return fail("failed to resolve the images: "+err.Error(), err)
		}
		var images []string
		for _, path := range items.Paths {
			if loraTrainImageExts[strings.ToLower(filepath.Ext(path))] {
				images = append(images, path)
			}
		}
		if len(images) == 0 {
			return fail("no images to train on (videos and other files are skipped)", nil)
		}
		if err := os.MkdirAll(runDir, 0o755); err != nil {
			return fail("failed to create "+runDir+": "+err.Error(), err)
		}
		list = filepath.Join(runDir, "images.jsonl")
		n, err := loraTrainList(q.Db, images, p.Captions == "library", list)
		if err != nil {
			return fail("failed to write the image list: "+err.Error(), err)
		}
		q.PushJobStdout(j.ID, fmt.Sprintf("%s: %d image(s) -> %s (%s LoRA%s)", tag, len(images), runDir, p.Kind, map[bool]string{true: fmt.Sprintf(", %d library caption(s)", n), false: ""}[p.Captions == "library"]))
	} else {
		q.PushJobStdout(j.ID, fmt.Sprintf("%s: continuing %s", tag, resumeAt))
	}
	if err := os.MkdirAll(runDir, 0o755); err == nil {
		_ = os.WriteFile(filepath.Join(runDir, ".job"), []byte(j.ID), 0o644)
	}
	meta, _ := json.Marshal(map[string]string{"job": j.ID})
	args := buildTrainLoraArgs(p, list, runDir, resumeAt, string(meta))

	select {
	case <-ctx.Done():
		return fail("task canceled", ctx.Err())
	default:
	}
	if q.PauseRequested(j.ID) {
		q.PushJobStdout(j.ID, tag+": paused before start - resume to continue")
		return jobqueue.ErrPaused
	}

	// The engine gets its own context: a pause or cancel first asks it for a checkpoint (STOP file), and the process
	// is only killed when it has not exited a few minutes later.
	procCtx, killProc := context.WithCancel(context.Background())
	defer killProc()
	done := make(chan struct{})
	defer close(done)
	var stopping, paused bool
	var smu sync.Mutex
	go func() {
		tick := time.NewTicker(2 * time.Second)
		defer tick.Stop()
		requestStop := func(pause bool) {
			smu.Lock()
			defer smu.Unlock()
			if stopping {
				return
			}
			stopping, paused = true, pause
			_ = os.WriteFile(filepath.Join(runDir, "STOP"), []byte("stop\n"), 0o644)
			q.PushJobStdout(j.ID, tag+": stopping after the current step (saving a checkpoint)")
			go func() {
				select {
				case <-done:
				case <-time.After(5 * time.Minute):
					killProc()
				}
			}()
		}
		for {
			select {
			case <-done:
				return
			case <-ctx.Done():
				requestStop(false)
			case <-tick.C:
				if q.PauseRequested(j.ID) {
					requestStop(true)
				}
			}
		}
	}()

	_ = q.SetJobProgress(j.ID, 0, 1)
	tool := aiToolRun{Bin: bin}
	out, waitErr := runAICLI(procCtx, q, j.ID, tag, tool, args, func(i, n int) {
		_ = q.SetJobProgress(j.ID, i, n)
	})
	smu.Lock()
	wasStopping, wasPaused := stopping, paused
	smu.Unlock()
	var lora string
	for _, l := range out {
		if strings.HasSuffix(strings.TrimSpace(l), ".safetensors") {
			lora = strings.TrimSpace(l)
		}
	}
	if wasStopping {
		if lora != "" {
			q.PushJobStdout(j.ID, tag+": checkpoint "+lora)
		}
		if wasPaused && ctx.Err() == nil {
			q.PushJobStdout(j.ID, tag+": paused - resume the job to continue training")
			return jobqueue.ErrPaused
		}
		return fail("task canceled; continue later with the resume option: "+filepath.Base(runDir), ctx.Err())
	}
	if waitErr != nil {
		return fail("failed: "+waitErr.Error(), waitErr)
	}
	if lora == "" {
		return fail("loki-tune wrote no LoRA", nil)
	}
	q.PushJobStdout(j.ID, tag+": completed -> "+lora)
	q.RegisterOutputFile(j.ID, lora)
	q.CompleteJob(j.ID)
	return nil
}
