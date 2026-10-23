package tasks

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	_ "image/png"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/stevecastle/shrike/deps"
	"github.com/stevecastle/shrike/heic"
	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/mediaext"
	"github.com/stevecastle/shrike/platform"
)

// --- scenes (the Scene editor) ---
//
// A scene is a layered composition built over one still: a base frame plus
// layers that each replace a masked part of it, either with a still edit
// (loki-retouch on a crop) or with motion (loki-reshoot animating a crop).
// The engines take no inpainting masks, so every layer works the same way:
// crop its region (the mask plus context around it) from the flattened
// stills below it, run the engine on the crop, and composite the result back
// through the layer's feathered mask. Build a scene up one region at a time
// until the whole frame moves.
//
// The Scene editor window (src/renderer/components/scene/) draws the masks
// and owns the document's editable fields; this file owns the files: each
// scene is a folder under <data>/scenes/<id>/ holding scene.json, the base
// frame, the crops sent to the engines and their results. Generations are
// ordinary retouch / reshoot jobs (visible on the Jobs page) run with
// `nolibrary`, so scratch crops never reach the library; reconcileScene folds
// finished jobs into their layer's versions whenever the scene is read.
// Exports (sceneRenderTask) are the only files written beside the source.

const (
	SceneKindStill  = "still"
	SceneKindMotion = "motion"

	sceneFPS = 24
	// sceneMaskMax caps the long edge of the full-frame mask PNGs the editor
	// sends; ffmpeg scales them back up to the frame.
	sceneMaskMax = 2048
)

// SceneRect is a rectangle in base-frame pixels.
type SceneRect struct {
	X int `json:"x"`
	Y int `json:"y"`
	W int `json:"w"`
	H int `json:"h"`
}

func (r SceneRect) empty() bool { return r.W <= 0 || r.H <= 0 }

// clamp limits r to a w×h frame, keeping at least 16 px each way.
func (r SceneRect) clamp(w, h int) SceneRect {
	if r.X < 0 {
		r.W += r.X
		r.X = 0
	}
	if r.Y < 0 {
		r.H += r.Y
		r.Y = 0
	}
	if r.X > w-16 {
		r.X = max(0, w-16)
	}
	if r.Y > h-16 {
		r.Y = max(0, h-16)
	}
	if r.X+r.W > w {
		r.W = w - r.X
	}
	if r.Y+r.H > h {
		r.H = h - r.Y
	}
	r.W = max(r.W, min(16, w))
	r.H = max(r.H, min(16, h))
	return r
}

// SceneVersion is one generation of a layer: the file the engine produced
// and the region it was generated for (the layer's mask may change later;
// the result stays where it was made).
type SceneVersion struct {
	Path     string    `json:"path"`
	Region   SceneRect `json:"region"`
	Prompt   string    `json:"prompt"`
	Duration float64   `json:"duration,omitempty"` // motion: seconds
	Audio    bool      `json:"audio,omitempty"`    // motion: has a soundtrack
	Note     string    `json:"note,omitempty"`     // e.g. "boomerang of v2"
	// Raw is the engine's own output a finished (scene-lock) version was made
	// from, so the finish can be redone without regenerating (RefinishVersion).
	Raw     string    `json:"raw,omitempty"`
	JobID   string    `json:"jobId,omitempty"`
	Created time.Time `json:"created"`
}

// SceneJob is a layer's generation in flight (server-owned).
type SceneJob struct {
	ID     string    `json:"id"`
	State  string    `json:"state"`
	Done   int       `json:"done"`
	Total  int       `json:"total"`
	Region SceneRect `json:"region"`
	Prompt string    `json:"prompt"`
	Input  string    `json:"input"`
	// Chain is every job of the generation in order (motion: reshoot, then
	// the scene-lock pass); ID is the last one, whose output is the result.
	Chain []string `json:"chain,omitempty"`
	// Requeued counts later steps queued again after being cancelled before
	// they ran (see reconcileLayerJob).
	Requeued int `json:"requeued,omitempty"`
	// Step is the command of the step running now (e.g. reshoot, scene-lock).
	Step string `json:"step,omitempty"`
	// Note is given to the version this job makes (e.g. "re-finished v2").
	Note string `json:"note,omitempty"`
	// Raw: the engine output this job finishes (a re-finish), kept on the version.
	Raw string `json:"raw,omitempty"`
	// Line is the job's latest log line, for the editor's status text.
	Line string `json:"line,omitempty"`
}

// SceneLayer is one masked edit. Kind still = an edited image region; motion
// = an animated one. The editor owns Name/Hidden/Region/Mask/Shapes/Feather/
// Prompt/Options/Active; Versions, Job and Error are server-owned.
type SceneLayer struct {
	ID     string    `json:"id"`
	Name   string    `json:"name"`
	Kind   string    `json:"kind"`
	Hidden bool      `json:"hidden,omitempty"`
	Region SceneRect `json:"region"` // what the next generation crops
	// Mask is a full-frame PNG data URL (long edge <= sceneMaskMax) whose
	// alpha is the layer's coverage, feather included.
	Mask    string          `json:"mask"`
	Shapes  json.RawMessage `json:"shapes,omitempty"` // the editor's strokes; opaque here
	Feather float64         `json:"feather"`
	Prompt  string          `json:"prompt"`
	Options map[string]any  `json:"options,omitempty"`

	Versions []SceneVersion `json:"versions"`
	Active   int            `json:"active"` // index into Versions; -1 = none shown
	Job      *SceneJob      `json:"job,omitempty"`
	Error    string         `json:"error,omitempty"`
}

func (l *SceneLayer) active() *SceneVersion {
	if l.Active < 0 || l.Active >= len(l.Versions) {
		return nil
	}
	return &l.Versions[l.Active]
}

// SceneRender is the latest export (server-owned).
type SceneRender struct {
	JobID  string `json:"jobId"`
	Format string `json:"format"`
	State  string `json:"state"`
	Done   int    `json:"done"`
	Total  int    `json:"total"`
	Output string `json:"output,omitempty"`
	Error  string `json:"error,omitempty"`
}

type Scene struct {
	ID         string        `json:"id"`
	Name       string        `json:"name"`
	Source     string        `json:"source"`
	SourceTime float64       `json:"sourceTime,omitempty"`
	Base       string        `json:"base"`
	Width      int           `json:"width"`
	Height     int           `json:"height"`
	Layers     []*SceneLayer `json:"layers"` // bottom to top
	// Duration of an export in seconds; 0 = the longest motion layer.
	Duration float64 `json:"duration"`
	// FreeLoops: each motion layer loops at its own length. Off (the
	// default), every layer is retimed to fit a whole number of its loops into
	// the scene length (loopSpeed), so the export itself loops seamlessly.
	FreeLoops bool         `json:"freeLoops,omitempty"`
	Render    *SceneRender `json:"render,omitempty"`
	Created   time.Time    `json:"created"`
	Updated   time.Time    `json:"updated"`
}

// SceneSummary is a scene in a list (no masks).
type SceneSummary struct {
	ID      string    `json:"id"`
	Name    string    `json:"name"`
	Source  string    `json:"source"`
	Base    string    `json:"base"`
	Layers  int       `json:"layers"`
	Motion  int       `json:"motion"`
	Updated time.Time `json:"updated"`
}

var (
	ErrSceneNotFound = errors.New("scene not found")
	ErrLayerNotFound = errors.New("layer not found")
	// ErrSceneBusy: the layer already has a generation in flight.
	ErrSceneBusy = errors.New("layer is already generating")
)

// sceneMu serialises every load-modify-save of scene documents: edits are
// rare and small, and one lock keeps a PUT from racing a reconcile.
var sceneMu sync.Mutex

var sceneIDRe = regexp.MustCompile(`^[a-f0-9-]{8,64}$`)

// sceneRootOverride lets tests keep scenes in a temp dir.
var sceneRootOverride string

func sceneRoot() string {
	if sceneRootOverride != "" {
		return sceneRootOverride
	}
	return filepath.Join(platform.GetDataDir(), "scenes")
}

func sceneDir(id string) (string, error) {
	if !sceneIDRe.MatchString(id) {
		return "", ErrSceneNotFound
	}
	return filepath.Join(sceneRoot(), id), nil
}

// inSceneDir reports whether p is a file inside the scene's folder.
func inSceneDir(id, p string) bool {
	dir, err := sceneDir(id)
	if err != nil || p == "" {
		return false
	}
	rel, err := filepath.Rel(dir, filepath.Clean(p))
	return err == nil && rel != "." && !strings.HasPrefix(rel, "..") && !filepath.IsAbs(rel)
}

func loadSceneLocked(id string) (*Scene, error) {
	dir, err := sceneDir(id)
	if err != nil {
		return nil, err
	}
	b, err := os.ReadFile(filepath.Join(dir, "scene.json"))
	if errors.Is(err, os.ErrNotExist) {
		return nil, ErrSceneNotFound
	}
	if err != nil {
		return nil, err
	}
	var s Scene
	if err := json.Unmarshal(b, &s); err != nil {
		return nil, fmt.Errorf("scene.json: %w", err)
	}
	if s.Layers == nil {
		s.Layers = []*SceneLayer{}
	}
	for _, l := range s.Layers {
		if l.Versions == nil {
			l.Versions = []SceneVersion{}
		}
	}
	return &s, nil
}

// saveSceneLocked writes scene.json atomically (temp file + rename).
func saveSceneLocked(s *Scene) error {
	dir, err := sceneDir(s.ID)
	if err != nil {
		return err
	}
	s.Updated = time.Now().UTC()
	b, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return err
	}
	tmp := filepath.Join(dir, fmt.Sprintf("scene.json.%d.tmp", time.Now().UnixNano()))
	if err := os.WriteFile(tmp, b, 0644); err != nil {
		return err
	}
	if err := os.Rename(tmp, filepath.Join(dir, "scene.json")); err != nil {
		os.Remove(tmp)
		return err
	}
	return nil
}

// runSceneFFmpeg runs the bundled ffmpeg quietly, returning its stderr in the
// error.
func runSceneFFmpeg(ctx context.Context, args []string) error {
	full := append([]string{"-hide_banner", "-loglevel", "error", "-y"}, args...)
	cmd := exec.CommandContext(ctx, deps.MustBundled("ffmpeg"), full...)
	platform.HideSubprocessWindow(cmd)
	out, err := cmd.CombinedOutput()
	if err != nil {
		msg := strings.TrimSpace(string(out))
		if len(msg) > 800 {
			msg = msg[len(msg)-800:]
		}
		return fmt.Errorf("ffmpeg: %v: %s", err, msg)
	}
	return nil
}

func pngSize(path string) (int, int, error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, 0, err
	}
	defer f.Close()
	cfg, _, err := image.DecodeConfig(f)
	if err != nil {
		return 0, 0, err
	}
	return cfg.Width, cfg.Height, nil
}

// CreateScene starts a scene over source: a still image, or the frame at `at`
// seconds of a video. The base frame is normalised to PNG once, so every
// later step reads one known format at known dimensions.
func CreateScene(ctx context.Context, source string, at float64) (*Scene, error) {
	source = absPath(source)
	info, err := os.Stat(source)
	if err != nil || info.IsDir() {
		return nil, fmt.Errorf("source not found: %s", source)
	}
	isVideo := mediaext.IsVideo(source)
	if !isVideo && !mediaext.IsImage(source) {
		return nil, fmt.Errorf("a scene starts from an image or a video frame")
	}
	id := uuid.NewString()
	dir, _ := sceneDir(id)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, err
	}
	ok := false
	defer func() {
		if !ok {
			os.RemoveAll(dir)
		}
	}()

	base := filepath.Join(dir, "base.png")
	switch {
	case isVideo:
		if at < 0 {
			at = 0
		}
		if err := runSceneFFmpeg(ctx, []string{"-ss", fmt.Sprintf("%.3f", at), "-i", source, "-frames:v", "1", base}); err != nil {
			return nil, err
		}
	case heic.IsHeic(source):
		// HEIC tile grids need the display renderer (see heic/display.go).
		jpg, rerr := heic.Render(deps.MustBundled("ffmpeg"), source)
		if rerr != nil {
			return nil, rerr
		}
		tmp := filepath.Join(dir, "source.jpg")
		if err := os.WriteFile(tmp, jpg, 0644); err != nil {
			return nil, err
		}
		err := runSceneFFmpeg(ctx, []string{"-i", tmp, "-frames:v", "1", base})
		os.Remove(tmp)
		if err != nil {
			return nil, err
		}
	default:
		at = 0
		if err := runSceneFFmpeg(ctx, []string{"-i", source, "-frames:v", "1", base}); err != nil {
			return nil, err
		}
	}
	w, h, err := pngSize(base)
	if err != nil {
		return nil, fmt.Errorf("base frame: %w", err)
	}
	now := time.Now().UTC()
	name := strings.TrimSuffix(filepath.Base(source), filepath.Ext(source))
	s := &Scene{
		ID: id, Name: name, Source: source, SourceTime: at, Base: base,
		Width: w, Height: h, Layers: []*SceneLayer{}, Created: now,
	}
	sceneMu.Lock()
	defer sceneMu.Unlock()
	if err := saveSceneLocked(s); err != nil {
		return nil, err
	}
	ok = true
	return s, nil
}

// ListScenes returns the scenes started from source (every scene when source
// is empty), most recently changed first.
func ListScenes(source string) []SceneSummary {
	entries, err := os.ReadDir(sceneRoot())
	if err != nil {
		return []SceneSummary{}
	}
	want := ""
	if source != "" {
		want = strings.ToLower(absPath(source))
	}
	out := []SceneSummary{}
	sceneMu.Lock()
	defer sceneMu.Unlock()
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		s, err := loadSceneLocked(e.Name())
		if err != nil {
			continue
		}
		if want != "" && strings.ToLower(s.Source) != want {
			continue
		}
		sum := SceneSummary{ID: s.ID, Name: s.Name, Source: s.Source, Base: s.Base, Layers: len(s.Layers), Updated: s.Updated}
		for _, l := range s.Layers {
			if l.Kind == SceneKindMotion {
				sum.Motion++
			}
		}
		out = append(out, sum)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Updated.After(out[j].Updated) })
	return out
}

// GetScene loads a scene, folding any finished generations into it first.
func GetScene(q *jobqueue.Queue, id string) (*Scene, error) {
	sceneMu.Lock()
	defer sceneMu.Unlock()
	s, err := loadSceneLocked(id)
	if err != nil {
		return nil, err
	}
	if reconcileScene(q, s) {
		if err := saveSceneLocked(s); err != nil {
			return nil, err
		}
	}
	return s, nil
}

func DeleteScene(q *jobqueue.Queue, id string) error {
	sceneMu.Lock()
	defer sceneMu.Unlock()
	s, err := loadSceneLocked(id)
	if err != nil {
		return err
	}
	for _, l := range s.Layers {
		cancelSceneJob(q, l.Job, "the scene was deleted")
	}
	dir, _ := sceneDir(id)
	return os.RemoveAll(dir)
}

// UpdateScene applies the editor's copy of the document. Only editor-owned
// fields are taken; a layer's versions, job and error stay the server's, so
// an edit can never drop a generation that finished meanwhile. Layers absent
// from the update are deleted (their running jobs cancelled).
func UpdateScene(q *jobqueue.Queue, id string, in *Scene) (*Scene, error) {
	sceneMu.Lock()
	defer sceneMu.Unlock()
	s, err := loadSceneLocked(id)
	if err != nil {
		return nil, err
	}
	reconcileScene(q, s)
	if name := strings.TrimSpace(in.Name); name != "" {
		s.Name = name
	}
	s.Duration = clampFloat(in.Duration, 0, 60)
	s.FreeLoops = in.FreeLoops

	old := map[string]*SceneLayer{}
	for _, l := range s.Layers {
		old[l.ID] = l
	}
	seen := map[string]bool{}
	next := make([]*SceneLayer, 0, len(in.Layers))
	for _, nl := range in.Layers {
		if nl == nil || nl.ID == "" || seen[nl.ID] || len(nl.ID) > 64 {
			continue
		}
		seen[nl.ID] = true
		l := old[nl.ID]
		if l == nil {
			l = &SceneLayer{ID: nl.ID, Versions: []SceneVersion{}, Active: -1}
		}
		l.Name = strings.TrimSpace(nl.Name)
		l.Kind = nl.Kind
		if l.Kind != SceneKindMotion {
			l.Kind = SceneKindStill
		}
		l.Hidden = nl.Hidden
		l.Region = nl.Region.clamp(s.Width, s.Height)
		if nl.Mask == "" || strings.HasPrefix(nl.Mask, "data:image/png;base64,") {
			l.Mask = nl.Mask
		}
		l.Shapes = nl.Shapes
		l.Feather = clampFloat(nl.Feather, 0, 512)
		l.Prompt = nl.Prompt
		l.Options = nl.Options
		if nl.Active >= -1 && nl.Active < len(l.Versions) {
			l.Active = nl.Active
		}
		next = append(next, l)
	}
	// A deleted layer takes its generations (and their crops) with it.
	for oid, l := range old {
		if seen[oid] {
			continue
		}
		cancelSceneJob(q, l.Job, fmt.Sprintf("layer %q was deleted in the Scene editor", l.Name))
		for _, v := range l.Versions {
			if inSceneDir(id, v.Path) {
				os.Remove(v.Path)
			}
		}
		if dir, err := sceneDir(id); err == nil {
			matches, _ := filepath.Glob(filepath.Join(dir, "*-"+safeLayerID(oid)+"-*"))
			for _, m := range matches {
				os.Remove(m)
			}
			os.Remove(filepath.Join(dir, "mask-"+safeLayerID(oid)+".png"))
			os.Remove(filepath.Join(dir, "voice-"+safeLayerID(oid)+".wav"))
		}
	}
	s.Layers = next
	if err := saveSceneLocked(s); err != nil {
		return nil, err
	}
	return s, nil
}

func jobStateName(st jobqueue.JobState) string {
	b, _ := st.MarshalJSON()
	return strings.Trim(string(b), `"`)
}

// reconcileScene folds finished generation / render jobs into the document.
// Returns whether anything changed.
func reconcileScene(q *jobqueue.Queue, s *Scene) bool {
	if q == nil {
		return false
	}
	changed := false
	for _, l := range s.Layers {
		if l.Job != nil && reconcileLayerJob(q, s, l) {
			changed = true
		}
	}
	if r := s.Render; r != nil && (r.State == "pending" || r.State == "in_progress" || r.State == "paused") {
		job, lines, ok := q.JobSnapshot(r.JobID, 3)
		switch {
		case !ok:
			r.State, r.Error = "error", "the render job was removed from the queue"
		case job.State == jobqueue.StateCompleted:
			r.State = "completed"
			if len(job.OutputFiles) > 0 {
				r.Output = job.OutputFiles[len(job.OutputFiles)-1]
			}
		case job.State == jobqueue.StateError || job.State == jobqueue.StateCancelled:
			r.State = jobStateName(job.State)
			if len(lines) > 0 {
				r.Error = lastMeaningfulLine(lines)
			}
		default:
			r.State, r.Done, r.Total = jobStateName(job.State), job.ProgressDone, job.ProgressTotal
		}
		changed = true
	}
	return changed
}

// sceneMaxRequeue bounds how often a generation's later step is queued again.
const sceneMaxRequeue = 2

// ids is every job of a generation, in order.
func (j *SceneJob) ids() []string {
	if len(j.Chain) > 0 {
		return j.Chain
	}
	return []string{j.ID}
}

// cancelSceneJob cancels every still-live step of a generation, noting why in
// each job's log (so a cancelled step on the Jobs page says who cancelled it).
// A chain is cancelled whole: a step left behind would keep the GPU busy for
// a result nothing is waiting for.
func cancelSceneJob(q *jobqueue.Queue, j *SceneJob, why string) {
	if q == nil || j == nil {
		return
	}
	for _, id := range j.ids() {
		job, _, ok := q.JobSnapshot(id, 0)
		if !ok || (job.State != jobqueue.StatePending && job.State != jobqueue.StateInProgress && job.State != jobqueue.StatePaused) {
			continue
		}
		_ = q.PushJobStdout(id, "scene: cancelled because "+why)
		_ = q.CancelJob(id)
	}
}

// reconcileLayerJob folds a layer's generation into it when it has ended, or
// copies its progress while it runs. Returns whether the layer changed.
//
// A generation is a chain (motion: reshoot -> scene-lock). Its state is the
// chain's, never one step's: while it runs, the first unfinished step's
// progress shows. If the engine's step fails or is cancelled, the steps
// still alive are cancelled too and the layer reports it. A later step that
// was cancelled before it ever ran is queued again once the engine is done
// (the engine's work is the expensive part; up to sceneMaxRequeue times). A
// later step that ran and failed keeps the engine's result as a version that
// says its edges are not locked, rather than throwing it away.
func reconcileLayerJob(q *jobqueue.Queue, s *Scene, l *SceneLayer) bool {
	ids := l.Job.ids()
	type step struct {
		job   jobqueue.Job
		lines []string
		ok    bool
	}
	steps := make([]step, len(ids))
	for i, id := range ids {
		steps[i].job, steps[i].lines, steps[i].ok = q.JobSnapshot(id, 4)
	}
	last := len(steps) - 1

	outputOf := func(st step) string {
		out := ""
		for _, f := range st.job.OutputFiles {
			if inSceneDir(s.ID, f) {
				out = f
			}
		}
		return out
	}
	addVersion := func(out, jobID, note string) {
		if note == "" {
			note = l.Job.Note
		}
		v := SceneVersion{Path: out, Region: l.Job.Region, Prompt: l.Job.Prompt, JobID: jobID, Note: note, Raw: l.Job.Raw, Created: time.Now().UTC()}
		// A chain's first step is the engine: its output is what was finished.
		if v.Raw == "" && len(steps) > 1 && steps[0].ok {
			if raw := outputOf(steps[0]); raw != "" && raw != out {
				v.Raw = raw
			}
		}
		if l.Kind == SceneKindMotion && deps.BundledOrEmpty("ffprobe") != "" {
			v.Duration = probeVideoDuration(context.Background(), out)
			v.Audio = probeHasAudio(out)
		}
		l.Versions = append(l.Versions, v)
		l.Active = len(l.Versions) - 1
		l.Error = ""
	}

	// Cleared from the queue before the scene looked: the result (if any) is
	// still where the engine writes it, beside the crop.
	if !steps[last].ok {
		if out := sceneJobOutput(l.Kind, l.Job.Input); out != "" {
			addVersion(out, l.Job.ID, "")
		} else {
			l.Error = "generation lost: the job was removed from the queue"
		}
		cancelSceneJob(q, l.Job, "its generation was removed from the queue")
		l.Job = nil
		return true
	}

	// The step that ended the chain early, if any.
	broken := -1
	for i, st := range steps {
		if st.ok && (st.job.State == jobqueue.StateError || st.job.State == jobqueue.StateCancelled) {
			broken = i
			break
		}
	}
	// A later step cancelled before it ran: not the user's intent to drop the
	// generation (the editor's Cancel clears the layer's job itself).
	if broken > 0 && steps[broken].job.State == jobqueue.StateCancelled && steps[broken].job.ClaimedAt.IsZero() && l.Job.Requeued < sceneMaxRequeue {
		gen := steps[0]
		switch {
		case gen.ok && (gen.job.State == jobqueue.StatePending || gen.job.State == jobqueue.StateInProgress || gen.job.State == jobqueue.StatePaused):
			// Keep following the engine; the step is queued again when it ends.
			broken = -1
			if l.Job.State != jobStateName(gen.job.State) || l.Job.Done != gen.job.ProgressDone {
				l.Job.State, l.Job.Done, l.Job.Total, l.Job.Step = jobStateName(gen.job.State), gen.job.ProgressDone, gen.job.ProgressTotal, gen.job.Command
				if len(gen.lines) > 0 {
					l.Job.Line = gen.lines[len(gen.lines)-1]
				}
				return true
			}
			return false
		case gen.ok && gen.job.State == jobqueue.StateCompleted && broken == last && outputOf(gen) != "":
			lost := steps[broken].job
			ids, err := q.AddWorkflow(jobqueue.Workflow{Tasks: []jobqueue.WorkflowTask{{Command: lost.Command, Arguments: lost.Arguments, Input: outputOf(gen)}}})
			if err == nil && len(ids) == 1 {
				_ = q.PushJobStdout(ids[0], fmt.Sprintf("scene: queued again: step %s was cancelled before it ran", lost.ID))
				l.Job.Chain = []string{gen.job.ID, ids[0]}
				l.Job.ID = ids[0]
				l.Job.Requeued++
				l.Job.State, l.Job.Done, l.Job.Total, l.Job.Line, l.Job.Step = "pending", 0, 0, "", lost.Command
				return true
			}
		}
	}
	if broken >= 0 {
		st := steps[broken]
		cancelSceneJob(q, l.Job, fmt.Sprintf("step %s of the same generation ended (%s)", st.job.Command, jobStateName(st.job.State)))
		if broken > 0 && steps[0].ok && steps[0].job.State == jobqueue.StateCompleted {
			if out := outputOf(steps[0]); out != "" {
				addVersion(out, steps[0].job.ID, fmt.Sprintf("edges not locked: the %s step was %s", st.job.Command, jobStateName(st.job.State)))
				l.Job = nil
				return true
			}
		}
		msg := "generation failed"
		if st.job.State == jobqueue.StateCancelled {
			msg = "generation cancelled"
		}
		if len(ids) > 1 {
			msg += " (" + st.job.Command + " step)"
		}
		if len(st.lines) > 0 {
			msg += ": " + lastMeaningfulLine(st.lines)
		}
		l.Error = msg
		l.Job = nil
		return true
	}

	if steps[last].job.State == jobqueue.StateCompleted {
		if out := outputOf(steps[last]); out != "" {
			addVersion(out, steps[last].job.ID, "")
		} else {
			l.Error = "the engine finished without a result"
		}
		l.Job = nil
		return true
	}

	// Running: report the first unfinished step.
	cur := steps[last]
	for _, st := range steps {
		if st.ok && st.job.State != jobqueue.StateCompleted {
			cur = st
			break
		}
	}
	state := jobStateName(cur.job.State)
	line := ""
	if len(cur.lines) > 0 {
		line = cur.lines[len(cur.lines)-1]
	}
	if l.Job.State == state && l.Job.Done == cur.job.ProgressDone && l.Job.Total == cur.job.ProgressTotal && l.Job.Line == line && l.Job.Step == cur.job.Command {
		return false
	}
	l.Job.State, l.Job.Done, l.Job.Total, l.Job.Line, l.Job.Step = state, cur.job.ProgressDone, cur.job.ProgressTotal, line, cur.job.Command
	return true
}

// sceneJobOutput is where retouch / reshoot put the result for a crop (see
// retouchSuffix and reshootTask's naming), or "" when there is none.
func sceneJobOutput(kind, input string) string {
	if input == "" {
		return ""
	}
	stem := strings.TrimSuffix(input, filepath.Ext(input))
	outs := []string{stem + "_edit.png"}
	if kind == SceneKindMotion {
		outs = []string{stem + "_locked.mp4", stem + "_reshoot.mp4"}
	}
	for _, out := range outs {
		if _, err := os.Stat(out); err == nil {
			return out
		}
	}
	return ""
}

func lastMeaningfulLine(lines []string) string {
	for i := len(lines) - 1; i >= 0; i-- {
		l := strings.TrimSpace(lines[i])
		if l != "" && !strings.Contains(l, "task canceled") {
			return l
		}
	}
	return strings.TrimSpace(lines[len(lines)-1])
}

func probeHasAudio(path string) bool {
	cmd := exec.Command(deps.MustBundled("ffprobe"), "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", path)
	platform.HideSubprocessWindow(cmd)
	out, err := cmd.Output()
	return err == nil && strings.TrimSpace(string(out)) != ""
}

// writeSceneMask decodes a layer's mask data URL into <dir>/mask-<layer>.png.
func writeSceneMask(dir string, l *SceneLayer) (string, error) {
	const prefix = "data:image/png;base64,"
	if !strings.HasPrefix(l.Mask, prefix) {
		return "", fmt.Errorf("layer %q has no mask", l.Name)
	}
	b, err := base64.StdEncoding.DecodeString(l.Mask[len(prefix):])
	if err != nil {
		return "", fmt.Errorf("layer %q: bad mask: %w", l.Name, err)
	}
	p := filepath.Join(dir, "mask-"+safeLayerID(l.ID)+".png")
	if err := os.WriteFile(p, b, 0644); err != nil {
		return "", err
	}
	return p, nil
}

var layerIDUnsafe = regexp.MustCompile(`[^A-Za-z0-9_-]`)

func safeLayerID(id string) string { return layerIDUnsafe.ReplaceAllString(id, "_") }

// sceneGraphLayer is one layer as the compositing graph sees it.
type sceneGraphLayer struct {
	Media  string
	Motion bool
	Audio  bool // mix its soundtrack (motion only)
	Region SceneRect
	Mask   string // mask PNG, full frame at any resolution; alpha = coverage
	// Duration of a motion clip, and Speed: its PTS multiplier (loopSpeed;
	// 0 or 1 = as generated).
	Duration float64
	Speed    float64
	// Offset (scene seconds) starts the loop at a different point; Volume
	// scales its soundtrack, 0 = as generated (motion only; see
	// sceneLayerTiming, which mutes a layer by leaving Audio off).
	Offset float64
	Volume float64
}

// sceneGraphArgs builds the ffmpeg inputs + filter graph compositing layers
// over base (w×h). video=false yields one frame (flatten); video=true loops
// stills and motion clips for dur seconds at sceneFPS. crop (non-empty)
// crops the result. Returns the args up to (not including) the output path.
func sceneGraphArgs(base string, w, h int, layers []sceneGraphLayer, video bool, dur float64, crop SceneRect) []string {
	var args []string
	still := func(p string) {
		if video {
			args = append(args, "-loop", "1", "-framerate", strconv.Itoa(sceneFPS))
		}
		args = append(args, "-i", p)
	}
	still(base)
	var f []string
	f = append(f, fmt.Sprintf("[0:v]scale=%d:%d,setsar=1,format=rgba[b0]", w, h))
	var audio []string
	in := 1
	for i, l := range layers {
		mi := in
		if l.Motion {
			if video {
				args = append(args, "-stream_loop", "-1")
				// The offset is a phase: where in its loop the clip is at
				// scene time 0, in the clip's own time.
				if ss := clipPhase(l); ss > 0.001 {
					args = append(args, "-ss", strconv.FormatFloat(ss, 'f', 4, 64))
				}
			}
			args = append(args, "-i", l.Media)
		} else {
			still(l.Media)
		}
		ki := in + 1
		still(l.Mask)
		in += 2
		r := l.Region
		retime := ""
		if l.Motion && video && l.Speed > 0 && math.Abs(l.Speed-1) > 1e-4 {
			retime = fmt.Sprintf("setpts=PTS*%.6f,", l.Speed)
		}
		f = append(f,
			fmt.Sprintf("[%d:v]%sscale=%d:%d:flags=lanczos,setsar=1,format=rgba[c%d]", mi, retime, r.W, r.H, i),
			fmt.Sprintf("[%d:v]format=rgba,alphaextract,scale=%d:%d,crop=%d:%d:%d:%d,setsar=1[a%d]", ki, w, h, r.W, r.H, r.X, r.Y, i),
			fmt.Sprintf("[c%d][a%d]alphamerge[l%d]", i, i, i),
			fmt.Sprintf("[b%d][l%d]overlay=%d:%d:format=auto:eof_action=repeat[b%d]", i, i, r.X, r.Y, i+1),
		)
		if video && l.Motion && l.Audio {
			var chain []string
			if retime != "" {
				chain = append(chain, atempoChain(1/l.Speed)...)
			}
			if l.Volume > 0 && math.Abs(l.Volume-1) > 1e-3 {
				chain = append(chain, fmt.Sprintf("volume=%.4f", l.Volume))
			}
			if len(chain) > 0 {
				f = append(f, fmt.Sprintf("[%d:a]%s[at%d]", mi, strings.Join(chain, ","), i))
				audio = append(audio, fmt.Sprintf("[at%d]", i))
			} else {
				audio = append(audio, fmt.Sprintf("[%d:a]", mi))
			}
		}
	}
	last := fmt.Sprintf("[b%d]", len(layers))
	tail := ""
	if !crop.empty() {
		tail = fmt.Sprintf("crop=%d:%d:%d:%d,", crop.W, crop.H, crop.X, crop.Y)
	}
	if video {
		f = append(f, last+tail+fmt.Sprintf("fps=%d,scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p[vout]", sceneFPS))
	} else {
		f = append(f, last+tail+"format=rgb24[vout]")
	}
	maps := []string{"-map", "[vout]"}
	if len(audio) == 1 {
		f = append(f, audio[0]+"anull[aout]")
		maps = append(maps, "-map", "[aout]")
	} else if len(audio) > 1 {
		f = append(f, strings.Join(audio, "")+fmt.Sprintf("amix=inputs=%d:duration=longest:normalize=0[aout]", len(audio)))
		maps = append(maps, "-map", "[aout]")
	}
	args = append(args, "-filter_complex", strings.Join(f, ";"))
	args = append(args, maps...)
	if video {
		args = append(args, "-t", formatFloatArg(dur), "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-movflags", "+faststart")
		if len(audio) > 0 {
			args = append(args, "-c:a", "aac", "-b:a", "192k")
		}
	} else {
		args = append(args, "-frames:v", "1")
	}
	return args
}

// sceneGraphLayers turns the visible, generated layers of s (all of them, or
// only stills when stillsOnly) into graph layers, writing their masks.
func sceneGraphLayers(dir string, layers []*SceneLayer, stillsOnly bool) ([]sceneGraphLayer, error) {
	var out []sceneGraphLayer
	for _, l := range layers {
		v := l.active()
		if l.Hidden || v == nil || (stillsOnly && l.Kind != SceneKindStill) {
			continue
		}
		if _, err := os.Stat(v.Path); err != nil {
			continue
		}
		mask, err := writeSceneMask(dir, l)
		if err != nil {
			return nil, err
		}
		g := sceneGraphLayer{Media: v.Path, Motion: l.Kind == SceneKindMotion, Region: v.Region, Mask: mask, Duration: v.Duration, Speed: 1}
		if g.Motion {
			g.Audio = v.Audio && sceneLayerSound(l.Options)
			t := sceneLayerTiming(l.Options)
			g.Speed, g.Offset, g.Volume = 1/t.Speed, t.Offset, t.Volume
			// Volume 0 mutes it (in the graph, 0 means "as generated").
			g.Audio = g.Audio && t.Volume > 0.001
		}
		out = append(out, g)
	}
	return out, nil
}

// sceneMotionSize picks loki-reshoot's --size for a region: the region's
// aspect at about mp megapixels, in multiples of 32 (the engine's grid).
func sceneMotionSize(r SceneRect, mp float64) (int, int) {
	if mp <= 0 {
		mp = 0.5
	}
	aspect := float64(r.W) / float64(r.H)
	aspect = clampFloat(aspect, 0.3, 3.4)
	w := math.Sqrt(mp * 1e6 * aspect)
	h := w / aspect
	snap := func(v float64) int { return clampInt(int(math.Round(v/32))*32, 256, 2048) }
	return snap(w), snap(h)
}

// sceneStillLongEdge picks loki-retouch's --long-edge for a crop: small
// regions are generated at >= 1024 px for detail (and scaled back down into
// the frame), huge ones capped at 2048; 0 = the crop's own size.
func sceneStillLongEdge(r SceneRect) int {
	long := max(r.W, r.H)
	switch {
	case long < 1024:
		return 1024
	case long > 2048:
		return 2048
	}
	return 0
}

// sceneMotionPrompt is the reshoot prompt for animating one region of a
// scene: <Picture 1> (the crop) is the first frame, the camera is locked off
// (the region is composited back into a still frame, so any camera move would
// swim against its surroundings), and only what is already in the crop moves,
// the way motion describes. Same sections as loki-reshoot's own templates.
//
// Sound (when generated) is the layer's own description if it has one, else
// the ambience of what moves; speech and music are ruled out unless the
// layer allows them.
type sceneMotionPromptInput struct {
	Motion, Subject string
	Sound           bool
	// SoundDesc: what it should sound like ("crackling fire, soft wind").
	SoundDesc string
	// Voices allows speech and music in the ambience (default: never).
	Voices bool
	// Lines are spoken in order (dialogue implies sound). VoiceRef: the
	// generation has <Audio 1>, a voice-timbre reference for the first speaker.
	Lines     []dialogueLine
	VoiceRef  bool
	LockEdges bool
	// Refs are the layer's own references, with the tokens the prompt uses
	// for them (sceneRefTokens).
	Refs []promptRef
}

// sceneRef is one reference a layer attaches to its generation (the crop
// itself is always the first: <image1> / <Picture 1>). Kind comes from the
// file's extension, never the client.
type sceneRef struct {
	Path string
	Kind string // image | video | audio
	Note string // what it is, in the user's words
}

// promptRef is a reference as the prompt template sees it.
type promptRef struct {
	Token, Kind, Note string
}

// Reference limits (besides the crop): the engines' own (loki-reshoot: 9
// images, 3 videos, 3 audio clips in all) and a sane cap for retouch.
const (
	sceneMaxStillRefs = 6
	sceneMaxRefImages = reshootMaxImages - 1
	sceneMaxRefVideos = reshootMaxVideos
	sceneMaxRefAudios = reshootMaxAudios
)

// sceneRefs reads a layer's references option (a JSON array of {path,
// note}); kinds come from the extensions and unknown files are dropped.
func sceneRefs(opts map[string]any) []sceneRef {
	raw, ok := opts["refs"].([]any)
	if !ok {
		return nil
	}
	var out []sceneRef
	for _, r := range raw {
		m, ok := r.(map[string]any)
		if !ok {
			continue
		}
		p := strings.TrimSpace(optString(m, "path"))
		kind := ""
		switch {
		case p == "":
		case mediaext.IsImage(p):
			kind = "image"
		case mediaext.IsVideo(p):
			kind = "video"
		case mediaext.IsAudio(p):
			kind = "audio"
		}
		if kind != "" {
			out = append(out, sceneRef{Path: p, Kind: kind, Note: strings.TrimSpace(optString(m, "note"))})
		}
	}
	return out
}

// sceneRefTokens is the token each reference answers to in a layer's prompt
// ("" = not usable by this kind of layer). Edit layers (loki-retouch) number
// images after the crop: <image2>, <image3>... Motion layers (loki-reshoot)
// number per kind: <Picture 2>... after the crop, <Video 1>..., and <Audio N>
// after the dialogue voice when there is one (it is <Audio 1>). Mirrored by
// refTokens in the editor (scene-api.ts).
func sceneRefTokens(layerKind string, refs []sceneRef, voice bool) []string {
	out := make([]string, len(refs))
	img, vid, aud := 2, 1, 1
	if voice {
		aud = 2
	}
	for i, r := range refs {
		switch {
		case layerKind != SceneKindMotion:
			if r.Kind == "image" {
				out[i] = fmt.Sprintf("<image%d>", img)
				img++
			}
		case r.Kind == "image":
			out[i] = fmt.Sprintf("<Picture %d>", img)
			img++
		case r.Kind == "video":
			out[i] = fmt.Sprintf("<Video %d>", vid)
			vid++
		case r.Kind == "audio":
			out[i] = fmt.Sprintf("<Audio %d>", aud)
			aud++
		}
	}
	return out
}

// checkSceneRefs validates a layer's references before a generation: the
// files exist, the kinds suit the layer and the engine's limits hold.
func checkSceneRefs(layerKind string, refs []sceneRef, voice bool) error {
	n := map[string]int{}
	for _, r := range refs {
		if _, err := os.Stat(r.Path); err != nil {
			return fmt.Errorf("reference not found: %s", filepath.Base(r.Path))
		}
		if layerKind != SceneKindMotion && r.Kind != "image" {
			return fmt.Errorf("an Edit layer takes image references only (%s)", filepath.Base(r.Path))
		}
		n[r.Kind]++
	}
	if voice {
		n["audio"]++
	}
	switch {
	case layerKind != SceneKindMotion && n["image"] > sceneMaxStillRefs:
		return fmt.Errorf("at most %d reference images", sceneMaxStillRefs)
	case n["image"] > sceneMaxRefImages:
		return fmt.Errorf("at most %d reference images", sceneMaxRefImages)
	case n["video"] > sceneMaxRefVideos:
		return fmt.Errorf("at most %d reference videos", sceneMaxRefVideos)
	case n["audio"] > sceneMaxRefAudios:
		return fmt.Errorf("at most %d audio references (the dialogue voice counts)", sceneMaxRefAudios)
	}
	return nil
}

// dialogueLine is one spoken line of a motion layer: who says it (empty =
// the person in the picture), in which language, and the words.
type dialogueLine struct {
	Speaker string `json:"speaker,omitempty"`
	Lang    string `json:"lang,omitempty"`
	Text    string `json:"text"`
}

// sceneDialogue reads a layer's dialogue option (a JSON array of lines),
// skipping empty lines.
func sceneDialogue(opts map[string]any) []dialogueLine {
	raw, ok := opts["dialogue"].([]any)
	if !ok {
		return nil
	}
	var out []dialogueLine
	for _, r := range raw {
		m, ok := r.(map[string]any)
		if !ok {
			continue
		}
		l := dialogueLine{Speaker: strings.TrimSpace(optString(m, "speaker")), Lang: strings.TrimSpace(optString(m, "lang")), Text: strings.TrimSpace(optString(m, "text"))}
		if l.Text != "" {
			out = append(out, l)
		}
	}
	return out
}

// sceneLayerSound reports whether a motion layer generates sound: asked for,
// or implied by dialogue.
func sceneLayerSound(opts map[string]any) bool {
	return optBool(opts, "sound") || len(sceneDialogue(opts)) > 0
}

// dialogueText renders lines in loki-reshoot's dialogue format: each voice a
// stable speaker ID (S1, S2... in order of first speech), the words in
// <d>[Language] ...</d> with closing punctuation. The default speaker (and a
// speaker named like the picture's person) is <Subject 1>, the person shown.
func dialogueText(lines []dialogueLine) string {
	ids := map[string]int{}
	var parts []string
	for i, l := range lines {
		who := strings.ToLower(l.Speaker)
		if who == "" {
			who = "the person in the picture"
		}
		id, seen := ids[who]
		if !seen {
			id = len(ids) + 1
			ids[who] = id
		}
		label := l.Speaker
		if label == "" || who == "the person in the picture" {
			label = "<Subject 1>"
		}
		lang := l.Lang
		if lang == "" {
			lang = "English"
		}
		text := strings.TrimSpace(l.Text)
		if !strings.ContainsAny(text[len(text)-1:], ".?!") {
			text += "."
		}
		verb := "says"
		if i > 0 {
			verb = "then says"
		}
		parts = append(parts, fmt.Sprintf("%s (S%d) %s, <d>[%s] %s</d>", label, id, verb, lang, text))
	}
	return strings.Join(parts, " ")
}

func sceneMotionPrompt(in sceneMotionPromptInput) string {
	motion := strings.TrimSpace(in.Motion)
	subject := strings.TrimSpace(in.Subject)
	talking := len(in.Lines) > 0
	sound, lockEdges := in.Sound || talking, in.LockEdges
	what := "the scene shown in <Picture 1>"
	if subject != "" {
		what = subject + ", as shown in <Picture 1>"
	}
	action := "Everything already visible moves naturally and subtly, the way it would in the real place: light shifts softly, and elements that would move on their own (water, clouds, foliage, smoke, fire, fabric) drift gently."
	if motion != "" {
		action = "The motion: " + strings.TrimRight(motion, ". ") + ". Only what is already visible in <Picture 1> moves; the rest of the frame stays as it is."
	}
	if lockEdges {
		// The crop's border is context around the mask: keeping the motion
		// away from it is what lets the region blend back into the still.
		action += " The motion stays in the middle of the frame and fades out towards its borders: everything near the edges of the frame stays perfectly still for the whole shot."
	}
	soundLine := "Silent."
	scape := "N/A"
	music := "N/A"
	if sound {
		desc := strings.TrimRight(strings.TrimSpace(in.SoundDesc), ". ")
		quiet := "; nobody speaks and there is no music."
		quietScape := "; no speech, no music."
		if in.Voices {
			quiet, quietScape = ".", "."
		} else if talking {
			// The lines are the only speech.
			quiet, quietScape = "; no other voices and no music.", "; no music."
		}
		if desc != "" {
			soundLine = "The sound: " + desc + ", realistic and present throughout" + quiet
			scape = desc + quietScape
			if in.Voices {
				music = "Only if the soundscape above calls for it."
			}
		} else {
			soundLine = "The sound is the natural, realistic ambience of exactly what moves in the shot, soft but present throughout" + quiet
			scape = "Natural ambience of the depicted elements and their motion" + quietScape
		}
	}
	defs := []string{"<Subject 1> is " + what + ", to be kept exactly as shown: its content, shapes, colors, lighting, texture, style and framing."}
	task := "[keyframe completion]"
	retain := []string{
		"<Subject 1> (appears in [Shot 1]): fully_preserved - content, appearance, style and framing are retained.",
		"<Picture 1> ([Shot 1] first frame): fully_preserved - the video begins from this exact composition and keeps its look.",
	}
	hasVisual, hasAudio := false, false
	for _, r := range in.Refs {
		note := strings.TrimRight(strings.TrimSpace(r.Note), ". ")
		switch r.Kind {
		case "image":
			hasVisual = true
			if note == "" {
				note = "used as the description says"
			}
			defs = append(defs, r.Token+" is a reference image: "+note+".")
			retain = append(retain, r.Token+": reference - only what the description takes from it appears.")
		case "video":
			hasVisual = true
			if note == "" {
				note = "its motion, style or content used as the description says"
			}
			defs = append(defs, r.Token+" is a reference video: "+note+".")
			retain = append(retain, r.Token+": reference - followed only as the description says.")
		case "audio":
			hasAudio = true
			if note == "" {
				note = "used as the description says"
			}
			defs = append(defs, r.Token+" is reference audio: "+note+".")
			retain = append(retain, r.Token+": reference - its sound is used only as the description says.")
		}
	}
	if hasVisual {
		task = strings.TrimSuffix(task, "]") + " + reference generation]"
	}
	if hasAudio && !strings.Contains(task, "audio reference") {
		task = strings.TrimSuffix(task, "]") + " + audio reference]"
	}
	unchanged := " Nothing new enters the frame: no people, faces, animals or objects appear that are not already in <Picture 1>, and nothing is replaced."
	if hasVisual {
		unchanged = " Apart from what the description takes from the references, nothing new enters the frame and nothing else is replaced."
	}
	speech := ""
	if talking {
		if in.VoiceRef {
			defs = append(defs, "<Audio 1> is the voice-timbre reference for the first speaker (S1).")
			if !strings.Contains(task, "audio reference") {
				task = strings.TrimSuffix(task, "]") + " + audio reference]"
			}
			retain = append(retain, "<Audio 1>: reference - the first speaker (S1) follows <Audio 1>'s voice timbre and delivery without copying the original signal or its words.")
		}
		speech = " " + dialogueText(in.Lines) + " Their lips, jaw and face move naturally and in sync with every word, with natural pauses; apart from speaking they keep their pose and expression."
	}
	return strings.Join([]string{
		"subject_definitions:",
		strings.Join(defs, "\n"),
		"",
		"summary:",
		task + " The target video is a faithful animation of <Picture 1>, one region of a larger still picture: one continuous shot that starts exactly from <Picture 1> and brings it to life. " + action + speech,
		"",
		"retention_analysis:",
		strings.Join(retain, "\n"),
		"",
		"detailed_description:",
		"The target video keeps the exact look, rendering style, colors, lighting and framing of <Picture 1>.",
		"[Shot 1] The shot begins exactly from <Picture 1> and stays one uninterrupted, unedited take with the same composition. " + action + speech + unchanged + " The edges of the frame stay continuous with what surrounds them. The camera is perfectly steady, locked off on a tripod, with no movement, zoom or shake at all. " + soundLine,
		"",
		"overall_soundscape:",
		scape,
		"",
		"non_diegetic_music:",
		music,
	}, "\n")
}

// sceneVoiceClipMax caps a voice reference: the engine takes audio
// references up to 15 s, and a timbre needs only a few seconds.
const sceneVoiceClipMax = 15.0

// sceneVoiceClip copies the first sceneVoiceClipMax seconds of a voice
// reference's sound (an audio file, or a video's soundtrack) into the scene
// folder as voice-<layer>.wav. Audio only: a video handed to the engine
// as-is would also become a visual reference.
func sceneVoiceClip(ctx context.Context, dir, lid, ref string) (string, error) {
	ref = absPath(ref)
	if _, err := os.Stat(ref); err != nil {
		return "", fmt.Errorf("voice reference not found: %s", ref)
	}
	if !mediaext.IsAudio(ref) && !mediaext.IsVideo(ref) {
		return "", fmt.Errorf("the voice reference must be an audio or video file")
	}
	out := filepath.Join(dir, "voice-"+safeLayerID(lid)+".wav")
	if err := runSceneFFmpeg(ctx, []string{"-i", ref, "-vn", "-t", formatFloatArg(sceneVoiceClipMax), "-ac", "1", "-ar", "24000", out}); err != nil {
		return "", fmt.Errorf("reading the voice reference: %w", err)
	}
	if info, err := os.Stat(out); err != nil || info.Size() < 1000 {
		os.Remove(out)
		return "", fmt.Errorf("the voice reference has no sound")
	}
	return out, nil
}

// GenerateLayer queues the next generation of layer lid: crops its region
// from the stills below it and submits a retouch (still) or reshoot (motion)
// job on the crop. The finished file becomes a new version on the next read.
func GenerateLayer(ctx context.Context, q *jobqueue.Queue, id, lid string) (*Scene, error) {
	sceneMu.Lock()
	defer sceneMu.Unlock()
	s, err := loadSceneLocked(id)
	if err != nil {
		return nil, err
	}
	reconcileScene(q, s)
	idx := -1
	for i, l := range s.Layers {
		if l.ID == lid {
			idx = i
		}
	}
	if idx < 0 {
		return nil, ErrLayerNotFound
	}
	l := s.Layers[idx]
	if l.Job != nil {
		return nil, ErrSceneBusy
	}
	prompt := strings.TrimSpace(l.Prompt)
	if l.Kind == SceneKindStill && prompt == "" {
		return nil, fmt.Errorf("describe the edit first")
	}
	region := l.Region.clamp(s.Width, s.Height)
	if region.empty() {
		return nil, fmt.Errorf("draw the area to change first")
	}
	dir, _ := sceneDir(id)

	below, err := sceneGraphLayers(dir, s.Layers[:idx], true)
	if err != nil {
		return nil, err
	}
	n := len(l.Versions) + 1
	stem := fmt.Sprintf("%s-%s-%d", l.Kind, safeLayerID(l.ID), n)
	input := filepath.Join(dir, stem+".png")
	for i := 2; ; i++ {
		if _, err := os.Stat(input); err != nil {
			break
		}
		input = filepath.Join(dir, fmt.Sprintf("%s-%s-%d.%d.png", l.Kind, safeLayerID(l.ID), n, i))
	}
	args := sceneGraphArgs(s.Base, s.Width, s.Height, below, false, 0, region)
	if err := runSceneFFmpeg(ctx, append(args, input)); err != nil {
		return nil, fmt.Errorf("cropping the region: %w", err)
	}

	var command string
	var jobArgs []string
	opt := func(k, v string) { jobArgs = append(jobArgs, "--"+k+"="+v) }
	seed := int64(-1)
	if _, ok := l.Options["seed"]; ok {
		seed = int64(optFloat(l.Options, "seed"))
	}
	switch l.Kind {
	case SceneKindStill:
		command = "retouch"
		opt("prompt", prompt)
		if refs := sceneRefs(l.Options); len(refs) > 0 {
			if err := checkSceneRefs(SceneKindStill, refs, false); err != nil {
				os.Remove(input)
				return nil, err
			}
			paths := make([]string, len(refs))
			for i, r := range refs {
				paths[i] = r.Path
			}
			opt("refs", strings.Join(paths, "\n"))
		}
		if le := sceneStillLongEdge(region); le > 0 {
			opt("longedge", strconv.Itoa(le))
		}
		steps := int(optFloat(l.Options, "steps"))
		if steps <= 0 {
			steps = 25
		}
		opt("steps", strconv.Itoa(clampInt(steps, 4, 80)))
	default:
		command = "reshoot"
		mp := optFloat(l.Options, "megapixels")
		w, h := sceneMotionSize(region, mp)
		// Not --animate: its living-photo template is about a person
		// (breathing, blinking, identity) and makes the model invent one
		// in a crop of sky or water. The crop is <Picture 1> of a prompt
		// written for a region of a larger, otherwise still picture.
		lines := sceneDialogue(l.Options)
		voice := ""
		if ref := strings.TrimSpace(optString(l.Options, "voiceRef")); ref != "" && len(lines) > 0 {
			v, err := sceneVoiceClip(ctx, dir, l.ID, ref)
			if err != nil {
				os.Remove(input)
				return nil, err
			}
			voice = v
		}
		refs := sceneRefs(l.Options)
		if err := checkSceneRefs(SceneKindMotion, refs, voice != ""); err != nil {
			os.Remove(input)
			return nil, err
		}
		// One list for the engine; it numbers each kind in list order, so the
		// voice goes first (<Audio 1>) and the rest follow sceneRefTokens.
		var paths []string
		if voice != "" {
			paths = append(paths, voice)
		}
		tokens := sceneRefTokens(SceneKindMotion, refs, voice != "")
		var prefs []promptRef
		hasVideo := false
		for i, r := range refs {
			paths = append(paths, r.Path)
			prefs = append(prefs, promptRef{Token: tokens[i], Kind: r.Kind, Note: r.Note})
			hasVideo = hasVideo || r.Kind == "video"
		}
		if len(paths) > 0 {
			opt("refs", strings.Join(paths, "\n"))
		}
		// A reference video is a visual reference: its soundtrack would become
		// an <Audio N> of its own and shift the numbering.
		if hasVideo {
			opt("novideoaudio", "true")
		}
		opt("prompt", sceneMotionPrompt(sceneMotionPromptInput{
			Motion: prompt, Subject: optString(l.Options, "describe"),
			Sound: optBool(l.Options, "sound"), SoundDesc: optString(l.Options, "soundPrompt"), Voices: optBool(l.Options, "voices"),
			Lines: lines, VoiceRef: voice != "", Refs: prefs,
			LockEdges: l.Options["lock"] != false,
		}))
		dur := optFloat(l.Options, "duration")
		if dur <= 0 {
			dur = 5
		}
		opt("duration", formatFloatArg(clampFloat(dur, 1, 15)))
		opt("size", fmt.Sprintf("%dx%d", w, h))
		if m := optString(l.Options, "model"); m == reshootModelFast {
			opt("model", reshootModelFast)
		}
		if !sceneLayerSound(l.Options) {
			opt("noaudio", "true")
		}
	}
	if seed >= 0 {
		opt("seed", strconv.FormatInt(seed, 10))
	}
	opt("nolibrary", "true")

	gen := jobqueue.WorkflowTask{ID: uuid.NewString(), Command: command, Arguments: jobArgs, Input: input}
	steps := []jobqueue.WorkflowTask{gen}
	// Motion: lock the static parts to the still (scene_lock.go) unless the
	// layer opts out; that is what keeps the region's edges static.
	// Detail (default on) brings it out at the region's own size with the
	// still's fine detail (scene_lock.go).
	freeze, detail := l.Options["lock"] != false, l.Options["detail"] != false
	if l.Kind == SceneKindMotion && (freeze || detail) {
		steps = append(steps, jobqueue.WorkflowTask{ID: uuid.NewString(), Command: "scene-lock",
			Arguments: []string{"--still=" + input, "--freeze=" + strconv.FormatBool(freeze), "--detail=" + strconv.FormatBool(detail),
				fmt.Sprintf("--size=%dx%d", region.W, region.H)}, Dependencies: []string{gen.ID}})
	}
	ids, err := q.AddWorkflow(jobqueue.Workflow{Tasks: steps})
	if err != nil || len(ids) == 0 {
		os.Remove(input)
		if err == nil {
			err = errors.New("the job queue returned no job")
		}
		return nil, err
	}
	l.Region = region
	l.Error = ""
	l.Job = &SceneJob{ID: ids[len(ids)-1], Chain: ids, State: "pending", Region: region, Prompt: prompt, Input: input}
	if err := saveSceneLocked(s); err != nil {
		return nil, err
	}
	return s, nil
}

// CancelLayer cancels a layer's generation in flight.
func CancelLayer(q *jobqueue.Queue, id, lid string) (*Scene, error) {
	sceneMu.Lock()
	defer sceneMu.Unlock()
	s, err := loadSceneLocked(id)
	if err != nil {
		return nil, err
	}
	for _, l := range s.Layers {
		if l.ID == lid && l.Job != nil {
			cancelSceneJob(q, l.Job, fmt.Sprintf("Cancel pressed on layer %q in the Scene editor", l.Name))
			l.Job = nil
			l.Error = ""
		}
	}
	if err := saveSceneLocked(s); err != nil {
		return nil, err
	}
	return s, nil
}

// KeepOnlyVersion discards every version of a layer but n (and their files).
func KeepOnlyVersion(id, lid string, n int) (*Scene, error) {
	sceneMu.Lock()
	defer sceneMu.Unlock()
	s, err := loadSceneLocked(id)
	if err != nil {
		return nil, err
	}
	for _, l := range s.Layers {
		if l.ID != lid {
			continue
		}
		if n < 0 || n >= len(l.Versions) {
			return nil, fmt.Errorf("no version %d", n+1)
		}
		keep := l.Versions[n]
		for i, v := range l.Versions {
			if i != n && v.Path != keep.Path && inSceneDir(id, v.Path) {
				os.Remove(v.Path)
				os.Remove(motionMapPath(v.Path))
			}
		}
		l.Versions = []SceneVersion{keep}
		l.Active = 0
		if err := saveSceneLocked(s); err != nil {
			return nil, err
		}
		return s, nil
	}
	return nil, ErrLayerNotFound
}

// versionPathShared reports whether another remaining version uses path.
func versionPathShared(vs []SceneVersion, path string) bool {
	for _, v := range vs {
		if v.Path == path {
			return true
		}
	}
	return false
}

// DeleteVersion drops version n of a layer (and its file).
func DeleteVersion(id, lid string, n int) (*Scene, error) {
	sceneMu.Lock()
	defer sceneMu.Unlock()
	s, err := loadSceneLocked(id)
	if err != nil {
		return nil, err
	}
	for _, l := range s.Layers {
		if l.ID != lid {
			continue
		}
		if n < 0 || n >= len(l.Versions) {
			return nil, fmt.Errorf("no version %d", n+1)
		}
		p := l.Versions[n].Path
		l.Versions = append(l.Versions[:n], l.Versions[n+1:]...)
		switch {
		case l.Active == n:
			l.Active = len(l.Versions) - 1
		case l.Active > n:
			l.Active--
		}
		if inSceneDir(id, p) && !versionPathShared(l.Versions, p) {
			os.Remove(p)
			os.Remove(motionMapPath(p))
		}
		if err := saveSceneLocked(s); err != nil {
			return nil, err
		}
		return s, nil
	}
	return nil, ErrLayerNotFound
}

// versionRaw finds the engine output a finished motion version was made
// from: recorded on it, else (versions from before it was recorded) the
// reshoot output beside its crop in the workflow temp folders.
func versionRaw(id string, v SceneVersion) string {
	if v.Raw != "" {
		if _, err := os.Stat(v.Raw); err == nil {
			return v.Raw
		}
	}
	crop := sceneVersionCrop(v)
	if crop == "" {
		return ""
	}
	dir, _ := sceneDir(id)
	stem := strings.TrimSuffix(filepath.Base(crop), filepath.Ext(crop))
	matches, _ := filepath.Glob(filepath.Join(dir, ".loki-temp", "*", stem+"_reshoot.mp4"))
	if len(matches) > 0 {
		return matches[len(matches)-1]
	}
	if p := filepath.Join(dir, stem+"_reshoot.mp4"); fileExists(p) {
		return p
	}
	return ""
}

// sceneVersionCrop is the crop (generation input) a version was made from:
// its file name up to the first _locked / _refined / _reshoot suffix + .png.
func sceneVersionCrop(v SceneVersion) string {
	base := filepath.Base(v.Path)
	for _, suf := range []string{"_locked", "_refined", "_reshoot"} {
		if i := strings.Index(base, suf); i > 0 {
			p := filepath.Join(filepath.Dir(v.Path), base[:i]+".png")
			if fileExists(p) {
				return p
			}
		}
	}
	return ""
}

func fileExists(p string) bool {
	_, err := os.Stat(p)
	return err == nil
}

// RefinishVersion redoes the finish (edge lock, detail) of motion version n
// from the engine's raw output with the layer's current settings, as a new
// version. CPU only: no regeneration.
func RefinishVersion(q *jobqueue.Queue, id, lid string, n int) (*Scene, error) {
	sceneMu.Lock()
	defer sceneMu.Unlock()
	s, err := loadSceneLocked(id)
	if err != nil {
		return nil, err
	}
	reconcileScene(q, s)
	for _, l := range s.Layers {
		if l.ID != lid {
			continue
		}
		if l.Kind != SceneKindMotion || n < 0 || n >= len(l.Versions) {
			return nil, fmt.Errorf("no motion version %d", n+1)
		}
		if l.Job != nil {
			return nil, ErrSceneBusy
		}
		v := l.Versions[n]
		raw, crop := versionRaw(id, v), sceneVersionCrop(v)
		if raw == "" || crop == "" {
			return nil, fmt.Errorf("the engine output of v%d is no longer on disk (loops and older versions can't be re-finished)", n+1)
		}
		stem := strings.TrimSuffix(crop, filepath.Ext(crop))
		out := stem + "_refined.mp4"
		for k := 2; fileExists(out); k++ {
			out = fmt.Sprintf("%s_refined%d.mp4", stem, k)
		}
		freeze, detail := l.Options["lock"] != false, l.Options["detail"] != false
		ids, err := q.AddWorkflow(jobqueue.Workflow{Tasks: []jobqueue.WorkflowTask{{
			Command: "scene-lock", Input: raw,
			Arguments: []string{"--still=" + crop, "--out=" + out, "--freeze=" + strconv.FormatBool(freeze), "--detail=" + strconv.FormatBool(detail),
				fmt.Sprintf("--size=%dx%d", v.Region.W, v.Region.H)},
		}}})
		if err != nil || len(ids) == 0 {
			if err == nil {
				err = errors.New("the job queue returned no job")
			}
			return nil, err
		}
		l.Error = ""
		l.Job = &SceneJob{ID: ids[0], Chain: ids, State: "pending", Region: v.Region, Prompt: v.Prompt, Input: crop,
			Step: "scene-lock", Note: fmt.Sprintf("re-finished v%d", n+1), Raw: raw}
		if err := saveSceneLocked(s); err != nil {
			return nil, err
		}
		return s, nil
	}
	return nil, ErrLayerNotFound
}

// VersionMotionMap returns where version n of motion layer lid moves (a PNG
// data URL, alpha = motion, covering the version's region) and that region.
func VersionMotionMap(ctx context.Context, id, lid string, n int) (string, SceneRect, error) {
	sceneMu.Lock()
	s, err := loadSceneLocked(id)
	sceneMu.Unlock()
	if err != nil {
		return "", SceneRect{}, err
	}
	for _, l := range s.Layers {
		if l.ID == lid && l.Kind == SceneKindMotion && n >= 0 && n < len(l.Versions) {
			v := l.Versions[n]
			url, err := MotionMapDataURL(ctx, v.Path)
			return url, v.Region, err
		}
	}
	return "", SceneRect{}, ErrLayerNotFound
}

// sceneDuration is an export's length: the scene's own, else the longest
// visible motion layer's clip, else 5 s.
func sceneDuration(s *Scene) float64 {
	if s.Duration > 0 {
		return s.Duration
	}
	d := 0.0
	for _, l := range s.Layers {
		if v := l.active(); v != nil && !l.Hidden && l.Kind == SceneKindMotion {
			// A layer played faster loops sooner.
			if ld := v.Duration / sceneLayerTiming(l.Options).Speed; ld > d {
				d = ld
			}
		}
	}
	if d <= 0 {
		d = 5
	}
	return d
}

// QueueSceneRender queues a scene-render job (format "video" or "still").
func QueueSceneRender(q *jobqueue.Queue, id, format string) (*Scene, error) {
	if format != "still" {
		format = "video"
	}
	sceneMu.Lock()
	defer sceneMu.Unlock()
	s, err := loadSceneLocked(id)
	if err != nil {
		return nil, err
	}
	reconcileScene(q, s)
	if r := s.Render; r != nil && (r.State == "pending" || r.State == "in_progress") {
		return nil, errors.New("an export is already running")
	}
	dir, _ := sceneDir(id)
	ids, err := q.AddWorkflow(jobqueue.Workflow{Tasks: []jobqueue.WorkflowTask{{
		Command: "scene-render", Arguments: []string{"--format=" + format}, Input: filepath.Join(dir, "scene.json"),
	}}})
	if err != nil || len(ids) == 0 {
		if err == nil {
			err = errors.New("the job queue returned no job")
		}
		return nil, err
	}
	s.Render = &SceneRender{JobID: ids[0], Format: format, State: "pending"}
	if err := saveSceneLocked(s); err != nil {
		return nil, err
	}
	return s, nil
}

var sceneRenderOptions = []TaskOption{
	{Name: "format", Label: "Format", Type: "enum", Choices: []string{"video", "still"}, Default: "video", Description: "video = an mp4 with every motion layer looping; still = a flattened PNG of the still edits (and the first frame of the motion)"},
}

// sceneRenderTask exports a scene (input: its scene.json) beside its source
// as <name>_scene.mp4 / _scene.png and adds it to the library.
func sceneRenderTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	const tag = "scene-render"
	ctx := j.Ctx
	if ctx == nil {
		ctx = context.Background()
	}
	fail := func(msg string, err error) error {
		q.PushJobStdout(j.ID, tag+": "+msg)
		q.ErrorJob(j.ID)
		if err == nil {
			err = fmt.Errorf("%s: %s", tag, msg)
		}
		return err
	}
	opts := ParseOptions(j, sceneRenderOptions)
	format := optString(opts, "format")

	id := filepath.Base(filepath.Dir(j.Input))
	sceneMu.Lock()
	s, err := loadSceneLocked(id)
	sceneMu.Unlock()
	if err != nil {
		return fail("cannot load scene "+id+": "+err.Error(), err)
	}
	dir, _ := sceneDir(id)
	sceneMu.Lock()
	layers, err := sceneGraphLayers(dir, s.Layers, false)
	sceneMu.Unlock()
	if err != nil {
		return fail(err.Error(), err)
	}
	if len(layers) == 0 {
		q.PushJobStdout(j.ID, tag+": no generated layers yet; exporting the base frame")
	}

	outDir := filepath.Dir(s.Source)
	if st, err := os.Stat(outDir); err != nil || !st.IsDir() {
		outDir = dir
	}
	stem := strings.TrimSuffix(filepath.Base(s.Source), filepath.Ext(s.Source))
	ext := ".mp4"
	video := format != "still"
	if !video {
		ext = ".png"
	}
	output := filepath.Join(outDir, stem+"_scene"+ext)
	if _, err := os.Stat(output); err == nil {
		output = resolveConflict(output)
	}

	dur := sceneDuration(s)
	if !s.FreeLoops {
		for i := range layers {
			if g := &layers[i]; g.Motion {
				// The layer's own speed first, then synced to the scene.
				own := g.Speed
				g.Speed = own * loopSpeed(dur, g.Duration*own)
			}
		}
	}
	args := sceneGraphArgs(s.Base, s.Width, s.Height, layers, video, dur, SceneRect{})
	if video {
		q.PushJobStdout(j.ID, fmt.Sprintf("%s: %d layer(s), %.2fs at %d fps -> %s", tag, len(layers), dur, sceneFPS, filepath.Base(output)))
	} else {
		q.PushJobStdout(j.ID, fmt.Sprintf("%s: flattening %d layer(s) -> %s", tag, len(layers), filepath.Base(output)))
	}
	_ = q.SetJobProgress(j.ID, 0, 1)
	if err := runSceneFFmpeg(ctx, append(args, output)); err != nil {
		os.Remove(output)
		if ctx.Err() != nil {
			return fail("task canceled", ctx.Err())
		}
		return fail(err.Error(), err)
	}
	info, err := os.Stat(output)
	if err != nil {
		return fail("no output produced", nil)
	}
	if err := insertMediaRecord(q.Db, output, info.Size()); err != nil {
		q.PushJobStdout(j.ID, tag+": failed to add to library: "+err.Error())
	}
	q.PushJobStdout(j.ID, tag+": completed -> "+output)
	q.RegisterOutputFile(j.ID, output, s.Source)
	broadcastMediaCreated([]string{output})
	_ = q.SetJobProgress(j.ID, 1, 1)
	q.CompleteJob(j.ID)
	return nil
}
