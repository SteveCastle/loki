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
	JobID    string    `json:"jobId,omitempty"`
	Created  time.Time `json:"created"`
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
	Duration float64      `json:"duration"`
	Render   *SceneRender `json:"render,omitempty"`
	Created  time.Time    `json:"created"`
	Updated  time.Time    `json:"updated"`
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
		if l.Job != nil && q != nil {
			_ = q.CancelJob(l.Job.ID)
		}
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
		if l.Job != nil && q != nil {
			_ = q.CancelJob(l.Job.ID)
		}
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
		if l.Job == nil {
			continue
		}
		job, lines, ok := q.JobSnapshot(l.Job.ID, 4)
		// A chain reports its first unfinished step (and the step that failed).
		for _, cid := range l.Job.Chain {
			if cid == l.Job.ID {
				break
			}
			cj, clines, cok := q.JobSnapshot(cid, 4)
			if !cok || cj.State == jobqueue.StateCompleted {
				continue
			}
			if cj.State == jobqueue.StateError || cj.State == jobqueue.StateCancelled || job.State == jobqueue.StatePending {
				job, lines, ok = cj, clines, true
			}
			break
		}
		if !ok {
			// Cleared from the queue before the scene looked: the result (if
			// any) is still where the engine writes it, beside the crop.
			job = jobqueue.Job{ID: l.Job.ID, State: jobqueue.StateError}
			if out := sceneJobOutput(l.Kind, l.Job.Input); out != "" {
				job.State, job.OutputFiles = jobqueue.StateCompleted, []string{out}
			} else {
				lines = []string{"the job was removed from the queue"}
			}
		}
		switch job.State {
		case jobqueue.StateCompleted:
			out := ""
			for _, f := range job.OutputFiles {
				if inSceneDir(s.ID, f) {
					out = f
				}
			}
			if out == "" {
				l.Error = "the engine finished without a result"
			} else {
				v := SceneVersion{Path: out, Region: l.Job.Region, Prompt: l.Job.Prompt, JobID: job.ID, Created: time.Now().UTC()}
				if l.Kind == SceneKindMotion {
					v.Duration = probeVideoDuration(context.Background(), out)
					v.Audio = probeHasAudio(out)
				}
				l.Versions = append(l.Versions, v)
				l.Active = len(l.Versions) - 1
				l.Error = ""
			}
			l.Job = nil
			changed = true
		case jobqueue.StateError, jobqueue.StateCancelled:
			msg := "generation failed"
			if job.State == jobqueue.StateCancelled {
				msg = "generation cancelled"
			}
			if len(lines) > 0 {
				msg += ": " + lastMeaningfulLine(lines)
			}
			l.Error = msg
			l.Job = nil
			changed = true
		default:
			st := jobStateName(job.State)
			line := ""
			if len(lines) > 0 {
				line = lines[len(lines)-1]
			}
			if l.Job.State != st || l.Job.Done != job.ProgressDone || l.Job.Total != job.ProgressTotal || l.Job.Line != line {
				l.Job.State, l.Job.Done, l.Job.Total, l.Job.Line = st, job.ProgressDone, job.ProgressTotal, line
				changed = true
			}
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
			}
			args = append(args, "-i", l.Media)
		} else {
			still(l.Media)
		}
		ki := in + 1
		still(l.Mask)
		in += 2
		r := l.Region
		f = append(f,
			fmt.Sprintf("[%d:v]scale=%d:%d,setsar=1,format=rgba[c%d]", mi, r.W, r.H, i),
			fmt.Sprintf("[%d:v]format=rgba,alphaextract,scale=%d:%d,crop=%d:%d:%d:%d,setsar=1[a%d]", ki, w, h, r.W, r.H, r.X, r.Y, i),
			fmt.Sprintf("[c%d][a%d]alphamerge[l%d]", i, i, i),
			fmt.Sprintf("[b%d][l%d]overlay=%d:%d:format=auto:eof_action=repeat[b%d]", i, i, r.X, r.Y, i+1),
		)
		if video && l.Motion && l.Audio {
			audio = append(audio, fmt.Sprintf("[%d:a]", mi))
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
		g := sceneGraphLayer{Media: v.Path, Motion: l.Kind == SceneKindMotion, Region: v.Region, Mask: mask}
		if g.Motion {
			g.Audio = v.Audio && optBool(l.Options, "sound")
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
func sceneMotionPrompt(motion, subject string, sound, lockEdges bool) string {
	motion = strings.TrimSpace(motion)
	subject = strings.TrimSpace(subject)
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
	if sound {
		soundLine = "The sound is the natural, realistic ambience of exactly what moves in the shot, soft but present throughout; nobody speaks and there is no music."
		scape = "Natural ambience of the depicted elements and their motion; no speech, no music."
	}
	return strings.Join([]string{
		"subject_definitions:",
		"<Subject 1> is " + what + ", to be kept exactly as shown: its content, shapes, colors, lighting, texture, style and framing.",
		"",
		"summary:",
		"[keyframe completion] The target video is a faithful animation of <Picture 1>, one region of a larger still picture: one continuous shot that starts exactly from <Picture 1> and brings it to life. " + action,
		"",
		"retention_analysis:",
		"<Subject 1> (appears in [Shot 1]): fully_preserved - content, appearance, style and framing are retained.",
		"<Picture 1> ([Shot 1] first frame): fully_preserved - the video begins from this exact composition and keeps its look.",
		"",
		"detailed_description:",
		"The target video keeps the exact look, rendering style, colors, lighting and framing of <Picture 1>.",
		"[Shot 1] The shot begins exactly from <Picture 1> and stays one uninterrupted, unedited take with the same composition. " + action + " Nothing new enters the frame: no people, faces, animals or objects appear that are not already in <Picture 1>, and nothing is replaced. The edges of the frame stay continuous with what surrounds them. The camera is perfectly steady, locked off on a tripod, with no movement, zoom or shake at all. " + soundLine,
		"",
		"overall_soundscape:",
		scape,
		"",
		"non_diegetic_music:",
		"N/A",
	}, "\n")
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
		opt("prompt", sceneMotionPrompt(prompt, optString(l.Options, "describe"), optBool(l.Options, "sound"), l.Options["lock"] != false))
		dur := optFloat(l.Options, "duration")
		if dur <= 0 {
			dur = 5
		}
		opt("duration", formatFloatArg(clampFloat(dur, 1, 15)))
		opt("size", fmt.Sprintf("%dx%d", w, h))
		if m := optString(l.Options, "model"); m == reshootModelFast {
			opt("model", reshootModelFast)
		}
		if !optBool(l.Options, "sound") {
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
	if l.Kind == SceneKindMotion && l.Options["lock"] != false {
		steps = append(steps, jobqueue.WorkflowTask{ID: uuid.NewString(), Command: "scene-lock",
			Arguments: []string{"--still=" + input}, Dependencies: []string{gen.ID}})
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
			_ = q.CancelJob(l.Job.ID)
			l.Job = nil
			l.Error = ""
		}
	}
	if err := saveSceneLocked(s); err != nil {
		return nil, err
	}
	return s, nil
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
		if inSceneDir(id, p) {
			os.Remove(p)
		}
		if err := saveSceneLocked(s); err != nil {
			return nil, err
		}
		return s, nil
	}
	return nil, ErrLayerNotFound
}

// BoomerangVersion adds a seamless-loop version of a motion layer's version
// n: the clip followed by itself reversed, so looping it never jumps.
func BoomerangVersion(ctx context.Context, id, lid string, n int) (*Scene, error) {
	sceneMu.Lock()
	s, err := loadSceneLocked(id)
	sceneMu.Unlock()
	if err != nil {
		return nil, err
	}
	var src SceneVersion
	found := false
	for _, l := range s.Layers {
		if l.ID == lid && l.Kind == SceneKindMotion && n >= 0 && n < len(l.Versions) {
			src, found = l.Versions[n], true
		}
	}
	if !found {
		return nil, ErrLayerNotFound
	}
	out := strings.TrimSuffix(src.Path, filepath.Ext(src.Path)) + "_loop.mp4"
	if _, err := os.Stat(out); err == nil {
		out = resolveConflict(out)
	}
	filter := "[0:v]split[f][r0];[r0]reverse[r];[f][r]concat=n=2:v=1:a=0[v]"
	args := []string{"-i", src.Path}
	maps := []string{"-map", "[v]"}
	if src.Audio {
		filter += ";[0:a]asplit[af][ar0];[ar0]areverse[ar];[af][ar]concat=n=2:v=0:a=1[a]"
		maps = append(maps, "-map", "[a]", "-c:a", "aac", "-b:a", "192k")
	}
	args = append(args, "-filter_complex", filter)
	args = append(args, maps...)
	args = append(args, "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", out)
	if err := runSceneFFmpeg(ctx, args); err != nil {
		return nil, err
	}

	sceneMu.Lock()
	defer sceneMu.Unlock()
	s, err = loadSceneLocked(id)
	if err != nil {
		os.Remove(out)
		return nil, err
	}
	for _, l := range s.Layers {
		if l.ID == lid {
			v := src
			v.Path, v.JobID, v.Created = out, "", time.Now().UTC()
			v.Duration = src.Duration * 2
			v.Note = fmt.Sprintf("boomerang of v%d", n+1)
			l.Versions = append(l.Versions, v)
			l.Active = len(l.Versions) - 1
		}
	}
	if err := saveSceneLocked(s); err != nil {
		return nil, err
	}
	return s, nil
}

// sceneDuration is an export's length: the scene's own, else the longest
// visible motion layer's clip, else 5 s.
func sceneDuration(s *Scene) float64 {
	if s.Duration > 0 {
		return s.Duration
	}
	d := 0.0
	for _, l := range s.Layers {
		if v := l.active(); v != nil && !l.Hidden && l.Kind == SceneKindMotion && v.Duration > d {
			d = v.Duration
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
