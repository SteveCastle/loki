package tasks

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"image"
	"image/png"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"

	"github.com/google/uuid"
	"github.com/stevecastle/shrike/jobqueue"
)

// --- Roto: a motion version's matte, tracked through its clip ---
//
// The editor picks what to keep in a motion version's clip (clicks, a box or
// a name, on any frame); a scene-track job runs loki-segment over the clip
// and writes one matte per clip frame (every tracked object together), plus
// the clip with that matte as alpha (VP9 webm) for the editor's stage. The
// export composites the clip through the layer's mask times the matte, so
// whatever moves can leave or enter the static mask's area cleanly.
//
// The job writes to fixed paths beside the version (sceneTrackPaths) and never
// edits the scene: reconcileTracks folds its state in when the scene is read,
// like an export's.

// SceneTrack is a motion version's roto (server-owned).
type SceneTrack struct {
	JobID string `json:"jobId"`
	State string `json:"state"`
	Done  int    `json:"done"`
	Total int    `json:"total"`
	Error string `json:"error,omitempty"`
	// Prompts are what it was tracked from: loki-segment --prompts entries
	// ({obj, frame, text | points [[x,y,1|0]] | box [x,y,w,h]}, clip pixels).
	Prompts json.RawMessage `json:"prompts"`
	// Dir holds the matte, 00000.png ... (one per clip frame); Video is the
	// clip with it as alpha. Set once the job completes.
	Dir    string `json:"dir,omitempty"`
	Frames int    `json:"frames,omitempty"`
	Video  string `json:"video,omitempty"`
}

// sceneTrackPaths is where version v's matte and its preview video live.
func sceneTrackPaths(v SceneVersion) (dir, video string) {
	stem := strings.TrimSuffix(v.Path, filepath.Ext(v.Path))
	return stem + "_matte", stem + "_matte.webm"
}

var sceneTrackOptions = []TaskOption{
	{Name: "prompts", Label: "Prompts file", Type: "string", Description: "loki-segment --prompts JSON"},
	{Name: "loop", Label: "Loop", Type: "bool", Description: "the clip is a seamless loop"},
}

// TrackVersion queues the roto of motion version n of layer lid from prompts.
func TrackVersion(q *jobqueue.Queue, id, lid string, n int, prompts json.RawMessage) (*Scene, error) {
	var list []map[string]any
	if err := json.Unmarshal(prompts, &list); err != nil || len(list) == 0 {
		return nil, fmt.Errorf("pick what to track first")
	}
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
		v := &l.Versions[n]
		if t := v.Track; t != nil && (t.State == "pending" || t.State == "in_progress") {
			cancelSceneJob(q, &SceneJob{ID: t.JobID}, "it was tracked again")
		}
		stem := strings.TrimSuffix(v.Path, filepath.Ext(v.Path))
		file := stem + "_track.json"
		if err := os.WriteFile(file, prompts, 0644); err != nil {
			return nil, err
		}
		loop := strings.Contains(strings.ToLower(v.Note), "loop") || strings.Contains(strings.ToLower(v.Note), "boomerang")
		ids, err := q.AddWorkflow(jobqueue.Workflow{Tasks: []jobqueue.WorkflowTask{{
			ID: uuid.NewString(), Command: "scene-track", Input: v.Path,
			Arguments: []string{"--prompts=" + file, "--loop=" + strconv.FormatBool(loop)},
		}}})
		if err != nil || len(ids) == 0 {
			if err == nil {
				err = fmt.Errorf("the job queue returned no job")
			}
			return nil, err
		}
		v.Track = &SceneTrack{JobID: ids[0], State: "pending", Prompts: prompts}
		if err := saveSceneLocked(s); err != nil {
			return nil, err
		}
		return s, nil
	}
	return nil, ErrLayerNotFound
}

// ClearTrack drops version n's roto (cancelling it if it runs).
func ClearTrack(q *jobqueue.Queue, id, lid string, n int) (*Scene, error) {
	sceneMu.Lock()
	defer sceneMu.Unlock()
	s, err := loadSceneLocked(id)
	if err != nil {
		return nil, err
	}
	for _, l := range s.Layers {
		if l.ID == lid && n >= 0 && n < len(l.Versions) {
			v := &l.Versions[n]
			if v.Track != nil {
				cancelSceneJob(q, &SceneJob{ID: v.Track.JobID}, "its roto was removed")
				dir, video := sceneTrackPaths(*v)
				_ = os.RemoveAll(dir)
				_ = os.Remove(video)
				v.Track = nil
			}
			return s, saveSceneLocked(s)
		}
	}
	return nil, ErrLayerNotFound
}

// reconcileTracks folds roto jobs into their versions. Returns whether anything changed.
func reconcileTracks(q *jobqueue.Queue, s *Scene) bool {
	changed := false
	for _, l := range s.Layers {
		for i := range l.Versions {
			t := l.Versions[i].Track
			if t == nil || (t.State != "pending" && t.State != "in_progress" && t.State != "paused") {
				continue
			}
			job, lines, ok := q.JobSnapshot(t.JobID, 3)
			switch {
			case !ok:
				t.State, t.Error = "error", "the roto job was removed from the queue"
			case job.State == jobqueue.StateCompleted:
				dir, video := sceneTrackPaths(l.Versions[i])
				t.State, t.Dir, t.Video, t.Frames = "completed", dir, video, countPNGs(dir)
				if !fileExists(video) {
					t.Video = ""
				}
			case job.State == jobqueue.StateError || job.State == jobqueue.StateCancelled:
				t.State = jobStateName(job.State)
				if len(lines) > 0 {
					t.Error = lastMeaningfulLine(lines)
				}
			default:
				t.State, t.Done, t.Total = jobStateName(job.State), job.ProgressDone, job.ProgressTotal
			}
			changed = true
		}
	}
	return changed
}

func countPNGs(dir string) int {
	m, _ := filepath.Glob(filepath.Join(dir, "*.png"))
	return len(m)
}

var trackProgressRe = regexp.MustCompile(`(\d+)/(\d+)`)

// sceneTrackTask tracks a motion clip (input) with loki-segment and writes
// its matte and matted preview (sceneTrackPaths).
func sceneTrackTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	const tag = "scene-track"
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
	opts := ParseOptions(j, sceneTrackOptions)
	video := j.Input
	prompts := optString(opts, "prompts")
	if video == "" || prompts == "" {
		return fail("needs a clip and a prompts file", nil)
	}
	// Installed on first use like the other engines (the tool and SAM 3.1's weights, ~1.8 GB).
	bin, err := segBinary()
	if err != nil {
		run, rerr := resolveAITool(ctx, q, j.ID, tag, segSpec)
		if rerr != nil {
			return fail(rerr.Error(), rerr)
		}
		bin = run.Bin
	}
	defer beginGPUHeavy()()

	v := SceneVersion{Path: video}
	dir, preview := sceneTrackPaths(v)
	raw := dir + ".tmp"
	_ = os.RemoveAll(raw)
	args := []string{"--prompts", prompts, "--json", "-o", raw}
	if optBool(opts, "loop") {
		args = append(args, "--loop")
	}
	args = append(args, video)
	q.PushJobStdout(j.ID, fmt.Sprintf("%s: tracking %s", tag, filepath.Base(video)))
	cmd := exec.CommandContext(ctx, bin, args...)
	cmd.Env = append(os.Environ(), aiToolEnv(segSpec)...)
	stderr, _ := cmd.StderrPipe()
	var stdout strings.Builder
	cmd.Stdout = &stdout
	if err := cmd.Start(); err != nil {
		return fail("starting loki-segment: "+err.Error(), err)
	}
	sc := bufio.NewScanner(stderr)
	sc.Split(scanLinesOrCR)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if m := trackProgressRe.FindStringSubmatch(line); m != nil {
			done, _ := strconv.Atoi(m[1])
			total, _ := strconv.Atoi(m[2])
			if total > 0 {
				_ = q.SetJobProgress(j.ID, done, total+1)
			}
		} else if line != "" {
			q.PushJobStdout(j.ID, tag+": "+line)
		}
	}
	if err := cmd.Wait(); err != nil {
		_ = os.RemoveAll(raw)
		if ctx.Err() != nil {
			return fail("task canceled", ctx.Err())
		}
		return fail("loki-segment failed: "+err.Error(), err)
	}
	var res struct {
		Frames  int `json:"frames"`
		Objects []struct {
			Frames string `json:"frames"`
		} `json:"objects"`
	}
	if err := json.Unmarshal([]byte(stdout.String()), &res); err != nil {
		return fail("could not read loki-segment's answer", err)
	}
	if len(res.Objects) == 0 {
		_ = os.RemoveAll(raw)
		return fail("nothing was found to track: pick it again (a click on it, or a box round it)", nil)
	}
	var objDirs []string
	for _, o := range res.Objects {
		objDirs = append(objDirs, o.Frames)
	}
	frames, err := mergeMattes(objDirs, raw+"-merged")
	if err != nil {
		return fail("merging the masks: "+err.Error(), err)
	}
	_ = os.RemoveAll(raw)
	_ = os.RemoveAll(dir)
	if err := os.Rename(raw+"-merged", dir); err != nil {
		return fail(err.Error(), err)
	}
	q.PushJobStdout(j.ID, fmt.Sprintf("%s: %d object(s), %d frames -> %s", tag, len(objDirs), frames, filepath.Base(dir)))

	// The stage's preview: the clip with the matte as alpha.
	info, err := probeVideoInfo(ctx, video)
	if err == nil {
		tmp := strings.TrimSuffix(preview, ".webm") + ".tmp.webm"
		args := []string{"-y", "-i", video, "-framerate", info.FPS, "-i", filepath.Join(dir, "%05d.png"),
			"-filter_complex", fmt.Sprintf("[1:v]format=gray,scale=%d:%d[m];[0:v][m]alphamerge,format=yuva420p[v]", info.W, info.H),
			"-map", "[v]"}
		if info.Audio {
			args = append(args, "-map", "0:a", "-c:a", "libopus", "-b:a", "128k")
		}
		args = append(args, "-c:v", "libvpx-vp9", "-crf", "30", "-b:v", "0", "-row-mt", "1", "-deadline", "good", "-cpu-used", "4", "-shortest", tmp)
		if err := runSceneFFmpeg(ctx, args); err != nil {
			q.PushJobStdout(j.ID, tag+": the preview video failed (the export still uses the matte): "+err.Error())
			os.Remove(tmp)
		} else if err := os.Rename(tmp, preview); err != nil {
			os.Remove(tmp)
		}
	}
	_ = q.SetJobProgress(j.ID, 1, 1)
	q.CompleteJob(j.ID)
	return nil
}

// mergeMattes writes, per frame, the union (max) of every object's mask into
// out/00000.png ... and returns the frame count.
func mergeMattes(objDirs []string, out string) (int, error) {
	if err := os.MkdirAll(out, 0755); err != nil {
		return 0, err
	}
	var names []string
	for _, d := range objDirs {
		m, _ := filepath.Glob(filepath.Join(d, "*.png"))
		if len(m) > len(names) {
			names = m
		}
	}
	sort.Strings(names)
	for i, first := range names {
		base := filepath.Base(first)
		var acc *image.Gray
		for _, d := range objDirs {
			f, err := os.Open(filepath.Join(d, base))
			if err != nil {
				continue
			}
			img, err := png.Decode(f)
			f.Close()
			if err != nil {
				return 0, err
			}
			b := img.Bounds()
			if acc == nil {
				acc = image.NewGray(image.Rect(0, 0, b.Dx(), b.Dy()))
			}
			for y := 0; y < b.Dy() && y < acc.Rect.Dy(); y++ {
				for x := 0; x < b.Dx() && x < acc.Rect.Dx(); x++ {
					r, _, _, _ := img.At(b.Min.X+x, b.Min.Y+y).RGBA()
					if g := uint8(r >> 8); g > acc.Pix[y*acc.Stride+x] {
						acc.Pix[y*acc.Stride+x] = g
					}
				}
			}
		}
		if acc == nil {
			continue
		}
		f, err := os.Create(filepath.Join(out, fmt.Sprintf("%05d.png", i)))
		if err != nil {
			return 0, err
		}
		err = png.Encode(f, acc)
		f.Close()
		if err != nil {
			return 0, err
		}
	}
	return len(names), nil
}

// scanLinesOrCR splits on \n or \r (progress lines often rewrite themselves with \r).
func scanLinesOrCR(data []byte, atEOF bool) (int, []byte, error) {
	for i, b := range data {
		if b == '\n' || b == '\r' {
			return i + 1, data[:i], nil
		}
	}
	if atEOF && len(data) > 0 {
		return len(data), data, nil
	}
	return 0, nil, nil
}

// VersionMatteFrame returns frame f of version n's roto matte as a mask piece
// (white, alpha = kept) covering rect, in clip pixels, and the clip's size.
func VersionMatteFrame(id, lid string, n, f int) (string, SceneRect, int, int, error) {
	sceneMu.Lock()
	s, err := loadSceneLocked(id)
	sceneMu.Unlock()
	if err != nil {
		return "", SceneRect{}, 0, 0, err
	}
	for _, l := range s.Layers {
		if l.ID != lid || n < 0 || n >= len(l.Versions) {
			continue
		}
		t := l.Versions[n].Track
		if t == nil || t.State != "completed" || t.Dir == "" {
			return "", SceneRect{}, 0, 0, fmt.Errorf("v%d has no roto yet", n+1)
		}
		path := filepath.Join(t.Dir, fmt.Sprintf("%05d.png", max(0, f)))
		fh, err := os.Open(path)
		if err != nil {
			return "", SceneRect{}, 0, 0, err
		}
		cfg, err := png.DecodeConfig(fh)
		fh.Close()
		if err != nil {
			return "", SceneRect{}, 0, 0, err
		}
		url, rect, err := maskFileDataURL(path)
		return url, rect, cfg.Width, cfg.Height, err
	}
	return "", SceneRect{}, 0, 0, ErrLayerNotFound
}
