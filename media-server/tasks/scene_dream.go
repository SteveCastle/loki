package tasks

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"

	"github.com/google/uuid"
	"github.com/stevecastle/shrike/jobqueue"
)

// An Edit layer can run on loki-dream (Kromoss) instead of loki-retouch
// (layer option engine = "dream"). Kromoss is a text -> image model: it takes
// no instruction and no reference images, so the layer's region is inpainted
// in place: the crop is the init image, the layer's mask (cropped to the
// region) says what is redrawn, and the prompt describes what should be there.
// Strength 1 redraws the masked area from scratch; a sketch (a finish layer
// over paint in the picture) keeps its shapes and colours at a lower strength.
// A colour guide (a finish layer whose paint is a reference) is the init
// image itself: the paint over the picture, lined up with the crop.

const SceneEngineDream = "dream"

// sceneDreamMP is the pixel budget a region is generated at: small regions
// are drawn bigger (detail) and scaled back into place, big ones smaller.
const sceneDreamMP = 1.0

// sceneDreamSize is the --size for a region: its shape at about sceneDreamMP,
// on a 16-pixel grid.
func sceneDreamSize(r SceneRect) (int, int) {
	a := float64(max(1, r.W)) / float64(max(1, r.H))
	w := math.Sqrt(sceneDreamMP * 1e6 * a)
	h := w / a
	snap := func(v float64) int { return clampInt(int(math.Round(v/16))*16, 256, 2048) }
	return snap(w), snap(h)
}

// sceneDreamPrompt is what Kromoss is told: a description of the region.
func sceneDreamPrompt(opts map[string]any, prompt string) string {
	if optString(opts, "finish") == "" {
		return prompt
	}
	what := strings.TrimRight(strings.TrimSpace(prompt), ".")
	if what == "" {
		what = "the scene"
	}
	return what + ", part of the same picture as its surroundings, in the same style, light and perspective"
}

// sceneDreamArgs sets the dream task's options for a layer's generation.
func sceneDreamArgs(ctx context.Context, s *Scene, l *SceneLayer, dir string, region SceneRect, input, prompt string, refs []sceneRef, opt func(k, v string)) error {
	if strings.TrimSpace(prompt) == "" && optString(l.Options, "finish") == "" {
		return fmt.Errorf("describe what should be there first")
	}
	mask, err := sceneMaskCrop(ctx, dir, l, s.Width, s.Height, region, strings.TrimSuffix(input, filepath.Ext(input))+"-mask.png")
	if err != nil {
		return err
	}
	opt("prompt", sceneDreamPrompt(l.Options, prompt))
	opt("mask", mask)
	// Kromoss keeps the guide's colours up to about 0.8 and only its layout
	// from about 0.9, so a sketch is redrawn harder than rough colour.
	strength := 1.0
	switch optString(l.Options, "finish") {
	case FinishSketch:
		strength = 0.9
	case FinishColour:
		strength = 0.8
	}
	if v, ok := l.Options["strength"].(float64); ok && v > 0 {
		strength = clampFloat(v, 0.05, 1)
	}
	opt("strength", formatFloatArg(strength))
	w, h := sceneDreamSize(region)
	opt("size", fmt.Sprintf("%dx%d", w, h))
	// Quality: Draft = the 8-step turbo LoRA, Standard = the model's own steps, Best = more.
	switch steps := int(optFloat(l.Options, "steps")); {
	case steps > 0 && steps <= 12:
		opt("turbo", "true")
	case steps >= 40:
		opt("steps", "50")
	}
	// A colour guide: Kromoss takes no references, so the guide (the paint
	// over the picture, lined up with the crop) IS the image it starts from.
	if optString(l.Options, "finish") == FinishColour {
		for _, r := range refs {
			if r.Layer != "" && r.Layer == optString(l.Options, "paint") && r.Path != "" {
				b, err := os.ReadFile(r.Path)
				if err != nil {
					return err
				}
				if err := os.WriteFile(input, b, 0644); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

// sceneMaskCrop writes the layer's mask, cropped to region, as a gray PNG
// (white = redraw) at out.
func sceneMaskCrop(ctx context.Context, dir string, l *SceneLayer, fw, fh int, region SceneRect, out string) (string, error) {
	full, err := writeSceneMask(dir, l)
	if err != nil {
		return "", err
	}
	vf := fmt.Sprintf("format=rgba,alphaextract,scale=%d:%d,crop=%d:%d:%d:%d,format=gray", fw, fh, region.W, region.H, region.X, region.Y)
	if err := runSceneFFmpeg(ctx, []string{"-y", "-i", full, "-vf", vf, "-frames:v", "1", out}); err != nil {
		return "", fmt.Errorf("cropping the mask: %w", err)
	}
	return out, nil
}

// DreamRequest is a new picture to start a scene from.
type DreamRequest struct {
	Prompt   string `json:"prompt"`
	Negative string `json:"negative,omitempty"`
	Size     string `json:"size,omitempty"`
	Turbo    bool   `json:"turbo,omitempty"`
	Seed     int64  `json:"seed,omitempty"`
}

// DreamStatus is how that picture is coming along.
type DreamStatus struct {
	JobID  string `json:"jobId"`
	State  string `json:"state"`
	Done   int    `json:"done"`
	Total  int    `json:"total"`
	Output string `json:"output,omitempty"`
	Error  string `json:"error,omitempty"`
}

var dreamSizeRe = regexp.MustCompile(`^\d{3,4}x\d{3,4}$`)

// QueueSceneDream queues a dream job (no input: a new image in the library's
// Generated folder) for a new scene to start from.
func QueueSceneDream(q *jobqueue.Queue, in DreamRequest) (*DreamStatus, error) {
	prompt := strings.TrimSpace(in.Prompt)
	if prompt == "" {
		return nil, fmt.Errorf("describe the picture first")
	}
	args := []string{"--prompt=" + prompt}
	if n := strings.TrimSpace(in.Negative); n != "" {
		args = append(args, "--negative="+n)
	}
	if dreamSizeRe.MatchString(in.Size) {
		args = append(args, "--size="+in.Size)
	}
	if in.Turbo {
		args = append(args, "--turbo=true")
	}
	if in.Seed > 0 {
		args = append(args, "--seed="+strconv.FormatInt(in.Seed, 10))
	}
	ids, err := q.AddWorkflow(jobqueue.Workflow{Tasks: []jobqueue.WorkflowTask{{ID: uuid.NewString(), Command: "dream", Arguments: args}}})
	if err != nil || len(ids) == 0 {
		if err == nil {
			err = fmt.Errorf("the job queue returned no job")
		}
		return nil, err
	}
	return &DreamStatus{JobID: ids[0], State: "pending"}, nil
}

// SceneDreamStatus reports a QueueSceneDream job.
func SceneDreamStatus(q *jobqueue.Queue, id string) (*DreamStatus, error) {
	job, lines, ok := q.JobSnapshot(id, 3)
	if !ok || job.Command != "dream" {
		return nil, ErrSceneNotFound
	}
	st := &DreamStatus{JobID: id, State: jobStateName(job.State), Done: job.ProgressDone, Total: job.ProgressTotal}
	switch job.State {
	case jobqueue.StateCompleted:
		if len(job.OutputFiles) > 0 {
			st.Output = job.OutputFiles[0]
		}
	case jobqueue.StateError, jobqueue.StateCancelled:
		if len(lines) > 0 {
			st.Error = lastMeaningfulLine(lines)
		}
	}
	return st, nil
}
