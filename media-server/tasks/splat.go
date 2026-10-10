package tasks

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/mediaext"
)

// --- splat (loki-splat) ---
//
// One video, or a set of photos of the same scene, -> one trained 3D
// Gaussian Splat `.ply` (frames -> COLMAP camera poses -> Brush training; see
// splat_pipeline.go). Like reshoot this is NOT per-file: every input of the
// job is one capture of a single scene, producing ONE `<first>_splat.ply`
// beside the first input (or in the workflow temp dir when chained). Lowkey
// Studio submits these and imports the result as a splat layer.
//
// The .ply is not a library media type, so it is registered as the job's
// output file (for workflows and the Studio, which downloads it through
// /media/file) but not inserted into the media table.

var splatOptions = []TaskOption{
	{Name: "quality", Label: "Quality", Type: "enum", Choices: []string{"draft", "standard", "high"}, Default: "standard", Description: "Training length: draft ~5k steps (minutes), standard ~15k, high ~30k (best detail, slowest)"},
	{Name: "frames", Label: "Max Video Frames", Type: "number", Default: 200.0, Description: "Frames sampled from a video input (the sharpest frame of each stretch is kept; 30-600)"},
	{Name: "maxres", Label: "Max Resolution", Type: "number", Default: 1600.0, Description: "Longest image side used for poses and training, in px (512-4096)"},
	{Name: "shdegree", Label: "SH Degree", Type: "number", Default: 3.0, Description: "View-dependent colour detail, 0 (flat colour, smallest file) to 3 (reflections and sheen)"},
	{Name: "steps", Label: "Steps", Type: "number", Default: 0.0, Description: "Exact training steps; 0 = from Quality"},
}

type splatParams struct {
	Quality  string
	Frames   int
	MaxRes   int
	SHDegree int
	Steps    int
}

func splatParamsFromOptions(opts map[string]any) splatParams {
	p := splatParams{Quality: strings.ToLower(strings.TrimSpace(optString(opts, "quality")))}
	switch p.Quality {
	case "draft", "standard", "high":
	default:
		p.Quality = "standard"
	}
	num := func(k string, def float64) float64 {
		if _, ok := opts[k]; !ok {
			return def
		}
		v := optFloat(opts, k)
		if v == 0 {
			return def
		}
		return v
	}
	p.Frames = clampInt(int(num("frames", 200)), 30, 600)
	p.MaxRes = clampInt(int(num("maxres", 1600)), 512, 4096)
	p.SHDegree = 3
	if _, ok := opts["shdegree"]; ok {
		p.SHDegree = clampInt(int(optFloat(opts, "shdegree")), 0, 3)
	}
	if s := int(optFloat(opts, "steps")); s > 0 {
		p.Steps = clampInt(s, 100, 100000)
	}
	return p
}

// classifySplatInputs keeps images and videos (deduplicated, input order)
// and reports the rest as skipped. A scene is ONE video or a set of photos.
func classifySplatInputs(paths []string) (images, videos, skipped []string) {
	seen := map[string]bool{}
	for _, p := range paths {
		key := strings.ToLower(filepath.Clean(p))
		if seen[key] {
			continue
		}
		seen[key] = true
		switch {
		case mediaext.IsImage(p):
			images = append(images, p)
		case mediaext.IsVideo(p):
			videos = append(videos, p)
		default:
			skipped = append(skipped, p)
		}
	}
	return
}

func splatTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	const tag = "splat"
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

	p := splatParamsFromOptions(ParseOptions(j, splatOptions))

	logf := func(format string, a ...any) { q.PushJobStdout(j.ID, tag+": "+fmt.Sprintf(format, a...)) }
	tools, err := resolveSplatTools(ctx, logf)
	if err != nil {
		return fail(err.Error(), err)
	}

	res, rerr := resolveJobItemsRaw(j, q)
	if rerr != nil {
		return fail("failed to resolve input: "+rerr.Error(), rerr)
	}
	var all []string
	for _, f := range res.Paths {
		f = absPath(f)
		if _, err := os.Stat(f); err != nil {
			q.PushJobStdout(j.ID, tag+": skipping missing file "+f)
			continue
		}
		all = append(all, f)
	}
	images, videos, skipped := classifySplatInputs(all)
	for _, s := range skipped {
		q.PushJobStdout(j.ID, tag+": skipping unsupported file "+filepath.Base(s))
	}
	var inputs []string
	switch {
	case len(videos) > 0:
		if len(videos) > 1 || len(images) > 0 {
			q.PushJobStdout(j.ID, tag+": a scene is one video or a set of photos — using "+filepath.Base(videos[0]))
		}
		inputs = videos[:1]
	case len(images) >= 3:
		inputs = images
	case len(images) > 0:
		return fail(fmt.Sprintf("needs at least 3 photos of the scene (got %d) — or one video", len(images)), nil)
	default:
		return fail("needs a video or a set of photos of one scene", nil)
	}
	first := inputs[0]

	// Progress: runSplatPipeline reports 0..1000 across its stages.
	const units = 1000
	_ = q.SetJobProgress(j.ID, 0, units)
	select {
	case <-ctx.Done():
		return fail("task canceled", ctx.Err())
	default:
	}
	if q.PauseRequested(j.ID) {
		q.PushJobStdout(j.ID, tag+": paused before start - resume to continue")
		return jobqueue.ErrPaused
	}

	terminal := !q.HasDependents(j.ID)
	outputDir, derr := aiOutputDir(j, terminal, first)
	if derr != nil {
		return fail("failed to create temp dir: "+derr.Error(), derr)
	}
	stem := strings.TrimSuffix(filepath.Base(first), filepath.Ext(first))
	output := filepath.Join(outputDir, stem+"_splat.ply")
	if terminal {
		if _, err := os.Stat(output); err == nil {
			output = resolveConflict(output)
		}
	}

	what := fmt.Sprintf("%d photos", len(inputs))
	if len(videos) > 0 {
		what = "video " + filepath.Base(first)
	}
	q.PushJobStdout(j.ID, fmt.Sprintf("%s: %s, quality %s, max %dpx -> %s", tag, what, p.Quality, p.MaxRes, filepath.Base(output)))

	work, werr := os.MkdirTemp("", "lowkey-splat-*")
	if werr != nil {
		return fail("failed to create a workspace: "+werr.Error(), werr)
	}
	defer os.RemoveAll(work)
	spec := splatPipelineSpec{Params: p, Output: output, WorkDir: work}
	if len(videos) > 0 {
		spec.Video = first
	} else {
		spec.Photos = inputs
	}
	perr := runSplatPipeline(ctx, tools, spec, logf, func(done int) {
		if done > units-1 {
			done = units - 1
		}
		_ = q.SetJobProgress(j.ID, done, units)
	})
	if ctx.Err() != nil {
		return fail("task canceled", ctx.Err())
	}
	if perr != nil {
		return fail(perr.Error(), perr)
	}
	if _, statErr := os.Stat(output); statErr != nil {
		return fail("no output produced", nil)
	}
	q.PushJobStdout(j.ID, tag+": completed -> "+output)
	q.RegisterOutputFile(j.ID, output, first)
	_ = q.SetJobProgress(j.ID, units, units)
	q.CompleteJob(j.ID)
	return nil
}
