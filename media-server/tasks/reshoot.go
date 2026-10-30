package tasks

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"

	"github.com/stevecastle/shrike/appconfig"
	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/mediaext"
)

// --- reshoot (loki-reshoot) ---
//
// Wraps the standalone `loki-reshoot` CLI (MiniMax H3, or the distilled
// 8-step FastH3 Omni with `model` = fast): reference images,
// videos and audio -> one video with sound. Unlike retouch this task is NOT
// per-file: every input of the job (plus the `refs` option) becomes one
// reference of a single generation, producing ONE `<first>_reshoot.mp4`
// beside the first input (or in the workflow temp dir when chained).

// Models loki-reshoot can run (`--model`). h3 is the original MiniMax H3
// ref2va (20 steps); fast is FastH3 Omni, distilled for exactly 8 steps.
const (
	reshootModelH3   = "h3"
	reshootModelFast = "fast"
)

// reshootDefaultSteps is the step count a model runs when none is given.
func reshootDefaultSteps(model string) int {
	if model == reshootModelFast {
		return 8
	}
	return 20
}

// ReshootModelChoice is one extra diffusion model a reshoot job can pick
// (`dit`), from Config.ReshootModels: a file and the recipe it samples with.
type ReshootModelChoice struct {
	File   string `json:"file"`
	Label  string `json:"label"`
	Recipe string `json:"recipe"` // reshootModelH3 | reshootModelFast
}

// ReshootModelChoices is Config.ReshootModels cleaned up: blank files
// dropped, duplicates dropped, recipe h3 unless fast, label = the file name
// without its extension unless given.
func ReshootModelChoices(c appconfig.Config) []ReshootModelChoice {
	out := []ReshootModelChoice{}
	seen := map[string]bool{}
	for _, m := range c.ReshootModels {
		file := strings.TrimSpace(m.File)
		if file == "" || seen[strings.ToLower(file)] {
			continue
		}
		seen[strings.ToLower(file)] = true
		ch := ReshootModelChoice{File: file, Label: strings.TrimSpace(m.Label), Recipe: reshootModelH3}
		if strings.EqualFold(strings.TrimSpace(m.Recipe), reshootModelFast) {
			ch.Recipe = reshootModelFast
		}
		if ch.Label == "" {
			ch.Label = strings.TrimSuffix(filepath.Base(file), filepath.Ext(file))
		}
		out = append(out, ch)
	}
	return out
}

// applyReshootDit checks a job's `dit` against the configured models and
// takes the model's recipe (and, when the steps were the old model's
// default, the new one's). Only listed files can be chosen: a job must not
// point the engine at an arbitrary path.
func applyReshootDit(c appconfig.Config, p *reshootParams) error {
	if p.Dit == "" {
		return nil
	}
	for _, m := range ReshootModelChoices(c) {
		if strings.EqualFold(m.File, p.Dit) {
			if m.Recipe != p.Model && p.Steps == reshootDefaultSteps(p.Model) {
				p.Steps = reshootDefaultSteps(m.Recipe)
			}
			p.Dit, p.Model = m.File, m.Recipe
			return nil
		}
	}
	return fmt.Errorf("%q is not one of the server's reshoot models (config.json reshootModels)", p.Dit)
}

// CLI limits on references per kind.
const (
	reshootMaxImages = 9
	reshootMaxVideos = 3
	reshootMaxAudios = 3
)

var reshootOptions = []TaskOption{
	{Name: "animate", Label: "Living Photo", Type: "bool", Default: false, Description: "Animate the first image with natural ambient life (the CLI writes the prompt; set Describe)"},
	{Name: "describe", Label: "Describe Subject", Type: "string", Default: "", Description: "With Living Photo: one sentence naming the subject and setting"},
	{Name: "shake", Label: "Camera Shake", Type: "enum", Choices: []string{"none", "subtle", "handheld"}, Default: "subtle", Description: "With Living Photo: camera behaviour"},
	{Name: "prompt", Label: "Prompt", Type: "string", Default: "", Description: "Full prompt (use <Picture 1>, <Video 1>, <Audio 1> tags); with Living Photo it adds direction"},
	{Name: "duration", Label: "Duration (s)", Type: "number", Default: 5.0, Description: "Seconds (0.25-15.08; snapped up to the model's frame grid)"},
	{Name: "size", Label: "Size", Type: "string", Default: "", Description: "Output size WxH (multiples of 32); empty = native / megapixels"},
	{Name: "native", Label: "Native Canvas", Type: "bool", Default: true, Description: "Snap to the model's native canvas for the first reference's aspect (ignored when Size is set)"},
	{Name: "megapixels", Label: "Megapixels", Type: "number", Default: 0.4, Description: "Pixel budget when Native is off and no Size is given"},
	{Name: "model", Label: "Model", Type: "enum", Choices: []string{reshootModelH3, reshootModelFast}, Default: reshootModelH3, Description: "h3 = MiniMax H3 (best quality, 20 steps); fast = FastH3 Omni, distilled for 8 steps: about 2.5x faster end to end (4-5x faster sampling), may be lower quality (needs the fasth3-omni dependency besides MiniMax H3's)"},
	{Name: "dit", Label: "Diffusion Model", Type: "string", Default: "", Description: "One of the server's extra reshoot models (config.json reshootModels, by file name): runs that file with its own recipe instead of H3 / FastH3 Omni; empty = the model above"},
	{Name: "steps", Label: "Steps", Type: "number", Default: 0.0, Description: "Sampling steps (4-60); 0 = the model's default (20 for h3, 8 for fast)"},
	{Name: "seed", Label: "Seed", Type: "number", Default: -1.0, Description: "Noise seed; -1 = random (logged)"},
	{Name: "refsize", Label: "Reference Image Size", Type: "enum", Choices: []string{"match", "max"}, Default: "match", Description: "match = scaled to the generation's area; max = up to 2048px (better identity, much slower)"},
	{Name: "noaudio", Label: "No Audio", Type: "bool", Default: false, Description: "Do not generate/mux audio"},
	{Name: "novideoaudio", Label: "Ignore Video Soundtracks", Type: "bool", Default: false, Description: "Do not use the soundtracks of reference videos"},
	{Name: "meta", Label: "Recipe (JSON)", Type: "string", Default: "", Description: "A JSON object stamped verbatim into the result's generation record as `meta` (the Transform flow's intent + settings, to rebuild a saved task from a result); dropped for engines too old to take it"},
	{Name: "nolibrary", Label: "Keep Out of Library", Type: "bool", Default: false, Description: "Scratch output (the Scene editor's layers): write the result beside the input but don't add it to the library or jump the viewer to it"},
	{Name: "refs", Label: "Extra References", Type: "string", Default: "", Description: "Extra reference images/videos/audio, one absolute path per line"},
}

type reshootParams struct {
	Model string // reshootModelH3 | reshootModelFast
	// Dit is a configured extra model's file (the `dit` option, checked by
	// applyReshootDit), passed as --dit; empty = Model's own file.
	Dit          string
	Animate      bool
	Describe     string
	Shake        string
	Prompt       string
	Duration     float64
	Size         string
	Native       bool
	Megapixels   float64
	Steps        int
	Seed         int64
	RefSize      string
	NoAudio      bool
	NoVideoAudio bool
	Refs         []string
	// NoLibrary keeps a finished result out of the library (scratch work).
	NoLibrary bool
	// Meta is the caller's recipe (JSON object) for the generation record;
	// MetaFile, when set by the task, carries a long one instead.
	Meta     string
	MetaFile string
}

func reshootParamsFromOptions(opts map[string]any) reshootParams {
	p := reshootParams{
		Model:        strings.ToLower(strings.TrimSpace(optString(opts, "model"))),
		Dit:          strings.TrimSpace(optString(opts, "dit")),
		Animate:      optBool(opts, "animate"),
		Describe:     strings.TrimSpace(optString(opts, "describe")),
		Shake:        strings.ToLower(strings.TrimSpace(optString(opts, "shake"))),
		Prompt:       strings.TrimSpace(optString(opts, "prompt")),
		Size:         strings.TrimSpace(optString(opts, "size")),
		Native:       true,
		Megapixels:   optFloat(opts, "megapixels"),
		RefSize:      strings.ToLower(strings.TrimSpace(optString(opts, "refsize"))),
		NoAudio:      optBool(opts, "noaudio"),
		NoVideoAudio: optBool(opts, "novideoaudio"),
		Refs:         splitPathList(optString(opts, "refs")),
		NoLibrary:    optBool(opts, "nolibrary"),
		Meta:         strings.TrimSpace(optString(opts, "meta")),
	}
	if v, ok := opts["native"].(bool); ok {
		p.Native = v
	}
	switch p.Shake {
	case "none", "subtle", "handheld":
	default:
		p.Shake = "subtle"
	}
	if p.Model != reshootModelFast {
		p.Model = reshootModelH3
	}
	if p.RefSize != "max" {
		p.RefSize = "match"
	}
	d := 5.0
	if _, ok := opts["duration"]; ok {
		d = optFloat(opts, "duration")
	}
	if d == 0 {
		d = 5
	}
	p.Duration = clampFloat(d, 0.25, 15.08)
	if p.Megapixels <= 0 {
		p.Megapixels = 0.4
	}
	steps := int(optFloat(opts, "steps"))
	if steps == 0 {
		steps = reshootDefaultSteps(p.Model)
	}
	p.Steps = clampInt(steps, 4, 60)
	p.Seed = -1
	if _, ok := opts["seed"]; ok {
		p.Seed = int64(optFloat(opts, "seed"))
	}
	return p
}

// classifyReshootInputs splits reference paths by kind (extension):
// images -> -i, videos -> -v, audio -> -a; anything else is skipped.
// Duplicate paths are dropped silently.
func classifyReshootInputs(paths []string) (images, videos, audios, skipped []string) {
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
		case mediaext.IsAudio(p):
			audios = append(audios, p)
		default:
			skipped = append(skipped, p)
		}
	}
	return
}

// capReshootInputs enforces the CLI's per-kind reference limits, returning
// the kept lists and the dropped paths.
func capReshootInputs(images, videos, audios []string) (imgs, vids, auds, dropped []string) {
	take := func(in []string, n int) []string {
		if len(in) <= n {
			return in
		}
		dropped = append(dropped, in[n:]...)
		return in[:n]
	}
	return take(images, reshootMaxImages), take(videos, reshootMaxVideos), take(audios, reshootMaxAudios), dropped
}

// buildReshootArgs builds the loki-reshoot argument list. With Animate the
// first image becomes --animate and the rest -i. Free text uses the
// `--flag=value` form; promptFile (non-empty) replaces the inline prompt.
func buildReshootArgs(p reshootParams, images, videos, audios []string, output, promptFile string) []string {
	var args []string
	// h3 is the engine's default: its args stay exactly what engines from
	// before --model accept.
	if p.Model == reshootModelFast {
		args = append(args, "--model", reshootModelFast)
	}
	if p.Dit != "" {
		args = append(args, "--dit", p.Dit)
	}
	imgs := images
	if p.Animate && len(imgs) > 0 {
		args = append(args, "--animate", imgs[0])
		imgs = imgs[1:]
		if p.Describe != "" {
			args = append(args, "--describe="+p.Describe)
		}
		args = append(args, "--shake", p.Shake)
	}
	if promptFile != "" {
		args = append(args, "--prompt-file", promptFile)
	} else if p.Prompt != "" {
		args = append(args, "--prompt="+p.Prompt)
	}
	if p.MetaFile != "" {
		args = append(args, "--meta-file", p.MetaFile)
	} else if p.Meta != "" {
		args = append(args, "--meta="+p.Meta)
	}
	args = append(args, "-d", formatFloatArg(p.Duration))
	switch {
	case p.Size != "":
		args = append(args, "--size", p.Size)
	case p.Native:
		args = append(args, "--native")
	default:
		args = append(args, "--megapixels", formatFloatArg(p.Megapixels))
	}
	args = append(args, "--steps", strconv.Itoa(p.Steps))
	if p.Seed >= 0 {
		args = append(args, "--seed", strconv.FormatInt(p.Seed, 10))
	}
	args = append(args, "--ref-image-size", p.RefSize)
	if p.NoAudio {
		args = append(args, "--no-audio")
	}
	if p.NoVideoAudio {
		args = append(args, "--no-video-audio")
	}
	for _, f := range imgs {
		args = append(args, "-i", f)
	}
	for _, f := range videos {
		args = append(args, "-v", f)
	}
	for _, f := range audios {
		args = append(args, "-a", f)
	}
	args = append(args, "-o", output)
	return args
}

// planReshootInputs resolves a job's paths plus the `refs` option into the
// kept reference lists (missing, unsupported and over-the-limit files are
// reported through logf and dropped). all is every existing file in input
// order (the output is named after the first kept one). Shared by the task
// and the prompt preview.
func planReshootInputs(paths []string, p reshootParams, logf func(string)) (images, videos, audios, all []string) {
	for _, f := range append(append([]string{}, paths...), p.Refs...) {
		f = absPath(f)
		if _, err := os.Stat(f); err != nil {
			logf("skipping missing file " + f)
			continue
		}
		all = append(all, f)
	}
	images, videos, audios, skipped := classifyReshootInputs(all)
	for _, s := range skipped {
		logf("skipping unsupported file " + filepath.Base(s))
	}
	images, videos, audios, dropped := capReshootInputs(images, videos, audios)
	for _, d := range dropped {
		logf(fmt.Sprintf("dropping %s (limit: %d images, %d videos, %d audio clips)", filepath.Base(d), reshootMaxImages, reshootMaxVideos, reshootMaxAudios))
	}
	return images, videos, audios, all
}

func reshootTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	// Selection sessions (loki-segment) let go of their VRAM while this runs.
	defer beginGPUHeavy()()
	const tag = "reshoot"
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

	p := reshootParamsFromOptions(ParseOptions(j, reshootOptions))
	if err := applyReshootDit(appconfig.Get(), &p); err != nil {
		return fail(err.Error(), err)
	}

	bin, err := resolveAITool(ctx, q, j.ID, tag, reshootSpec(p.Model, p.Dit))
	if err != nil {
		return fail(err.Error(), err)
	}
	defer bin.Release()
	if p.Model == reshootModelFast {
		if !aiToolHelpHas(ctx, bin, "--model") {
			return fail("this loki-reshoot is too old for the fast model (FastH3 Omni); update it (Dependencies page, or your own build on PATH) or use the H3 model", nil)
		}
		if p.Dit == "" {
			q.PushJobStdout(j.ID, fmt.Sprintf("%s: model FastH3 Omni, %d steps", tag, p.Steps))
		}
	}
	if p.Dit != "" {
		q.PushJobStdout(j.ID, fmt.Sprintf("%s: model %s (%s recipe), %d steps", tag, p.Dit, p.Model, p.Steps))
	}

	res, rerr := resolveJobItemsRaw(j, q)
	if rerr != nil {
		return fail("failed to resolve input: "+rerr.Error(), rerr)
	}
	if res.FromQuery {
		q.PushJobStdout(j.ID, fmt.Sprintf("%s: query: %s", tag, res.Query))
	}

	images, videos, audios, all := planReshootInputs(res.Paths, p, func(msg string) { q.PushJobStdout(j.ID, tag+": "+msg) })

	if len(images)+len(videos)+len(audios) == 0 {
		return fail("needs at least one reference image, video or audio file", nil)
	}
	if p.Animate && len(images) == 0 {
		return fail("living photo (animate) needs an image input", nil)
	}
	if !p.Animate && p.Prompt == "" {
		return fail("needs a prompt (or enable animate for a living photo)", nil)
	}

	// The output is named after the first kept reference, in input order.
	kept := map[string]bool{}
	for _, l := range [][]string{images, videos, audios} {
		for _, f := range l {
			kept[f] = true
		}
	}
	first := ""
	for _, f := range all {
		if kept[f] {
			first = f
			break
		}
	}
	if p.Animate {
		first = images[0]
	}

	seed, random := resolveSeed(p.Seed)
	p.Seed = seed
	if random {
		q.PushJobStdout(j.ID, fmt.Sprintf("%s: seed %d (random)", tag, seed))
	} else {
		q.PushJobStdout(j.ID, fmt.Sprintf("%s: seed %d", tag, seed))
	}

	units := p.Steps + 2
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
	var created []string
	defer func() {
		if terminal && !p.NoLibrary {
			broadcastMediaCreated(created)
		}
	}()

	outputDir, derr := aiOutputDir(j, terminal, first)
	if derr != nil {
		return fail("failed to create temp dir: "+derr.Error(), derr)
	}
	stem := strings.TrimSuffix(filepath.Base(first), filepath.Ext(first))
	output := filepath.Join(outputDir, stem+"_reshoot.mp4")
	if terminal {
		if _, err := os.Stat(output); err == nil {
			output = resolveConflict(output)
		}
	}

	promptFile, perr := writePromptFile(p.Prompt)
	if perr != nil {
		return fail("failed to write prompt file: "+perr.Error(), perr)
	}
	if promptFile != "" {
		defer os.Remove(promptFile)
	}
	cleanupMeta, merr := prepareMeta(ctx, q, j.ID, tag, bin, &p.Meta, &p.MetaFile)
	if merr != nil {
		return fail("failed to write recipe file: "+merr.Error(), merr)
	}
	defer cleanupMeta()

	args := buildReshootArgs(p, images, videos, audios, output, promptFile)
	mode := "prompt"
	if p.Animate {
		mode = "living photo"
	}
	q.PushJobStdout(j.ID, fmt.Sprintf("%s: %s, %d image(s), %d video(s), %d audio -> %s", tag, mode, len(images), len(videos), len(audios), filepath.Base(output)))
	_ = q.SetJobProgress(j.ID, 1, units)

	out, waitErr := runAICLI(ctx, q, j.ID, tag, bin, args, func(i, n int) {
		done := 1 + i*p.Steps/n
		if done > units-1 {
			done = units - 1
		}
		_ = q.SetJobProgress(j.ID, done, units)
	})
	for _, l := range out {
		if l != "" {
			q.PushJobStdout(j.ID, tag+": output "+l)
		}
	}
	if ctx.Err() != nil {
		return fail("task canceled", ctx.Err())
	}
	if waitErr != nil {
		return fail("failed: "+waitErr.Error(), waitErr)
	}
	info, statErr := os.Stat(output)
	if statErr != nil {
		return fail("no output produced", nil)
	}
	if terminal && !p.NoLibrary {
		if err := insertMediaRecord(q.Db, output, info.Size()); err != nil {
			q.PushJobStdout(j.ID, tag+": failed to add to library: "+err.Error())
		}
	}
	q.PushJobStdout(j.ID, tag+": completed -> "+output)
	q.RegisterOutputFile(j.ID, output, first)
	created = append(created, output)
	_ = q.SetJobProgress(j.ID, units, units)
	q.CompleteJob(j.ID)
	return nil
}
