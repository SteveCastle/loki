package tasks

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/stevecastle/shrike/appconfig"
	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/platform"
)

// --- dream (loki-dream) ---
//
// Wraps the standalone `loki-dream` CLI (Kromoss, a MOSS-Image fine-tune:
// text -> image). Three modes, by what the job is given:
//
//   - no input: a new image from the prompt, written to the generations
//     folder (dreamDir) and added to the library;
//   - an input image: img2img (the image noised to `strength`, then
//     denoised toward the prompt), written beside it;
//   - an input image and a `mask`: inpainting (only the white of the mask is
//     regenerated), which is how the Scene editor's Edit layers use it.
//
// It shares the diffusion GPU bucket with retouch / reshoot (registry.go).

var dreamTool = aiToolSpec{Bin: "loki-dream", Model: "kromoss"}

var dreamOptions = []TaskOption{
	{Name: "prompt", Label: "Prompt", Type: "string", Default: "", Description: "What to make: a description of the picture (with an input image: of the whole result)"},
	{Name: "negative", Label: "Negative Prompt", Type: "string", Default: "", Description: "What to avoid (ignored in turbo mode)"},
	{Name: "size", Label: "Size", Type: "string", Default: "", Description: "Output size WxH; empty = 1024x1024 (with an input image: its size)"},
	{Name: "steps", Label: "Steps", Type: "number", Default: 0.0, Description: "Sampling steps; 0 = the model's default (turbo: 8)"},
	{Name: "cfg", Label: "Guidance", Type: "number", Default: 0.0, Description: "Classifier-free guidance; 0 = the model's default (turbo: 1)"},
	{Name: "turbo", Label: "Turbo", Type: "bool", Default: false, Description: "8 steps with the distilled turbo LoRA: several times faster, a little less detail"},
	{Name: "seed", Label: "Seed", Type: "number", Default: -1.0, Description: "Noise seed; -1 = random (logged)"},
	{Name: "count", Label: "Count", Type: "number", Default: 1.0, Description: "Images to make (1-8; consecutive seeds, one model load)"},
	{Name: "loras", Label: "LoRAs", Type: "string", Default: "", Description: "LoRA files, one per line, each optionally path:scale"},
	{Name: "strength", Label: "Strength", Type: "number", Default: 0.8, Description: "With an input image: how much is redrawn (0 = nothing, 1 = everything; the composition survives up to about 0.9)"},
	{Name: "mask", Label: "Mask", Type: "string", Default: "", Description: "With an input image: a mask image (white = regenerate) for inpainting"},
	{Name: "outdir", Label: "Output Folder", Type: "string", Default: "", Description: "Where new images (no input) go; empty = a Generated folder in the default library root"},
	{Name: "meta", Label: "Recipe (JSON)", Type: "string", Default: "", Description: "A JSON object stamped into the result's generation record as `meta`"},
	{Name: "nolibrary", Label: "Keep Out of Library", Type: "bool", Default: false, Description: "Scratch output (the Scene editor's layers): don't add it to the library"},
}

type dreamParams struct {
	Prompt, Negative, Size string
	Steps                  int
	CFG                    float64
	Turbo                  bool
	Seed                   int64
	Count                  int
	LoRAs                  []string
	Strength               float64
	Mask                   string
	OutDir                 string
	Meta, MetaFile         string
	NoLibrary              bool
	// Model is the --dit swap (Config.DreamModel); empty = the engine's default.
	Model string
}

func dreamParamsFromOptions(opts map[string]any) dreamParams {
	p := dreamParams{
		Prompt:    strings.TrimSpace(optString(opts, "prompt")),
		Negative:  strings.TrimSpace(optString(opts, "negative")),
		Size:      strings.TrimSpace(optString(opts, "size")),
		Steps:     int(optFloat(opts, "steps")),
		CFG:       optFloat(opts, "cfg"),
		Turbo:     optBool(opts, "turbo"),
		Seed:      int64(optFloat(opts, "seed")),
		Count:     clampInt(int(optFloat(opts, "count")), 1, 8),
		LoRAs:     splitPathList(optString(opts, "loras")),
		Strength:  clampFloat(optFloat(opts, "strength"), 0, 1),
		Mask:      strings.TrimSpace(optString(opts, "mask")),
		OutDir:    strings.TrimSpace(optString(opts, "outdir")),
		Meta:      strings.TrimSpace(optString(opts, "meta")),
		NoLibrary: optBool(opts, "nolibrary"),
	}
	if _, ok := opts["strength"]; !ok {
		p.Strength = 0.8
	}
	if p.Steps < 0 {
		p.Steps = 0
	}
	if p.Steps > 0 {
		p.Steps = clampInt(p.Steps, 1, 100)
	}
	return p
}

// buildDreamArgs is one loki-dream invocation: init (and mask) for img2img /
// inpainting, else text -> image; output is a file (one image) or a folder.
func buildDreamArgs(p dreamParams, init, output, promptFile string) []string {
	var args []string
	if promptFile != "" {
		args = append(args, "--prompt-file", promptFile)
	} else {
		args = append(args, "--prompt", p.Prompt)
	}
	if p.Negative != "" && !p.Turbo {
		args = append(args, "--negative", p.Negative)
	}
	if p.Size != "" {
		args = append(args, "--size", p.Size)
	}
	if p.Turbo {
		args = append(args, "--turbo")
	}
	if p.Steps > 0 {
		args = append(args, "--steps", strconv.Itoa(p.Steps))
	}
	if p.CFG > 0 {
		args = append(args, "--cfg", formatFloatArg(p.CFG))
	}
	args = append(args, "--seed", strconv.FormatInt(p.Seed, 10))
	if p.Count > 1 && init == "" {
		args = append(args, "--count", strconv.Itoa(p.Count))
	}
	for _, l := range p.LoRAs {
		args = append(args, "--lora", l)
	}
	if p.Model != "" {
		args = append(args, "--dit", p.Model)
	}
	if init != "" {
		args = append(args, "--init", init, "--strength", formatFloatArg(p.Strength))
		if p.Mask != "" {
			args = append(args, "--mask", p.Mask)
		}
	}
	// Same recipe flags as loki-retouch / loki-reshoot.
	if p.MetaFile != "" {
		args = append(args, "--meta-file", p.MetaFile)
	} else if p.Meta != "" {
		args = append(args, "--meta="+p.Meta)
	}
	return append(args, "-o", output)
}

// dreamDir is where new images go: a Generated folder in the default library
// root when it is a local folder, else in the server's data folder.
func dreamDir() string {
	if storageReg != nil {
		if b := storageReg.DefaultBackend(); b != nil {
			if root := b.Root(); root.Type != "s3" && root.Path != "" {
				if st, err := os.Stat(root.Path); err == nil && st.IsDir() {
					return filepath.Join(root.Path, "Generated")
				}
			}
		}
	}
	return filepath.Join(platform.GetDataDir(), "generated")
}

func dreamTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	defer beginGPUHeavy()()
	const tag = "dream"
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
	p := dreamParamsFromOptions(ParseOptions(j, dreamOptions))
	if p.Prompt == "" {
		return fail("needs a prompt", nil)
	}
	if p.Model = strings.TrimSpace(appconfig.Get().DreamModel); p.Model != "" {
		q.PushJobStdout(j.ID, tag+": model "+p.Model)
	}
	tool, err := resolveAITool(ctx, q, j.ID, tag, dreamTool)
	if err != nil {
		return fail(err.Error(), err)
	}
	defer tool.Release()
	res, rerr := resolveJobItemsRaw(j, q)
	if rerr != nil {
		return fail("failed to resolve input: "+rerr.Error(), rerr)
	}
	var inits []string
	for _, f := range res.Paths {
		if isImageExt(filepath.Ext(f)) {
			inits = append(inits, absPath(f))
		} else {
			q.PushJobStdout(j.ID, tag+": skipping (not a still image): "+filepath.Base(f))
		}
	}

	seed, random := resolveSeed(p.Seed)
	q.PushJobStdout(j.ID, fmt.Sprintf("%s: seed %d%s", tag, seed, map[bool]string{true: " (random)", false: ""}[random]))
	promptFile, perr := writePromptFile(p.Prompt)
	if perr != nil {
		return fail("failed to write prompt file: "+perr.Error(), perr)
	}
	if promptFile != "" {
		defer os.Remove(promptFile)
	}
	cleanupMeta, merr := prepareMeta(ctx, q, j.ID, tag, tool, &p.Meta, &p.MetaFile)
	if merr != nil {
		return fail("failed to write recipe file: "+merr.Error(), merr)
	}
	defer cleanupMeta()

	terminal := !q.HasDependents(j.ID)
	var created []string
	defer func() {
		if terminal && !p.NoLibrary {
			broadcastMediaCreated(created)
		}
	}()
	// One run per input image, or one text -> image run (Count images).
	type run struct{ init, output string }
	var runs []run
	if len(inits) == 0 {
		dir := p.OutDir
		if dir == "" {
			dir = dreamDir()
		}
		if err := os.MkdirAll(dir, 0755); err != nil {
			return fail("cannot create "+dir+": "+err.Error(), err)
		}
		out := dir
		if p.Count == 1 {
			out = filepath.Join(dir, fmt.Sprintf("dream_%s_%d.png", time.Now().Format("20060102-150405"), seed))
		}
		runs = append(runs, run{output: out})
	} else {
		for _, in := range inits {
			dir, derr := aiOutputDir(j, terminal, in)
			if derr != nil {
				return fail("failed to create temp dir: "+derr.Error(), derr)
			}
			out := filepath.Join(dir, strings.TrimSuffix(filepath.Base(in), filepath.Ext(in))+"_dream.png")
			if _, err := os.Stat(out); err == nil {
				out = resolveConflict(out)
			}
			runs = append(runs, run{init: in, output: out})
		}
	}

	_ = q.SetJobProgress(j.ID, 0, len(runs)*100)
	for i, r := range runs {
		if ctx.Err() != nil {
			return fail("task canceled", ctx.Err())
		}
		rp := p
		rp.Seed = seed + int64(i)
		args := buildDreamArgs(rp, r.init, r.output, promptFile)
		if r.init != "" {
			q.PushJobStdout(j.ID, fmt.Sprintf("%s: %s %s -> %s", tag, map[bool]string{true: "inpainting", false: "reimagining"}[p.Mask != ""], filepath.Base(r.init), filepath.Base(r.output)))
		} else {
			q.PushJobStdout(j.ID, fmt.Sprintf("%s: generating %d image(s)", tag, p.Count))
		}
		base := i * 100
		out, werr := runAICLI(ctx, q, j.ID, tag, tool, args, func(s, n int) {
			if n > 0 {
				_ = q.SetJobProgress(j.ID, base+s*99/n, len(runs)*100)
			}
		})
		if ctx.Err() != nil {
			return fail("task canceled", ctx.Err())
		}
		if werr != nil {
			return fail("loki-dream failed: "+werr.Error(), werr)
		}
		// stdout: the path of every image written, one per line.
		for _, line := range out {
			path := strings.TrimSpace(line)
			info, err := os.Stat(path)
			if path == "" || err != nil || info.IsDir() {
				continue
			}
			if terminal && !p.NoLibrary {
				if err := insertMediaRecord(q.Db, path, info.Size()); err != nil {
					q.PushJobStdout(j.ID, tag+": failed to add to library: "+err.Error())
				}
			}
			src := r.init
			if src == "" {
				src = path
			}
			q.RegisterOutputFile(j.ID, path, src)
			created = append(created, path)
			q.PushJobStdout(j.ID, tag+": wrote "+path)
		}
	}
	if len(created) == 0 {
		return fail("no image was produced", nil)
	}
	_ = q.SetJobProgress(j.ID, len(runs)*100, len(runs)*100)
	q.CompleteJob(j.ID)
	return nil
}
