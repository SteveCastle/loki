package tasks

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/mediaext"
	"github.com/stevecastle/shrike/platform"
)

// --- prompt preview (POST /api/transform/prompt) ---
//
// Shows what the model will actually be given for a retouch / reshoot job
// before it is queued: the job's arguments go through the same planning and
// argument building as the task, and the engine itself expands the prompt
// (`--show-prompt`, which loads no models and returns in milliseconds). So
// presets, --append, living-photo expansion and reference tokens all read
// exactly as they will run.

// PromptReference is one image/clip/audio the model sees, with the token the
// prompt uses for it.
type PromptReference struct {
	Token string `json:"token"`
	Path  string `json:"path"`
	Kind  string `json:"kind"` // image | video-frame | video | audio
}

// PromptPreview is the expanded prompt of the first run of a job.
type PromptPreview struct {
	Engine string `json:"engine"`
	// Prompt is the instruction as the engine expands it.
	Prompt string `json:"prompt"`
	// ModelInput is the whole text-encoder turn (retouch): every reference
	// announced by its token with an image block, then the prompt.
	ModelInput string            `json:"modelInput,omitempty"`
	References []PromptReference `json:"references"`
	// Runs is how many engine runs the job makes (one per image in a batch).
	Runs int `json:"runs"`
	// FromEngine is false when the engine is not installed and the prompt
	// was assembled here (only possible when nothing needs expanding).
	FromEngine bool   `json:"fromEngine"`
	Note       string `json:"note,omitempty"`
}

// ErrPreviewUnsupported marks requests the preview cannot answer (bad input,
// unknown task); the handler maps it to a 4xx.
var ErrPreviewUnsupported = errors.New("prompt preview")

func previewErr(format string, a ...any) error {
	return fmt.Errorf("%w: %s", ErrPreviewUnsupported, fmt.Sprintf(format, a...))
}

// retouchSystemPrompt / retouchVisionBlock / retouchModelInput mirror
// loki-retouch's text-encoder template (tools/loki-retouch/src/text_encoder.rs).
const (
	retouchSystemPrompt = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
	retouchVisionBlock  = "<|vision_start|><|image_pad|><|vision_end|>"
)

func retouchModelInput(prompt string, nImages int) string {
	refs := make([]string, nImages)
	for i := range refs {
		refs[i] = fmt.Sprintf("<image%d>%s", i+1, retouchVisionBlock)
	}
	return retouchSystemPrompt + "<|im_start|>user\n" + strings.Join(refs, " ") + prompt + "<|im_end|>\n<|im_start|>assistant\n"
}

// PreviewPrompt previews a retouch or reshoot job given as /create would
// queue it: the task id, its option arguments and the newline-separated
// input paths.
func PreviewPrompt(ctx context.Context, command string, args []string, input string) (PromptPreview, error) {
	j := &jobqueue.Job{Command: command, Arguments: args, Input: input}
	if _, ok := extractQueryFromJob(j); ok {
		return PromptPreview{}, previewErr("a query input cannot be previewed; select the files")
	}
	files := parseExplicitPaths(input, false)
	if len(files) == 0 {
		return PromptPreview{}, previewErr("no input files")
	}
	switch command {
	case "retouch":
		return previewRetouch(ctx, retouchParamsFromOptions(ParseOptions(j, retouchOptions)), files)
	case "reshoot":
		return previewReshoot(ctx, reshootParamsFromOptions(ParseOptions(j, reshootOptions)), files)
	}
	return PromptPreview{}, previewErr("no prompt preview for task %q", command)
}

func previewRetouch(ctx context.Context, p retouchParams, files []string) (PromptPreview, error) {
	out := PromptPreview{Engine: "loki-retouch"}
	if p.Prompt == "" && p.Preset == "" {
		return out, previewErr("needs a prompt or a preset")
	}
	items, refs := planRetouchRuns(files, p, func(string) {})
	var runnable []retouchItem
	for _, it := range items {
		if isImageExt(filepath.Ext(it.src)) || mediaext.IsVideo(it.src) {
			runnable = append(runnable, it)
		}
	}
	if len(runnable) == 0 {
		return out, previewErr("no image to edit")
	}
	out.Runs = len(runnable)
	item := runnable[0]
	src := absPath(item.src)
	itemRefs := append(append([]string{}, item.refs...), refs...)

	input := src
	kind := "image"
	if mediaext.IsVideo(src) {
		kind = "video-frame"
		frame, err := sampleVideoFrame(ctx, src, p.Time)
		if err != nil {
			return out, fmt.Errorf("sampling a frame from %s: %w", filepath.Base(src), err)
		}
		defer os.RemoveAll(filepath.Dir(frame))
		input = frame
	}
	out.References = append(out.References, PromptReference{Token: "<image1>", Path: src, Kind: kind})
	for i, r := range itemRefs {
		out.References = append(out.References, PromptReference{Token: fmt.Sprintf("<image%d>", i+2), Path: r, Kind: "image"})
	}

	if tool, ok := locateAITool(retouchTool); ok {
		ip := p
		ip.Seed = 0
		prompt, err := runShowPrompt(ctx, tool, p.Prompt, func(promptFile string) []string {
			return buildRetouchArgs(ip, input, itemRefs, filepath.Join(os.TempDir(), "loki-preview.png"), promptFile)
		})
		if err != nil {
			return out, err
		}
		out.Prompt, out.FromEngine = prompt, true
	} else if p.Preset == "" {
		// A free-form prompt is used verbatim; only --append is added.
		out.Prompt = p.Prompt
		if p.Append != "" {
			out.Prompt += "\n\n" + p.Append
		}
		out.Note = "loki-retouch is not installed yet; this is the prompt as it will be sent."
	} else {
		out.Note = "The " + strconv.Quote(p.Preset) + " preset's prompt is written by loki-retouch, which is not installed yet."
	}
	if out.Prompt != "" {
		out.ModelInput = retouchModelInput(out.Prompt, len(out.References))
	}
	if out.Runs > 1 {
		out.Note = strings.TrimSpace(out.Note + fmt.Sprintf(" Shown for the first of %d images: each is edited on its own as <image1>.", out.Runs))
	}
	return out, nil
}

func previewReshoot(ctx context.Context, p reshootParams, files []string) (PromptPreview, error) {
	out := PromptPreview{Engine: "loki-reshoot", Runs: 1}
	images, videos, audios, _ := planReshootInputs(files, p, func(string) {})
	if len(images)+len(videos)+len(audios) == 0 {
		return out, previewErr("needs at least one reference image, video or audio file")
	}
	if p.Animate && len(images) == 0 {
		return out, previewErr("living photo (animate) needs an image input")
	}
	if !p.Animate && p.Prompt == "" {
		return out, previewErr("needs a prompt")
	}
	for i, f := range images {
		out.References = append(out.References, PromptReference{Token: fmt.Sprintf("<Picture %d>", i+1), Path: f, Kind: "image"})
	}
	for i, f := range videos {
		out.References = append(out.References, PromptReference{Token: fmt.Sprintf("<Video %d>", i+1), Path: f, Kind: "video"})
	}
	for i, f := range audios {
		out.References = append(out.References, PromptReference{Token: fmt.Sprintf("<Audio %d>", i+1), Path: f, Kind: "audio"})
	}

	if tool, ok := locateAITool(reshootTool); ok {
		ip := p
		ip.Seed = 0
		prompt, err := runShowPrompt(ctx, tool, p.Prompt, func(promptFile string) []string {
			return buildReshootArgs(ip, images, videos, audios, filepath.Join(os.TempDir(), "loki-preview.mp4"), promptFile)
		})
		if err != nil {
			return out, err
		}
		out.Prompt, out.FromEngine = prompt, true
	} else if !p.Animate {
		out.Prompt = p.Prompt
		out.Note = "loki-reshoot is not installed yet; this is the prompt as it will be sent."
	} else {
		out.Note = "The living-photo prompt is written by loki-reshoot, which is not installed yet."
	}
	return out, nil
}

// runShowPrompt runs the engine with the job's own arguments plus
// --show-prompt and returns what it prints. Long prompts go through a prompt
// file exactly like the task does.
func runShowPrompt(ctx context.Context, tool aiToolRun, prompt string, build func(promptFile string) []string) (string, error) {
	promptFile, err := writePromptFile(prompt)
	if err != nil {
		return "", err
	}
	if promptFile != "" {
		defer os.Remove(promptFile)
	}
	ctx, cancel := context.WithTimeout(ctx, 20*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, tool.Bin, append([]string{"--show-prompt"}, build(promptFile)...)...)
	if len(tool.Env) > 0 {
		cmd.Env = append(os.Environ(), tool.Env...)
	}
	platform.HideSubprocessWindow(cmd)
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	if err := cmd.Run(); err != nil {
		msg := strings.TrimSpace(stderr.String())
		if msg == "" {
			msg = err.Error()
		}
		return "", fmt.Errorf("%s --show-prompt: %s", filepath.Base(tool.Bin), msg)
	}
	return strings.TrimRight(strings.ReplaceAll(stdout.String(), "\r\n", "\n"), "\n"), nil
}
