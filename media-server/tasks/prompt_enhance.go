package tasks

import (
	"context"
	"embed"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/stevecastle/shrike/mediaext"
)

// --- prompt enhancement (POST /api/transform/enhance-prompt) ---
//
// Rewrites a rough Transform prompt into one written the way the target model
// was trained to read, with loki-scribe. The user asks for it (a button in the
// Transform flow): it is never a step of a generation, so generation stays as
// fast as before and the prompt that runs is always one the user has seen. The
// rewrite itself runs as an enhance-prompt job (prompt_enhance_task.go), so the
// text model takes its turn with the other model work.
//
// Each target model has a guide in prompt_guides/<model>.md, distilled from
// its official prompting docs; the guide is the heart of the system prompt.

//go:embed prompt_guides/*.md
var promptGuideFS embed.FS

// EnhanceReference is one input the prompt may name by its token.
type EnhanceReference struct {
	Token string `json:"token"`
	Kind  string `json:"kind"` // image | video | audio
	// Path (optional) is the input's local file: images and videos (a frame)
	// are shown to the text model so the rewrite can use what they contain.
	Path string `json:"path,omitempty"`
}

// enhanceMaxImages bounds how many pictures the text model reads at once.
const enhanceMaxImages = 6

// EnhanceRequest is POST /api/transform/enhance-prompt's body.
type EnhanceRequest struct {
	Prompt string `json:"prompt"`
	// Model is a prompt_guides key: h3-video, h3-image, qwen-image-edit,
	// qwen-image, moss-image, flux2-klein, krea2, anima, sdxl.
	Model string `json:"model"`
	// Task is what the prompt drives: generate | reimagine | edit | video.
	Task       string             `json:"task"`
	References []EnhanceReference `json:"references"`
	// Image (optional) is a local file the model looks at to ground the
	// rewrite, besides the references' own files: the picture being redrawn
	// (which has no token), or an edit's image when it has no reference entry.
	Image string `json:"image"`
	// ImageToken is the token Image goes by in the prompt, if it has one.
	ImageToken string `json:"imageToken"`
}

// EnhanceResult is the rewritten prompt.
type EnhanceResult struct {
	Prompt string `json:"prompt"`
	Model  string `json:"model"`
	// MissingTokens are reference tokens the input prompt used that the
	// rewrite dropped (the UI warns; the user decides).
	MissingTokens []string `json:"missingTokens,omitempty"`
}

// ErrEnhanceInput marks requests that cannot be enhanced (the handler maps it
// to a 4xx).
var ErrEnhanceInput = errors.New("prompt enhancement")

// enhanceMaxTokens bounds the rewrite; H3 video prompts run several hundred
// words with their audio and camera lines.
const enhanceMaxTokens = 1200

// PromptGuideModels lists the models that have a guide.
func PromptGuideModels() []string {
	entries, _ := promptGuideFS.ReadDir("prompt_guides")
	out := make([]string, 0, len(entries))
	for _, e := range entries {
		out = append(out, strings.TrimSuffix(e.Name(), ".md"))
	}
	return out
}

func promptGuide(model string) (string, bool) {
	if strings.ContainsAny(model, `/\.`) || model == "" {
		return "", false
	}
	b, err := promptGuideFS.ReadFile("prompt_guides/" + model + ".md")
	if err != nil {
		return "", false
	}
	// The sources comment is for maintainers, not the model.
	return strings.TrimSpace(htmlComment.ReplaceAllString(string(b), "")), true
}

var htmlComment = regexp.MustCompile(`(?s)<!--.*?-->\n?`)

var enhanceTaskLines = map[string]string{
	"generate":  "The prompt creates a new picture from text alone.",
	"reimagine": "The prompt redraws an existing picture (image-to-image): it must describe the whole intended result, not just the change.",
	"edit":      "The prompt is an edit instruction applied to an existing picture.",
	"video":     "The prompt directs a video generated from the references, with its own sound.",
}

// enhanceImage is one picture shown to the text model, in attachment order.
type enhanceImage struct {
	path  string // as given (a video for a frame)
	token string // "" = the untokened source picture
	kind  string // image | video
}

// enhanceImages lists the pictures the text model reads: the request's
// Image first, then every image or video reference with a file, each once.
func enhanceImages(req EnhanceRequest) []enhanceImage {
	var out []enhanceImage
	seen := map[string]bool{}
	add := func(p, token, kind string) {
		if p == "" || len(out) >= enhanceMaxImages {
			return
		}
		k := strings.ToLower(p)
		if seen[k] {
			// the same file as a reference: give the earlier entry its token
			for i := range out {
				if strings.ToLower(out[i].path) == k && out[i].token == "" {
					out[i].token = token
				}
			}
			return
		}
		seen[k] = true
		out = append(out, enhanceImage{path: p, token: token, kind: kind})
	}
	if req.Image != "" {
		add(req.Image, req.ImageToken, "image")
	}
	for _, r := range req.References {
		if r.Kind == "image" || r.Kind == "video" {
			add(r.Path, r.Token, r.Kind)
		}
	}
	return out
}

func enhanceSystemPrompt(req EnhanceRequest, guide string) string {
	var b strings.Builder
	b.WriteString("You are an expert prompt writer for generative image and video models. ")
	b.WriteString("Rewrite the user's prompt into the best possible prompt for the model described in the guide below, following the guide's structure and rules.\n\n")
	b.WriteString("Hard rules:\n")
	b.WriteString("- Keep the user's intent, subject and every concrete detail they gave. Add only what the guide says the model needs (composition, lighting, camera, motion, sound, style words) and keep additions consistent with the request.\n")
	b.WriteString("- Never add people, nudity, brand names, logos or lettering the user did not ask for. Keep any text the user wants rendered exactly as written.\n")
	b.WriteString("- Write the prompt in English, unless the user asks for lettering in another language.\n")
	b.WriteString("- Reply with the final prompt only: no preamble, explanation, title, markdown or surrounding quotes.\n")
	if line, ok := enhanceTaskLines[req.Task]; ok {
		b.WriteString("\nTask: " + line + "\n")
	}
	if len(req.References) > 0 {
		b.WriteString("\nThe model is given these inputs, and the prompt refers to them by these exact tokens:\n")
		for _, r := range req.References {
			fmt.Fprintf(&b, "- %s (%s)\n", r.Token, r.Kind)
		}
		b.WriteString("Use these tokens verbatim (same spelling, brackets and spacing) wherever the prompt refers to an input, keep every token the user used, and never invent other tokens. If the guide shows a different reference syntax, use these tokens instead.\n")
	}
	if imgs := enhanceImages(req); len(imgs) > 0 {
		if len(imgs) == 1 {
			b.WriteString("\nOne image is attached: ")
		} else {
			b.WriteString("\nThese images are attached, in this order:\n")
		}
		for i, im := range imgs {
			what := "the picture the prompt works on"
			switch {
			case im.token != "" && im.kind == "video":
				what = "a frame from " + im.token
			case im.token != "":
				what = im.token
			}
			if len(imgs) == 1 {
				b.WriteString(what + ".\n")
			} else {
				fmt.Fprintf(&b, "%d. %s\n", i+1, what)
			}
		}
		b.WriteString("Look at them to build the prompt from what they really show: who or what is in each reference (appearance, clothing, objects, style), the setting and the light, so the prompt can say precisely what to take from which input. Write only what the prompt needs: do not describe what should stay unchanged at length, and do not describe a reference in place of naming it by its token.\n")
	}
	b.WriteString("\n--- Model guide ---\n\n")
	b.WriteString(guide)
	b.WriteString("\n")
	return b.String()
}

var enhanceLabel = regexp.MustCompile(`(?i)^(?:\*\*)?(?:optimized |enhanced |final |rewritten )?prompt(?:\*\*)?\s*:(?:\*\*)?\s*`)

// cleanEnhanced strips what a chat model wraps an answer in.
func cleanEnhanced(s string) string {
	s = strings.TrimSpace(s)
	if i := strings.Index(s, "</think>"); i >= 0 {
		s = strings.TrimSpace(s[i+len("</think>"):])
	}
	s = strings.TrimPrefix(s, "```text")
	s = strings.TrimPrefix(s, "```")
	s = strings.TrimSuffix(s, "```")
	s = strings.TrimSpace(s)
	s = enhanceLabel.ReplaceAllString(s, "")
	return strings.TrimSpace(trimQuotes(s))
}

func trimQuotes(s string) string {
	for _, q := range [][2]string{{`"`, `"`}, {"“", "”"}} {
		if strings.HasPrefix(s, q[0]) && strings.HasSuffix(s, q[1]) && len(s) > len(q[0])+len(q[1]) {
			inner := s[len(q[0]) : len(s)-len(q[1])]
			// Only an outer pair: a prompt that quotes lettering inside keeps it.
			if !strings.Contains(inner, q[0]) && !strings.Contains(inner, q[1]) {
				return inner
			}
		}
	}
	return s
}

// validateEnhance checks what can be checked before any model work.
func validateEnhance(req EnhanceRequest) (string, error) {
	if strings.TrimSpace(req.Prompt) == "" {
		return "", fmt.Errorf("%w: write a prompt first", ErrEnhanceInput)
	}
	guide, ok := promptGuide(req.Model)
	if !ok {
		return "", fmt.Errorf("%w: no prompt guide for model %q", ErrEnhanceInput, req.Model)
	}
	return guide, nil
}

// EnhancePrompt rewrites req.Prompt for req.Model, here and now: callers go
// through the job queue (RunEnhanceJob); this is the job's body.
func EnhancePrompt(ctx context.Context, req EnhanceRequest) (EnhanceResult, error) {
	return enhancePrompt(ctx, req, nil)
}

func enhancePrompt(ctx context.Context, req EnhanceRequest, logf func(string)) (EnhanceResult, error) {
	out := EnhanceResult{Model: req.Model}
	guide, err := validateEnhance(req)
	if err != nil {
		return out, err
	}
	text := textRequest{
		System:      enhanceSystemPrompt(req, guide),
		Prompt:      "Rewrite this prompt:\n\n" + strings.TrimSpace(req.Prompt),
		MaxTokens:   enhanceMaxTokens,
		Temperature: 0.7,
		NoRepeat:    true,
	}
	ctx, cancel := context.WithTimeout(ctx, 10*time.Minute)
	defer cancel()
	if req.Image != "" {
		if _, statErr := os.Stat(req.Image); statErr != nil {
			return out, fmt.Errorf("%w: image not found: %s", ErrEnhanceInput, req.Image)
		}
	}
	var cleanup []string
	defer func() {
		for _, p := range cleanup {
			_ = os.Remove(p)
		}
	}()
	for _, im := range enhanceImages(req) {
		p, tmp, err := visionImage(ctx, im.path)
		if err != nil {
			return out, fmt.Errorf("reading %s: %w", filepath.Base(im.path), err)
		}
		cleanup = append(cleanup, tmp...)
		text.Images = append(text.Images, p)
	}
	if n := len(text.Images); n > 1 {
		// several pictures share the context: fewer vision tokens each
		text.MaxImageTokens = max(384, min(1024, 4096/n))
	}
	answer, err := generateText(ctx, logf, text)
	if err != nil {
		return out, err
	}
	out.Prompt = cleanEnhanced(answer)
	if out.Prompt == "" {
		return out, errors.New("the text model returned an empty prompt")
	}
	for _, r := range req.References {
		if strings.Contains(req.Prompt, r.Token) && !strings.Contains(out.Prompt, r.Token) {
			out.MissingTokens = append(out.MissingTokens, r.Token)
		}
	}
	return out, nil
}

// visionImage readies one file for the text model the way imageText does:
// images it can read as they are (resized if large), anything else (a video,
// heic, tiff) through an ffmpeg frame. tmp lists files to remove afterwards.
func visionImage(ctx context.Context, path string) (string, []string, error) {
	var tmp []string
	src := path
	if !mediaext.IsDirectVisionImage(path) {
		frame, err := extractVideoFrame(ctx, path, "")
		if err != nil {
			return "", nil, fmt.Errorf("extracting a frame: %w", err)
		}
		tmp = append(tmp, frame)
		src = frame
	}
	resized, err := resizeImageIfNeeded(src)
	if err != nil {
		return "", tmp, fmt.Errorf("resizing: %w", err)
	}
	if resized != src {
		tmp = append(tmp, resized)
	}
	return resized, tmp, nil
}
