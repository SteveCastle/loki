package tasks

import (
	"context"
	"embed"
	"errors"
	"fmt"
	"os"
	"regexp"
	"strings"
	"time"
)

// --- prompt enhancement (POST /api/transform/enhance-prompt) ---
//
// Rewrites a rough Transform prompt into one written the way the target model
// was trained to read, with loki-scribe. The user asks for it (a button in the
// Transform flow): it never runs as part of a job, so generation stays as fast
// as before and the prompt that runs is always one the user has seen.
//
// Each target model has a guide in prompt_guides/<model>.md, distilled from
// its official prompting docs; the guide is the heart of the system prompt.

//go:embed prompt_guides/*.md
var promptGuideFS embed.FS

// EnhanceReference is one input the prompt may name by its token.
type EnhanceReference struct {
	Token string `json:"token"`
	Kind  string `json:"kind"` // image | video | audio
}

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
	// rewrite: the picture being edited or redrawn, or the first reference.
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
	if req.Image != "" {
		if req.ImageToken != "" {
			fmt.Fprintf(&b, "\nThe attached image is %s. Look at it to get details right (who or what is there, the setting, the light), but write only what the prompt needs: do not describe what should stay unchanged at length.\n", req.ImageToken)
		} else {
			b.WriteString("\nThe attached image is the picture the prompt works on. Look at it to get details right, but write only what the prompt needs.\n")
		}
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

// EnhancePrompt rewrites req.Prompt for req.Model.
func EnhancePrompt(ctx context.Context, req EnhanceRequest) (EnhanceResult, error) {
	out := EnhanceResult{Model: req.Model}
	if strings.TrimSpace(req.Prompt) == "" {
		return out, fmt.Errorf("%w: write a prompt first", ErrEnhanceInput)
	}
	guide, ok := promptGuide(req.Model)
	if !ok {
		return out, fmt.Errorf("%w: no prompt guide for model %q", ErrEnhanceInput, req.Model)
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
	var answer string
	var err error
	if req.Image != "" {
		if _, statErr := os.Stat(req.Image); statErr != nil {
			return out, fmt.Errorf("%w: image not found: %s", ErrEnhanceInput, req.Image)
		}
		answer, err = imageText(ctx, nil, "", req.Image, text)
	} else {
		answer, err = generateText(ctx, nil, text)
	}
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
