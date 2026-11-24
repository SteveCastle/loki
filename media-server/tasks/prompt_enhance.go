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
// was trained to read, with loki-scribe. The user asks for it: the Enhance
// button puts the rewrite in the prompt box to review before running, and the
// review step's "Enhance first" box sends each queued run through its own
// rewrite (seeded like the run, so runs of one prompt read differently) before
// the generation is queued with it. A generation never enhances on its own.
// The rewrite runs as an enhance-prompt job (prompt_enhance_task.go), so the
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
	// Seed (optional) seeds the text model's sampling: the same request and
	// seed give the same rewrite, another seed another one. 0 = the engine's
	// default seed.
	Seed uint64 `json:"seed,omitempty"`
	// Chunks (video, optional) are the frame counts of a long video's chunks
	// (loki-reshoot's plan; see extend.rs). Two or more make the prompt a
	// chunked one: Prompt is the part every chunk shares (or the whole story,
	// to be split), Segments each chunk's own part (see ComposeChunkPrompt).
	Chunks []int `json:"chunks,omitempty"`
	// Segments are the user's own texts for chunks 1, 2, ... (optional).
	Segments []string `json:"segments,omitempty"`
}

// EnhanceResult is the rewritten prompt.
type EnhanceResult struct {
	// Prompt is the rewrite; for a chunked prompt, the part every chunk shares.
	Prompt string `json:"prompt"`
	// Segments are a chunked prompt's per-chunk parts, chunks 1, 2, ...
	Segments []string `json:"segments,omitempty"`
	Model    string   `json:"model"`
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
	if spans := chunkSpans(req.Chunks); len(spans) > 1 {
		total := spans[len(spans)-1][1]
		fmt.Fprintf(&b, "\nThis video is %s long, longer than the model makes at once, so it is generated in %d consecutive segments, each from its own prompt:\n", secs(total), len(spans))
		for i, sp := range spans {
			fmt.Fprintf(&b, "- segment %d: %s to %s (%s)\n", i+1, secs(sp[0]), secs(sp[1]), secs(sp[1]-sp[0]))
		}
		b.WriteString("Each segment continues seamlessly from the last moment of the one before (same shot, no cut). The model sees one segment at a time: the SHARED part with that segment's own part put in place of the line {chunk}. Write the prompt in that form:\n")
		fmt.Fprintf(&b, "- First the SHARED part: everything that holds for the whole video, in the guide's section order (who or what is in it with their reference tokens, appearance, clothing and voice; the setting; the style and light; the ongoing soundscape and the music). Where the time-dependent sections go (for this guide: summary and detailed_description), write a line containing only {chunk}.\n")
		fmt.Fprintf(&b, "- Then a line containing only ===, then the %d segment parts in order, separated by lines containing only ---. A segment part holds only the time-dependent sections, for that segment's time span. No headings, numbers or labels of your own.\n", len(spans))
		b.WriteString("- Never repeat the shared descriptions inside the segment parts; refer to people and things by the same labels.\n")
		b.WriteString("- Divide the user's action, dialogue and sound events over the segments in story order, about as much as fits each segment's length: each happens in exactly one segment and is never repeated. Each spoken line goes whole into one segment.\n")
		b.WriteString("- Segment 2 onwards picks up mid-scene where the previous segment ended: say where people are and what they are doing at that moment, then what happens next. Never describe an opening, an establishing shot, an introduction, a cut or an ending before the last segment.\n")
		b.WriteString("- If the user already wrote text for some segments, keep its content in those segments.\n")
	}
	b.WriteString("\n--- Model guide ---\n\n")
	b.WriteString(guide)
	b.WriteString("\n")
	return b.String()
}

// enhanceContextFrames is loki-reshoot's long-video context: every chunk after
// the first re-makes this many frames of the previous one before going on.
const enhanceContextFrames = 22

// chunkSpans turns a long video's chunk frame counts into the [start, end]
// seconds each chunk adds to the video.
func chunkSpans(chunks []int) [][2]float64 {
	var out [][2]float64
	end := 0.0
	for i, f := range chunks {
		n := f
		if i > 0 {
			n -= enhanceContextFrames
		}
		if n <= 0 {
			return nil
		}
		out = append(out, [2]float64{end, end + float64(n)/24})
		end += float64(n) / 24
	}
	return out
}

func secs(s float64) string { return fmt.Sprintf("%.1f s", s) }

// ChunkSlot is the line of a chunked prompt's shared part where each chunk's
// own part goes; without it the chunk's part follows the shared one.
const ChunkSlot = "{chunk}"

// ComposeChunkPrompt is the prompt one chunk of a long video runs with: the
// shared part with the chunk's own part in its {chunk} line (or after it).
// The renderer's composeChunkPrompt (intents.ts) is the same rule.
func ComposeChunkPrompt(shared, segment string) string {
	segment = strings.TrimSpace(segment)
	lines := strings.Split(strings.ReplaceAll(shared, "\r\n", "\n"), "\n")
	for i, l := range lines {
		if strings.TrimSpace(l) == ChunkSlot {
			out := append(append(append([]string{}, lines[:i]...), segment), lines[i+1:]...)
			if segment == "" {
				out = append(append([]string{}, lines[:i]...), lines[i+1:]...)
			}
			return strings.TrimSpace(strings.Join(out, "\n"))
		}
	}
	var parts []string
	for _, p := range []string{strings.TrimSpace(shared), segment} {
		if p != "" {
			parts = append(parts, p)
		}
	}
	return strings.Join(parts, "\n\n")
}

// enhanceSegmentLabel is a heading a model puts above a part anyway.
var enhanceSegmentLabel = regexp.MustCompile(`(?i)^[#*\s]*(?:(?:segment|chunk|part|clip)\s*\d+\b|shared(?: part)?\b)[^\n]{0,60}\n`)

// enhanceSharedEnd is the line between a chunked rewrite's shared part and its
// segments.
var enhanceSharedEnd = regexp.MustCompile(`(?m)^[ \t]*={3,}[ \t]*$`)

// splitEnhancedChunks splits a long video's rewrite into its shared part and
// n segment parts: extra segments join the last one, missing ones are empty.
// An answer without the === line is all segments (each then a whole prompt).
func splitEnhancedChunks(answer string, n int) (string, []string) {
	// (the reasoning and a code fence wrap the whole answer)
	answer = cleanEnhanced(answer)
	shared, rest := "", answer
	if loc := enhanceSharedEnd.FindStringIndex(answer); loc != nil {
		shared, rest = answer[:loc[0]], answer[loc[1]:]
	}
	clean := func(p string) string {
		return cleanEnhanced(strings.TrimSpace(enhanceSegmentLabel.ReplaceAllString(strings.TrimSpace(p), "")))
	}
	var parts []string
	for _, p := range splitChunkPrompts(rest) {
		if p = clean(p); p != "" {
			parts = append(parts, p)
		}
	}
	if len(parts) > n {
		parts = append(parts[:n-1], strings.Join(parts[n-1:], "\n\n"))
	}
	for len(parts) < n {
		parts = append(parts, "")
	}
	return clean(shared), parts
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
	if strings.TrimSpace(req.Prompt+strings.Join(req.Segments, "")) == "" {
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
	user := "Rewrite this prompt:\n\n" + strings.TrimSpace(req.Prompt)
	maxTokens := enhanceMaxTokens
	segments := len(chunkSpans(req.Chunks))
	if segments > 1 {
		user = fmt.Sprintf("Write the shared part and the %d segment parts for this video.", segments)
		if p := strings.TrimSpace(req.Prompt); p != "" {
			user += "\n\nThe user's prompt (what is shared, or the whole story to divide):\n\n" + p
		}
		for i, sg := range req.Segments {
			if sg = strings.TrimSpace(sg); sg != "" && i < segments {
				user += fmt.Sprintf("\n\nWhat the user wrote for segment %d:\n%s", i+1, sg)
			}
		}
		maxTokens = min(600+700*segments, 6000)
	}
	text := textRequest{
		System:      enhanceSystemPrompt(req, guide),
		Prompt:      user,
		MaxTokens:   maxTokens,
		Temperature: 0.7,
		NoRepeat:    true,
		Seed:        req.Seed,
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
	if segments > 1 {
		out.Prompt, out.Segments = splitEnhancedChunks(answer, segments)
		written := 0
		for _, sg := range out.Segments {
			if sg != "" {
				written++
			}
		}
		if logf != nil && written < segments {
			logf(fmt.Sprintf("note: the text model wrote %d of %d segment parts", written, segments))
		}
	} else {
		out.Prompt = cleanEnhanced(answer)
	}
	if strings.TrimSpace(out.Prompt+strings.Join(out.Segments, "")) == "" {
		return out, errors.New("the text model returned an empty prompt")
	}
	in := strings.Join(append([]string{req.Prompt}, req.Segments...), "\n")
	all := strings.Join(append([]string{out.Prompt}, out.Segments...), "\n")
	for _, r := range req.References {
		if strings.Contains(in, r.Token) && !strings.Contains(all, r.Token) {
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
