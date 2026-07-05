package tasks

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/color"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/stevecastle/shrike/appconfig"
)

// Inference provider identifiers. Persisted in Config.InferenceProvider and
// driven from the Inference tab in the config UI. Add a new constant + a
// matching case in callVisionLLM to wire in a new backend.
const (
	InferenceProviderOff      = "off"
	InferenceProviderOllama   = "ollama"
	InferenceProviderRunPod   = "runpod"
	InferenceProviderLMStudio = "lmstudio"
	InferenceProviderLlamaCpp = "llamacpp"
)

// Host buckets used for jobqueue concurrency on inference work. Each
// provider gets its own bucket so its limit can be tuned independently —
// local single-GPU backends typically want 1 at a time, while RunPod
// serverless can take many in parallel because it scales out per request.
// A new engine adds one constant here, one case in InferenceHost, one
// field on appconfig.InferenceConcurrency, and one bump in ApplyHostLimits.
const (
	HostBucketOllama   = "ollama"
	HostBucketRunPod   = "runpod"
	HostBucketLMStudio = "lmstudio"
	HostBucketLlamaCpp = "llamacpp"
	// HostBucketEmbed is the local visual-embedding bucket. It is intentionally
	// separate from the LLM inference buckets so embedding doesn't compete with
	// autotag/description jobs and isn't capped by the LLM provider's setting.
	// One embed job at a time is enough: each job parallelizes internally via
	// its own worker pool (see runEmbedPool).
	HostBucketEmbed = "embed"
	// HostBucketAutotag is the local ONNX auto-tagging bucket, separate from the
	// LLM inference bucket for the same reasons as embed. Each autotag job
	// parallelizes internally via its own worker pool (see runAutotagPool).
	HostBucketAutotag = "autotag"
	// HostBucketFaces is the local ONNX face-scanning bucket, separate for the
	// same reasons. Each faces job parallelizes internally (see runFacesPool).
	HostBucketFaces = "faces"
)

// InferenceHost returns the concurrency bucket name for the currently
// configured vision provider. Job-creation code calls this (via the
// tasks.ResolveHost registry) when assigning a job's Host field; the
// returned bucket is then governed by the matching SetHostLimit cap.
//
// The "off" / unknown cases land on "localhost" so the job still claims
// (and then fails fast inside callVisionLLM with ErrInferenceDisabled).
func InferenceHost() string {
	switch strings.ToLower(strings.TrimSpace(appconfig.Get().InferenceProvider)) {
	case InferenceProviderRunPod:
		return HostBucketRunPod
	case InferenceProviderOllama:
		return HostBucketOllama
	case InferenceProviderLMStudio:
		return HostBucketLMStudio
	case InferenceProviderLlamaCpp:
		return HostBucketLlamaCpp
	default:
		return "localhost"
	}
}

// ErrInferenceDisabled is returned by callVisionLLM when the user has set
// the inference provider to "off". Callers can match on this to surface a
// friendlier "configure a provider" message instead of treating it as a
// hard failure.
var ErrInferenceDisabled = errors.New("inference disabled: set an InferenceProvider in config")

// callVisionLLM is the single entry point for image-conditioned LLM calls
// (description, autotag). It dispatches based on the configured inference
// provider. The caller supplies a deadline via ctx.
func callVisionLLM(ctx context.Context, imagePath, prompt string) (string, error) {
	cfg := appconfig.Get()
	provider := strings.ToLower(strings.TrimSpace(cfg.InferenceProvider))
	switch provider {
	case "", InferenceProviderOff:
		return "", ErrInferenceDisabled
	case InferenceProviderOllama:
		return callOllamaVisionRaw(ctx, imagePath, prompt, cfg.OllamaBaseURL, cfg.OllamaModel)
	case InferenceProviderRunPod:
		if strings.TrimSpace(cfg.RunPodEndpoint) == "" || strings.TrimSpace(cfg.RunPodAPIKey) == "" {
			return "", fmt.Errorf("runpod provider selected but endpoint or api key is empty")
		}
		return callRunPodVision(ctx, imagePath, prompt, cfg.RunPodEndpoint, cfg.RunPodAPIKey)
	case InferenceProviderLMStudio:
		if strings.TrimSpace(cfg.LMStudioBaseURL) == "" {
			return "", fmt.Errorf("lmstudio provider selected but base URL is empty")
		}
		return callOpenAICompatibleVision(ctx, imagePath, prompt, cfg.LMStudioBaseURL, cfg.LMStudioAPIKey, cfg.LMStudioModel)
	case InferenceProviderLlamaCpp:
		if strings.TrimSpace(cfg.LlamaCppBaseURL) == "" {
			return "", fmt.Errorf("llamacpp provider selected but base URL is empty")
		}
		return callOpenAICompatibleVision(ctx, imagePath, prompt, cfg.LlamaCppBaseURL, cfg.LlamaCppAPIKey, cfg.LlamaCppModel)
	default:
		return "", fmt.Errorf("unknown inference provider %q", cfg.InferenceProvider)
	}
}

// callOpenAICompatibleVision posts a chat-completions vision payload to any
// server that speaks the OpenAI /v1/chat/completions shape — currently
// LM Studio and the llama.cpp `server` binary, but the same client handles
// any future OpenAI-compatible local engine. The auth header is only sent
// when apiKey is non-empty so this works for unauthenticated local servers
// as well as proxied / remote setups.
//
// Distinct from callRunPodVision, which wraps the same payload in RunPod's
// {input: ...} envelope and supports async /run polling. If we ever pick up
// more OpenAI-shaped providers (vLLM, TGI, etc.) they reuse this helper.
func callOpenAICompatibleVision(ctx context.Context, imagePath, prompt, baseURL, apiKey, model string) (string, error) {
	img, err := loadImageForInference(imagePath)
	if err != nil {
		return "", fmt.Errorf("inference: %w", err)
	}
	endpoint := strings.TrimRight(baseURL, "/") + "/v1/chat/completions"
	img.logRequest("openai-compatible", model, endpoint, prompt)

	payload := map[string]any{
		"model":  model,
		"stream": false,
		"messages": []map[string]any{
			{
				"role": "user",
				"content": []map[string]any{
					{"type": "text", "text": prompt},
					{"type": "image_url", "image_url": map[string]any{"url": img.dataURI()}},
				},
			},
		},
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("failed to marshal chat-completions payload: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", endpoint, bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("failed to build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if strings.TrimSpace(apiKey) != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("inference request failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("inference error: status=%d, body=%s", resp.StatusCode, string(raw))
	}

	rawBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("reading inference response failed: %w", err)
	}
	var parsed struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
		Error *struct {
			Message string `json:"message"`
		} `json:"error,omitempty"`
	}
	if err := json.Unmarshal(rawBody, &parsed); err != nil {
		return "", fmt.Errorf("could not unmarshal chat-completions response: %w", err)
	}
	if parsed.Error != nil && parsed.Error.Message != "" {
		return "", fmt.Errorf("inference returned error: %s", parsed.Error.Message)
	}
	if len(parsed.Choices) == 0 {
		return "", fmt.Errorf("inference response missing choices: %s", string(rawBody))
	}
	content := strings.TrimSpace(parsed.Choices[0].Message.Content)
	logVisionResponse("openai-compatible", len(rawBody), content)
	if content == "" {
		return "", fmt.Errorf("inference response empty content: %s", string(rawBody))
	}
	return content, nil
}

// callOllamaVisionRaw issues an /api/generate request to a local-or-remote
// Ollama server and returns the model's response field. Equivalent to the
// in-line plumbing that used to live in metadata_ops.go and autotag_vision.go.
func callOllamaVisionRaw(ctx context.Context, imagePath, prompt, baseURL, model string) (string, error) {
	img, err := loadImageForInference(imagePath)
	if err != nil {
		return "", fmt.Errorf("ollama: %w", err)
	}
	base := strings.TrimRight(baseURL, "/")
	img.logRequest("ollama", model, base+"/api/generate", prompt)
	reqJSON := fmt.Sprintf(`{"model":"%s","stream":false,"prompt":%s,"images":["%s"]}`,
		model, strconv.Quote(prompt), img.base64())
	req, err := http.NewRequestWithContext(ctx, "POST", base+"/api/generate", strings.NewReader(reqJSON))
	if err != nil {
		return "", fmt.Errorf("failed to build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("ollama request failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("ollama error: status=%d, body=%s", resp.StatusCode, string(body))
	}
	var response struct {
		Response string `json:"response"`
	}
	respData, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("reading response body failed: %w", err)
	}
	if err := json.Unmarshal(respData, &response); err != nil {
		return "", fmt.Errorf("could not unmarshal Ollama response: %w", err)
	}
	logVisionResponse("ollama", len(respData), strings.TrimSpace(response.Response))
	return response.Response, nil
}

// runPodAsyncEndpoint matches RunPod async endpoints (those ending in `/run`).
// Async endpoints return a job id and require polling /status/<id>; sync
// endpoints (`/runsync`) return the model output inline.
var runPodAsyncEndpoint = regexp.MustCompile(`/run(?:\/?$|\?)`)

// callRunPodVision posts an OpenAI-style chat-completions payload to a
// RunPod serverless worker (e.g. SvenBrnn/runpod-worker-ollama) and returns
// the model's text response. Mirrors the structure tested in
// thespian/send-image.js.
func callRunPodVision(ctx context.Context, imagePath, prompt, endpoint, apiKey string) (string, error) {
	img, err := loadImageForInference(imagePath)
	if err != nil {
		return "", fmt.Errorf("runpod: %w", err)
	}
	img.logRequest("runpod", "", endpoint, prompt)

	payload := map[string]any{
		"input": map[string]any{
			"messages": []map[string]any{
				{
					"role": "user",
					"content": []map[string]any{
						{"type": "text", "text": prompt},
						{"type": "image_url", "image_url": map[string]any{"url": img.dataURI()}},
					},
				},
			},
		},
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("failed to marshal RunPod payload: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", endpoint, bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("failed to build RunPod request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+apiKey)

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("RunPod request failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("RunPod error: status=%d, body=%s", resp.StatusCode, string(raw))
	}

	rawBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("reading RunPod response failed: %w", err)
	}

	var initial runPodResponse
	if err := json.Unmarshal(rawBody, &initial); err != nil {
		return "", fmt.Errorf("could not unmarshal RunPod response: %w", err)
	}

	final := initial
	if runPodAsyncEndpoint.MatchString(endpoint) && initial.ID != "" {
		base := strings.TrimRight(strings.TrimSuffix(strings.TrimRight(endpoint, "/"), "/run"), "/")
		statusURL := fmt.Sprintf("%s/status/%s", base, initial.ID)
		polled, err := pollRunPod(ctx, statusURL, apiKey)
		if err != nil {
			return "", err
		}
		final = *polled
	}

	if final.Status != "" && final.Status != "COMPLETED" {
		return "", fmt.Errorf("RunPod job %s (id=%s)", final.Status, final.ID)
	}
	text := extractRunPodText(final)
	logVisionResponse("runpod", len(rawBody), text)
	if text == "" {
		// Surface the raw body so a misshaped worker response is debuggable.
		return "", fmt.Errorf("RunPod response missing text output: %s", string(rawBody))
	}
	return text, nil
}

type runPodResponse struct {
	ID     string          `json:"id,omitempty"`
	Status string          `json:"status,omitempty"`
	Output json.RawMessage `json:"output,omitempty"`
	Error  string          `json:"error,omitempty"`
}

// pollRunPod polls a /status/<id> URL until the job reaches a terminal state
// or ctx is cancelled.
func pollRunPod(ctx context.Context, statusURL, apiKey string) (*runPodResponse, error) {
	ticker := time.NewTicker(1500 * time.Millisecond)
	defer ticker.Stop()
	for {
		req, err := http.NewRequestWithContext(ctx, "GET", statusURL, nil)
		if err != nil {
			return nil, fmt.Errorf("build RunPod status request: %w", err)
		}
		req.Header.Set("Authorization", "Bearer "+apiKey)
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			return nil, fmt.Errorf("RunPod status fetch: %w", err)
		}
		body, readErr := io.ReadAll(resp.Body)
		resp.Body.Close()
		if readErr != nil {
			return nil, fmt.Errorf("read RunPod status body: %w", readErr)
		}
		if resp.StatusCode != http.StatusOK {
			return nil, fmt.Errorf("RunPod status error: status=%d, body=%s", resp.StatusCode, string(body))
		}
		var r runPodResponse
		if err := json.Unmarshal(body, &r); err != nil {
			return nil, fmt.Errorf("unmarshal RunPod status: %w", err)
		}
		switch r.Status {
		case "COMPLETED":
			return &r, nil
		case "FAILED", "CANCELLED", "TIMED_OUT":
			return nil, fmt.Errorf("RunPod job %s: %s", r.Status, r.Error)
		}
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-ticker.C:
		}
	}
}

// extractRunPodText pulls the model's text response out of a RunPod
// response. The expected shape is
//
//	output: { choices: [ { message: { content: "..." } } ] }
//
// but workers sometimes wrap output in an array, so we handle that too.
func extractRunPodText(r runPodResponse) string {
	if len(r.Output) == 0 {
		return ""
	}
	type chatChoice struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	}
	type chatOutput struct {
		Choices []chatChoice `json:"choices"`
	}
	// Try object shape first.
	var single chatOutput
	if err := json.Unmarshal(r.Output, &single); err == nil && len(single.Choices) > 0 {
		if c := strings.TrimSpace(single.Choices[0].Message.Content); c != "" {
			return c
		}
	}
	// Fall back to array-of-outputs shape.
	var arr []chatOutput
	if err := json.Unmarshal(r.Output, &arr); err == nil {
		for _, o := range arr {
			if len(o.Choices) > 0 {
				if c := strings.TrimSpace(o.Choices[0].Message.Content); c != "" {
					return c
				}
			}
		}
	}
	return ""
}

// imagePayload is the validated, instrumented image we hand to a vision
// backend. It exists so every provider (Ollama, OpenAI-compatible, RunPod)
// loads bytes the same way, logs the same diagnostics, and labels the MIME
// type consistently — the three used to each call os.ReadFile + mimeFromExt
// inline with no logging, which is why a "the model acted like it never got
// an image" run left nothing to inspect.
type imagePayload struct {
	path        string
	data        []byte
	extMime     string // MIME guessed from the file extension (mimeFromExt)
	sniffedMime string // MIME sniffed from the first bytes (http.DetectContentType)
	format      string // decoded format ("png","jpeg","webp",…); "" if undecodable
	colorModel  string // decoded color model ("ycbcr","cmyk","rgba",…); "" if undecodable
	width       int
	height      int
}

// colorModelName maps a decoded color.Model to a short label. CMYK is the one
// worth watching for in JPEGs: Go decodes it fine, but the stb_image-based
// decoders in several local vision backends mishandle CMYK JPEGs and hand the
// model a blank/garbage frame — a plausible cause of "the model answered as if
// it never saw the (jpg) image" that leaves no error behind.
func colorModelName(m color.Model) string {
	switch m {
	case color.YCbCrModel:
		return "ycbcr"
	case color.CMYKModel:
		return "cmyk"
	case color.RGBAModel:
		return "rgba"
	case color.NRGBAModel:
		return "nrgba"
	case color.GrayModel:
		return "gray"
	case color.Gray16Model:
		return "gray16"
	default:
		return "other"
	}
}

// loadImageForInference reads the image, validates it is non-empty, sniffs its
// real content type, and best-effort decodes its header for the true format +
// dimensions. A decode failure is NOT fatal (the backend may still accept the
// raw bytes) but it is a loud signal in the logs. The decoders for
// png/jpeg/gif/webp/bmp/tiff are registered package-wide via the blank imports
// in metadata_ops.go, so DecodeConfig here understands every format the
// describe path can produce.
func loadImageForInference(imagePath string) (*imagePayload, error) {
	data, err := os.ReadFile(imagePath)
	if err != nil {
		return nil, fmt.Errorf("could not read image %q: %w", imagePath, err)
	}
	if len(data) == 0 {
		return nil, fmt.Errorf("image %q is empty (0 bytes) — nothing to send to the model", imagePath)
	}
	p := &imagePayload{
		path:        imagePath,
		data:        data,
		extMime:     mimeFromExt(imagePath),
		sniffedMime: http.DetectContentType(data),
	}
	if cfg, format, derr := image.DecodeConfig(bytes.NewReader(data)); derr == nil {
		p.format, p.width, p.height = format, cfg.Width, cfg.Height
		p.colorModel = colorModelName(cfg.ColorModel)
	}
	return p, nil
}

// bestMime prefers the type sniffed from the actual bytes over the one guessed
// from the extension — this fixes mislabeled files and the
// application/octet-stream fallback. When the sniff can't recognize the bytes
// as an image (e.g. avif/heic, which http.DetectContentType doesn't know) it
// falls back to the extension-derived MIME so behavior is never worse than
// before.
func (p *imagePayload) bestMime() string {
	if strings.HasPrefix(p.sniffedMime, "image/") {
		return p.sniffedMime
	}
	return p.extMime
}

func (p *imagePayload) base64() string {
	return base64.StdEncoding.EncodeToString(p.data)
}

func (p *imagePayload) dataURI() string {
	return fmt.Sprintf("data:%s;base64,%s", p.bestMime(), p.base64())
}

// logRequest emits a single line confirming exactly what is about to leave for
// the model: byte count, encoded length, the MIME we attached (and whether the
// extension disagreed), the decoded format/dimensions, and the prompt length.
// "decoded=UNDECODABLE" means no registered decoder could read the bytes — a
// strong hint the backend won't be able to either.
func (p *imagePayload) logRequest(provider, model, endpoint, prompt string) {
	decoded := "UNDECODABLE(no registered decoder matched the bytes)"
	if p.format != "" {
		decoded = fmt.Sprintf("%s/%s %dx%d", p.format, p.colorModel, p.width, p.height)
	}
	mimeNote := p.bestMime()
	if p.bestMime() != p.extMime {
		mimeNote = fmt.Sprintf("%s (extension implied %s)", p.bestMime(), p.extMime)
	}
	log.Printf("[vision:request] provider=%s model=%q endpoint=%q file=%q bytes=%d base64Len=%d mime=%s decoded=%s promptLen=%d",
		provider, model, endpoint, filepath.Base(p.path), len(p.data),
		base64.StdEncoding.EncodedLen(len(p.data)), mimeNote, decoded, len(prompt))
}

// logVisionResponse records what came back so a blind-but-successful run is
// distinguishable from a healthy one: the raw HTTP body size, the extracted
// content length, and a short preview (which surfaces "I don't see an image"
// style refusals without dumping a whole description into the log).
func logVisionResponse(provider string, rawLen int, content string) {
	preview := strings.ReplaceAll(content, "\n", " ")
	if len(preview) > 200 {
		preview = preview[:200] + "…"
	}
	log.Printf("[vision:response] provider=%s rawBytes=%d contentLen=%d preview=%q",
		provider, rawLen, len(content), preview)
}

// noImageResponseMarkers are first-person phrases a vision model emits when it
// got the prompt but no image — i.e. the backend never handed the image to the
// model (a non-vision model, or a preprocessor that dropped it, e.g. on an
// extreme aspect ratio). They are deliberately phrased from the model's "I
// wasn't given an image" perspective so they don't match descriptions that
// merely mention images ("a sign that reads no photography").
var noImageResponseMarkers = []string{
	"no image was provided",
	"no image provided",
	"since no image",
	"cannot see an image",
	"can't see an image",
	"cannot see any image",
	"can't see any image",
	"don't see an image",
	"do not see an image",
	"don't see any image",
	"do not see any image",
	"unable to view the image",
	"unable to see the image",
	"unable to view an image",
	"haven't provided an image",
	"have not provided an image",
	"didn't provide an image",
	"did not provide an image",
	"wasn't provided an image",
	"was not provided an image",
	"wasn't provided with an image",
	"was not provided with an image",
	"didn't receive an image",
	"did not receive an image",
	"no image attached",
	"no image was attached",
}

// looksLikeNoImageResponse reports whether content matches a known "I didn't
// get an image" pattern. Heuristic by design: it gates saving an obviously
// blind description, it does not judge correctness. Callers treat a hit as a
// failure so the bad text is never persisted to the library.
func looksLikeNoImageResponse(content string) bool {
	lc := strings.ToLower(content)
	for _, m := range noImageResponseMarkers {
		if strings.Contains(lc, m) {
			return true
		}
	}
	return false
}

func mimeFromExt(path string) string {
	switch strings.ToLower(strings.TrimPrefix(filepath.Ext(path), ".")) {
	case "jpg", "jpeg":
		return "image/jpeg"
	case "png":
		return "image/png"
	case "webp":
		return "image/webp"
	case "gif":
		return "image/gif"
	case "bmp":
		return "image/bmp"
	default:
		return "application/octet-stream"
	}
}
