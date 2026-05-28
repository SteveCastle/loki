package tasks

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
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
	data, err := os.ReadFile(imagePath)
	if err != nil {
		return "", fmt.Errorf("could not read image for inference: %w", err)
	}
	b64 := base64.StdEncoding.EncodeToString(data)
	mime := mimeFromExt(imagePath)
	dataURI := fmt.Sprintf("data:%s;base64,%s", mime, b64)

	payload := map[string]any{
		"model":  model,
		"stream": false,
		"messages": []map[string]any{
			{
				"role": "user",
				"content": []map[string]any{
					{"type": "text", "text": prompt},
					{"type": "image_url", "image_url": map[string]any{"url": dataURI}},
				},
			},
		},
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("failed to marshal chat-completions payload: %w", err)
	}

	endpoint := strings.TrimRight(baseURL, "/") + "/v1/chat/completions"
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
	if content == "" {
		return "", fmt.Errorf("inference response empty content: %s", string(rawBody))
	}
	return content, nil
}

// callOllamaVisionRaw issues an /api/generate request to a local-or-remote
// Ollama server and returns the model's response field. Equivalent to the
// in-line plumbing that used to live in metadata_ops.go and autotag_vision.go.
func callOllamaVisionRaw(ctx context.Context, imagePath, prompt, baseURL, model string) (string, error) {
	data, err := os.ReadFile(imagePath)
	if err != nil {
		return "", fmt.Errorf("could not read image for Ollama: %w", err)
	}
	b64 := base64.StdEncoding.EncodeToString(data)
	reqJSON := fmt.Sprintf(`{"model":"%s","stream":false,"prompt":%s,"images":["%s"]}`,
		model, strconv.Quote(prompt), b64)
	base := strings.TrimRight(baseURL, "/")
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
	data, err := os.ReadFile(imagePath)
	if err != nil {
		return "", fmt.Errorf("could not read image for RunPod: %w", err)
	}
	b64 := base64.StdEncoding.EncodeToString(data)
	mime := mimeFromExt(imagePath)
	dataURI := fmt.Sprintf("data:%s;base64,%s", mime, b64)

	payload := map[string]any{
		"input": map[string]any{
			"messages": []map[string]any{
				{
					"role": "user",
					"content": []map[string]any{
						{"type": "text", "text": prompt},
						{"type": "image_url", "image_url": map[string]any{"url": dataURI}},
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
