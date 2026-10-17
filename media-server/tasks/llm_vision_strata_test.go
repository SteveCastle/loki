package tasks

import (
	"context"
	"encoding/json"
	"image"
	"image/png"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stevecastle/shrike/appconfig"
)

func writeTestPNG(t *testing.T) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "img.png")
	f, err := os.Create(p)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if err := png.Encode(f, image.NewRGBA(image.Rect(0, 0, 4, 4))); err != nil {
		t.Fatal(err)
	}
	return p
}

// fakeStrata mimics the parts of Strata's serve/server.py the vision path
// relies on: /v1/* behind a Bearer key, OpenAI image_url parts, and thinking
// kept out of `content` (it goes to `reasoning_content`). It records the
// decoded request so tests can assert on the payload.
func fakeStrata(t *testing.T, apiKey string, got *map[string]any) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" || r.Method != http.MethodPost {
			http.NotFound(w, r)
			return
		}
		if apiKey != "" && r.Header.Get("Authorization") != "Bearer "+apiKey {
			w.WriteHeader(http.StatusUnauthorized)
			w.Write([]byte(`{"error":{"message":"missing or wrong API key"}}`))
			return
		}
		if err := json.NewDecoder(r.Body).Decode(got); err != nil {
			t.Errorf("decode request: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"A red ball on grass."},"finish_reason":"stop"}]}`))
	}))
	t.Cleanup(srv.Close)
	return srv
}

func setStrataConfig(t *testing.T, baseURL, apiKey string) {
	t.Helper()
	cur := appconfig.Get()
	t.Cleanup(func() { appconfig.Set(cur) })
	cfg := cur
	cfg.InferenceProvider = InferenceProviderStrata
	cfg.StrataBaseURL = baseURL
	cfg.StrataAPIKey = apiKey
	cfg.StrataModel = ""
	appconfig.Set(cfg)
}

func TestCallVisionLLMStrata(t *testing.T) {
	var got map[string]any
	srv := fakeStrata(t, "secret", &got)
	setStrataConfig(t, srv.URL+"/", "secret") // trailing slash must not double up

	out, err := callVisionLLM(context.Background(), writeTestPNG(t), "describe")
	if err != nil {
		t.Fatalf("callVisionLLM: %v", err)
	}
	if out != "A red ball on grass." {
		t.Errorf("content = %q", out)
	}
	if got["reasoning_effort"] != "none" {
		t.Errorf("reasoning_effort = %v; want \"none\" so thinking can't eat max_tokens", got["reasoning_effort"])
	}
	if got["max_tokens"] != float64(visionMaxOutputTokens) {
		t.Errorf("max_tokens = %v; want %d", got["max_tokens"], visionMaxOutputTokens)
	}
	if got["stream"] != false {
		t.Errorf("stream = %v; want false", got["stream"])
	}
	msgs, _ := got["messages"].([]any)
	if len(msgs) != 1 {
		t.Fatalf("messages = %v", got["messages"])
	}
	parts, _ := msgs[0].(map[string]any)["content"].([]any)
	var sawImage bool
	for _, p := range parts {
		m, _ := p.(map[string]any)
		if m["type"] == "image_url" {
			url, _ := m["image_url"].(map[string]any)["url"].(string)
			sawImage = strings.HasPrefix(url, "data:image/png;base64,")
		}
	}
	if !sawImage {
		t.Errorf("no image_url data-URI part in %v", parts)
	}
}

func TestCallVisionLLMStrataWrongKey(t *testing.T) {
	var got map[string]any
	srv := fakeStrata(t, "secret", &got)
	setStrataConfig(t, srv.URL, "")

	_, err := callVisionLLM(context.Background(), writeTestPNG(t), "describe")
	if err == nil || !strings.Contains(err.Error(), "status=401") {
		t.Fatalf("err = %v; want a 401 inference error", err)
	}
}

func TestCallVisionLLMStrataEmptyBaseURL(t *testing.T) {
	setStrataConfig(t, "  ", "")
	_, err := callVisionLLM(context.Background(), writeTestPNG(t), "describe")
	if err == nil || !strings.Contains(err.Error(), "strata provider selected but base URL is empty") {
		t.Fatalf("err = %v", err)
	}
}

// The Strata-only thinking switch must not leak into the other
// OpenAI-compatible providers, whose payload stays exactly as before.
func TestOpenAICompatibleNoExtrasForLlamaCpp(t *testing.T) {
	var got map[string]any
	srv := fakeStrata(t, "", &got)
	cur := appconfig.Get()
	t.Cleanup(func() { appconfig.Set(cur) })
	cfg := cur
	cfg.InferenceProvider = InferenceProviderLlamaCpp
	cfg.LlamaCppBaseURL = srv.URL
	appconfig.Set(cfg)

	if _, err := callVisionLLM(context.Background(), writeTestPNG(t), "describe"); err != nil {
		t.Fatalf("callVisionLLM: %v", err)
	}
	if _, ok := got["reasoning_effort"]; ok {
		t.Errorf("llamacpp request carried reasoning_effort: %v", got["reasoning_effort"])
	}
}
