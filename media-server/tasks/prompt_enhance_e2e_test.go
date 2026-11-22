package tasks

import (
	"context"
	"os"
	"testing"
)

// Opt-in: LOKI_ENHANCE_E2E=1 runs the real text model. LOKI_ENHANCE_IMG1 and
// LOKI_ENHANCE_IMG2 (optional) add a two-reference edit.
func TestEnhancePromptEndToEnd(t *testing.T) {
	if os.Getenv("LOKI_ENHANCE_E2E") == "" {
		t.Skip("set LOKI_ENHANCE_E2E=1")
	}
	reqs := []EnhanceRequest{
		{Prompt: "fox in snow at sunrise", Model: "flux2-klein", Task: "generate"},
	}
	if a, b := os.Getenv("LOKI_ENHANCE_IMG1"), os.Getenv("LOKI_ENHANCE_IMG2"); a != "" && b != "" {
		reqs = append(reqs, EnhanceRequest{
			Prompt: "put the thing from <image2> into <image1>", Model: "qwen-image-edit", Task: "edit",
			References: []EnhanceReference{{Token: "<image1>", Kind: "image", Path: a}, {Token: "<image2>", Kind: "image", Path: b}},
		})
	}
	for _, req := range reqs {
		res, err := EnhancePrompt(context.Background(), req)
		if err != nil {
			t.Fatalf("%s: %v", req.Model, err)
		}
		t.Logf("%s:\n%s\nmissing=%v\n", req.Model, res.Prompt, res.MissingTokens)
	}
}
