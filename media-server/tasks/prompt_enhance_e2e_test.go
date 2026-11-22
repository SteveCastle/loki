package tasks

import (
	"context"
	"os"
	"testing"
)

// Opt-in: LOKI_ENHANCE_E2E=1 runs the real text model.
func TestEnhancePromptEndToEnd(t *testing.T) {
	if os.Getenv("LOKI_ENHANCE_E2E") == "" {
		t.Skip("set LOKI_ENHANCE_E2E=1")
	}
	for _, req := range []EnhanceRequest{
		{Prompt: "fox in snow at sunrise", Model: "flux2-klein", Task: "generate"},
		{Prompt: "make <image1> look like a rainy night", Model: "qwen-image-edit", Task: "edit", References: []EnhanceReference{{Token: "<image1>", Kind: "image"}}},
		{Prompt: "<Picture 1> waves at the camera and says hi", Model: "h3-video", Task: "video", References: []EnhanceReference{{Token: "<Picture 1>", Kind: "image"}}},
	} {
		res, err := EnhancePrompt(context.Background(), req)
		if err != nil {
			t.Fatalf("%s: %v", req.Model, err)
		}
		t.Logf("%s:\n%s\nmissing=%v\n", req.Model, res.Prompt, res.MissingTokens)
	}
}
