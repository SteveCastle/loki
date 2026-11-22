package tasks

import (
	"strings"
	"testing"
)

// The renderer's enhance-prompt.ts targets these guide keys.
func TestPromptGuidesForEveryTarget(t *testing.T) {
	for _, m := range []string{"h3-video", "h3-image", "qwen-image-edit", "moss-image", "flux2-klein", "krea2", "anima", "sdxl"} {
		g, ok := promptGuide(m)
		if !ok || len(g) < 200 {
			t.Errorf("guide %q missing or too short", m)
		}
		if strings.Contains(g, "<!--") {
			t.Errorf("guide %q still carries its sources comment", m)
		}
	}
	if _, ok := promptGuide("../scribe"); ok {
		t.Error("guide key escaped the folder")
	}
}

func TestCleanEnhanced(t *testing.T) {
	cases := map[string]string{
		"A fox in snow.":                           "A fox in snow.",
		"Prompt: A fox in snow.":                   "A fox in snow.",
		"**Optimized prompt:** A fox in snow.":     "A fox in snow.",
		`"A fox in snow."`:                         "A fox in snow.",
		`A sign reading "OPEN" on a door`:          `A sign reading "OPEN" on a door`,
		"<think>\nhmm\n</think>\n\nA fox in snow.": "A fox in snow.",
		"```\nA fox in snow.\n```":                 "A fox in snow.",
	}
	for in, want := range cases {
		if got := cleanEnhanced(in); got != want {
			t.Errorf("cleanEnhanced(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestEnhanceSystemPromptNamesTokens(t *testing.T) {
	sys := enhanceSystemPrompt(EnhanceRequest{Task: "edit", References: []EnhanceReference{{Token: "<image1>", Kind: "image"}}, Image: "x.png", ImageToken: "<image1>"}, "GUIDE")
	for _, want := range []string{"<image1> (image)", "One image is attached: <image1>.", "edit instruction", "GUIDE"} {
		if !strings.Contains(sys, want) {
			t.Errorf("system prompt lacks %q", want)
		}
	}
}

func TestEnhanceImagesFromReferences(t *testing.T) {
	req := EnhanceRequest{
		Task: "video",
		References: []EnhanceReference{
			{Token: "<Picture 1>", Kind: "image", Path: "a.jpg"},
			{Token: "<Video 1>", Kind: "video", Path: "b.mp4"},
			{Token: "<Audio 1>", Kind: "audio", Path: "c.mp3"},
			{Token: "<Picture 2>", Kind: "image"},
		},
	}
	imgs := enhanceImages(req)
	if len(imgs) != 2 || imgs[0].token != "<Picture 1>" || imgs[1].kind != "video" {
		t.Fatalf("enhanceImages = %+v", imgs)
	}
	sys := enhanceSystemPrompt(req, "GUIDE")
	for _, want := range []string{"1. <Picture 1>", "2. a frame from <Video 1>"} {
		if !strings.Contains(sys, want) {
			t.Errorf("system prompt lacks %q", want)
		}
	}
	// The source picture given as Image picks up its reference token.
	req = EnhanceRequest{Image: "A.jpg", References: []EnhanceReference{{Token: "<image1>", Kind: "image", Path: "a.jpg"}}}
	if imgs := enhanceImages(req); len(imgs) != 1 || imgs[0].token != "<image1>" {
		t.Errorf("source + reference = %+v", imgs)
	}
}
