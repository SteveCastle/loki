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
	for _, want := range []string{"<image1> (image)", "The attached image is <image1>", "edit instruction", "GUIDE"} {
		if !strings.Contains(sys, want) {
			t.Errorf("system prompt lacks %q", want)
		}
	}
}
