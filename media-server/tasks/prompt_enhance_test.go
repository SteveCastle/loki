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

// A long video asks for one prompt per chunk, with each chunk's time span.
func TestEnhanceLongVideoSegments(t *testing.T) {
	// loki-reshoot's plan for 30 s (736 frames)
	chunks := []int{209, 209, 192, 192}
	spans := chunkSpans(chunks)
	if len(spans) != 4 || spans[1][0] != spans[0][1] || int(spans[3][1]*24+0.5) != 736 {
		t.Fatalf("spans = %v", spans)
	}
	sys := enhanceSystemPrompt(EnhanceRequest{Task: "video", Chunks: chunks}, "GUIDE")
	for _, want := range []string{"30.7 s long", "4 consecutive segments", "segment 2: 8.7 s to 16.5 s", "{chunk}", "only ===", "the 4 segment parts"} {
		if !strings.Contains(sys, want) {
			t.Errorf("system prompt lacks %q:\n%s", want, sys)
		}
	}
	if strings.Contains(enhanceSystemPrompt(EnhanceRequest{Task: "video", Chunks: []int{362}}, "GUIDE"), "segment") {
		t.Error("a plain video asks for segments")
	}

	answer := "<think>plan</think>\nShared part:\nsubject_definitions: S1 is a woman.\n{chunk}\noverall_soundscape: rain.\n===\nSegment 1 (0-8.7 s):\nShe walks in.\n---\n**Segment 2**\nShe sits.\n---\nShe talks.\n---\nShe laughs.\n---\nShe leaves."
	shared, got := splitEnhancedChunks(answer, 4)
	want := []string{"She walks in.", "She sits.", "She talks.", "She laughs.\n\nShe leaves."}
	if shared != "subject_definitions: S1 is a woman.\n{chunk}\noverall_soundscape: rain." || strings.Join(got, "|") != strings.Join(want, "|") {
		t.Errorf("splitEnhancedChunks = %q %q, want %q", shared, got, want)
	}
	// no === line: all segments; too few: padded
	if shared, got := splitEnhancedChunks("A.\n---\nB.", 3); shared != "" || strings.Join(got, "|") != "A.|B.|" {
		t.Errorf("without shared part = %q %q", shared, got)
	}

	if got := ComposeChunkPrompt("defs\n{chunk}\nsound", "action"); got != "defs\naction\nsound" {
		t.Errorf("compose in slot = %q", got)
	}
	if got := ComposeChunkPrompt("defs\n  {chunk}  \nsound", ""); got != "defs\nsound" {
		t.Errorf("compose empty slot = %q", got)
	}
	if got := ComposeChunkPrompt("a woman in a red coat", "she waves"); got != "a woman in a red coat\n\nshe waves" {
		t.Errorf("compose without slot = %q", got)
	}
	if got := ComposeChunkPrompt("", "she waves"); got != "she waves" {
		t.Errorf("compose without shared = %q", got)
	}

	// "Enhance first": the run gets each chunk's whole prompt
	args := enhancedArgs([]string{"--duration", "30", "--prompt", "rough", "--chunkprompts", "rough 2"}, EnhanceResult{Prompt: "S\n{chunk}", Segments: []string{"a", "b", "c"}})
	if strings.Join(args, " ") != "--duration 30 --prompt S\na --chunkprompts S\nb\n---\nS\nc" {
		t.Errorf("enhancedArgs = %q", args)
	}
	if args := enhancedArgs([]string{"--prompt", "rough"}, EnhanceResult{Prompt: "p1"}); strings.Join(args, " ") != "--prompt p1" {
		t.Errorf("plain enhancedArgs = %q", args)
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

func TestEnhanceProgressFromEngineLines(t *testing.T) {
	p := enhanceProgress{task: "video"}
	steps := []struct {
		line string
		want int
		ok   bool
	}{
		{"[text] loading qwen.gguf (17.0 GB) ...", 0, false},
		{"[text]   layers 16/64", 7, true},
		{"[text]   layers 64/64", 30, true},
		{"[text] [request] prompt 2400 tokens (2 image(s)) -> generating", 35, true},
		{"[text] [request] 32 tokens so far", 39, true},
		{"[text] [request] 450 tokens so far", 72, true},
		{"[text] [request] 4000 tokens so far", 94, true},
		{"[text]   layers 64/64", 0, false}, // never backwards
	}
	for _, s := range steps {
		got, ok := p.update(s.line)
		if ok != s.ok || (ok && got != s.want) {
			t.Errorf("%q -> %d %v; want %d %v", s.line, got, ok, s.want, s.ok)
		}
	}
}
