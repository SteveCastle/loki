package tasks

import (
	"context"
	"errors"
	"path/filepath"
	"strings"
	"testing"
)

func TestRetouchModelInputMirrorsTheEngineTemplate(t *testing.T) {
	got := retouchModelInput("put <image2> into <image1>", 2)
	want := "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n" +
		"<|im_start|>user\n<image1><|vision_start|><|image_pad|><|vision_end|> <image2><|vision_start|><|image_pad|><|vision_end|>" +
		"put <image2> into <image1><|im_end|>\n<|im_start|>assistant\n"
	if got != want {
		t.Errorf("model input =\n%q\nwant\n%q", got, want)
	}
}

func TestPreviewPromptRetouchCombineUsesReferences(t *testing.T) {
	argsLog := installFakeAI(t)
	dir := t.TempDir()
	a := writeFileForTest(t, filepath.Join(dir, "me.png"), "x")
	b := writeFileForTest(t, filepath.Join(dir, "jacket.png"), "x")

	pv, err := PreviewPrompt(context.Background(), "retouch",
		uiFieldsToArgs(map[string]string{"prompt": "put the jacket of <image2> on <image1>", "combine": "1", "steps": "25", "seed": "9"}),
		a+"\n"+b)
	if err != nil {
		t.Fatal(err)
	}
	if !pv.FromEngine || pv.Prompt != "EXPANDED put the jacket of <image2> on <image1>" {
		t.Errorf("preview = %+v", pv)
	}
	if pv.Runs != 1 || len(pv.References) != 2 || pv.References[0].Token != "<image1>" || pv.References[0].Path != a ||
		pv.References[1].Token != "<image2>" || pv.References[1].Path != b {
		t.Errorf("references = %+v (runs %d)", pv.References, pv.Runs)
	}
	if !strings.Contains(pv.ModelInput, "<image2><|vision_start|>") {
		t.Errorf("model input = %q", pv.ModelInput)
	}

	// The engine was asked exactly what the task would run, plus --show-prompt.
	calls := readFakeArgs(t, argsLog)
	if len(calls) != 1 || calls[0][0] != "--show-prompt" {
		t.Fatalf("calls = %q", calls)
	}
	if refs := argValues(calls[0], "--ref"); len(refs) != 1 || refs[0] != b {
		t.Errorf("--ref = %q; want [%s]", refs, b)
	}
	if last := calls[0][len(calls[0])-1]; last != a {
		t.Errorf("primary input = %q", last)
	}
}

func TestPreviewPromptRetouchBatchShowsFirstRun(t *testing.T) {
	installFakeAI(t)
	dir := t.TempDir()
	a := writeFileForTest(t, filepath.Join(dir, "a.png"), "x")
	b := writeFileForTest(t, filepath.Join(dir, "b.png"), "x")
	pv, err := PreviewPrompt(context.Background(), "retouch", uiFieldsToArgs(map[string]string{"preset": "restore"}), a+"\n"+b)
	if err != nil {
		t.Fatal(err)
	}
	if pv.Runs != 2 || len(pv.References) != 1 || pv.Prompt != "EXPANDED built-in prompt" || !strings.Contains(pv.Note, "first of 2") {
		t.Errorf("preview = %+v", pv)
	}
}

func TestPreviewPromptWithoutEngine(t *testing.T) {
	prev := locateAITool
	locateAITool = func(aiToolSpec) (aiToolRun, bool) { return aiToolRun{}, false }
	t.Cleanup(func() { locateAITool = prev })
	dir := t.TempDir()
	a := writeFileForTest(t, filepath.Join(dir, "a.png"), "x")

	pv, err := PreviewPrompt(context.Background(), "retouch", uiFieldsToArgs(map[string]string{"prompt": "add snow", "append": "keep grain"}), a)
	if err != nil {
		t.Fatal(err)
	}
	if pv.FromEngine || pv.Prompt != "add snow\n\nkeep grain" {
		t.Errorf("free-form fallback = %+v", pv)
	}
	pv, err = PreviewPrompt(context.Background(), "retouch", uiFieldsToArgs(map[string]string{"preset": "upscale"}), a)
	if err != nil || pv.Prompt != "" || !strings.Contains(pv.Note, "not installed") {
		t.Errorf("preset fallback = %+v, %v", pv, err)
	}
}

func TestPreviewPromptReshootTokens(t *testing.T) {
	argsLog := installFakeAI(t)
	dir := t.TempDir()
	img := writeFileForTest(t, filepath.Join(dir, "p.png"), "x")
	clip := writeFileForTest(t, filepath.Join(dir, "c.mp4"), "x")
	song := writeFileForTest(t, filepath.Join(dir, "s.mp3"), "x")
	pv, err := PreviewPrompt(context.Background(), "reshoot",
		uiFieldsToArgs(map[string]string{"prompt": "<Picture 1> dances to <Audio 1>", "duration": "5"}), img+"\n"+clip+"\n"+song)
	if err != nil {
		t.Fatal(err)
	}
	var toks []string
	for _, r := range pv.References {
		toks = append(toks, r.Token)
	}
	if strings.Join(toks, ",") != "<Picture 1>,<Video 1>,<Audio 1>" || pv.Prompt != "EXPANDED <Picture 1> dances to <Audio 1>" {
		t.Errorf("preview = %+v", pv)
	}
	if calls := readFakeArgs(t, argsLog); len(calls) != 1 || calls[0][0] != "--show-prompt" {
		t.Errorf("calls = %q", calls)
	}
}

func TestPreviewPromptRejectsBadRequests(t *testing.T) {
	for _, tc := range []struct{ cmd, input string }{
		{"retouch", ""},
		{"ffmpeg", "a.png"},
	} {
		if _, err := PreviewPrompt(context.Background(), tc.cmd, nil, tc.input); !errors.Is(err, ErrPreviewUnsupported) {
			t.Errorf("%s %q: err = %v; want ErrPreviewUnsupported", tc.cmd, tc.input, err)
		}
	}
}
