package tasks

import (
	"context"
	"strings"
	"testing"

	"github.com/stevecastle/shrike/appconfig"
)

func TestTextCapturePrompt(t *testing.T) {
	cur := appconfig.Get()
	t.Cleanup(func() { appconfig.Set(cur) })
	cfg := cur
	cfg.TextCapturePrompt = "BASE"
	appconfig.Set(cfg)

	if got := textCapturePrompt("", ""); got != "BASE" {
		t.Errorf("no language: %q", got)
	}
	if got := textCapturePrompt("", "Original"); got != "BASE" {
		t.Errorf("original: %q", got)
	}
	got := textCapturePrompt("", "Spanish")
	if !strings.HasPrefix(got, "BASE\n\n") || !strings.Contains(got, "translate every piece of text into Spanish") {
		t.Errorf("translation instruction missing: %q", got)
	}
	if got := textCapturePrompt("MINE", "French"); !strings.HasPrefix(got, "MINE") || !strings.Contains(got, "French") {
		t.Errorf("custom prompt: %q", got)
	}
	cfg.TextCapturePrompt = ""
	appconfig.Set(cfg)
	if got := textCapturePrompt("", ""); got != appconfig.DefaultTextCapturePrompt {
		t.Error("an empty configured prompt falls back to the default")
	}
	if op, ok := itemOps[textCaptureOp]; !ok || op.Applies == nil || !op.Applies("page.png") || op.Applies("clip.mp4") {
		t.Error("capture-text applies to images only")
	}
	if r := ResolveResources(textCaptureOp, nil, ""); !contains(r, HostBucketScribe) {
		t.Errorf("capture-text resources %v; want the scribe bucket", r)
	}
}

func contains(xs []string, v string) bool {
	for _, x := range xs {
		if x == v {
			return true
		}
	}
	return false
}

func TestTrimRepeatedTail(t *testing.T) {
	loop := "PANEL 7\nWOMAN: Hi\n" + strings.Repeat("TEXT: Ah♡\n", 40) + "TEXT: A"
	got, looped := trimRepeatedTail(strings.TrimSpace(loop))
	// the cut-off final copy breaks the line run, so the char pass catches it;
	// either way a single copy of the loop must remain
	if !looped || strings.Count(got, "Ah♡") > 2 || !strings.HasPrefix(got, "PANEL 7\nWOMAN: Hi") {
		t.Errorf("loop not trimmed (%v): %q", looped, got)
	}
	whole := "A\n" + strings.Repeat("TEXT: Ah♡\n", 10)
	if got, looped := trimRepeatedTail(strings.TrimSpace(whole)); !looped || got != "A\nTEXT: Ah♡" {
		t.Errorf("line loop: %v %q", looped, got)
	}
	legit := "PANEL 1\nSFX: BANG\nSFX: BANG\nSFX: BANG\nMAN: Run"
	if got, looped := trimRepeatedTail(legit); looped || got != legit {
		t.Errorf("legit repeats trimmed: %q", got)
	}
	phrase := "WOMAN: " + strings.Repeat("ha ", 60)
	if got, looped := trimRepeatedTail(strings.TrimSpace(phrase)); !looped || len(got) > 20 {
		t.Errorf("phrase loop: %v %q", looped, got)
	}
}

func TestScribeSupportedArgsDropsDRY(t *testing.T) {
	args := append([]string{"--json", "--temp", "0.2"}, scribeDRYArgs...)
	args = append(args, "-p", "x")
	// a binary that does not exist has no --help: treated as not supporting it
	got := scribeSupportedArgs(context.Background(), aiToolRun{Bin: "Z:/nope/loki-scribe.exe"}, args, func(string) {})
	if strings.Join(got, " ") != "--json --temp 0.2 -p x" {
		t.Errorf("got %v", got)
	}
}
