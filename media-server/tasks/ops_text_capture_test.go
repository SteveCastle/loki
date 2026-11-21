package tasks

import (
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
