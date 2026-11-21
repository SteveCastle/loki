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

	// transcription never carries a translation instruction (that is pass 2)
	if got := textCapturePrompt(""); got != "BASE" {
		t.Errorf("configured prompt: %q", got)
	}
	if got := textCapturePrompt("MINE"); got != "MINE" {
		t.Errorf("custom prompt: %q", got)
	}
	cfg.TextCapturePrompt = ""
	cfg.TextTranslatePrompt = ""
	appconfig.Set(cfg)
	if got := textCapturePrompt(""); got != appconfig.DefaultTextCapturePrompt {
		t.Error("an empty configured prompt falls back to the default")
	}
	got := textTranslatePrompt("Spanish", "PANEL 1\nA: hola")
	if strings.Contains(got, "{language}") || !strings.Contains(got, "into Spanish") || !strings.HasSuffix(got, "TEXT:\nPANEL 1\nA: hola") {
		t.Errorf("translate prompt: %q", got)
	}
	if textCaptureLanguage("Original") != "" || textCaptureLanguage(" English ") != "English" {
		t.Error("language normalization")
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

func TestTranslatedLines(t *testing.T) {
	// the reply shape the translation prompt asks for (copied originals, ">> " translations)
	reply := "PANEL 1\nWOMAN IN HOODIE: 실례 했습니다.\n>> WOMAN IN HOODIE: Thanks for having me.\nBLONDE MAN: 그럼\n>> BLONDE MAN: Well then.\n\nPANEL 2\nSFX: ガギン\n>> SFX: CLUNK\n"
	want := "PANEL 1\nWOMAN IN HOODIE: Thanks for having me.\nBLONDE MAN: Well then.\n\nPANEL 2\nSFX: CLUNK"
	if got := translatedLines(reply); got != want {
		t.Errorf("got\n%s\nwant\n%s", got, want)
	}
	// a reply that ignored the format is kept whole
	if got := translatedLines("PANEL 1\nA: hello"); got != "PANEL 1\nA: hello" {
		t.Errorf("plain reply: %q", got)
	}
	// CRLF and blank-line runs
	if got := translatedLines("PANEL 1\r\nX: y\r\n>> X: z\r\n\r\n\r\nPANEL 2\r\n>> SFX: BZZT"); got != "PANEL 1\nX: z\n\nPANEL 2\nSFX: BZZT" {
		t.Errorf("crlf: %q", got)
	}
}
