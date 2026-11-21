package tasks

import (
	"os"
	"path/filepath"
	"testing"
)

func TestScribeLogKey(t *testing.T) {
	for _, p := range []string{`C:\Users\x\media\Cat.JPG`, `C:\Users\x\media\cat.jpg`, "/job/in/0/cat.jpg", "cat.jpg"} {
		if got := scribeLogKey(p); got != "cat.jpg" {
			t.Errorf("scribeLogKey(%q) = %q; want cat.jpg", p, got)
		}
	}
}

func TestRemoteScribeArgsKeepsModelFilesHome(t *testing.T) {
	dir := t.TempDir()
	// an LM Studio layout: <dir>/models/<publisher>/<repo>/<file>
	model := dir + "/models/pub/repo/m.gguf"
	mustWrite(t, model)
	other := dir + "/elsewhere/x.gguf"
	mustWrite(t, other)
	got := remoteScribeArgs([]string{"--json", "--model", model, "--mmproj", other, "-p", "hi", "img.jpg"})
	if got[2] != "pub/repo/m.gguf" || got[4] != "x.gguf" {
		t.Errorf("remoteScribeArgs = %v", got)
	}
	if got[6] != "hi" || got[7] != "img.jpg" {
		t.Errorf("other args changed: %v", got)
	}
}

func mustWrite(t *testing.T, p string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestSplitShard(t *testing.T) {
	for name, want := range map[string]bool{
		"m-00001-of-00002.gguf": false, "m-00002-of-00002.gguf": true, "m-00010-of-00012.gguf": true, "m-q4_k_m.gguf": false,
	} {
		if got := splitShard.MatchString(name); got != want {
			t.Errorf("splitShard(%q) = %v; want %v", name, got, want)
		}
	}
}
