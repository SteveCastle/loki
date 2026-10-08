package handlers

import (
	"encoding/json"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/stevecastle/shrike/deps/models"
)

// Verifying an item the user supplied (a PATH engine and weights beside it)
// checks their files in place instead of reporting the managed folder missing.
func TestVerifyUserProvided(t *testing.T) {
	dir := t.TempDir()
	exe := "loki-retouch"
	if runtime.GOOS == "windows" {
		exe += ".exe"
	}
	os.WriteFile(filepath.Join(dir, exe), []byte("x"), 0o755)
	t.Setenv("PATH", dir)
	t.Setenv("LOKI_MODELS", "")
	models.SetDataDirForTest(t.TempDir())
	t.Cleanup(func() { models.SetDataDirForTest("") })

	verify := func(id string) map[string]any {
		r := httptest.NewRequest("POST", "/api/deps/models/"+id+"/verify", nil)
		r.SetPathValue("id", id)
		w := httptest.NewRecorder()
		HandleModelVerify(w, r)
		var out map[string]any
		if err := json.Unmarshal(w.Body.Bytes(), &out); err != nil {
			t.Fatal(err)
		}
		return out
	}

	tool := verify("loki-retouch")
	if tool["source"] != "user" || tool["location"] == "" {
		t.Errorf("tool: %v", tool)
	}
	for f, res := range tool["files"].(map[string]any) {
		if res != "ok (on PATH)" {
			t.Errorf("%s = %v", f, res)
		}
	}

	m, _ := models.Lookup("qwen-image-2.1")
	os.MkdirAll(filepath.Join(dir, "models"), 0o755)
	for _, f := range m.EffectiveFiles() {
		os.WriteFile(filepath.Join(dir, "models", f.RelPath), []byte("tiny"), 0o644)
	}
	res := verify("qwen-image-2.1")
	if res["source"] != "user" {
		t.Fatalf("model: %v", res)
	}
	for f, v := range res["files"].(map[string]any) {
		if v == "missing" || v == nil {
			t.Errorf("%s reported %v", f, v)
		}
	}

	// Not provided by the user and not installed: the managed answer.
	other := verify("minimax-h3-ref2va")
	if other["source"] != nil {
		t.Errorf("unexpected user source: %v", other)
	}
}
