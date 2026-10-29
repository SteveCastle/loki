package tasks

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/stevecastle/shrike/jobqueue"
)

func TestBuildDreamArgs(t *testing.T) {
	p := dreamParams{Prompt: "a lighthouse at dusk", Negative: "blurry", Size: "1344x768", Seed: 7, Count: 1, Strength: 0.75}
	args := buildDreamArgs(p, "", "out.png", "")
	for _, want := range []string{"--prompt", "a lighthouse at dusk", "--negative", "blurry", "--size", "1344x768", "--seed", "7", "-o", "out.png"} {
		if !slices.Contains(args, want) {
			t.Errorf("args %v miss %q", args, want)
		}
	}
	if slices.Contains(args, "--init") || slices.Contains(args, "--count") || slices.Contains(args, "--dit") {
		t.Errorf("text -> image args: %v", args)
	}
	// A swapped model (Config.DreamModel) rides along as --dit.
	p.Model = "finetune.safetensors"
	if args = buildDreamArgs(p, "", "out.png", ""); !slices.Contains(args, "--dit") || !slices.Contains(args, "finetune.safetensors") {
		t.Errorf("model args: %v", args)
	}
	p.Model = ""
	// Turbo: no negative branch; inpainting: init, strength, mask.
	p.Turbo, p.Mask, p.Strength = true, "m.png", 1
	args = buildDreamArgs(p, "in.png", "o.png", "")
	if slices.Contains(args, "--negative") || !slices.Contains(args, "--turbo") {
		t.Errorf("turbo args: %v", args)
	}
	for _, want := range []string{"--init", "in.png", "--strength", "1", "--mask", "m.png"} {
		if !slices.Contains(args, want) {
			t.Errorf("inpaint args %v miss %q", args, want)
		}
	}
}

func lastArgs(t *testing.T, log string) []string {
	t.Helper()
	b, err := os.ReadFile(log)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(b)), "\n")
	var args []string
	if err := json.Unmarshal([]byte(lines[len(lines)-1]), &args); err != nil {
		t.Fatal(err)
	}
	return args
}

// A prompt and no input: a new image in the output folder, in the library.
func TestDreamTaskTextToImageFake(t *testing.T) {
	argsLog := installFakeAI(t)
	db := setupItemOpsDB(t)
	out := t.TempDir()
	q, j := newItemOpsJob(t, db, "dream", []string{"--prompt", "a red fox in snow", "--seed", "42", "--outdir", out}, "")
	if err := dreamTask(j, q, &sync.Mutex{}); err != nil {
		t.Fatalf("dream: %v\n%s", err, jobLog(q, j.ID))
	}
	files, _ := filepath.Glob(filepath.Join(out, "dream_*_42.png"))
	if len(files) != 1 {
		t.Fatalf("outputs %v\n%s", files, jobLog(q, j.ID))
	}
	if got := q.GetJob(j.ID); got.State != jobqueue.StateCompleted || len(got.OutputFiles) != 1 {
		t.Fatalf("job %v outputs %v", got.State, got.OutputFiles)
	}
	var n int
	if err := db.QueryRow(`SELECT COUNT(*) FROM media WHERE path = ?`, files[0]).Scan(&n); err != nil || n != 1 {
		t.Fatalf("library rows for %s: %d (%v)", files[0], n, err)
	}
	if args := lastArgs(t, argsLog); !slices.Contains(args, "--prompt") || slices.Contains(args, "--init") {
		t.Fatalf("args %v", args)
	}
}

// An input image and a mask: inpainting beside the input.
func TestDreamTaskInpaintFake(t *testing.T) {
	argsLog := installFakeAI(t)
	db := setupItemOpsDB(t)
	dir := t.TempDir()
	in := writeFileForTest(t, filepath.Join(dir, "crop.png"), "png")
	mask := writeFileForTest(t, filepath.Join(dir, "mask.png"), "png")
	q, j := newItemOpsJob(t, db, "dream", []string{"--prompt", "a stone well", "--mask", mask, "--strength", "1", "--nolibrary", "true"}, in)
	if err := dreamTask(j, q, &sync.Mutex{}); err != nil {
		t.Fatalf("dream: %v\n%s", err, jobLog(q, j.ID))
	}
	if _, err := os.Stat(filepath.Join(dir, "crop_dream.png")); err != nil {
		t.Fatalf("no crop_dream.png: %v\n%s", err, jobLog(q, j.ID))
	}
	args := lastArgs(t, argsLog)
	for _, want := range []string{"--init", in, "--mask", mask, "--strength", "1"} {
		if !slices.Contains(args, want) {
			t.Errorf("args %v miss %q", args, want)
		}
	}
	var n int
	_ = db.QueryRow(`SELECT COUNT(*) FROM media`).Scan(&n)
	if n != 0 {
		t.Errorf("nolibrary: %d library rows", n)
	}
}
