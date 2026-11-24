package tasks

import (
	"database/sql"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stevecastle/shrike/jobqueue"
	_ "modernc.org/sqlite"
)

// The fields the viewer's Train LoRA dialog sends (src/renderer/components/train-lora/), as POST /create fields.
func uiTrainLoraArgs(fields map[string]string) []string {
	j := &jobqueue.Job{Arguments: uiFieldsToArgs(fields)}
	p := trainLoraParamsFromOptions(ParseOptions(j, trainLoraOptions))
	return buildTrainLoraArgs(p, "images.jsonl", "runs/"+loraRunName(p), "", `{"job":"j1"}`)
}

func TestUIContractTrainLora(t *testing.T) {
	a := uiTrainLoraArgs(map[string]string{
		"name": "Old Man", "trigger": "zkx", "kind": "character", "class": "man",
		"captions": "auto", "steps": "1500", "rank": "32", "resolution": "768",
	})
	joined := strings.Join(a, "|")
	for _, want := range []string{"images.jsonl|-o|runs/Old_Man", "--name|Old_Man", "--kind|character", "--trigger|zkx", "--class|man",
		"--captions|auto", "--steps|1500", "--rank|32", "--resolution|768", `--meta|{"job":"j1"}`} {
		if !strings.Contains(joined, want) {
			t.Errorf("missing %q in %s", want, joined)
		}
	}
	// defaults: character, auto captions, rank 16, 1024 px, steps left to the engine
	a = uiTrainLoraArgs(map[string]string{"trigger": "ohwx"})
	joined = strings.Join(a, "|")
	for _, want := range []string{"-o|runs/ohwx", "--kind|character", "--rank|16", "--resolution|1024", "--captions|auto"} {
		if !strings.Contains(joined, want) {
			t.Errorf("default: missing %q in %s", want, joined)
		}
	}
	if strings.Contains(joined, "--steps") || strings.Contains(joined, "--lr") || strings.Contains(joined, "--flip") {
		t.Errorf("default: unexpected flags in %s", joined)
	}
	// library captions travel in the list, not as a flag; none is passed through
	if j := strings.Join(uiTrainLoraArgs(map[string]string{"captions": "library"}), "|"); !strings.Contains(j, "--captions|auto") {
		t.Errorf("library captions: %s", j)
	}
	if j := strings.Join(uiTrainLoraArgs(map[string]string{"captions": "none", "kind": "style", "flip": "true"}), "|"); !strings.Contains(j, "--captions|none") || !strings.Contains(j, "--kind|style") || !strings.Contains(j, "--flip") {
		t.Errorf("none/style/flip: %s", j)
	}
}

func TestTrainLoraResumeArgs(t *testing.T) {
	p := trainLoraParamsFromOptions(map[string]any{"steps": 3000.0, "rank": 64.0, "kind": "style"})
	a := buildTrainLoraArgs(p, "", "runs/x", "runs/x/checkpoints/step-000500", "")
	j := strings.Join(a, "|")
	if !strings.HasPrefix(j, "--resume|runs/x/checkpoints/step-000500") || !strings.Contains(j, "--steps|3000") {
		t.Errorf("resume args: %s", j)
	}
	// the run's own settings are not repeated on resume
	if strings.Contains(j, "--rank") || strings.Contains(j, "--kind") || strings.Contains(j, "-o|") {
		t.Errorf("resume repeats run settings: %s", j)
	}
}

func TestTrainLoraParams(t *testing.T) {
	p := trainLoraParamsFromOptions(map[string]any{"kind": "bogus", "rank": 20.0, "resolution": "2048", "captions": "x"})
	if p.Kind != "character" || p.Rank != 24 || p.Resolution != 1024 || p.Captions != "auto" {
		t.Errorf("normalized params: %+v", p)
	}
	for in, want := range map[string]string{"zkx": "zkx", "My Style!": "My_Style", "": "lora", "..": "lora"} {
		if got := loraRunName(trainLoraParams{Trigger: in}); got != want {
			t.Errorf("run name %q = %q, want %q", in, got, want)
		}
	}
	if got := loraRunName(trainLoraParams{Name: "set a", Trigger: "zkx"}); got != "set_a" {
		t.Errorf("name wins over trigger: %q", got)
	}
}

func TestTrainLoraListUsesLibraryDescriptions(t *testing.T) {
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if _, err := db.Exec(`CREATE TABLE media (path TEXT PRIMARY KEY, description TEXT)`); err != nil {
		t.Fatal(err)
	}
	_, _ = db.Exec(`INSERT INTO media VALUES ('/a.png', 'a red apple'), ('/b.png', '  ')`)
	dir := t.TempDir()
	list := filepath.Join(dir, "l.jsonl")
	n, err := loraTrainList(db, []string{"/a.png", "/b.png", "/c.png"}, true, list)
	if err != nil || n != 1 {
		t.Fatalf("captioned %d, err %v", n, err)
	}
	b, _ := os.ReadFile(list)
	lines := strings.Split(strings.TrimSpace(string(b)), "\n")
	if len(lines) != 3 || !strings.Contains(lines[0], `"caption":"a red apple"`) || strings.Contains(lines[1], "caption") {
		t.Errorf("list:\n%s", b)
	}
	// without library captions the list holds paths only
	if n, _ := loraTrainList(db, []string{"/a.png"}, false, list); n != 0 {
		t.Errorf("captions without library: %d", n)
	}
}

func TestTrainLoraRunJobMeta(t *testing.T) {
	dir := t.TempDir()
	if runJobMeta(dir) != "" {
		t.Error("no run.json: no job")
	}
	_ = os.WriteFile(filepath.Join(dir, "run.json"), []byte(`{"name":"x","meta":{"job":"abc"}}`), 0o644)
	if got := runJobMeta(dir); got != "abc" {
		t.Errorf("job = %q", got)
	}
}

func TestTrainLoraCheckpointSteps(t *testing.T) {
	run := t.TempDir()
	for _, s := range []string{"000250", "000500", "000750"} {
		d := filepath.Join(run, "checkpoints", "step-"+s)
		_ = os.MkdirAll(d, 0o755)
		if s != "000750" { // step 750 has no optimizer state: not resumable
			_ = os.WriteFile(filepath.Join(d, "state.safetensors"), nil, 0o644)
		}
	}
	if got := latestCheckpointStep(run); got != 500 {
		t.Errorf("latest = %d", got)
	}
	if got := checkpointStep(filepath.Join(run, "checkpoints", "step-000250")); got != 250 {
		t.Errorf("step = %d", got)
	}
	if checkpointStep(run) != -1 || latestCheckpointStep(t.TempDir()) != -1 {
		t.Error("not checkpoints")
	}
	_ = os.WriteFile(filepath.Join(run, ".job"), []byte("job-7\n"), 0o644)
	if readJobMarker(run) != "job-7" {
		t.Error("marker")
	}
}
