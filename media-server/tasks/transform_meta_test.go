package tasks

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stevecastle/shrike/jobqueue"
)

// The Transform flow stamps its recipe (intent + settings) through the
// `meta` field: it must reach both engines as --meta (or --meta-file when
// long), and never reach an engine build that would reject the flag.

const testRecipe = `{"kind":"loki-transform","v":1,"intent":"edit","settings":{"prompt":"make it \"night\"\nwith rain"},"paths":["C:\\a.png"]}`

func TestTransformMetaReachesEngines(t *testing.T) {
	retouch := uiRetouchArgs(map[string]string{"prompt": "x", "meta": testRecipe}, "in.png", nil)
	if !containsArg(retouch, "--meta="+testRecipe) {
		t.Errorf("retouch args %q lack the recipe", retouch)
	}

	j := &jobqueue.Job{Arguments: uiFieldsToArgs(map[string]string{"prompt": "x", "meta": testRecipe})}
	p := reshootParamsFromOptions(ParseOptions(j, reshootOptions))
	reshoot := buildReshootArgs(p, []string{"a.png"}, nil, nil, "o.mp4", "")
	if !containsArg(reshoot, "--meta="+testRecipe) {
		t.Errorf("reshoot args %q lack the recipe", reshoot)
	}

	p.MetaFile = "recipe.json"
	reshoot = buildReshootArgs(p, []string{"a.png"}, nil, nil, "o.mp4", "")
	if !strings.Contains(strings.Join(reshoot, "|"), "--meta-file|recipe.json") || containsArg(reshoot, "--meta="+testRecipe) {
		t.Errorf("with a meta file, args = %q", reshoot)
	}

	// No recipe, no flag.
	plain := uiRetouchArgs(map[string]string{"prompt": "x"}, "in.png", nil)
	for _, a := range plain {
		if strings.HasPrefix(a, "--meta") {
			t.Errorf("args %q carry a meta flag without a recipe", plain)
		}
	}
}

func containsArg(args []string, want string) bool {
	for _, a := range args {
		if a == want {
			return true
		}
	}
	return false
}

func withHelp(t *testing.T, help string, err error) *int {
	t.Helper()
	calls := 0
	prev := probeAIToolHelp
	probeAIToolHelp = func(context.Context, aiToolRun) (string, error) {
		calls++
		return help, err
	}
	metaSupportMu.Lock()
	metaSupport = map[string]bool{}
	metaSupportMu.Unlock()
	t.Cleanup(func() { probeAIToolHelp = prev })
	return &calls
}

func TestPrepareMetaOldEngineDropsRecipe(t *testing.T) {
	withHelp(t, "Usage: loki-retouch [OPTIONS]\n  --json  Print JSON\n", nil)
	q := jobqueue.NewQueue()
	meta, file := testRecipe, ""
	cleanup, err := prepareMeta(context.Background(), q, "job", "retouch", aiToolRun{Bin: "old-engine"}, &meta, &file)
	if err != nil {
		t.Fatal(err)
	}
	defer cleanup()
	if meta != "" || file != "" {
		t.Errorf("old engine: meta=%q file=%q; want both empty", meta, file)
	}
}

func TestPrepareMetaNewEngine(t *testing.T) {
	calls := withHelp(t, "  --meta <JSON>  Your own JSON object\n  --meta-file <FILE>\n", nil)
	q := jobqueue.NewQueue()
	tool := aiToolRun{Bin: "new-engine"}

	meta, file := testRecipe, ""
	cleanup, err := prepareMeta(context.Background(), q, "job", "retouch", tool, &meta, &file)
	if err != nil {
		t.Fatal(err)
	}
	cleanup()
	if meta != testRecipe || file != "" {
		t.Errorf("short recipe: meta=%q file=%q; want inline", meta, file)
	}

	long := `{"settings":{"prompt":"` + strings.Repeat("a", aiLongPromptBytes) + `"}}`
	meta, file = long, ""
	cleanup, err = prepareMeta(context.Background(), q, "job", "retouch", tool, &meta, &file)
	if err != nil {
		t.Fatal(err)
	}
	if file == "" {
		t.Fatal("long recipe did not go through a file")
	}
	if b, _ := os.ReadFile(file); string(b) != long {
		t.Errorf("recipe file holds %d bytes; want the recipe", len(b))
	}
	cleanup()
	if _, err := os.Stat(file); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("recipe file %s not removed", filepath.Base(file))
	}
	if *calls != 1 {
		t.Errorf("--help probed %d times; want 1 (cached)", *calls)
	}
}

func TestAIToolSupportsMetaRetriesFailedProbe(t *testing.T) {
	calls := withHelp(t, "", errors.New("boom"))
	tool := aiToolRun{Bin: "flaky-engine"}
	if aiToolSupportsMeta(context.Background(), tool) || aiToolSupportsMeta(context.Background(), tool) {
		t.Fatal("a failed probe must report no support")
	}
	if *calls != 2 {
		t.Errorf("probed %d times; a failed probe must not be cached", *calls)
	}
}
