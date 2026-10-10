package tasks

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/stevecastle/shrike/deps/models"
)

// TestSplatPipelineEndToEnd trains a real splat. Opt-in, needs a GPU and
// network on first run (COLMAP + Brush install from the manifest):
//
//	LOKI_SPLAT_E2E=<video>  LOKI_SPLAT_E2E_DATA=<persistent deps dir> \
//	  go test ./tasks -run SplatPipelineEndToEnd -v -timeout 60m
func TestSplatPipelineEndToEnd(t *testing.T) {
	video := os.Getenv("LOKI_SPLAT_E2E")
	if video == "" {
		t.Skip("set LOKI_SPLAT_E2E=<video> to run")
	}
	data := os.Getenv("LOKI_SPLAT_E2E_DATA")
	if data == "" {
		data = t.TempDir()
	}
	models.SetDataDirForTest(data)
	t.Cleanup(func() { models.SetDataDirForTest("") })
	models.RebuildState()

	logf := func(format string, a ...any) { t.Logf(format, a...) }
	ctx, cancel := context.WithTimeout(context.Background(), 55*time.Minute)
	defer cancel()
	tools, err := resolveSplatTools(ctx, logf)
	if err != nil {
		t.Fatalf("resolveSplatTools: %v", err)
	}
	t.Logf("tools: %+v", tools)

	out := filepath.Join(t.TempDir(), "scene_splat.ply")
	if keep := os.Getenv("LOKI_SPLAT_E2E_OUT"); keep != "" {
		out = keep
	}
	last := -1
	err = runSplatPipeline(ctx, tools, splatPipelineSpec{
		Params:  splatParamsFromOptions(map[string]any{"quality": "draft", "frames": 90.0, "maxres": 960.0}),
		Video:   video,
		Output:  out,
		WorkDir: e2eWorkDir(t),
	}, logf, func(done int) {
		if done < last {
			t.Errorf("progress went backwards: %d after %d", done, last)
		}
		last = done
	})
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	f, err := os.Open(out)
	if err != nil {
		t.Fatalf("no output: %v", err)
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	vertices := ""
	for sc.Scan() {
		line := sc.Text()
		if strings.HasPrefix(line, "element vertex ") {
			vertices = strings.TrimPrefix(line, "element vertex ")
		}
		if line == "end_header" {
			break
		}
	}
	if vertices == "" {
		t.Fatalf("output is not a splat PLY")
	}
	st, _ := f.Stat()
	t.Logf("wrote %s: %s splats, %.1f MB, final progress %d", out, vertices, float64(st.Size())/1e6, last)
	fmt.Println("E2E_PLY", out)
}

// e2eWorkDir keeps the workspace for inspection when LOKI_SPLAT_E2E_WORK is set.
func e2eWorkDir(t *testing.T) string {
	if d := os.Getenv("LOKI_SPLAT_E2E_WORK"); d != "" {
		_ = os.RemoveAll(d)
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
		return d
	}
	return t.TempDir()
}
