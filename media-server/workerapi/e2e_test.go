package workerapi

import (
	"context"
	"image"
	"image/color"
	"image/png"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// Opt-in end-to-end tests against a running loki-worker with real engines
// and models (a GPU):
//
//	LOKI_WORKER_E2E=http://127.0.0.1:8080 LOKI_WORKER_E2E_TOKEN=... go test ./workerapi -run E2E -v
func e2eNode(t *testing.T) *Node {
	u := os.Getenv("LOKI_WORKER_E2E")
	if u == "" {
		t.Skip("set LOKI_WORKER_E2E to a loki-worker URL")
	}
	return &Node{Name: "e2e", URL: u, Token: os.Getenv("LOKI_WORKER_E2E_TOKEN")}
}

func writeTestPNG(t *testing.T, path string) {
	img := image.NewRGBA(image.Rect(0, 0, 512, 384))
	for y := 0; y < 384; y++ {
		for x := 0; x < 512; x++ {
			c := color.RGBA{90, 140, 200, 255}
			if (x-256)*(x-256)+(y-192)*(y-192) < 100*100 {
				c = color.RGBA{240, 200, 40, 255}
			}
			img.Set(x, y, c)
		}
	}
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if err := png.Encode(f, img); err != nil {
		t.Fatal(err)
	}
}

func TestE2ERetouchOnNode(t *testing.T) {
	n := e2eNode(t)
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Minute)
	defer cancel()
	dir := t.TempDir()
	in := filepath.Join(dir, "my photo.png")
	out := filepath.Join(dir, "result", "night.png")
	writeTestPNG(t, in)
	for round := 1; round <= 2; round++ {
		os.Remove(out)
		plan := PlanArgs([]string{"-p", "make it night", "--steps", "4", "-o", out, in})
		var logs []string
		t0 := time.Now()
		res, err := RunOnNode(ctx, n, plan, "retouch", false, Hooks{Log: func(l string) { logs = append(logs, l) }, Status: func(s string) { t.Log("status:", s) }})
		if err != nil {
			t.Fatalf("round %d: %v\n%s", round, err, strings.Join(logs, "\n"))
		}
		t.Logf("round %d: %.1fs (engine %.1fs), stdout %q", round, time.Since(t0).Seconds(), res.Secs, res.Stdout)
		if len(res.Stdout) != 1 || res.Stdout[0] != out {
			t.Fatalf("stdout %q, want the local output %q", res.Stdout, out)
		}
		if st, err := os.Stat(out); err != nil || st.Size() == 0 {
			t.Fatalf("no output at %s: %v", out, err)
		}
	}
}

func TestE2EShowPromptLight(t *testing.T) {
	n := e2eNode(t)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	plan := PlanArgs([]string{"--preset", "restore", "--show-prompt"})
	res, err := RunOnNode(ctx, n, plan, "loki-retouch", true, Hooks{})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(strings.Join(res.Stdout, "\n"), "<image1>") {
		t.Fatalf("prompt preview: %q", res.Stdout)
	}
}

func TestE2ESegmentSession(t *testing.T) {
	n := e2eNode(t)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	dir := t.TempDir()
	in := filepath.Join(dir, "pic.png")
	writeTestPNG(t, in)
	id, err := n.OpenSession(ctx, "segment")
	if err != nil {
		t.Fatal(err)
	}
	defer n.CloseSession(context.Background(), id)
	if err := n.PutSessionFile(ctx, id, "in/pic.png", in); err != nil {
		t.Fatal(err)
	}
	r, err := n.CallSession(ctx, id, map[string]any{"cmd": "open", "path": SessionDir + "/in/pic.png"}, 3*time.Minute)
	if err != nil || r.Error != "" {
		t.Fatalf("open: %v %s", err, r.Error)
	}
	t0 := time.Now()
	r, err = n.CallSession(ctx, id, map[string]any{"cmd": "segment", "point": []int{256, 192}, "out": SessionDir + "/out"}, time.Minute)
	if err != nil || r.Error != "" {
		t.Fatalf("segment: %v %v", err, r)
	}
	t.Logf("pick in %v: %v, files %v", time.Since(t0), r.Reply["objects"], r.Files)
	objs, _ := r.Reply["objects"].([]any)
	if len(objs) != 1 {
		t.Fatalf("objects: %v", r.Reply)
	}
	mask, _ := objs[0].(map[string]any)["mask"].(string)
	if !strings.HasPrefix(mask, SessionDir+"/out/") {
		t.Fatalf("mask path %q is not a session path", mask)
	}
	got := 0
	err = n.SessionFiles(ctx, id, "out", func(rel string, r io.Reader) error {
		b, _ := io.ReadAll(r)
		if len(b) > 0 {
			got++
		}
		return nil
	})
	if err != nil || got == 0 {
		t.Fatalf("session files: %v (%d)", err, got)
	}
}
