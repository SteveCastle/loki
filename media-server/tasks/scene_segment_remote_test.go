package tasks

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stevecastle/shrike/appconfig"
	"github.com/stevecastle/shrike/workerapi"
)

// fakeSessionNode serves loki-worker's session API with a fake segmenter.
type fakeSessionNode struct {
	mu      sync.Mutex
	dir     string
	puts    int
	opens   int
	opened  string
	srv     *httptest.Server
	restart bool // the next segment answers "no image open" (engine restarted)
}

func newFakeSessionNode(t *testing.T) *fakeSessionNode {
	n := &fakeSessionNode{dir: t.TempDir()}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/info", func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(workerapi.Info{Worker: "fake", Ready: true, Slots: 1, Tools: []workerapi.ToolInfo{{Name: "loki-segment"}}})
	})
	mux.HandleFunc("POST /v1/sessions", func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]string{"id": "s1"})
	})
	mux.HandleFunc("PUT /v1/sessions/s1/files/{path...}", func(w http.ResponseWriter, r *http.Request) {
		n.mu.Lock()
		n.puts++
		n.mu.Unlock()
		_, _ = workerapi.WriteFileAtomic(filepath.Join(n.dir, filepath.FromSlash(r.PathValue("path"))), r.Body)
	})
	mux.HandleFunc("POST /v1/sessions/s1/call", func(w http.ResponseWriter, r *http.Request) {
		var call workerapi.SessionCall
		_ = json.NewDecoder(r.Body).Decode(&call)
		n.mu.Lock()
		defer n.mu.Unlock()
		exp := func(s string) string { return strings.Replace(s, workerapi.SessionDir, filepath.ToSlash(n.dir), 1) }
		switch call.Req["cmd"] {
		case "open":
			p := exp(call.Req["path"].(string))
			if _, err := os.Stat(p); err != nil {
				_ = json.NewEncoder(w).Encode(workerapi.SessionReply{Error: err.Error()})
				return
			}
			n.opens++
			n.opened = p
			_ = json.NewEncoder(w).Encode(workerapi.SessionReply{Reply: map[string]any{"ok": true}})
		case "segment":
			if n.restart || n.opened == "" {
				n.restart = false
				n.opened = ""
				_ = json.NewEncoder(w).Encode(workerapi.SessionReply{Error: "no image open: send {\"cmd\":\"open\",\"path\":...} first"})
				return
			}
			out := call.Req["out"].(string) // $SESSION/out/<n>
			rel := strings.TrimPrefix(out, workerapi.SessionDir+"/") + "/1.png"
			_, _ = workerapi.WriteFileAtomic(filepath.Join(n.dir, filepath.FromSlash(rel)), strings.NewReader("MASK"))
			_ = json.NewEncoder(w).Encode(workerapi.SessionReply{
				Reply: map[string]any{"width": 4, "height": 4, "objects": []any{map[string]any{"id": 1, "mask": out + "/1.png"}}},
				Files: []workerapi.OutputFile{{Path: rel, Size: 4, Slot: -1}},
			})
		}
	})
	mux.HandleFunc("GET /v1/sessions/s1/files.tar", func(w http.ResponseWriter, r *http.Request) {
		prefix := r.URL.Query().Get("prefix")
		_ = workerapi.WriteTar(w, n.dir, []string{prefix + "/1.png"})
	})
	mux.HandleFunc("DELETE /v1/sessions/s1", func(w http.ResponseWriter, r *http.Request) {})
	n.srv = httptest.NewServer(mux)
	t.Cleanup(n.srv.Close)
	return n
}

func TestSegmentSessionOnWorker(t *testing.T) {
	node := newFakeSessionNode(t)
	usePools(t, nil, appconfig.EnginesConfig{Strategy: "pool", Pools: []appconfig.WorkerPool{{Name: "gpu", Nodes: []appconfig.WorkerNode{{Name: "n", URL: node.srv.URL}}}}})
	t.Cleanup(func() {
		remoteSegs.Lock()
		remoteSegs.m = map[string]*remoteSeg{}
		remoteSegs.Unlock()
	})
	// a local session would need a local engine: there is none
	oldBin := segBinary
	t.Cleanup(func() { segBinary = oldBin })
	segBinary = func() (string, error) { t.Fatal("a pooled pick started a local loki-segment"); return "", nil }
	dir := t.TempDir()
	pic := filepath.Join(dir, "seg-picture.png")
	os.WriteFile(pic, []byte("picture v1"), 0644)
	out := filepath.Join(dir, "seg-out")

	pickOnce := func() string {
		_ = os.RemoveAll(out)
		resp, err := segRequest("scene1", pic, map[string]any{"out": out, "points": [][]float64{{1, 1, 1}}})
		if err != nil {
			t.Fatal(err)
		}
		mask := resp["objects"].([]any)[0].(map[string]any)["mask"].(string)
		if b, err := os.ReadFile(mask); err != nil || string(b) != "MASK" || !strings.HasPrefix(mask, out) {
			t.Fatalf("mask %q: %q %v", mask, b, err)
		}
		return mask
	}
	pickOnce()
	pickOnce()
	if node.puts != 1 || node.opens != 1 {
		t.Fatalf("an unchanged picture was sent %d times, opened %d times", node.puts, node.opens)
	}
	// the picture changes (layers re-flattened): sent and opened again
	time.Sleep(20 * time.Millisecond)
	os.WriteFile(pic, []byte("picture v2, longer"), 0644)
	pickOnce()
	if node.puts != 2 || node.opens != 2 {
		t.Fatalf("a changed picture: %d sends, %d opens", node.puts, node.opens)
	}
	// the worker's engine restarted under the session: reopened transparently
	node.mu.Lock()
	node.restart = true
	node.mu.Unlock()
	pickOnce()
	if node.opens != 3 {
		t.Fatalf("not reopened after an engine restart: %d opens", node.opens)
	}
}
