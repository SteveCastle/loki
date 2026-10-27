package tasks

import (
	"context"
	"crypto/sha1"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/stevecastle/shrike/appconfig"
	"github.com/stevecastle/shrike/workerapi"
)

// --- Scene editor selection on a GPU worker ---
//
// With loki-segment routed to a pool, each scene's selection session lives
// on a worker node (serverless workers cannot hold one): the picture (or
// clip) is uploaded once and again only when it changes, every pick writes
// into its own folder there, and the masks come back to the local out
// folder the request named, so SegmentScene reads them exactly as it reads
// a local session's.

type remoteSeg struct {
	mu     sync.Mutex
	node   *workerapi.Node
	key    string // target key, for logs
	id     string // the worker's session id
	opened string // local path open in it
	sig    string // size+mtime of what was uploaded for it
	picks  int
	used   time.Time
}

var remoteSegs = struct {
	sync.Mutex
	m map[string]*remoteSeg
}{m: map[string]*remoteSeg{}}

var remoteSegReaper sync.Once

// remoteSegTarget picks the node a new session goes to: the least busy
// reachable node. ok=false means the pick runs here (hybrid, or no node and
// the strategy falls back).
func remoteSegTarget() (*engineTarget, bool, error) {
	engine := workerapi.EngineName(segSpec.Bin)
	if engineDisp.route(engine) == appconfig.EngineStrategyHybrid {
		return nil, false, nil
	}
	cands, unreachable := engineDisp.candidates(engine)
	var best *engineTarget
	bestLoad := 1 << 30
	for _, t := range cands {
		if t.kind == "local" {
			continue
		}
		if t.kind != "node" {
			continue
		}
		t.mu.Lock()
		load := t.inflight
		t.mu.Unlock()
		if load < bestLoad {
			best, bestLoad = t, load
		}
	}
	if best != nil {
		return best, true, nil
	}
	for _, t := range cands {
		if t.kind == "local" {
			return nil, false, nil
		}
	}
	msg := "object selection needs a GPU: no worker node is reachable for loki-segment (serverless endpoints cannot hold a selection session)"
	if len(unreachable) > 0 {
		msg += ": " + strings.Join(unreachable, "; ")
	}
	return nil, true, errors.New(msg)
}

func fileSig(p string) string {
	st, err := os.Stat(p)
	if err != nil {
		return ""
	}
	return fmt.Sprintf("%d-%d", st.Size(), st.ModTime().UnixNano())
}

func (r *remoteSeg) close() {
	if r.id == "" {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	_ = r.node.CloseSession(ctx, r.id)
	r.id, r.opened, r.sig = "", "", ""
}

// remoteSegRequest answers one segment request for scene id on a worker.
// handled=false: run it locally instead.
func remoteSegRequest(id, path string, req map[string]any) (resp map[string]any, handled bool, err error) {
	remoteSegReaper.Do(func() {
		go func() {
			for range time.Tick(30 * time.Second) {
				remoteSegs.Lock()
				for k, s := range remoteSegs.m {
					if s.mu.TryLock() {
						if time.Since(s.used) > segSessionIdle {
							s.close()
							delete(remoteSegs.m, k)
						}
						s.mu.Unlock()
					}
				}
				remoteSegs.Unlock()
			}
		}()
	})
	remoteSegs.Lock()
	s := remoteSegs.m[id]
	remoteSegs.Unlock()
	if s == nil {
		t, remote, terr := remoteSegTarget()
		if !remote {
			return nil, false, nil
		}
		if terr != nil {
			return nil, true, terr
		}
		s = &remoteSeg{node: t.node, key: t.key}
		remoteSegs.Lock()
		if prev := remoteSegs.m[id]; prev != nil {
			s = prev
		} else {
			remoteSegs.m[id] = s
		}
		remoteSegs.Unlock()
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.used = time.Now()
	resp, err = s.request(path, req)
	if err != nil {
		// the worker went away or restarted: start over once on a fresh session
		var he *workerapi.HTTPError
		if errors.As(err, &he) && he.Status == 404 || strings.Contains(err.Error(), "connect") {
			s.id, s.opened, s.sig = "", "", ""
			resp, err = s.request(path, req)
		}
	}
	return resp, true, err
}

func (s *remoteSeg) request(path string, req map[string]any) (map[string]any, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	if s.id == "" {
		id, err := s.node.OpenSession(ctx, segSpec.Bin)
		if err != nil {
			return nil, fmt.Errorf("opening a selection session on %s: %w", s.key, err)
		}
		s.id, s.opened, s.sig = id, "", ""
	}
	open := func() error {
		sum := sha1.Sum([]byte(path))
		rel := "in/" + hex.EncodeToString(sum[:4]) + "-" + workerapi.SafeName(filepath.Base(path))
		if err := s.node.PutSessionFile(ctx, s.id, rel, path); err != nil {
			return fmt.Errorf("sending %s to %s: %w", filepath.Base(path), s.key, err)
		}
		r, err := s.node.CallSession(ctx, s.id, map[string]any{"cmd": "open", "path": workerapi.SessionDir + "/" + rel}, 3*time.Minute)
		if err != nil {
			return err
		}
		if r.Error != "" {
			return errors.New(r.Error)
		}
		s.opened, s.sig = path, fileSig(path)
		return nil
	}
	if s.opened != path || s.sig != fileSig(path) {
		if err := open(); err != nil {
			return nil, err
		}
	}
	// each pick its own folder on the worker; the masks land in the local one
	localOut, _ := req["out"].(string)
	s.picks++
	pick := fmt.Sprintf("out/%d", s.picks)
	call := map[string]any{}
	for k, v := range req {
		call[k] = v
	}
	call["cmd"] = "segment" // as the local session sends it
	if localOut != "" {
		call["out"] = workerapi.SessionDir + "/" + pick
	}
	r, err := s.node.CallSession(ctx, s.id, call, time.Minute)
	if err == nil && r.Error != "" && strings.Contains(r.Error, "no image open") {
		// the worker's engine restarted under the session
		if err = open(); err == nil {
			r, err = s.node.CallSession(ctx, s.id, call, time.Minute)
		}
	}
	if err != nil {
		return nil, err
	}
	if r.Error != "" {
		return nil, errors.New(r.Error)
	}
	mapFn := func(rel string) (string, bool) {
		rest, ok := strings.CutPrefix(rel, pick+"/")
		if !ok || localOut == "" {
			return "", false
		}
		return filepath.Join(localOut, filepath.FromSlash(rest)), true
	}
	if localOut != "" && len(r.Files) > 0 {
		err := s.node.SessionFiles(ctx, s.id, pick, func(rel string, rd io.Reader) error {
			local, ok := mapFn(rel)
			if !ok {
				return nil
			}
			_, werr := workerapi.WriteFileAtomic(local, rd)
			return werr
		})
		if err != nil {
			return nil, fmt.Errorf("fetching the masks from %s: %w", s.key, err)
		}
	}
	return rewriteValue(r.Reply, mapFn).(map[string]any), nil
}

// rewriteValue maps $SESSION paths in every string of a JSON value.
func rewriteValue(v any, mapFn func(string) (string, bool)) any {
	switch x := v.(type) {
	case string:
		return workerapi.RewritePaths(x, mapFn)
	case []any:
		out := make([]any, len(x))
		for i := range x {
			out[i] = rewriteValue(x[i], mapFn)
		}
		return out
	case map[string]any:
		out := make(map[string]any, len(x))
		for k, e := range x {
			out[k] = rewriteValue(e, mapFn)
		}
		return out
	case nil:
		return map[string]any{}
	}
	return v
}
