package main

import (
	"context"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/stevecastle/shrike/workerapi"
)

// Interactive sessions: loki-segment's --serve vocabulary (open, segment,
// track) held open between calls, so a picture is encoded once and each
// click costs milliseconds. A session lives in its engine's process (the
// one resident model serves them all) and has its own directory for the
// pictures uploaded to it and the masks written for it.

const sessionIdle = 15 * time.Minute

type session struct {
	id   string
	tool string
	dir  string
	slot *slot
	gen  int // the engine process it lives in
	mu   sync.Mutex
	used time.Time
}

type sessions struct {
	w  *worker
	mu sync.Mutex
	m  map[string]*session
}

func newSessions(w *worker) *sessions { return &sessions{w: w, m: map[string]*session{}} }

func (ss *sessions) count() int {
	ss.mu.Lock()
	defer ss.mu.Unlock()
	return len(ss.m)
}

func (ss *sessions) open(tool string) (*session, error) {
	tool = workerapi.EngineName(tool)
	if tool != "loki-segment" {
		return nil, fmt.Errorf("sessions are for loki-segment, not %s", tool)
	}
	if _, ok := ss.w.tools[tool]; !ok {
		return nil, fmt.Errorf("%s is not available on this worker", tool)
	}
	if len(ss.w.slots) == 0 {
		return nil, fmt.Errorf("no GPU")
	}
	// the slot with the fewest sessions
	ss.mu.Lock()
	defer ss.mu.Unlock()
	load := map[*slot]int{}
	for _, s := range ss.m {
		load[s.slot]++
	}
	best := ss.w.slots[0]
	for _, sl := range ss.w.slots[1:] {
		if load[sl] < load[best] {
			best = sl
		}
	}
	id := "s" + randID(6)
	dir := filepath.Join(ss.w.cfg.dataDir, "sessions", id)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, err
	}
	s := &session{id: id, tool: tool, dir: dir, slot: best, used: time.Now()}
	ss.m[id] = s
	return s, nil
}

func (ss *sessions) get(id string) *session {
	ss.mu.Lock()
	defer ss.mu.Unlock()
	return ss.m[id]
}

func (ss *sessions) close(id string) bool {
	ss.mu.Lock()
	s := ss.m[id]
	delete(ss.m, id)
	ss.mu.Unlock()
	if s == nil {
		return false
	}
	e := s.slot.engines[s.tool]
	if e.State() != "stopped" && e.gen == s.gen {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		_, _ = e.Call(ctx, map[string]any{"cmd": "close-session", "session": s.id}, nil)
		cancel()
	}
	_ = os.RemoveAll(s.dir)
	return true
}

// dropEngine forgets the in-process state of sessions living in e (it is
// about to stop); their next call reopens what it needs.
func (ss *sessions) dropEngine(e *engine) {
	ss.mu.Lock()
	defer ss.mu.Unlock()
	for _, s := range ss.m {
		if s.slot.engines[s.tool] == e {
			s.gen = -1
		}
	}
}

func (ss *sessions) expire() {
	ss.mu.Lock()
	var old []string
	for id, s := range ss.m {
		if time.Since(s.used) > sessionIdle {
			old = append(old, id)
		}
	}
	ss.mu.Unlock()
	for _, id := range old {
		ss.close(id)
	}
}

// call runs one request in the session. On a small GPU it waits for the
// slot (a generation there holds the VRAM); on a large one it runs beside it.
func (ss *sessions) call(ctx context.Context, s *session, req map[string]any) (*workerapi.SessionReply, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.used = time.Now()
	if !ss.w.coexist(s.slot) {
		s.slot.mu.Lock()
		defer s.slot.mu.Unlock()
		// the segment engine is the one on this GPU while a session works
		for t, other := range s.slot.engines {
			if t != s.tool && other.State() != "stopped" {
				other.Stop()
			}
		}
	}
	e := s.slot.engines[s.tool]
	start := time.Now().Add(-time.Second)
	expanded := expandValue(req, s.dir).(map[string]any)
	r, err := e.Call(ctx, map[string]any{"cmd": "session", "session": s.id, "req": expanded}, nil)
	if err != nil {
		return nil, err
	}
	s.gen = e.gen
	out := &workerapi.SessionReply{}
	if !r.OK {
		out.Error = placeholder(r.Error, s.dir, workerapi.SessionDir)
		return out, nil
	}
	if r.Reply != nil {
		out.Reply = placeholderValue(r.Reply, s.dir).(map[string]any)
	}
	// files the request wrote (masks, frame folders)
	_ = filepath.WalkDir(s.dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		info, err := d.Info()
		if err != nil || info.ModTime().Before(start) {
			return nil
		}
		rel, _ := filepath.Rel(s.dir, p)
		rel = filepath.ToSlash(rel)
		if strings.HasPrefix(rel, "in/") {
			return nil
		}
		out.Files = append(out.Files, workerapi.OutputFile{Path: rel, Size: info.Size(), Slot: -1})
		return nil
	})
	return out, nil
}

// expandValue replaces $SESSION in every string of a JSON value.
func expandValue(v any, dir string) any {
	d := filepath.ToSlash(dir)
	switch x := v.(type) {
	case string:
		return strings.ReplaceAll(x, workerapi.SessionDir, d)
	case []any:
		out := make([]any, len(x))
		for i := range x {
			out[i] = expandValue(x[i], dir)
		}
		return out
	case map[string]any:
		out := make(map[string]any, len(x))
		for k, e := range x {
			out[k] = expandValue(e, dir)
		}
		return out
	}
	return v
}

// placeholderValue maps this machine's session paths back to $SESSION/...
func placeholderValue(v any, dir string) any {
	switch x := v.(type) {
	case string:
		return placeholder(x, dir, workerapi.SessionDir)
	case []any:
		out := make([]any, len(x))
		for i := range x {
			out[i] = placeholderValue(x[i], dir)
		}
		return out
	case map[string]any:
		out := make(map[string]any, len(x))
		for k, e := range x {
			out[k] = placeholderValue(e, dir)
		}
		return out
	}
	return v
}
