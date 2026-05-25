package bundled

import (
	"context"
	"os/exec"
	"strings"
	"sync"
	"time"

	"github.com/stevecastle/shrike/platform"
)

var (
	statusCache   []Status
	statusCacheMu sync.RWMutex
)

func VerifyAll() []Status {
	out := make([]Status, 0, len(Manifest))
	for _, b := range Manifest {
		out = append(out, verifyOne(b))
	}
	statusCacheMu.Lock()
	statusCache = out
	statusCacheMu.Unlock()
	return out
}

func CachedStatus() []Status {
	statusCacheMu.RLock()
	defer statusCacheMu.RUnlock()
	out := make([]Status, len(statusCache))
	copy(out, statusCache)
	return out
}

// SetCachedStatusForTest replaces the cache. Tests only.
func SetCachedStatusForTest(in []Status) {
	statusCacheMu.Lock()
	defer statusCacheMu.Unlock()
	statusCache = in
}

func verifyOne(b Bundled) Status {
	s := Status{ID: b.ID, Name: b.Name}
	path, err := Resolve(b.ID)
	if err != nil {
		s.State = "missing"
		s.Error = err.Error()
		return s
	}
	s.Path = path
	removeQuarantine(path)

	if len(b.VersionArgs) == 0 {
		s.State = "ready"
		return s
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, path, b.VersionArgs...)
	platform.HideSubprocessWindow(cmd)
	out, err := cmd.CombinedOutput()
	if err != nil {
		s.State = "broken"
		s.Error = trimErr(string(out), err)
		return s
	}
	s.State = "ready"
	s.Version = firstLine(string(out))
	return s
}

func firstLine(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return strings.TrimSpace(s[:i])
	}
	return strings.TrimSpace(s)
}

func trimErr(out string, err error) string {
	msg := strings.TrimSpace(out)
	if msg == "" {
		return err.Error()
	}
	if len(msg) > 200 {
		msg = msg[:200] + "..."
	}
	return msg
}
