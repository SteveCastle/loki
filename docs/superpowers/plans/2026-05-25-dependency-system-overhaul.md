# Dependency & Setup System Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the on-demand binary downloader with three focused packages (bundled / optional / models), a robust model downloader (manifest + checksum + atomic + resume), a skippable welcome wizard, and a CI release pipeline that bundles native binaries per platform — fixing the macOS Gatekeeper crash class along the way.

**Architecture:** Three new packages under `media-server/deps/` (`bundled`, `optional`, `models`) with no cross-imports; a thin `status` aggregator the UI reads; new `/api/deps/*` HTTP handlers; CI scripts that fetch and pin upstream binaries by SHA-256; a React onboarding route in the existing SPA.

**Tech Stack:** Go 1.x (server, build tags for platform split), React + XState (SPA renderer at `src/renderer/`), SQLite (existing, untouched), GitHub Actions (release pipeline), bash + PowerShell (fetch scripts).

**Spec:** `docs/superpowers/specs/2026-05-25-dependency-system-overhaul-design.md`

---

## File Structure

**New files (Go):**
```
media-server/deps/bundled/bundled.go              Resolve, IDs, types
media-server/deps/bundled/manifest.go             var Manifest []Bundled
media-server/deps/bundled/verify.go               VerifyAll
media-server/deps/bundled/quarantine_darwin.go    removeQuarantine
media-server/deps/bundled/quarantine_other.go     no-op stub
media-server/deps/bundled/bundled_test.go
media-server/deps/bundled/verify_test.go

media-server/deps/optional/optional.go            Detect, Status, IDs
media-server/deps/optional/manifest.go            var Manifest []Optional
media-server/deps/optional/hints.go               static install hints
media-server/deps/optional/optional_test.go

media-server/deps/models/manifest.go              embedded JSON parse
media-server/deps/models/manifest.json            embedded manifest
media-server/deps/models/store.go                 paths, atomic install
media-server/deps/models/downloader.go            resumable HTTP + sha256
media-server/deps/models/progress.go              tracker + SSE channel
media-server/deps/models/state.go                 state.json, serialized writes
media-server/deps/models/manifest_test.go
media-server/deps/models/store_test.go
media-server/deps/models/downloader_test.go
media-server/deps/models/state_test.go

media-server/deps/status/status.go                Snapshot()
media-server/deps/status/status_test.go

media-server/deps/facade.go                       MustBundled, ModelPath shims

media-server/handlers/deps_handlers.go            /api/deps/* endpoints
media-server/handlers/deps_handlers_test.go
media-server/routes_deps.go                       RegisterDepsRoutes(mux)

media-server/scripts/bundled-versions.json        pinned versions + sha256
media-server/scripts/fetch-bundled-deps.sh        Linux/macOS fetch
media-server/scripts/fetch-bundled-deps.ps1       Windows fetch
media-server/scripts/smoke-bundled.sh             post-build smoke test
media-server/.github/workflows/release-server.yml release workflow
```

**New files (SPA, React):**
```
src/renderer/onboarding/OnboardingWizard.tsx
src/renderer/onboarding/BundledPanel.tsx
src/renderer/onboarding/OptionalPanel.tsx
src/renderer/onboarding/ModelsPanel.tsx
src/renderer/onboarding/useDepsStatus.ts        polling + SSE hook
src/renderer/onboarding/api.ts                  fetch helpers
src/renderer/onboarding/styles.module.css
src/renderer/onboarding/__tests__/OnboardingWizard.test.tsx
```

**Files deleted at the end (do NOT delete until callers migrated):**
```
media-server/deps/deps.go
media-server/deps/metadata.go
media-server/deps/paths.go
media-server/deps/exec.go
media-server/deps/exec_windows.go
media-server/deps/exec_linux.go
media-server/deps/exec_darwin.go
media-server/deps/ffmpeg.go
media-server/deps/whisper.go
media-server/deps/onnx.go
media-server/deps/onnxtag.go
media-server/deps/ytdlp.go
media-server/deps/gallerydl.go
media-server/deps/ollama.go
media-server/deps/dce.go
media-server/deps/deps_test.go
media-server/downloads/                          (entire package)
```

**Files modified:**
```
media-server/main.go             (Windows) — remove /setup redirect, /dependencies handlers, setupMode flag; call bundled.VerifyAll; register new routes
media-server/main_darwin.go      same
media-server/main_linux.go       same
media-server/tasks/*.go          ~40 call sites — rewrite GetXxxPath → MustBundled/ModelPath/Detect
media-server/loki_api.go         any dep references
```

---

## Phase 1: Foundation — `bundled` package

### Task 1: Bundled package skeleton + types + Resolve

**Files:**
- Create: `media-server/deps/bundled/bundled.go`
- Create: `media-server/deps/bundled/manifest.go`
- Test: `media-server/deps/bundled/bundled_test.go`

- [ ] **Step 1: Write the failing test**

Create `media-server/deps/bundled/bundled_test.go`:

```go
package bundled

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestResolve_ReturnsPathRelativeToExecDir(t *testing.T) {
	dir := t.TempDir()
	binDir := filepath.Join(dir, "bin")
	if err := os.MkdirAll(binDir, 0o755); err != nil {
		t.Fatal(err)
	}
	name := "ffmpeg"
	if runtime.GOOS == "windows" {
		name = "ffmpeg.exe"
	}
	if err := os.WriteFile(filepath.Join(binDir, name), []byte("fake"), 0o755); err != nil {
		t.Fatal(err)
	}

	old := execDirOverride
	execDirOverride = dir
	defer func() { execDirOverride = old }()

	got, err := Resolve("ffmpeg")
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	want := filepath.Join(binDir, name)
	if got != want {
		t.Errorf("Resolve = %q, want %q", got, want)
	}
}

func TestResolve_ReturnsErrMissingWhenFileAbsent(t *testing.T) {
	dir := t.TempDir()
	old := execDirOverride
	execDirOverride = dir
	defer func() { execDirOverride = old }()

	_, err := Resolve("ffmpeg")
	if err == nil {
		t.Fatal("expected error for missing file")
	}
	if !IsMissing(err) {
		t.Errorf("expected IsMissing(err)=true, got: %v", err)
	}
}

func TestResolve_UnknownIDReturnsErrUnknown(t *testing.T) {
	_, err := Resolve("nope-not-a-dep")
	if err == nil {
		t.Fatal("expected error")
	}
	if IsMissing(err) {
		t.Error("unknown id should not be IsMissing")
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd media-server && go test ./deps/bundled/...`
Expected: compile error (package doesn't exist yet).

- [ ] **Step 3: Implement `bundled.go`**

Create `media-server/deps/bundled/bundled.go`:

```go
// Package bundled resolves paths to native binaries shipped alongside the
// server executable. It never downloads; it only locates and validates.
package bundled

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

// Bundled is one binary or shared library bundled with the server release.
type Bundled struct {
	ID          string // stable key callers use, e.g. "ffmpeg"
	Name        string // display name, e.g. "FFmpeg"
	RelPath     string // path under <execDir>/bin, e.g. "ffmpeg.exe"
	VersionArgs []string // args to pass to probe version, e.g. []string{"-version"}
}

// Status describes the runtime state of one bundled entry.
type Status struct {
	ID      string
	Name    string
	Path    string
	State   string // "ready" | "missing" | "broken"
	Version string
	Error   string
}

var (
	ErrUnknown = errors.New("bundled: unknown dependency id")
	errMissing = errors.New("bundled: file not found")

	execDirOverride string // tests
	execDirOnce     sync.Once
	execDirCached   string
)

// IsMissing reports whether err indicates a bundled file that was not present
// at the resolved path (vs. an unknown id, which is a programmer error).
func IsMissing(err error) bool { return errors.Is(err, errMissing) }

// Resolve returns the absolute path to the bundled binary for id, or an error.
// The returned path is guaranteed to exist when err == nil.
func Resolve(id string) (string, error) {
	entry, ok := lookup(id)
	if !ok {
		return "", fmt.Errorf("%w: %q", ErrUnknown, id)
	}
	path := filepath.Join(execDir(), "bin", entry.RelPath)
	info, err := os.Stat(path)
	if err != nil || info.IsDir() {
		return "", fmt.Errorf("%w: %s (id=%s)", errMissing, path, id)
	}
	return path, nil
}

// IDs returns the IDs of every bundled entry in manifest order.
func IDs() []string {
	out := make([]string, 0, len(Manifest))
	for _, b := range Manifest {
		out = append(out, b.ID)
	}
	return out
}

func lookup(id string) (Bundled, bool) {
	for _, b := range Manifest {
		if b.ID == id {
			return b, true
		}
	}
	return Bundled{}, false
}

func execDir() string {
	if execDirOverride != "" {
		return execDirOverride
	}
	execDirOnce.Do(func() {
		exe, err := os.Executable()
		if err != nil {
			execDirCached = "."
			return
		}
		execDirCached = filepath.Dir(exe)
	})
	return execDirCached
}
```

- [ ] **Step 4: Implement `manifest.go`**

Create `media-server/deps/bundled/manifest.go`:

```go
package bundled

import "runtime"

// Manifest enumerates every bundled binary or shared library. RelPath is
// resolved against <execDir>/bin. macOS may omit entries (e.g. ffplay) by
// constructing the slice conditionally.
var Manifest = func() []Bundled {
	exe := ""
	if runtime.GOOS == "windows" {
		exe = ".exe"
	}
	libExt := ".so"
	switch runtime.GOOS {
	case "windows":
		libExt = ".dll"
	case "darwin":
		libExt = ".dylib"
	}

	entries := []Bundled{
		{ID: "ffmpeg", Name: "FFmpeg", RelPath: "ffmpeg" + exe, VersionArgs: []string{"-version"}},
		{ID: "ffprobe", Name: "FFprobe", RelPath: "ffprobe" + exe, VersionArgs: []string{"-version"}},
		{ID: "exiftool", Name: "ExifTool", RelPath: "exiftool" + exe, VersionArgs: []string{"-ver"}},
		{ID: "onnxtag", Name: "ONNX Tagger", RelPath: "onnxtag" + exe, VersionArgs: []string{"--version"}},
		{ID: "onnxruntime", Name: "ONNX Runtime", RelPath: "onnxruntime" + libExt, VersionArgs: nil},
	}
	if runtime.GOOS != "darwin" {
		entries = append(entries, Bundled{ID: "ffplay", Name: "FFplay", RelPath: "ffplay" + exe, VersionArgs: []string{"-version"}})
	}
	return entries
}()
```

- [ ] **Step 5: Run tests to verify pass**

Run: `cd media-server && go test ./deps/bundled/...`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add media-server/deps/bundled/
git commit -m "feat(deps): bundled package — Resolve + manifest

Resolves bundled binaries from <execDir>/bin. Never downloads. ErrUnknown
for missing manifest entries, IsMissing(err) for absent files. Manifest
is a single var enumerating every bundled dep per platform."
```

### Task 2: macOS quarantine removal

**Files:**
- Create: `media-server/deps/bundled/quarantine_darwin.go`
- Create: `media-server/deps/bundled/quarantine_other.go`

- [ ] **Step 1: Implement darwin variant**

Create `media-server/deps/bundled/quarantine_darwin.go`:

```go
//go:build darwin

package bundled

import (
	"os/exec"
	"time"
)

// removeQuarantine strips com.apple.quarantine from path. Best-effort;
// xattr returns an error if the attribute is absent — that's fine.
// We swallow errors and don't log here; verify.go logs the call.
func removeQuarantine(path string) {
	cmd := exec.Command("xattr", "-d", "com.apple.quarantine", path)
	// Bounded — if xattr hangs for any reason, don't block boot.
	done := make(chan struct{})
	_ = cmd.Start()
	go func() { _ = cmd.Wait(); close(done) }()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		_ = cmd.Process.Kill()
	}
}
```

- [ ] **Step 2: Implement no-op variant**

Create `media-server/deps/bundled/quarantine_other.go`:

```go
//go:build !darwin

package bundled

// removeQuarantine is a no-op on non-macOS platforms.
func removeQuarantine(_ string) {}
```

- [ ] **Step 3: Verify both build**

Run: `cd media-server && go build ./deps/bundled/...`
Expected: no output (success).

Run: `cd media-server && GOOS=darwin go build ./deps/bundled/...`
Expected: no output (success).

- [ ] **Step 4: Commit**

```bash
git add media-server/deps/bundled/quarantine_darwin.go media-server/deps/bundled/quarantine_other.go
git commit -m "feat(deps): strip com.apple.quarantine on macOS

Fixes the Gatekeeper-kills-binary crash class on darwin by removing
the quarantine xattr from each bundled binary at boot. Best-effort,
bounded to 2s, swallows errors. No-op on other platforms."
```

### Task 3: VerifyAll boot-time check

**Files:**
- Create: `media-server/deps/bundled/verify.go`
- Test: `media-server/deps/bundled/verify_test.go`

- [ ] **Step 1: Write the failing test**

Create `media-server/deps/bundled/verify_test.go`:

```go
package bundled

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestVerifyAll_ReportsReadyForPresentBinaries(t *testing.T) {
	dir := t.TempDir()
	binDir := filepath.Join(dir, "bin")
	if err := os.MkdirAll(binDir, 0o755); err != nil {
		t.Fatal(err)
	}
	// On non-windows we can fake a binary that prints "1.0"; on Windows
	// we skip the exec probe by using a known-broken VersionArgs=nil entry.
	// To keep this test cross-platform, write a small shell stub only on unix.
	if runtime.GOOS == "windows" {
		t.Skip("verify exec stub not portable to windows in unit test")
	}
	stub := filepath.Join(binDir, "ffmpeg")
	script := "#!/bin/sh\necho fake-version 1.2.3\n"
	if err := os.WriteFile(stub, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	// Stub out manifest to a single entry so the test is independent.
	old := Manifest
	Manifest = []Bundled{{ID: "ffmpeg", Name: "FFmpeg", RelPath: "ffmpeg", VersionArgs: []string{"--noop"}}}
	defer func() { Manifest = old }()

	prevExec := execDirOverride
	execDirOverride = dir
	defer func() { execDirOverride = prevExec }()

	statuses := VerifyAll()
	if len(statuses) != 1 {
		t.Fatalf("want 1 status, got %d", len(statuses))
	}
	if statuses[0].State != "ready" {
		t.Errorf("state=%q error=%q want ready", statuses[0].State, statuses[0].Error)
	}
}

func TestVerifyAll_ReportsMissing(t *testing.T) {
	dir := t.TempDir()
	old := Manifest
	Manifest = []Bundled{{ID: "ffmpeg", Name: "FFmpeg", RelPath: "ffmpeg", VersionArgs: nil}}
	defer func() { Manifest = old }()
	prev := execDirOverride
	execDirOverride = dir
	defer func() { execDirOverride = prev }()

	statuses := VerifyAll()
	if statuses[0].State != "missing" {
		t.Errorf("state=%q, want missing", statuses[0].State)
	}
}
```

- [ ] **Step 2: Run test to verify failure**

Run: `cd media-server && go test ./deps/bundled/...`
Expected: compile error (VerifyAll not defined).

- [ ] **Step 3: Implement verify.go**

Create `media-server/deps/bundled/verify.go`:

```go
package bundled

import (
	"context"
	"os/exec"
	"strings"
	"sync"
	"time"
)

var (
	statusCache   []Status
	statusCacheMu sync.RWMutex
)

// VerifyAll checks every bundled entry: file present, quarantine stripped
// (macOS), version probe succeeds. Results are cached and returned.
// Safe to call multiple times; the cache is replaced on each call.
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

// CachedStatus returns the most recent VerifyAll result. If VerifyAll has not
// been called yet, returns an empty slice.
func CachedStatus() []Status {
	statusCacheMu.RLock()
	defer statusCacheMu.RUnlock()
	out := make([]Status, len(statusCache))
	copy(out, statusCache)
	return out
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
		// Shared library; presence is enough.
		s.State = "ready"
		return s
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, path, b.VersionArgs...)
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
```

- [ ] **Step 4: Run tests**

Run: `cd media-server && go test ./deps/bundled/...`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add media-server/deps/bundled/verify.go media-server/deps/bundled/verify_test.go
git commit -m "feat(deps): VerifyAll boot-time bundled check

Reports ready/missing/broken per entry. Runs the version probe with a
5s timeout. Caches the result so the status API doesn't re-execute
binaries on every poll."
```

---

## Phase 2: Foundation — `optional` package

### Task 4: Optional package — types, Detect, manifest, hints

**Files:**
- Create: `media-server/deps/optional/optional.go`
- Create: `media-server/deps/optional/manifest.go`
- Create: `media-server/deps/optional/hints.go`
- Test: `media-server/deps/optional/optional_test.go`

- [ ] **Step 1: Write the failing test**

Create `media-server/deps/optional/optional_test.go`:

```go
package optional

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// writeFakeBin writes a stub script that prints a version string and exits 0.
func writeFakeBin(t *testing.T, dir, name, versionOutput string) string {
	t.Helper()
	if runtime.GOOS == "windows" {
		path := filepath.Join(dir, name+".bat")
		content := "@echo off\r\necho " + versionOutput + "\r\n"
		if err := os.WriteFile(path, []byte(content), 0o755); err != nil {
			t.Fatal(err)
		}
		return path
	}
	path := filepath.Join(dir, name)
	content := "#!/bin/sh\necho " + versionOutput + "\n"
	if err := os.WriteFile(path, []byte(content), 0o755); err != nil {
		t.Fatal(err)
	}
	return path
}

func withPath(t *testing.T, dir string) {
	t.Helper()
	old := os.Getenv("PATH")
	sep := ":"
	if runtime.GOOS == "windows" {
		sep = ";"
	}
	if err := os.Setenv("PATH", dir+sep+old); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Setenv("PATH", old) })
}

func TestDetect_FindsBinaryOnPath(t *testing.T) {
	dir := t.TempDir()
	writeFakeBin(t, dir, "yt-dlp", "2026.05.01")
	withPath(t, dir)

	s, err := Detect("yt-dlp")
	if err != nil {
		t.Fatal(err)
	}
	if !s.Installed {
		t.Error("expected Installed=true")
	}
	if !strings.Contains(s.Version, "2026.05.01") {
		t.Errorf("Version=%q want 2026.05.01", s.Version)
	}
	if s.Hint.DocsURL == "" {
		t.Error("expected DocsURL populated even when installed")
	}
}

func TestDetect_ReturnsNotInstalledWithHint(t *testing.T) {
	// Empty PATH so binary cannot be found.
	t.Setenv("PATH", t.TempDir())

	s, err := Detect("yt-dlp")
	if err != nil {
		t.Fatal(err)
	}
	if s.Installed {
		t.Error("expected Installed=false")
	}
	if len(s.Hint.Commands) == 0 {
		t.Error("expected install commands")
	}
}

func TestDetect_UnknownIDIsError(t *testing.T) {
	_, err := Detect("not-a-tool")
	if err == nil {
		t.Fatal("expected error for unknown id")
	}
}
```

- [ ] **Step 2: Run test to verify failure**

Run: `cd media-server && go test ./deps/optional/...`
Expected: compile error.

- [ ] **Step 3: Implement optional.go**

Create `media-server/deps/optional/optional.go`:

```go
// Package optional detects user-installed CLI tools on PATH and provides
// per-OS install hints. It NEVER installs anything.
package optional

import (
	"context"
	"errors"
	"fmt"
	"os/exec"
	"strings"
	"time"
)

var ErrUnknown = errors.New("optional: unknown tool id")

// Optional describes one detectable tool.
type Optional struct {
	ID          string
	Name        string
	Binary      string   // executable name as found on PATH
	VersionArgs []string // probe args
	Description string
	DocsURL     string
}

// Status is the runtime detection result.
type Status struct {
	ID        string
	Name      string
	Installed bool
	Path      string
	Version   string
	Hint      InstallHint
}

// InstallHint is shown in the UI when the tool is missing (or always, as
// reference). Commands are per-OS copy-paste install lines.
type InstallHint struct {
	Description string
	Commands    []OSCmd
	DocsURL     string
}

// OSCmd is a single install command for a single OS.
type OSCmd struct {
	OS      string // "darwin" | "windows" | "linux"
	Label   string // e.g. "Homebrew", "winget", "pipx"
	Command string // copy-paste line
}

// IDs returns all known optional tool ids.
func IDs() []string {
	out := make([]string, 0, len(Manifest))
	for _, o := range Manifest {
		out = append(out, o.ID)
	}
	return out
}

// Detect runs PATH lookup + version probe for id.
func Detect(id string) (Status, error) {
	entry, ok := lookup(id)
	if !ok {
		return Status{}, fmt.Errorf("%w: %q", ErrUnknown, id)
	}
	s := Status{ID: entry.ID, Name: entry.Name, Hint: hintFor(entry)}
	path, err := exec.LookPath(entry.Binary)
	if err != nil {
		return s, nil
	}
	s.Installed = true
	s.Path = path
	if len(entry.VersionArgs) > 0 {
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		out, perr := exec.CommandContext(ctx, path, entry.VersionArgs...).CombinedOutput()
		if perr == nil {
			s.Version = firstLine(string(out))
		}
	}
	return s, nil
}

func lookup(id string) (Optional, bool) {
	for _, o := range Manifest {
		if o.ID == id {
			return o, true
		}
	}
	return Optional{}, false
}

func firstLine(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return strings.TrimSpace(s[:i])
	}
	return strings.TrimSpace(s)
}
```

- [ ] **Step 4: Implement manifest.go**

Create `media-server/deps/optional/manifest.go`:

```go
package optional

var Manifest = []Optional{
	{
		ID:          "yt-dlp",
		Name:        "yt-dlp",
		Binary:      "yt-dlp",
		VersionArgs: []string{"--version"},
		Description: "Video downloader. Required to import from YouTube, Twitch, and many other sites.",
		DocsURL:     "https://github.com/yt-dlp/yt-dlp#installation",
	},
	{
		ID:          "gallery-dl",
		Name:        "gallery-dl",
		Binary:      "gallery-dl",
		VersionArgs: []string{"--version"},
		Description: "Image gallery downloader for sites like DeviantArt, Pixiv, Reddit.",
		DocsURL:     "https://github.com/mikf/gallery-dl#installation",
	},
	{
		ID:          "ollama",
		Name:        "Ollama",
		Binary:      "ollama",
		VersionArgs: []string{"--version"},
		Description: "Local large language model runtime. Enables AI captioning and chat features.",
		DocsURL:     "https://ollama.com/download",
	},
}
```

- [ ] **Step 5: Implement hints.go**

Create `media-server/deps/optional/hints.go`:

```go
package optional

func hintFor(o Optional) InstallHint {
	h := InstallHint{Description: o.Description, DocsURL: o.DocsURL}
	switch o.ID {
	case "yt-dlp":
		h.Commands = []OSCmd{
			{OS: "darwin", Label: "Homebrew", Command: "brew install yt-dlp"},
			{OS: "windows", Label: "winget", Command: "winget install yt-dlp"},
			{OS: "linux", Label: "pipx", Command: "pipx install yt-dlp"},
		}
	case "gallery-dl":
		h.Commands = []OSCmd{
			{OS: "darwin", Label: "Homebrew", Command: "brew install gallery-dl"},
			{OS: "windows", Label: "pip", Command: "pip install --user gallery-dl"},
			{OS: "linux", Label: "pipx", Command: "pipx install gallery-dl"},
		}
	case "ollama":
		h.Commands = []OSCmd{
			{OS: "darwin", Label: "Homebrew", Command: "brew install ollama"},
			{OS: "windows", Label: "Download", Command: "Download installer from https://ollama.com/download/windows"},
			{OS: "linux", Label: "Shell installer", Command: "curl -fsSL https://ollama.com/install.sh | sh"},
		}
	}
	return h
}
```

- [ ] **Step 6: Run tests**

Run: `cd media-server && go test ./deps/optional/...`
Expected: PASS (3 tests).

- [ ] **Step 7: Commit**

```bash
git add media-server/deps/optional/
git commit -m "feat(deps): optional package — PATH detect + install hints

yt-dlp, gallery-dl, ollama are looked up on PATH; missing tools return
per-OS install commands. Never auto-installs."
```

---

## Phase 3: Foundation — `models` package

### Task 5: Models manifest (schema + embedded JSON parse)

**Files:**
- Create: `media-server/deps/models/manifest.go`
- Create: `media-server/deps/models/manifest.json`
- Test: `media-server/deps/models/manifest_test.go`

- [ ] **Step 1: Write failing test**

Create `media-server/deps/models/manifest_test.go`:

```go
package models

import (
	"net/url"
	"strings"
	"testing"
)

func TestManifest_ParsesAndHasModels(t *testing.T) {
	if len(Manifest) == 0 {
		t.Fatal("expected at least one model in manifest")
	}
	for _, m := range Manifest {
		if m.ID == "" {
			t.Error("empty model id")
		}
		if m.Version == "" {
			t.Errorf("model %s: empty version", m.ID)
		}
		if len(m.Files) == 0 {
			t.Errorf("model %s: no files", m.ID)
		}
		for _, f := range m.Files {
			if _, err := url.Parse(f.URL); err != nil || !strings.HasPrefix(f.URL, "http") {
				t.Errorf("model %s file %s: bad url %q", m.ID, f.RelPath, f.URL)
			}
			if f.RelPath == "" {
				t.Errorf("model %s: empty rel_path", m.ID)
			}
			if f.SHA256 == "" {
				t.Errorf("model %s file %s: empty sha256", m.ID, f.RelPath)
			}
		}
	}
}

func TestLookup(t *testing.T) {
	if _, ok := Lookup("nope"); ok {
		t.Error("expected !ok for unknown id")
	}
	m, ok := Lookup(Manifest[0].ID)
	if !ok || m.ID != Manifest[0].ID {
		t.Errorf("Lookup roundtrip failed")
	}
}
```

- [ ] **Step 2: Run test (compile fail expected)**

Run: `cd media-server && go test ./deps/models/...`
Expected: compile error.

- [ ] **Step 3: Implement manifest.go**

Create `media-server/deps/models/manifest.go`:

```go
// Package models implements the on-demand AI model downloader.
package models

import (
	_ "embed"
	"encoding/json"
	"fmt"
)

//go:embed manifest.json
var manifestJSON []byte

// Model is one downloadable model bundle.
type Model struct {
	ID          string  `json:"id"`
	Name        string  `json:"name"`
	Description string  `json:"description"`
	Version     string  `json:"version"`
	Consumers   []string `json:"consumers"`
	SizeBytes   int64   `json:"size_bytes"`
	Files       []File  `json:"files"`
}

// File is one downloadable file within a model.
type File struct {
	URL     string `json:"url"`
	RelPath string `json:"rel_path"`
	SHA256  string `json:"sha256"`
}

// Manifest is the parsed model registry.
var Manifest []Model

func init() {
	var doc struct {
		SchemaVersion int     `json:"schema_version"`
		Models        []Model `json:"models"`
	}
	if err := json.Unmarshal(manifestJSON, &doc); err != nil {
		panic(fmt.Sprintf("deps/models: manifest.json invalid: %v", err))
	}
	if doc.SchemaVersion != 1 {
		panic(fmt.Sprintf("deps/models: unsupported schema_version %d", doc.SchemaVersion))
	}
	Manifest = doc.Models
}

// Lookup returns the model with the given id.
func Lookup(id string) (Model, bool) {
	for _, m := range Manifest {
		if m.ID == id {
			return m, true
		}
	}
	return Model{}, false
}

// IDs returns every model id in manifest order.
func IDs() []string {
	out := make([]string, 0, len(Manifest))
	for _, m := range Manifest {
		out = append(out, m.ID)
	}
	return out
}
```

- [ ] **Step 4: Create initial manifest.json (one seed entry)**

Create `media-server/deps/models/manifest.json` with the WD tagger entry that exists in current code. SHA-256s will be populated by Task 7. For now use a clearly-marked placeholder so tests can require non-empty:

```json
{
  "schema_version": 1,
  "models": [
    {
      "id": "wd-eva02-large-tagger-v3",
      "name": "WD EVA02 Large Tagger v3",
      "description": "Image autotagging classifier (general + character tags).",
      "version": "1.0.0",
      "consumers": ["autotag"],
      "size_bytes": 1257385984,
      "files": [
        {
          "url": "https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3/resolve/main/model.onnx",
          "rel_path": "model.onnx",
          "sha256": "UNVERIFIED"
        },
        {
          "url": "https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3/resolve/main/selected_tags.csv",
          "rel_path": "selected_tags.csv",
          "sha256": "UNVERIFIED"
        },
        {
          "url": "https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3/resolve/main/config.json",
          "rel_path": "config.json",
          "sha256": "UNVERIFIED"
        }
      ]
    }
  ]
}
```

Note: the value `UNVERIFIED` satisfies the "non-empty" check. Real SHA-256s are computed and inlined during Task 7.

- [ ] **Step 5: Run tests**

Run: `cd media-server && go test ./deps/models/...`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add media-server/deps/models/manifest.go media-server/deps/models/manifest.json media-server/deps/models/manifest_test.go
git commit -m "feat(deps): models manifest schema + embedded JSON

Seeds the WD-EVA02 tagger entry. SHA-256 values are placeholders
(\"UNVERIFIED\") until populated by the manifest-hashing task; the
schema check requires them non-empty so they will be replaced before
the manifest goes live."
```

### Task 6: Models store (paths + atomic install primitives)

**Files:**
- Create: `media-server/deps/models/store.go`
- Test: `media-server/deps/models/store_test.go`

- [ ] **Step 1: Write the failing test**

Create `media-server/deps/models/store_test.go`:

```go
package models

import (
	"os"
	"path/filepath"
	"testing"
)

func TestModelDir_ComposesUnderDataDir(t *testing.T) {
	dir := t.TempDir()
	SetDataDirForTest(dir)
	t.Cleanup(func() { SetDataDirForTest("") })

	got := ModelDir("wd-eva02-large-tagger-v3")
	want := filepath.Join(dir, "models", "wd-eva02-large-tagger-v3")
	if got != want {
		t.Errorf("ModelDir = %q want %q", got, want)
	}
}

func TestPath_ReturnsErrNotInstalledWhenAbsent(t *testing.T) {
	dir := t.TempDir()
	SetDataDirForTest(dir)
	t.Cleanup(func() { SetDataDirForTest("") })

	_, err := Path("wd-eva02-large-tagger-v3", "model.onnx")
	if err == nil {
		t.Fatal("expected error")
	}
	if !IsNotInstalled(err) {
		t.Errorf("expected IsNotInstalled(err)=true, got %v", err)
	}
}

func TestPath_ReturnsAbsolutePathWhenPresent(t *testing.T) {
	dir := t.TempDir()
	SetDataDirForTest(dir)
	t.Cleanup(func() { SetDataDirForTest("") })

	mdir := ModelDir("wd-eva02-large-tagger-v3")
	if err := os.MkdirAll(mdir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(mdir, "model.onnx"), []byte("fake"), 0o644); err != nil {
		t.Fatal(err)
	}

	got, err := Path("wd-eva02-large-tagger-v3", "model.onnx")
	if err != nil {
		t.Fatal(err)
	}
	want := filepath.Join(mdir, "model.onnx")
	if got != want {
		t.Errorf("Path = %q want %q", got, want)
	}
}

func TestAtomicWrite_RenamesAfterClose(t *testing.T) {
	dir := t.TempDir()
	final := filepath.Join(dir, "out.bin")

	w, err := NewAtomicWriter(final)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := w.Write([]byte("hello")); err != nil {
		t.Fatal(err)
	}
	// Before Commit, the final file should not exist.
	if _, err := os.Stat(final); !os.IsNotExist(err) {
		t.Errorf("final exists before commit: err=%v", err)
	}
	if err := w.Commit(); err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(final)
	if err != nil || string(b) != "hello" {
		t.Fatalf("post-commit read: %q err=%v", b, err)
	}
}

func TestAtomicWrite_AbortCleansPartial(t *testing.T) {
	dir := t.TempDir()
	final := filepath.Join(dir, "out.bin")
	w, err := NewAtomicWriter(final)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := w.Write([]byte("x")); err != nil {
		t.Fatal(err)
	}
	if err := w.Abort(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(final + ".partial"); !os.IsNotExist(err) {
		t.Errorf(".partial still exists after Abort")
	}
}
```

- [ ] **Step 2: Run tests (compile fail)**

Run: `cd media-server && go test ./deps/models/...`
Expected: compile error.

- [ ] **Step 3: Implement store.go**

Create `media-server/deps/models/store.go`:

```go
package models

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"

	"github.com/stevecastle/shrike/platform"
)

var (
	errNotInstalled = errors.New("models: not installed")

	dataDirOverride string
	dataDirMu       sync.RWMutex
)

// SetDataDirForTest overrides the user data dir. Pass "" to clear.
func SetDataDirForTest(dir string) {
	dataDirMu.Lock()
	dataDirOverride = dir
	dataDirMu.Unlock()
}

func dataDir() string {
	dataDirMu.RLock()
	defer dataDirMu.RUnlock()
	if dataDirOverride != "" {
		return dataDirOverride
	}
	return platform.GetDataDir()
}

// IsNotInstalled reports whether err indicates a missing model file.
func IsNotInstalled(err error) bool { return errors.Is(err, errNotInstalled) }

// ModelDir is the directory that holds one model's files.
func ModelDir(id string) string { return filepath.Join(dataDir(), "models", id) }

// Path returns the absolute path to relPath inside the named model. Returns
// IsNotInstalled error if the file is not present.
func Path(id, relPath string) (string, error) {
	full := filepath.Join(ModelDir(id), relPath)
	info, err := os.Stat(full)
	if err != nil || info.IsDir() {
		return "", fmt.Errorf("%w: %s/%s", errNotInstalled, id, relPath)
	}
	return full, nil
}

// AtomicWriter streams bytes to <final>.partial, then renames to <final> on
// Commit, or deletes the partial on Abort.
type AtomicWriter struct {
	final   string
	partial string
	f       *os.File
}

// NewAtomicWriter opens (or truncates and re-creates) <final>.partial for write.
func NewAtomicWriter(final string) (*AtomicWriter, error) {
	if err := os.MkdirAll(filepath.Dir(final), 0o755); err != nil {
		return nil, err
	}
	partial := final + ".partial"
	// Truncate on open: this is the "write from scratch" path. Resume callers
	// should not use NewAtomicWriter; see OpenAtomicResume below.
	f, err := os.OpenFile(partial, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		return nil, err
	}
	return &AtomicWriter{final: final, partial: partial, f: f}, nil
}

// OpenAtomicResume opens <final>.partial for append, returning the current size
// so the caller can issue a Range request. The returned AtomicWriter behaves
// like NewAtomicWriter on Commit/Abort.
func OpenAtomicResume(final string) (*AtomicWriter, int64, error) {
	if err := os.MkdirAll(filepath.Dir(final), 0o755); err != nil {
		return nil, 0, err
	}
	partial := final + ".partial"
	f, err := os.OpenFile(partial, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return nil, 0, err
	}
	info, err := f.Stat()
	if err != nil {
		_ = f.Close()
		return nil, 0, err
	}
	return &AtomicWriter{final: final, partial: partial, f: f}, info.Size(), nil
}

func (w *AtomicWriter) Write(p []byte) (int, error) { return w.f.Write(p) }

// Commit closes the temp file and renames it to its final path.
func (w *AtomicWriter) Commit() error {
	if err := w.f.Sync(); err != nil {
		_ = w.f.Close()
		_ = os.Remove(w.partial)
		return err
	}
	if err := w.f.Close(); err != nil {
		_ = os.Remove(w.partial)
		return err
	}
	return os.Rename(w.partial, w.final)
}

// Abort closes and removes the partial file.
func (w *AtomicWriter) Abort() error {
	_ = w.f.Close()
	if err := os.Remove(w.partial); err != nil && !os.IsNotExist(err) {
		return err
	}
	return nil
}

// Ensure io.Writer interface.
var _ io.Writer = (*AtomicWriter)(nil)
```

- [ ] **Step 4: Run tests**

Run: `cd media-server && go test ./deps/models/...`
Expected: PASS (manifest tests + 5 store tests).

- [ ] **Step 5: Commit**

```bash
git add media-server/deps/models/store.go media-server/deps/models/store_test.go
git commit -m "feat(deps): models store — paths + atomic writer

ModelDir/Path return absolute paths under <dataDir>/models/<id>/.
AtomicWriter streams to .partial and renames on Commit; Abort deletes
the partial. OpenAtomicResume opens append mode and returns existing
size so the downloader can issue a Range request."
```

### Task 7: Models downloader (HTTP + sha256 + resume + retry)

**Files:**
- Create: `media-server/deps/models/downloader.go`
- Test: `media-server/deps/models/downloader_test.go`

- [ ] **Step 1: Write the failing test**

Create `media-server/deps/models/downloader_test.go`:

```go
package models

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
)

func sha256Hex(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

func TestDownloadFile_HappyPath(t *testing.T) {
	body := []byte("hello world")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write(body)
	}))
	defer srv.Close()

	dir := t.TempDir()
	dst := filepath.Join(dir, "file.bin")
	got, err := downloadFile(context.Background(), srv.URL, dst, sha256Hex(body), nil)
	if err != nil {
		t.Fatal(err)
	}
	if got != int64(len(body)) {
		t.Errorf("bytes=%d want %d", got, len(body))
	}
	b, _ := os.ReadFile(dst)
	if string(b) != string(body) {
		t.Errorf("content=%q want %q", b, body)
	}
	if _, err := os.Stat(dst + ".partial"); !os.IsNotExist(err) {
		t.Errorf("partial still exists")
	}
}

func TestDownloadFile_ResumesFromPartial(t *testing.T) {
	body := []byte("0123456789ABCDEF")
	var rangeRequests int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Range") != "" {
			atomic.AddInt32(&rangeRequests, 1)
			// "bytes=8-" → serve last half
			w.Header().Set("Content-Range", "bytes 8-15/16")
			w.WriteHeader(http.StatusPartialContent)
			_, _ = w.Write(body[8:])
			return
		}
		_, _ = w.Write(body)
	}))
	defer srv.Close()

	dir := t.TempDir()
	dst := filepath.Join(dir, "file.bin")
	if err := os.WriteFile(dst+".partial", body[:8], 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := downloadFile(context.Background(), srv.URL, dst, sha256Hex(body), nil); err != nil {
		t.Fatal(err)
	}
	if atomic.LoadInt32(&rangeRequests) != 1 {
		t.Errorf("expected 1 Range request, got %d", rangeRequests)
	}
	b, _ := os.ReadFile(dst)
	if string(b) != string(body) {
		t.Errorf("got %q want %q", b, body)
	}
}

func TestDownloadFile_ChecksumMismatchDeletesPartial(t *testing.T) {
	body := []byte("payload")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write(body)
	}))
	defer srv.Close()

	dir := t.TempDir()
	dst := filepath.Join(dir, "file.bin")
	_, err := downloadFile(context.Background(), srv.URL, dst, sha256Hex([]byte("wrong-payload")), nil)
	if err == nil {
		t.Fatal("expected checksum error")
	}
	if !strings.Contains(err.Error(), "checksum") {
		t.Errorf("err = %v, want checksum error", err)
	}
	for _, p := range []string{dst, dst + ".partial"} {
		if _, err := os.Stat(p); !os.IsNotExist(err) {
			t.Errorf("%s exists after mismatch", p)
		}
	}
}

func TestDownloadFile_CancelDeletesPartial(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", "1000000")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(make([]byte, 64*1024))
		// Hang for the rest until client cancels.
		<-r.Context().Done()
	}))
	defer srv.Close()

	dir := t.TempDir()
	dst := filepath.Join(dir, "file.bin")
	ctx, cancel := context.WithCancel(context.Background())
	errCh := make(chan error, 1)
	go func() {
		_, err := downloadFile(ctx, srv.URL, dst, sha256Hex([]byte("noop")), nil)
		errCh <- err
	}()
	cancel()
	if err := <-errCh; err == nil {
		t.Fatal("expected cancellation error")
	}
	if _, err := os.Stat(dst + ".partial"); !os.IsNotExist(err) {
		t.Errorf(".partial not cleaned up after cancel")
	}
}

func TestInstallModel_FailsForUnknownID(t *testing.T) {
	dir := t.TempDir()
	SetDataDirForTest(dir)
	t.Cleanup(func() { SetDataDirForTest("") })
	if err := InstallModel(context.Background(), "no-such-model", nil); err == nil {
		t.Fatal("expected error for unknown id")
	}
}

// Round-trip: install a fake one-file model and read the .meta.json back.
func TestInstallModel_WritesMetaAndFiles(t *testing.T) {
	body := []byte("fake-onnx-bytes")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write(body)
	}))
	defer srv.Close()

	dir := t.TempDir()
	SetDataDirForTest(dir)
	t.Cleanup(func() { SetDataDirForTest("") })

	// Substitute Manifest for this test.
	oldManifest := Manifest
	Manifest = []Model{
		{
			ID: "fake", Name: "Fake", Version: "1.0", SizeBytes: int64(len(body)),
			Files: []File{{URL: srv.URL, RelPath: "model.bin", SHA256: sha256Hex(body)}},
		},
	}
	defer func() { Manifest = oldManifest }()

	if err := InstallModel(context.Background(), "fake", nil); err != nil {
		t.Fatal(err)
	}
	meta, err := os.ReadFile(filepath.Join(ModelDir("fake"), ".meta.json"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(meta), `"version":"1.0"`) {
		t.Errorf("meta missing version: %s", meta)
	}
	got, _ := io.ReadAll(strings.NewReader(string(body)))
	b, _ := os.ReadFile(filepath.Join(ModelDir("fake"), "model.bin"))
	if string(b) != string(got) {
		t.Errorf("model.bin content mismatch")
	}
}
```

- [ ] **Step 2: Run tests (compile fail expected)**

Run: `cd media-server && go test ./deps/models/...`
Expected: compile error.

- [ ] **Step 3: Implement downloader.go**

Create `media-server/deps/models/downloader.go`:

```go
package models

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"time"
)

// ProgressFn receives per-file byte counts. Safe to be nil.
type ProgressFn func(file string, bytesDone, bytesTotal int64)

var ErrChecksumMismatch = errors.New("models: sha256 checksum mismatch")

// Install lock per model id so two concurrent callers for the same model
// don't tread on each other; different models may install in parallel.
var (
	installLocks   = map[string]*sync.Mutex{}
	installLocksMu sync.Mutex
)

func lockForModel(id string) *sync.Mutex {
	installLocksMu.Lock()
	defer installLocksMu.Unlock()
	if l, ok := installLocks[id]; ok {
		return l
	}
	l := &sync.Mutex{}
	installLocks[id] = l
	return l
}

// InstallModel downloads every file of the named model, verifies its
// SHA-256, atomically installs it, and writes <ModelDir>/.meta.json on
// success.
func InstallModel(ctx context.Context, id string, progress ProgressFn) error {
	m, ok := Lookup(id)
	if !ok {
		return fmt.Errorf("models: unknown id %q", id)
	}
	l := lockForModel(id)
	l.Lock()
	defer l.Unlock()

	for _, f := range m.Files {
		dst := filepath.Join(ModelDir(id), f.RelPath)
		if _, err := downloadFileWithRetry(ctx, f.URL, dst, f.SHA256, progress); err != nil {
			return fmt.Errorf("file %s: %w", f.RelPath, err)
		}
	}
	return writeMeta(id, m)
}

// downloadFileWithRetry wraps downloadFile with exponential backoff:
// 1s, 4s, 16s. Network errors retry; HTTP 4xx (except 416) and checksum
// mismatch don't.
func downloadFileWithRetry(ctx context.Context, url, dst, want string, progress ProgressFn) (int64, error) {
	delays := []time.Duration{0, time.Second, 4 * time.Second, 16 * time.Second}
	var lastErr error
	for i, d := range delays {
		if d > 0 {
			select {
			case <-time.After(d):
			case <-ctx.Done():
				return 0, ctx.Err()
			}
		}
		n, err := downloadFile(ctx, url, dst, want, progress)
		if err == nil {
			return n, nil
		}
		lastErr = err
		if errors.Is(err, ErrChecksumMismatch) || isPermanentHTTPError(err) || errors.Is(err, context.Canceled) {
			return 0, err
		}
		_ = i
	}
	return 0, lastErr
}

type httpStatusError struct {
	Status int
	URL    string
}

func (e *httpStatusError) Error() string {
	return fmt.Sprintf("HTTP %d from %s", e.Status, e.URL)
}

func isPermanentHTTPError(err error) bool {
	var hs *httpStatusError
	if errors.As(err, &hs) {
		return hs.Status >= 400 && hs.Status < 500 && hs.Status != http.StatusRequestedRangeNotSatisfiable
	}
	return false
}

// downloadFile streams url -> dst with optional resume from <dst>.partial,
// verifying the running SHA-256 against want before renaming.
func downloadFile(ctx context.Context, url, dst, want string, progress ProgressFn) (int64, error) {
	// Try to resume.
	w, existing, err := OpenAtomicResume(dst)
	if err != nil {
		return 0, err
	}
	// If existing > 0, hash the existing bytes first so the running hash matches the full stream.
	h := sha256.New()
	if existing > 0 {
		if err := hashExisting(w.partial, h); err != nil {
			_ = w.Abort()
			return 0, err
		}
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		_ = w.Abort()
		return 0, err
	}
	if existing > 0 {
		req.Header.Set("Range", "bytes="+strconv.FormatInt(existing, 10)+"-")
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		_ = w.Abort()
		return 0, err
	}
	defer resp.Body.Close()

	switch resp.StatusCode {
	case http.StatusOK:
		// Server ignored our Range — truncate and restart.
		if existing > 0 {
			_ = w.Abort()
			w, _, err = OpenAtomicResume(dst) // reopen empty
			if err != nil {
				return 0, err
			}
			_ = os.Truncate(w.partial, 0)
			h = sha256.New()
			existing = 0
		}
	case http.StatusPartialContent:
		// Good — append.
	case http.StatusRequestedRangeNotSatisfiable:
		// Existing file size matches or exceeds; verify checksum below.
	default:
		_ = w.Abort()
		return 0, &httpStatusError{Status: resp.StatusCode, URL: url}
	}

	total := existing
	if resp.ContentLength > 0 {
		total += resp.ContentLength
	}
	mw := io.MultiWriter(w, h)
	written, err := copyWithProgress(ctx, mw, resp.Body, existing, total, progress, filepath.Base(dst))
	if err != nil {
		_ = w.Abort()
		return 0, err
	}

	got := hex.EncodeToString(h.Sum(nil))
	if got != want {
		_ = w.Abort()
		return 0, fmt.Errorf("%w: got %s want %s", ErrChecksumMismatch, got, want)
	}
	if err := w.Commit(); err != nil {
		return 0, err
	}
	return existing + written, nil
}

func hashExisting(path string, h io.Writer) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = io.Copy(h, f)
	return err
}

// copyWithProgress copies src→dst in 64KB chunks, calling progress every chunk.
func copyWithProgress(ctx context.Context, dst io.Writer, src io.Reader, start, total int64, p ProgressFn, name string) (int64, error) {
	buf := make([]byte, 64*1024)
	var written int64
	for {
		if err := ctx.Err(); err != nil {
			return written, err
		}
		nr, er := src.Read(buf)
		if nr > 0 {
			nw, ew := dst.Write(buf[:nr])
			written += int64(nw)
			if p != nil {
				p(name, start+written, total)
			}
			if ew != nil {
				return written, ew
			}
		}
		if er == io.EOF {
			return written, nil
		}
		if er != nil {
			return written, er
		}
	}
}

func writeMeta(id string, m Model) error {
	type meta struct {
		Version        string `json:"version"`
		InstalledAt    string `json:"installed_at"`
		SHA256Verified bool   `json:"sha256_verified"`
	}
	doc := meta{Version: m.Version, InstalledAt: time.Now().UTC().Format(time.RFC3339), SHA256Verified: true}
	b, err := json.Marshal(doc)
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(ModelDir(id), ".meta.json"), b, 0o644)
}
```

- [ ] **Step 4: Run tests**

Run: `cd media-server && go test ./deps/models/...`
Expected: PASS. The cancellation test may be slow (a few seconds) — that is fine.

- [ ] **Step 5: Populate real SHA-256s for the seed model**

Run the manifest hasher on the seed model and inline real hashes. This is a one-shot helper, kept in `media-server/scripts/`:

Create `media-server/scripts/hash-manifest.sh`:

```sh
#!/usr/bin/env bash
# Reads media-server/deps/models/manifest.json, downloads each file to a temp
# dir, computes SHA-256, and prints a patched manifest to stdout. Pipe to
# `tee media-server/deps/models/manifest.json` once verified.
set -euo pipefail
in="${1:-media-server/deps/models/manifest.json}"
tmpdir="$(mktemp -d)"
trap "rm -rf $tmpdir" EXIT
jq -c '.models[] | {id, files: .files[]}' "$in" | while read -r line; do
  id=$(echo "$line" | jq -r '.id')
  url=$(echo "$line" | jq -r '.files.url')
  rel=$(echo "$line" | jq -r '.files.rel_path')
  echo "fetching $id/$rel ..." 1>&2
  out="$tmpdir/$(echo "$url" | shasum | cut -c1-12)"
  curl -fsSL "$url" -o "$out"
  sum=$(shasum -a 256 "$out" | awk '{print $1}')
  echo "$id $rel $sum"
done
```

Run:
```bash
chmod +x media-server/scripts/hash-manifest.sh
media-server/scripts/hash-manifest.sh
```

Edit `media-server/deps/models/manifest.json` and replace every `"UNVERIFIED"` with the matching SHA-256 from the script's output.

- [ ] **Step 6: Run tests again to confirm manifest still parses**

Run: `cd media-server && go test ./deps/models/...`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add media-server/deps/models/downloader.go media-server/deps/models/downloader_test.go media-server/deps/models/manifest.json media-server/scripts/hash-manifest.sh
git commit -m "feat(deps): models downloader — resume, sha256 verify, retry

Streams each model file to .partial, computes running SHA-256, retries
transient errors with backoff (1s/4s/16s), bails on checksum mismatch
or HTTP 4xx (except 416). Range-resumes mid-download. Cancellation
deletes the partial. Real SHA-256s populated in manifest.json from
upstream HuggingFace files."
```

### Task 8: Models progress tracker + SSE channel

**Files:**
- Create: `media-server/deps/models/progress.go`

(No dedicated unit test — exercised end-to-end via the handler tests in Task 11. The tracker is thin glue.)

- [ ] **Step 1: Implement progress.go**

Create `media-server/deps/models/progress.go`:

```go
package models

import (
	"context"
	"sync"
	"time"
)

// InstallState enumerates the lifecycle of a model install.
type InstallState string

const (
	StateQueued      InstallState = "queued"
	StateDownloading InstallState = "downloading"
	StateVerifying   InstallState = "verifying"
	StateInstalled   InstallState = "installed"
	StateFailed      InstallState = "failed"
	StateCancelled   InstallState = "cancelled"
)

// Install is the current snapshot of one install attempt.
type Install struct {
	ID          string       `json:"id"`
	State       InstallState `json:"state"`
	CurrentFile string       `json:"current_file,omitempty"`
	BytesDone   int64        `json:"bytes_done"`
	BytesTotal  int64        `json:"bytes_total"`
	Error       string       `json:"error,omitempty"`
	StartedAt   time.Time    `json:"started_at"`
	UpdatedAt   time.Time    `json:"updated_at"`
}

type tracker struct {
	mu      sync.RWMutex
	rows    map[string]*Install
	cancels map[string]context.CancelFunc
	subs    map[chan Install]struct{}
}

var Tracker = &tracker{
	rows:    map[string]*Install{},
	cancels: map[string]context.CancelFunc{},
	subs:    map[chan Install]struct{}{},
}

// StartInstall launches an install for id in a goroutine. If one is already
// active for id, returns the existing one without starting another.
func (t *tracker) StartInstall(id string) (*Install, error) {
	t.mu.Lock()
	if row, ok := t.rows[id]; ok && (row.State == StateDownloading || row.State == StateQueued || row.State == StateVerifying) {
		t.mu.Unlock()
		return row, nil
	}
	row := &Install{ID: id, State: StateQueued, StartedAt: time.Now().UTC(), UpdatedAt: time.Now().UTC()}
	t.rows[id] = row
	ctx, cancel := context.WithCancel(context.Background())
	t.cancels[id] = cancel
	t.mu.Unlock()

	t.broadcast(*row)
	go t.run(ctx, id)
	return row, nil
}

// Cancel cancels the in-flight install for id if any.
func (t *tracker) Cancel(id string) bool {
	t.mu.Lock()
	cancel, ok := t.cancels[id]
	t.mu.Unlock()
	if !ok {
		return false
	}
	cancel()
	return true
}

// Snapshot returns the current Install for id, or false.
func (t *tracker) Snapshot(id string) (Install, bool) {
	t.mu.RLock()
	defer t.mu.RUnlock()
	if row, ok := t.rows[id]; ok {
		return *row, true
	}
	return Install{}, false
}

// All returns a copy of every tracked Install.
func (t *tracker) All() []Install {
	t.mu.RLock()
	defer t.mu.RUnlock()
	out := make([]Install, 0, len(t.rows))
	for _, r := range t.rows {
		out = append(out, *r)
	}
	return out
}

// Subscribe returns a channel that receives every state transition. The
// returned cleanup func must be called to remove the subscription.
func (t *tracker) Subscribe() (<-chan Install, func()) {
	ch := make(chan Install, 16)
	t.mu.Lock()
	t.subs[ch] = struct{}{}
	t.mu.Unlock()
	return ch, func() {
		t.mu.Lock()
		delete(t.subs, ch)
		t.mu.Unlock()
		close(ch)
	}
}

func (t *tracker) broadcast(snap Install) {
	t.mu.RLock()
	defer t.mu.RUnlock()
	for ch := range t.subs {
		// Drop messages on a slow consumer rather than block the producer.
		select {
		case ch <- snap:
		default:
		}
	}
}

func (t *tracker) update(id string, f func(*Install)) {
	t.mu.Lock()
	row, ok := t.rows[id]
	if !ok {
		t.mu.Unlock()
		return
	}
	f(row)
	row.UpdatedAt = time.Now().UTC()
	snap := *row
	t.mu.Unlock()
	t.broadcast(snap)
}

func (t *tracker) run(ctx context.Context, id string) {
	t.update(id, func(r *Install) { r.State = StateDownloading })
	progressFn := func(file string, done, total int64) {
		t.update(id, func(r *Install) {
			r.CurrentFile = file
			r.BytesDone = done
			r.BytesTotal = total
		})
	}
	err := InstallModel(ctx, id, progressFn)
	t.mu.Lock()
	delete(t.cancels, id)
	t.mu.Unlock()
	t.update(id, func(r *Install) {
		switch {
		case err == nil:
			r.State = StateInstalled
			r.Error = ""
		case ctx.Err() != nil:
			r.State = StateCancelled
		default:
			r.State = StateFailed
			r.Error = err.Error()
		}
	})
}
```

- [ ] **Step 2: Verify it builds**

Run: `cd media-server && go build ./deps/models/...`
Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add media-server/deps/models/progress.go
git commit -m "feat(deps): models progress tracker + subscribers

In-memory tracker keyed by model id. StartInstall is idempotent
(returns the existing row if one is in flight). Cancel cancels the
context. Subscribers receive copies on a buffered channel; slow
consumers drop messages."
```

### Task 9: Models state (rebuild from disk, persisted cache)

**Files:**
- Create: `media-server/deps/models/state.go`
- Test: `media-server/deps/models/state_test.go`

- [ ] **Step 1: Write failing test**

Create `media-server/deps/models/state_test.go`:

```go
package models

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestRebuildState_DerivesFromFilesystem(t *testing.T) {
	dir := t.TempDir()
	SetDataDirForTest(dir)
	t.Cleanup(func() { SetDataDirForTest("") })

	oldMan := Manifest
	Manifest = []Model{
		{ID: "m1", Version: "1.0", Files: []File{{RelPath: "a", URL: "x", SHA256: "y"}}},
		{ID: "m2", Version: "2.0", Files: []File{{RelPath: "a", URL: "x", SHA256: "y"}}},
	}
	defer func() { Manifest = oldMan }()

	// Install m1 fully, leave m2 missing.
	if err := os.MkdirAll(ModelDir("m1"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(ModelDir("m1"), "a"), []byte("data"), 0o644); err != nil {
		t.Fatal(err)
	}
	meta := []byte(`{"version":"1.0","installed_at":"2026-05-25T00:00:00Z","sha256_verified":true}`)
	if err := os.WriteFile(filepath.Join(ModelDir("m1"), ".meta.json"), meta, 0o644); err != nil {
		t.Fatal(err)
	}

	out := RebuildState()
	if out["m1"] != StatusInstalled {
		t.Errorf("m1 status=%q want installed", out["m1"])
	}
	if out["m2"] != StatusMissing {
		t.Errorf("m2 status=%q want missing", out["m2"])
	}
}

func TestRebuildState_PartialInstallIsMissing(t *testing.T) {
	dir := t.TempDir()
	SetDataDirForTest(dir)
	t.Cleanup(func() { SetDataDirForTest("") })

	oldMan := Manifest
	Manifest = []Model{{ID: "m1", Version: "1.0", Files: []File{
		{RelPath: "a"}, {RelPath: "b"},
	}}}
	defer func() { Manifest = oldMan }()

	if err := os.MkdirAll(ModelDir("m1"), 0o755); err != nil {
		t.Fatal(err)
	}
	_ = os.WriteFile(filepath.Join(ModelDir("m1"), "a"), []byte("only-a"), 0o644)
	// Missing "b" and missing .meta.json → not installed.

	out := RebuildState()
	if out["m1"] != StatusMissing {
		t.Errorf("partial install must be missing, got %q", out["m1"])
	}
}

func TestRebuildState_VersionMismatchIsMissing(t *testing.T) {
	dir := t.TempDir()
	SetDataDirForTest(dir)
	t.Cleanup(func() { SetDataDirForTest("") })

	oldMan := Manifest
	Manifest = []Model{{ID: "m1", Version: "2.0", Files: []File{{RelPath: "a"}}}}
	defer func() { Manifest = oldMan }()

	_ = os.MkdirAll(ModelDir("m1"), 0o755)
	_ = os.WriteFile(filepath.Join(ModelDir("m1"), "a"), []byte("data"), 0o644)
	old, _ := json.Marshal(map[string]any{"version": "1.0", "installed_at": "t", "sha256_verified": true})
	_ = os.WriteFile(filepath.Join(ModelDir("m1"), ".meta.json"), old, 0o644)

	out := RebuildState()
	if out["m1"] != StatusMissing {
		t.Errorf("version mismatch must be missing, got %q", out["m1"])
	}
}
```

- [ ] **Step 2: Run test (fail expected)**

Run: `cd media-server && go test ./deps/models/...`
Expected: compile error.

- [ ] **Step 3: Implement state.go**

Create `media-server/deps/models/state.go`:

```go
package models

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sync"
)

// ModelStatus is the simple per-model installed/missing summary.
type ModelStatus string

const (
	StatusInstalled ModelStatus = "installed"
	StatusMissing   ModelStatus = "missing"
)

var (
	stateCache   map[string]ModelStatus
	stateCacheMu sync.RWMutex
)

// RebuildState walks the model directory and recomputes installed/missing
// for every manifest entry. Always succeeds; results are cached.
func RebuildState() map[string]ModelStatus {
	out := make(map[string]ModelStatus, len(Manifest))
	for _, m := range Manifest {
		out[m.ID] = statusFor(m)
	}
	persist(out)
	stateCacheMu.Lock()
	stateCache = out
	stateCacheMu.Unlock()
	return out
}

// Cached returns the most recent RebuildState result, rebuilding lazily if empty.
func Cached() map[string]ModelStatus {
	stateCacheMu.RLock()
	if stateCache != nil {
		out := make(map[string]ModelStatus, len(stateCache))
		for k, v := range stateCache {
			out[k] = v
		}
		stateCacheMu.RUnlock()
		return out
	}
	stateCacheMu.RUnlock()
	return RebuildState()
}

func statusFor(m Model) ModelStatus {
	dir := ModelDir(m.ID)
	if _, err := os.Stat(dir); err != nil {
		return StatusMissing
	}
	for _, f := range m.Files {
		if _, err := os.Stat(filepath.Join(dir, f.RelPath)); err != nil {
			return StatusMissing
		}
	}
	metaPath := filepath.Join(dir, ".meta.json")
	b, err := os.ReadFile(metaPath)
	if err != nil {
		return StatusMissing
	}
	var meta struct {
		Version string `json:"version"`
	}
	if err := json.Unmarshal(b, &meta); err != nil {
		return StatusMissing
	}
	if meta.Version != m.Version {
		return StatusMissing
	}
	return StatusInstalled
}

// persist writes a cache of the derived state to <dataDir>/models/state.json.
// Failures are swallowed: the file is just a cache.
func persist(out map[string]ModelStatus) {
	root := filepath.Join(dataDir(), "models")
	_ = os.MkdirAll(root, 0o755)
	b, err := json.MarshalIndent(out, "", "  ")
	if err != nil {
		return
	}
	tmp := filepath.Join(root, "state.json.tmp")
	if err := os.WriteFile(tmp, b, 0o644); err != nil {
		return
	}
	_ = os.Rename(tmp, filepath.Join(root, "state.json"))
}
```

- [ ] **Step 4: Run tests**

Run: `cd media-server && go test ./deps/models/...`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add media-server/deps/models/state.go media-server/deps/models/state_test.go
git commit -m "feat(deps): models state — rebuild from disk, cached

Walks <dataDir>/models/<id>/ to derive installed/missing per manifest
entry. Partial installs and version mismatches both report missing.
Persisted state.json is a derived cache; corruption falls back to a
rebuild without erroring."
```

---

## Phase 4: Status aggregator

### Task 10: status.Snapshot — cross-category aggregation

**Files:**
- Create: `media-server/deps/status/status.go`
- Test: `media-server/deps/status/status_test.go`

- [ ] **Step 1: Write failing test**

Create `media-server/deps/status/status_test.go`:

```go
package status

import (
	"testing"

	"github.com/stevecastle/shrike/deps/bundled"
	"github.com/stevecastle/shrike/deps/models"
)

func TestSnapshot_IncludesAllCategories(t *testing.T) {
	// Override sources to known values.
	bundled.SetCachedStatusForTest([]bundled.Status{{ID: "ffmpeg", Name: "FFmpeg", State: "ready", Version: "7.1"}})
	t.Cleanup(func() { bundled.SetCachedStatusForTest(nil) })

	models.SetCachedStateForTest(map[string]models.ModelStatus{"fake-model": models.StatusInstalled})
	t.Cleanup(func() { models.SetCachedStateForTest(nil) })

	snap := Snapshot()
	var sawBundled, sawOptional, sawModel bool
	for _, s := range snap {
		switch s.Category {
		case "bundled":
			sawBundled = true
		case "optional":
			sawOptional = true
		case "model":
			sawModel = true
		}
	}
	if !sawBundled || !sawOptional || !sawModel {
		t.Errorf("missing categories: bundled=%v optional=%v model=%v", sawBundled, sawOptional, sawModel)
	}
}
```

- [ ] **Step 2: Add testing seams to bundled and models**

Append to `media-server/deps/bundled/verify.go`:

```go
// SetCachedStatusForTest replaces the cache. Tests only.
func SetCachedStatusForTest(in []Status) {
	statusCacheMu.Lock()
	defer statusCacheMu.Unlock()
	statusCache = in
}
```

Append to `media-server/deps/models/state.go`:

```go
// SetCachedStateForTest overrides the cache. Tests only.
func SetCachedStateForTest(in map[string]ModelStatus) {
	stateCacheMu.Lock()
	defer stateCacheMu.Unlock()
	stateCache = in
}
```

- [ ] **Step 3: Run test (compile fail expected for status pkg)**

Run: `cd media-server && go test ./deps/status/...`
Expected: compile error.

- [ ] **Step 4: Implement status.go**

Create `media-server/deps/status/status.go`:

```go
// Package status aggregates dep state across bundled / optional / model
// for the UI. It NEVER triggers installs; it only reads snapshots.
package status

import (
	"github.com/stevecastle/shrike/deps/bundled"
	"github.com/stevecastle/shrike/deps/models"
	"github.com/stevecastle/shrike/deps/optional"
)

// Item is one row the UI renders.
type Item struct {
	ID        string `json:"id"`
	Category  string `json:"category"`   // "bundled" | "optional" | "model"
	Name      string `json:"name"`
	State     string `json:"state"`      // ready | missing | broken | installed | not_installed | downloading | failed
	Version   string `json:"version,omitempty"`
	SizeBytes int64  `json:"size_bytes,omitempty"`
	Path      string `json:"path,omitempty"`
	Error     string `json:"error,omitempty"`
	Detail    any    `json:"detail,omitempty"`
}

// Snapshot returns one Item per dep across every category.
func Snapshot() []Item {
	out := make([]Item, 0, 16)

	for _, b := range bundled.CachedStatus() {
		out = append(out, Item{
			ID: b.ID, Category: "bundled", Name: b.Name,
			State: b.State, Version: b.Version, Path: b.Path, Error: b.Error,
		})
	}
	for _, o := range optional.Manifest {
		s, _ := optional.Detect(o.ID)
		state := "not_installed"
		if s.Installed {
			state = "installed"
		}
		out = append(out, Item{
			ID: s.ID, Category: "optional", Name: s.Name,
			State: state, Version: s.Version, Path: s.Path, Detail: s.Hint,
		})
	}
	cached := models.Cached()
	for _, m := range models.Manifest {
		state := string(cached[m.ID])
		if state == "" {
			state = string(models.StatusMissing)
		}
		// Overlay in-flight progress if any.
		if inst, ok := models.Tracker.Snapshot(m.ID); ok {
			state = string(inst.State)
			out = append(out, Item{
				ID: m.ID, Category: "model", Name: m.Name,
				State: state, SizeBytes: m.SizeBytes,
				Detail: inst, Error: inst.Error,
			})
			continue
		}
		out = append(out, Item{
			ID: m.ID, Category: "model", Name: m.Name,
			State: state, SizeBytes: m.SizeBytes,
		})
	}
	return out
}
```

- [ ] **Step 5: Run tests**

Run: `cd media-server && go test ./deps/...`
Expected: PASS across bundled, optional, models, status.

- [ ] **Step 6: Commit**

```bash
git add media-server/deps/status/ media-server/deps/bundled/verify.go media-server/deps/models/state.go
git commit -m "feat(deps): status.Snapshot aggregates all three categories

One Item per dep, tagged by category. In-flight model installs are
overlaid from the progress tracker so the UI sees a single source
of truth."
```

---

## Phase 5: HTTP handlers

### Task 11: deps_handlers — REST endpoints

**Files:**
- Create: `media-server/handlers/deps_handlers.go`
- Create: `media-server/handlers/deps_handlers_test.go`

- [ ] **Step 1: Write failing test**

Create `media-server/handlers/deps_handlers_test.go`:

```go
package handlers

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestGetStatus_ReturnsJSONArray(t *testing.T) {
	rr := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/api/deps/status", nil)
	HandleDepsStatus(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rr.Code, rr.Body)
	}
	var arr []map[string]any
	if err := json.Unmarshal(rr.Body.Bytes(), &arr); err != nil {
		t.Fatalf("body not JSON array: %v", err)
	}
}

func TestPostDownload_UnknownIDReturns404(t *testing.T) {
	rr := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/deps/models/no-such-model/download", nil)
	req.SetPathValue("id", "no-such-model")
	HandleModelDownload(rr, req)
	if rr.Code != http.StatusNotFound {
		t.Errorf("status=%d want 404", rr.Code)
	}
	if !strings.Contains(rr.Body.String(), "unknown") {
		t.Errorf("body=%q want unknown", rr.Body)
	}
}

func TestPostDownload_KnownIDReturns202(t *testing.T) {
	// This test relies on a real manifest entry; we don't actually start a
	// download against the network — we override the manifest in the
	// downloader_test which can't share with handlers package directly.
	// Instead, just assert the path doesn't 500 for the seed id.
	rr := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/deps/models/wd-eva02-large-tagger-v3/download", nil)
	req.SetPathValue("id", "wd-eva02-large-tagger-v3")
	HandleModelDownload(rr, req)
	if rr.Code != http.StatusAccepted && rr.Code != http.StatusOK {
		t.Errorf("status=%d want 202 or 200, body=%s", rr.Code, rr.Body)
	}
}
```

- [ ] **Step 2: Run test (compile fail expected)**

Run: `cd media-server && go test ./handlers/...`
Expected: compile error (handlers package may not exist).

- [ ] **Step 3: Implement deps_handlers.go**

Create `media-server/handlers/deps_handlers.go`:

```go
// Package handlers hosts HTTP handlers that are decoupled from main.go and
// can be shared by every platform build.
package handlers

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/stevecastle/shrike/deps/bundled"
	"github.com/stevecastle/shrike/deps/models"
	"github.com/stevecastle/shrike/deps/status"
)

// HandleDepsStatus serves GET /api/deps/status.
func HandleDepsStatus(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, status.Snapshot())
}

// HandleModelDownload serves POST /api/deps/models/{id}/download.
func HandleModelDownload(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if _, ok := models.Lookup(id); !ok {
		writeErr(w, http.StatusNotFound, fmt.Errorf("unknown model id %q", id))
		return
	}
	// Already installed? Return 200 with current snapshot.
	if cached := models.Cached(); cached[id] == models.StatusInstalled {
		if inst, ok := models.Tracker.Snapshot(id); ok {
			writeJSON(w, http.StatusOK, inst)
			return
		}
		writeJSON(w, http.StatusOK, map[string]string{"state": "installed"})
		return
	}
	inst, err := models.Tracker.StartInstall(id)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusAccepted, inst)
}

// HandleModelCancel serves POST /api/deps/models/{id}/cancel.
func HandleModelCancel(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if !models.Tracker.Cancel(id) {
		writeErr(w, http.StatusNotFound, fmt.Errorf("no active install for %q", id))
		return
	}
	w.WriteHeader(http.StatusAccepted)
}

// HandleModelDelete serves DELETE /api/deps/models/{id}.
func HandleModelDelete(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if _, ok := models.Lookup(id); !ok {
		writeErr(w, http.StatusNotFound, fmt.Errorf("unknown model id %q", id))
		return
	}
	if err := os.RemoveAll(models.ModelDir(id)); err != nil && !os.IsNotExist(err) {
		writeErr(w, http.StatusInternalServerError, err)
		return
	}
	models.RebuildState()
	w.WriteHeader(http.StatusNoContent)
}

// HandleModelVerify serves POST /api/deps/models/{id}/verify.
// Re-hashes every file and compares against the manifest SHA-256.
func HandleModelVerify(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	m, ok := models.Lookup(id)
	if !ok {
		writeErr(w, http.StatusNotFound, fmt.Errorf("unknown model id %q", id))
		return
	}
	result := struct {
		ID    string            `json:"id"`
		Files map[string]string `json:"files"` // file → "ok" | error message
	}{ID: id, Files: map[string]string{}}
	for _, f := range m.Files {
		path := filepath.Join(models.ModelDir(id), f.RelPath)
		if err := models.VerifySHA256(path, f.SHA256); err != nil {
			if errors.Is(err, os.ErrNotExist) {
				result.Files[f.RelPath] = "missing"
			} else {
				result.Files[f.RelPath] = err.Error()
			}
		} else {
			result.Files[f.RelPath] = "ok"
		}
	}
	writeJSON(w, http.StatusOK, result)
}

// HandleModelProgressSSE serves GET /api/deps/models/progress as Server-Sent
// Events. One event per state transition for any model install.
func HandleModelProgressSSE(w http.ResponseWriter, r *http.Request) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	ch, done := models.Tracker.Subscribe()
	defer done()

	// Send current snapshot of every tracked install immediately.
	for _, inst := range models.Tracker.All() {
		writeSSE(w, flusher, inst)
	}

	ping := time.NewTicker(15 * time.Second)
	defer ping.Stop()
	for {
		select {
		case <-r.Context().Done():
			return
		case inst, ok := <-ch:
			if !ok {
				return
			}
			writeSSE(w, flusher, inst)
		case <-ping.C:
			_, _ = w.Write([]byte(": ping\n\n"))
			flusher.Flush()
		}
	}
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeErr(w http.ResponseWriter, status int, err error) {
	writeJSON(w, status, map[string]string{"error": err.Error()})
}

func writeSSE(w http.ResponseWriter, f http.Flusher, payload any) {
	b, err := json.Marshal(payload)
	if err != nil {
		return
	}
	_, _ = w.Write([]byte("data: "))
	_, _ = w.Write(b)
	_, _ = w.Write([]byte("\n\n"))
	f.Flush()
	_ = strings.TrimSpace // keep imports used
}
```

- [ ] **Step 4: Add VerifySHA256 helper to models package**

Append to `media-server/deps/models/downloader.go`:

```go
// VerifySHA256 hashes the file at path and compares against want. Returns
// os.ErrNotExist if the file is missing.
func VerifySHA256(path, want string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return err
	}
	got := hex.EncodeToString(h.Sum(nil))
	if got != want {
		return fmt.Errorf("%w: got %s want %s", ErrChecksumMismatch, got, want)
	}
	return nil
}
```

- [ ] **Step 5: Run tests**

Run: `cd media-server && go test ./handlers/... ./deps/...`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add media-server/handlers/ media-server/deps/models/downloader.go
git commit -m "feat(handlers): /api/deps endpoints

REST: status, download, cancel, verify, delete. SSE: progress stream
keyed off the in-memory Tracker. Idempotent: download of an installed
model returns 200; second download of an in-flight one returns its
existing row."
```

### Task 12: Shared route registration

**Files:**
- Create: `media-server/routes_deps.go`

(No build-tag — this file compiles on every OS.)

- [ ] **Step 1: Implement routes_deps.go**

Create `media-server/routes_deps.go`:

```go
package main

import (
	"net/http"

	"github.com/stevecastle/shrike/handlers"
)

// RegisterDepsRoutes wires every /api/deps/* handler onto the given mux.
// Called from each platform's main file after the SPA router is mounted.
//
// Path patterns use Go 1.22+ wildcards.
func RegisterDepsRoutes(mux *http.ServeMux) {
	mux.HandleFunc("GET /api/deps/status", handlers.HandleDepsStatus)
	mux.HandleFunc("GET /api/deps/models/progress", handlers.HandleModelProgressSSE)
	mux.HandleFunc("POST /api/deps/models/{id}/download", handlers.HandleModelDownload)
	mux.HandleFunc("POST /api/deps/models/{id}/cancel", handlers.HandleModelCancel)
	mux.HandleFunc("POST /api/deps/models/{id}/verify", handlers.HandleModelVerify)
	mux.HandleFunc("DELETE /api/deps/models/{id}", handlers.HandleModelDelete)
}
```

- [ ] **Step 2: Verify build on all platforms**

Run:
```bash
cd media-server && go build .
GOOS=linux  go build .
GOOS=darwin go build .
```
Expected: success on all three. (You may get errors from existing legacy code that still references the deleted modules — defer fixing those until Task 14 if so. If the build is already broken, note the errors and proceed.)

- [ ] **Step 3: Commit**

```bash
git add media-server/routes_deps.go
git commit -m "feat(routes): shared RegisterDepsRoutes helper

Mounts /api/deps/* on a *http.ServeMux. Called from main.go,
main_darwin.go, main_linux.go so handler registration is not
duplicated across platform files."
```

---

## Phase 6: CI bundling pipeline

### Task 13: bundled-versions.json — pinned versions and SHA-256s

**Files:**
- Create: `media-server/scripts/bundled-versions.json`

- [ ] **Step 1: Write the file with placeholders for sha256, real URLs**

Create `media-server/scripts/bundled-versions.json`:

```json
{
  "schema_version": 1,
  "binaries": {
    "ffmpeg": {
      "windows-amd64": {
        "version": "n7.1",
        "url": "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip",
        "sha256": "TO_FILL",
        "archive": "zip",
        "extract": [
          {"from": "*/bin/ffmpeg.exe",  "to": "ffmpeg.exe"},
          {"from": "*/bin/ffprobe.exe", "to": "ffprobe.exe"},
          {"from": "*/bin/ffplay.exe",  "to": "ffplay.exe"}
        ]
      },
      "linux-amd64": {
        "version": "n7.1",
        "url": "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz",
        "sha256": "TO_FILL",
        "archive": "tar.xz",
        "extract": [
          {"from": "*/bin/ffmpeg",  "to": "ffmpeg"},
          {"from": "*/bin/ffprobe", "to": "ffprobe"},
          {"from": "*/bin/ffplay",  "to": "ffplay"}
        ]
      },
      "darwin-arm64": {
        "version": "evermeet-latest",
        "url": "https://evermeet.cx/ffmpeg/ffmpeg-7.1.zip",
        "sha256": "TO_FILL",
        "archive": "zip",
        "extract": [{"from": "ffmpeg", "to": "ffmpeg"}]
      },
      "darwin-amd64": {
        "version": "evermeet-latest",
        "url": "https://evermeet.cx/ffmpeg/ffmpeg-7.1.zip",
        "sha256": "TO_FILL",
        "archive": "zip",
        "extract": [{"from": "ffmpeg", "to": "ffmpeg"}]
      }
    },
    "ffprobe-macos": {
      "darwin-arm64": {
        "version": "evermeet-latest",
        "url": "https://evermeet.cx/ffmpeg/ffprobe-7.1.zip",
        "sha256": "TO_FILL",
        "archive": "zip",
        "extract": [{"from": "ffprobe", "to": "ffprobe"}]
      },
      "darwin-amd64": {
        "version": "evermeet-latest",
        "url": "https://evermeet.cx/ffmpeg/ffprobe-7.1.zip",
        "sha256": "TO_FILL",
        "archive": "zip",
        "extract": [{"from": "ffprobe", "to": "ffprobe"}]
      }
    },
    "exiftool": {
      "windows-amd64": {
        "version": "13.04",
        "url": "https://exiftool.org/exiftool-13.04_64.zip",
        "sha256": "TO_FILL",
        "archive": "zip",
        "extract": [
          {"from": "exiftool-13.04_64/exiftool(-k).exe", "to": "exiftool.exe"},
          {"from": "exiftool-13.04_64/exiftool_files",   "to": "exiftool_files", "type": "dir"}
        ]
      },
      "linux-amd64": {
        "version": "13.04",
        "url": "https://exiftool.org/Image-ExifTool-13.04.tar.gz",
        "sha256": "TO_FILL",
        "archive": "tar.gz",
        "extract": [
          {"from": "Image-ExifTool-13.04/exiftool",   "to": "exiftool"},
          {"from": "Image-ExifTool-13.04/lib",        "to": "exiftool_files", "type": "dir"}
        ]
      },
      "darwin-arm64": {
        "version": "13.04",
        "url": "https://exiftool.org/Image-ExifTool-13.04.tar.gz",
        "sha256": "TO_FILL",
        "archive": "tar.gz",
        "extract": [
          {"from": "Image-ExifTool-13.04/exiftool", "to": "exiftool"},
          {"from": "Image-ExifTool-13.04/lib",      "to": "exiftool_files", "type": "dir"}
        ]
      },
      "darwin-amd64": {
        "version": "13.04",
        "url": "https://exiftool.org/Image-ExifTool-13.04.tar.gz",
        "sha256": "TO_FILL",
        "archive": "tar.gz",
        "extract": [
          {"from": "Image-ExifTool-13.04/exiftool", "to": "exiftool"},
          {"from": "Image-ExifTool-13.04/lib",      "to": "exiftool_files", "type": "dir"}
        ]
      }
    },
    "onnxtag": {
      "windows-amd64": {
        "version": "1.0.0",
        "url": "https://github.com/SteveCastle/loki/releases/download/onnxtag-v1.0.0/onnxtag.exe",
        "sha256": "TO_FILL",
        "archive": "none",
        "extract": [{"from": "onnxtag.exe", "to": "onnxtag.exe"}]
      },
      "linux-amd64": {
        "version": "1.0.0",
        "url": "https://github.com/SteveCastle/loki/releases/download/onnxtag-v1.0.0/onnxtag",
        "sha256": "TO_FILL",
        "archive": "none",
        "extract": [{"from": "onnxtag", "to": "onnxtag"}]
      },
      "darwin-arm64": {
        "version": "1.0.0",
        "url": "https://github.com/SteveCastle/loki/releases/download/onnxtag-v1.0.0/onnxtag-darwin-arm64",
        "sha256": "TO_FILL",
        "archive": "none",
        "extract": [{"from": "onnxtag-darwin-arm64", "to": "onnxtag"}]
      },
      "darwin-amd64": {
        "version": "1.0.0",
        "url": "https://github.com/SteveCastle/loki/releases/download/onnxtag-v1.0.0/onnxtag-darwin-amd64",
        "sha256": "TO_FILL",
        "archive": "none",
        "extract": [{"from": "onnxtag-darwin-amd64", "to": "onnxtag"}]
      }
    },
    "onnxruntime": {
      "windows-amd64": {
        "version": "1.21.0",
        "url": "https://github.com/microsoft/onnxruntime/releases/download/v1.21.0/onnxruntime-win-x64-1.21.0.zip",
        "sha256": "TO_FILL",
        "archive": "zip",
        "extract": [{"from": "onnxruntime-win-x64-1.21.0/lib/onnxruntime.dll", "to": "onnxruntime.dll"}]
      },
      "linux-amd64": {
        "version": "1.21.0",
        "url": "https://github.com/microsoft/onnxruntime/releases/download/v1.21.0/onnxruntime-linux-x64-1.21.0.tgz",
        "sha256": "TO_FILL",
        "archive": "tar.gz",
        "extract": [{"from": "onnxruntime-linux-x64-1.21.0/lib/libonnxruntime.so*", "to": "onnxruntime.so", "match": "newest"}]
      },
      "darwin-arm64": {
        "version": "1.21.0",
        "url": "https://github.com/microsoft/onnxruntime/releases/download/v1.21.0/onnxruntime-osx-arm64-1.21.0.tgz",
        "sha256": "TO_FILL",
        "archive": "tar.gz",
        "extract": [{"from": "onnxruntime-osx-arm64-1.21.0/lib/libonnxruntime.*.dylib", "to": "onnxruntime.dylib", "match": "newest"}]
      },
      "darwin-amd64": {
        "version": "1.21.0",
        "url": "https://github.com/microsoft/onnxruntime/releases/download/v1.21.0/onnxruntime-osx-x86_64-1.21.0.tgz",
        "sha256": "TO_FILL",
        "archive": "tar.gz",
        "extract": [{"from": "onnxruntime-osx-x86_64-1.21.0/lib/libonnxruntime.*.dylib", "to": "onnxruntime.dylib", "match": "newest"}]
      }
    }
  }
}
```

- [ ] **Step 2: Commit (placeholders intentional — Task 14's script fills them)**

```bash
git add media-server/scripts/bundled-versions.json
git commit -m "feat(ci): pinned bundled binary versions

Per-OS-arch entries for ffmpeg, exiftool, onnxtag, onnxruntime, plus
the macOS ffprobe split. SHA-256s are 'TO_FILL' until populated by
fetch-bundled-deps.sh's --verify pass."
```

### Task 14: fetch-bundled-deps script (Linux/macOS)

**Files:**
- Create: `media-server/scripts/fetch-bundled-deps.sh`

- [ ] **Step 1: Implement the script**

Create `media-server/scripts/fetch-bundled-deps.sh`:

```sh
#!/usr/bin/env bash
# Fetch bundled binaries for one GOOS-GOARCH target into media-server/bin/<target>/.
# Reads media-server/scripts/bundled-versions.json. Verifies SHA-256 if not
# "TO_FILL"; with --update prints discovered SHA-256s to stdout instead.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONF="$ROOT/media-server/scripts/bundled-versions.json"
TARGET="${1:-}"
MODE="${2:-verify}"   # verify | update
if [ -z "$TARGET" ]; then
  echo "usage: $0 <goos-goarch> [verify|update]" >&2
  exit 2
fi

OUTDIR="$ROOT/media-server/bin/$TARGET"
mkdir -p "$OUTDIR"

tmpdir="$(mktemp -d)"
trap "rm -rf $tmpdir" EXIT

# List binaries that have an entry for $TARGET.
bins=$(jq -r --arg t "$TARGET" '.binaries | to_entries[] | select(.value[$t] != null) | .key' "$CONF")
for bin in $bins; do
  url=$(jq -r --arg t "$TARGET" --arg b "$bin" '.binaries[$b][$t].url' "$CONF")
  archive=$(jq -r --arg t "$TARGET" --arg b "$bin" '.binaries[$b][$t].archive' "$CONF")
  want_sum=$(jq -r --arg t "$TARGET" --arg b "$bin" '.binaries[$b][$t].sha256' "$CONF")

  archive_path="$tmpdir/$bin.archive"
  echo "fetching $bin ($url) ..."
  curl -fsSL "$url" -o "$archive_path"
  got_sum=$(shasum -a 256 "$archive_path" | awk '{print $1}')

  if [ "$MODE" = "update" ]; then
    echo "SHA256 $bin $TARGET $got_sum"
  else
    if [ "$want_sum" != "TO_FILL" ] && [ "$want_sum" != "$got_sum" ]; then
      echo "SHA256 mismatch for $bin $TARGET" >&2
      echo "  want: $want_sum" >&2
      echo "  got:  $got_sum" >&2
      exit 1
    fi
  fi

  # Extract.
  extract_dir="$tmpdir/$bin"
  mkdir -p "$extract_dir"
  case "$archive" in
    zip)    unzip -q "$archive_path" -d "$extract_dir" ;;
    tar.gz) tar -xzf "$archive_path" -C "$extract_dir" ;;
    tar.xz) tar -xJf "$archive_path" -C "$extract_dir" ;;
    none)   cp "$archive_path" "$extract_dir/$(basename "$url")" ;;
    *) echo "unknown archive type $archive" >&2; exit 1 ;;
  esac

  # For each extract entry, glob and copy.
  jq -r --arg t "$TARGET" --arg b "$bin" '.binaries[$b][$t].extract[] | [.from, .to, (.type // "file")] | @tsv' "$CONF" |
  while IFS=$'\t' read -r from to type; do
    matches=( $(cd "$extract_dir" && ls -1 $from 2>/dev/null || true) )
    if [ ${#matches[@]} -eq 0 ]; then
      echo "no match for $from in $bin" >&2
      exit 1
    fi
    src="$extract_dir/${matches[0]}"
    dst="$OUTDIR/$to"
    if [ "$type" = "dir" ]; then
      rm -rf "$dst"
      cp -R "$src" "$dst"
    else
      cp "$src" "$dst"
      chmod +x "$dst" || true
    fi
  done
done

echo "Bundled binaries for $TARGET written to $OUTDIR"
```

- [ ] **Step 2: Make executable and run in update mode for each target on a Mac/Linux runner**

```bash
chmod +x media-server/scripts/fetch-bundled-deps.sh
media-server/scripts/fetch-bundled-deps.sh darwin-arm64 update
media-server/scripts/fetch-bundled-deps.sh darwin-amd64 update
media-server/scripts/fetch-bundled-deps.sh linux-amd64 update
```

Inline each `SHA256 …` line's hash into `bundled-versions.json`, replacing `TO_FILL`. Then re-run in verify mode for each target:

```bash
media-server/scripts/fetch-bundled-deps.sh darwin-arm64 verify
media-server/scripts/fetch-bundled-deps.sh darwin-amd64 verify
media-server/scripts/fetch-bundled-deps.sh linux-amd64 verify
```

Each should exit 0 with output `Bundled binaries for <target> written to …`.

- [ ] **Step 3: Commit the script and updated SHA-256s**

```bash
git add media-server/scripts/fetch-bundled-deps.sh media-server/scripts/bundled-versions.json
git commit -m "feat(ci): fetch-bundled-deps.sh

Downloads and extracts per-OS-arch binaries described in
bundled-versions.json into media-server/bin/<target>/. SHA-256
verification enforced (verify mode) or printed (update mode).
SHA-256s populated for darwin/linux."
```

### Task 15: fetch-bundled-deps.ps1 (Windows runner equivalent)

**Files:**
- Create: `media-server/scripts/fetch-bundled-deps.ps1`

- [ ] **Step 1: Implement the script**

Create `media-server/scripts/fetch-bundled-deps.ps1`:

```powershell
# Windows equivalent of fetch-bundled-deps.sh. PowerShell 5.1 compatible.
param(
  [Parameter(Mandatory=$true)][string]$Target,
  [ValidateSet("verify","update")][string]$Mode = "verify"
)
$ErrorActionPreference = "Stop"

$root  = (Resolve-Path "$PSScriptRoot\..\..").Path
$conf  = Join-Path $root "media-server\scripts\bundled-versions.json"
$outDir = Join-Path $root "media-server\bin\$Target"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$data = Get-Content $conf -Raw | ConvertFrom-Json
$tmp  = New-Item -ItemType Directory -Path ([System.IO.Path]::GetTempPath() + [System.Guid]::NewGuid().ToString())

try {
  foreach ($bin in $data.binaries.PSObject.Properties.Name) {
    $entry = $data.binaries.$bin.$Target
    if ($null -eq $entry) { continue }

    $archivePath = Join-Path $tmp "$bin.archive"
    Write-Host "fetching $bin ($($entry.url)) ..."
    Invoke-WebRequest -Uri $entry.url -OutFile $archivePath
    $gotSum = (Get-FileHash -Algorithm SHA256 $archivePath).Hash.ToLower()

    if ($Mode -eq "update") {
      Write-Host "SHA256 $bin $Target $gotSum"
    } else {
      if ($entry.sha256 -ne "TO_FILL" -and $entry.sha256 -ne $gotSum) {
        Write-Error "SHA256 mismatch for $bin $Target`n  want: $($entry.sha256)`n  got:  $gotSum"
      }
    }

    $extractDir = Join-Path $tmp $bin
    New-Item -ItemType Directory -Force -Path $extractDir | Out-Null
    switch ($entry.archive) {
      "zip"    { Expand-Archive -Path $archivePath -DestinationPath $extractDir -Force }
      "tar.gz" { tar -xzf $archivePath -C $extractDir }
      "tar.xz" { tar -xJf $archivePath -C $extractDir }
      "none"   { Copy-Item $archivePath (Join-Path $extractDir ([IO.Path]::GetFileName($entry.url))) }
      default  { Write-Error "unknown archive type $($entry.archive)" }
    }

    foreach ($ex in $entry.extract) {
      $type = if ($ex.type) { $ex.type } else { "file" }
      $matches = Get-ChildItem -Path $extractDir -Recurse -Filter ([IO.Path]::GetFileName($ex.from)) -ErrorAction SilentlyContinue
      if (-not $matches) { Write-Error "no match for $($ex.from) in $bin" }
      $src = $matches[0].FullName
      $dst = Join-Path $outDir $ex.to
      if ($type -eq "dir") {
        if (Test-Path $dst) { Remove-Item -Recurse -Force $dst }
        Copy-Item -Recurse $src $dst
      } else {
        Copy-Item -Force $src $dst
      }
    }
  }
  Write-Host "Bundled binaries for $Target written to $outDir"
} finally {
  Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}
```

- [ ] **Step 2: Run on Windows runner (or local Windows shell) and update SHA-256s**

```powershell
powershell -File media-server\scripts\fetch-bundled-deps.ps1 -Target windows-amd64 -Mode update
```

Inline the printed SHA-256s into `bundled-versions.json` for the windows-amd64 entries.

```powershell
powershell -File media-server\scripts\fetch-bundled-deps.ps1 -Target windows-amd64 -Mode verify
```

Expected exit 0.

- [ ] **Step 3: Commit**

```bash
git add media-server/scripts/fetch-bundled-deps.ps1 media-server/scripts/bundled-versions.json
git commit -m "feat(ci): fetch-bundled-deps.ps1 (Windows)

PowerShell 5.1 port of the bash fetcher. Same JSON contract.
windows-amd64 SHA-256s populated."
```

### Task 16: release-server.yml workflow + smoke script

**Files:**
- Create: `media-server/.github/workflows/release-server.yml`
- Create: `media-server/scripts/smoke-bundled.sh`

- [ ] **Step 1: Implement smoke-bundled.sh**

Create `media-server/scripts/smoke-bundled.sh`:

```sh
#!/usr/bin/env bash
# Boots a freshly-built server in the release layout, polls /api/deps/status,
# asserts every bundled dep is "ready" or "broken" (NOT "missing"), then
# stops the server. Used as a post-build CI check.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
INSTALL_DIR="${1:-$ROOT/media-server}"

cd "$INSTALL_DIR"
PORT=18762
LOWKEY_PORT=$PORT ./lowkeymediaserver &
PID=$!
trap "kill $PID 2>/dev/null || true" EXIT

# Wait up to 10s for the listener.
for i in $(seq 1 20); do
  if curl -fsS "http://127.0.0.1:$PORT/api/deps/status" >/tmp/status.json 2>/dev/null; then
    break
  fi
  sleep 0.5
done

if ! [ -s /tmp/status.json ]; then
  echo "server did not serve /api/deps/status within 10s" >&2
  exit 1
fi

# Every bundled entry must NOT be "missing".
missing=$(jq -r '[.[] | select(.category=="bundled" and .state=="missing")] | length' /tmp/status.json)
if [ "$missing" -ne 0 ]; then
  echo "smoke failed: $missing bundled deps missing" >&2
  jq '[.[] | select(.category=="bundled" and .state=="missing")]' /tmp/status.json >&2
  exit 1
fi
echo "smoke OK"
```

- [ ] **Step 2: Create the release workflow**

Create `media-server/.github/workflows/release-server.yml`:

```yaml
name: Release Media Server

on:
  push:
    tags: [ 'mediaserver-v*' ]
  workflow_dispatch: {}

jobs:
  build:
    strategy:
      fail-fast: false
      matrix:
        include:
          - os: windows-latest
            target: windows-amd64
            goos: windows
            goarch: amd64
            ext: .exe
            fetch_cmd: "powershell -File media-server/scripts/fetch-bundled-deps.ps1 -Target windows-amd64 -Mode verify"
          - os: ubuntu-latest
            target: linux-amd64
            goos: linux
            goarch: amd64
            ext: ""
            fetch_cmd: "media-server/scripts/fetch-bundled-deps.sh linux-amd64 verify"
          - os: macos-14   # Apple Silicon
            target: darwin-arm64
            goos: darwin
            goarch: arm64
            ext: ""
            fetch_cmd: "media-server/scripts/fetch-bundled-deps.sh darwin-arm64 verify"
          - os: macos-13   # Intel
            target: darwin-amd64
            goos: darwin
            goarch: amd64
            ext: ""
            fetch_cmd: "media-server/scripts/fetch-bundled-deps.sh darwin-amd64 verify"
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-go@v5
        with: { go-version: '1.22' }
      - name: Install jq (unix)
        if: runner.os != 'Windows'
        run: |
          if ! command -v jq >/dev/null; then
            sudo apt-get update && sudo apt-get install -y jq || brew install jq
          fi
      - name: Build SPA
        run: |
          yarn install --frozen-lockfile
          yarn build:web
      - name: Fetch bundled binaries
        shell: bash
        run: ${{ matrix.fetch_cmd }}
      - name: Build server
        working-directory: media-server
        env:
          GOOS: ${{ matrix.goos }}
          GOARCH: ${{ matrix.goarch }}
        run: go build -o lowkeymediaserver${{ matrix.ext }} .
      - name: Stage release dir
        shell: bash
        run: |
          mkdir -p release/${{ matrix.target }}/bin
          cp media-server/lowkeymediaserver${{ matrix.ext }} release/${{ matrix.target }}/
          cp -R media-server/bin/${{ matrix.target }}/* release/${{ matrix.target }}/bin/
      - name: Smoke test (unix only)
        if: runner.os != 'Windows'
        shell: bash
        run: media-server/scripts/smoke-bundled.sh release/${{ matrix.target }}
      - name: Archive
        shell: bash
        run: |
          cd release
          if [ "${{ matrix.goos }}" = "windows" ]; then
            7z a lowkey-media-server-${{ matrix.target }}.zip ${{ matrix.target }}/*
          else
            tar -czf lowkey-media-server-${{ matrix.target }}.tar.gz ${{ matrix.target }}
          fi
      - name: Upload artifact
        uses: actions/upload-artifact@v4
        with:
          name: server-${{ matrix.target }}
          path: release/lowkey-media-server-${{ matrix.target }}.*
```

- [ ] **Step 3: Commit**

```bash
chmod +x media-server/scripts/smoke-bundled.sh
git add media-server/scripts/smoke-bundled.sh media-server/.github/workflows/release-server.yml
git commit -m "ci: release-server workflow + smoke test

Per-target matrix builds the SPA, fetches and verifies bundled
binaries, builds the Go server, stages a release dir, runs a smoke
test (unix) hitting /api/deps/status, and uploads the archive.
macOS jobs run on macos-14 (arm64) and macos-13 (Intel)."
```

---

## Phase 7: Boot wiring + caller migration

### Task 17: deps facade — backwards-compatible helpers

**Files:**
- Create: `media-server/deps/facade.go`

The current `deps/` package exposes ~6 `GetXxxPath()` helpers. The new code uses `bundled.Resolve`, `optional.Detect`, `models.Path`. We add a thin facade so the rewrite in Task 18 is a mechanical pattern change.

- [ ] **Step 1: Implement facade.go**

Create `media-server/deps/facade.go`:

```go
// Package deps re-exports thin helpers over the new bundled/optional/models
// packages. The legacy types (Dependency, registry, MetadataStore) have been
// removed; this file is the only public surface during the migration.
package deps

import (
	"fmt"

	"github.com/stevecastle/shrike/deps/bundled"
	"github.com/stevecastle/shrike/deps/models"
	"github.com/stevecastle/shrike/deps/optional"
)

// MustBundled returns the absolute path to the named bundled binary. Panics
// if missing — after bundled.VerifyAll runs at boot, callers can assume
// every entry is present.
func MustBundled(id string) string {
	p, err := bundled.Resolve(id)
	if err != nil {
		panic(fmt.Sprintf("deps: bundled %q unresolvable: %v", id, err))
	}
	return p
}

// BundledOrEmpty returns the path or "" if the binary is missing. Useful for
// callers that gracefully degrade (e.g. ffplay-on-macOS).
func BundledOrEmpty(id string) string {
	p, err := bundled.Resolve(id)
	if err != nil {
		return ""
	}
	return p
}

// ModelPath returns the absolute path to relPath inside the named model, or
// IsModelNotInstalled(err) → true if the model isn't present.
func ModelPath(id, relPath string) (string, error) {
	p, err := models.Path(id, relPath)
	if err != nil {
		return "", err
	}
	return p, nil
}

// IsModelNotInstalled forwards to models.IsNotInstalled.
func IsModelNotInstalled(err error) bool { return models.IsNotInstalled(err) }

// DetectOptional forwards to optional.Detect.
func DetectOptional(id string) (optional.Status, error) { return optional.Detect(id) }
```

- [ ] **Step 2: Build**

Run: `cd media-server && go build ./deps/`
Expected: success (the legacy files in `deps/` are still present and may cause failures referencing `downloads`; they will be deleted in Task 22). If the build fails because `deps/deps.go` references missing things, do NOT delete deps.go yet — that's Task 22. Just confirm `facade.go` is syntactically valid:

Run: `cd media-server && go vet ./deps/facade.go`
Expected: no output, or only complaints unrelated to facade.go itself.

- [ ] **Step 3: Commit**

```bash
git add media-server/deps/facade.go
git commit -m "feat(deps): facade with MustBundled / ModelPath / DetectOptional

Thin shim over bundled/optional/models so callers can be migrated
mechanically before the legacy package files are deleted."
```

### Task 18: Rewrite all call sites

**Scope:** every `.go` file outside `media-server/deps/` and `media-server/downloads/` that references the legacy API.

The legacy patterns and their replacements:

| Old | New |
|-----|-----|
| `deps.GetFFmpegPath()` / similar | `deps.MustBundled("ffmpeg")` |
| `deps.Get("yt-dlp")` + `.Check()` | `s, _ := deps.DetectOptional("yt-dlp"); if !s.Installed { … }` |
| `depspkg.EnsureAvailable(ctx, q, id)` for binaries | delete the call (bundled is always present) |
| `depspkg.EnsureAvailable(ctx, q, id)` for models | `_, err := deps.ModelPath(id, mainFile); if deps.IsModelNotInstalled(err) { return ErrModelMissing }` |
| `deps.GetMetadataStore()` | delete; metadata is gone |
| `deps.CheckAnyMissing(ctx)` | replace with `return false` then delete the call site entirely in Task 20 |
| `downloads.*` | delete the call; the new code does not need it |

- [ ] **Step 1: Enumerate every call site**

Run:
```bash
cd media-server && grep -rn "depspkg\|\"github.com/stevecastle/shrike/deps\"\|\"github.com/stevecastle/shrike/downloads\"" --include="*.go" . | grep -v "/deps/" | grep -v "/downloads/" | grep -v "_test.go" > /tmp/callsites.txt
wc -l /tmp/callsites.txt
cat /tmp/callsites.txt
```

Use this list as the work queue. For each file, open it and apply the table above. Common patterns to watch for:

- A struct field `OnnxModelPath string` set from `deps.GetXxx` → set instead from `deps.ModelPath("wd-eva02-large-tagger-v3", "model.onnx")` with the error surfaced.
- A handler that returned "dependency not installed" via `EnsureAvailable` should return a typed `ErrModelMissing` (defined inline if needed: `var ErrModelMissing = errors.New("model not installed")`) so the HTTP layer can map it to a 412 with payload `{"model": "...", "install_url": "/api/deps/models/.../download"}`.
- macOS-specific code paths that called the legacy `GetFFplayPath` should accept `BundledOrEmpty` and disable the feature when empty.

- [ ] **Step 2: Run go build on every platform after each batch**

Don't try to fix all 200+ references in one pass. Process by package:

```bash
cd media-server
# After each package's edits:
go build ./tasks/...
GOOS=linux  go build ./tasks/...
GOOS=darwin go build ./tasks/...
```

- [ ] **Step 3: Run tests after the full sweep**

```bash
cd media-server && go test ./...
```

Fix any compile/test errors before proceeding. Tests that exercise the legacy `Dependency.Check` flow can be deleted in Task 22 — comment them out for now with `// TODO Task 22: delete with legacy deps removal`.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor(deps): migrate ~200 call sites to facade

Mechanical rewrite of GetXxxPath/EnsureAvailable/Check patterns to
MustBundled/ModelPath/DetectOptional. Legacy deps.go and downloads/
files are left in place; Task 22 removes them once nothing imports
them."
```

### Task 19: Boot integration — VerifyAll, RebuildState, register routes, drop /setup

**Files:** `media-server/main.go`, `media-server/main_darwin.go`, `media-server/main_linux.go`

The same edits apply to each platform file (they're parallel build-tag variants).

- [ ] **Step 1: Add boot calls to each main_*.go**

In each `main.go` / `main_darwin.go` / `main_linux.go`, find the place where the HTTP listener is started (search for `srv := &http.Server` or `http.ListenAndServe`). Immediately before that, add:

```go
// Verify bundled deps and rebuild model state. Both are non-fatal.
go func() {
    bundled.VerifyAll()
    models.RebuildState()
    // One-line legacy file notice and rename (idempotent).
    legacy := filepath.Join(platform.GetDataDir(), "dependencies.json")
    if _, err := os.Stat(legacy); err == nil {
        _ = os.Rename(legacy, legacy+".bak")
        log.Printf("deps: legacy dependencies.json renamed to dependencies.json.bak")
    }
}()
```

Add the imports if missing:
```go
"github.com/stevecastle/shrike/deps/bundled"
"github.com/stevecastle/shrike/deps/models"
"github.com/stevecastle/shrike/platform"
```

- [ ] **Step 2: Register the new routes after the SPA mux is built**

After the line that mounts the SPA (search for `mux.HandleFunc` near the SPA `/` handler), add:

```go
RegisterDepsRoutes(mux)
```

- [ ] **Step 3: Remove the forced /setup middleware**

In each main_*.go, search for `setupMode` and `CheckAnyMissing`. Remove:
- the `setupMode` var declaration and assignment
- the middleware that redirects to `/setup`
- the `/setup` route handler itself
- the `/dependencies` and `/dependencies/download` handlers (they're being replaced by `/api/deps/*`)
- any imports that become unused after these removals (run `go build` to surface them; remove with `goimports -w`)

- [ ] **Step 4: Build all platforms**

```bash
cd media-server && go build .
GOOS=linux  go build .
GOOS=darwin go build .
```

Expected: success.

- [ ] **Step 5: Manually test boot locally**

```bash
cd media-server && go run . &
sleep 2
curl -sS http://127.0.0.1:8060/api/deps/status | head -c 500
kill %1
```

Expected: a JSON array containing entries for ffmpeg, ffprobe, exiftool, onnxtag, onnxruntime, yt-dlp, gallery-dl, ollama, and the seed model. Bundled entries will be "missing" if you haven't run the fetch script locally — that's fine, the server boots either way.

- [ ] **Step 6: Commit**

```bash
git add media-server/main.go media-server/main_darwin.go media-server/main_linux.go
git commit -m "feat(server): wire VerifyAll, RebuildState, /api/deps; drop /setup

Boot now triggers a non-fatal verification of bundled deps and
rebuilds the model install state. /api/deps/* routes are registered
once via the shared helper. The legacy /setup forced redirect,
/dependencies UI, and setupMode flag are removed. Legacy
dependencies.json is renamed to .bak on first boot."
```

### Task 20: Delete legacy deps and downloads packages

**Files:**
- Delete: every file enumerated in the "Files deleted at the end" list near the top of this plan.

- [ ] **Step 1: Verify nothing outside deps/ imports the legacy types**

```bash
cd media-server && grep -rn "deps\.\(Dependency\|Register\|GetAll\|GetMetadataStore\|EnsureAvailable\|StatusInstalled\|GetMissingRequired\|CheckAnyMissing\|GetFilePath\|GetInstallPath\)" --include="*.go" . | grep -v "/deps/" | grep -v "_test.go"
```

Expected: empty output. If anything matches, it's a missed call site from Task 18; fix it before deleting.

```bash
grep -rn '"github.com/stevecastle/shrike/downloads"' --include="*.go" .
```

Expected: empty (or only matches under `media-server/downloads/`).

- [ ] **Step 2: Delete the files**

```bash
cd media-server
rm deps/deps.go deps/deps_test.go deps/metadata.go deps/paths.go deps/exec.go deps/exec_windows.go deps/exec_linux.go deps/exec_darwin.go
rm deps/ffmpeg.go deps/whisper.go deps/onnx.go deps/onnxtag.go deps/ytdlp.go deps/gallerydl.go deps/ollama.go deps/dce.go
rm -rf downloads
```

- [ ] **Step 3: Build and test all platforms**

```bash
cd media-server && go test ./...
GOOS=linux  go build .
GOOS=darwin go build .
```

Expected: all PASS / build success.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore(deps): delete legacy deps and downloads packages

Replaced by bundled/optional/models/status. The Dependency struct,
init()-based registry, dependencies.json metadata, and the entire
downloads/ package (manager + extract + http) are gone."
```

---

## Phase 8: SPA welcome wizard

The wizard lives in `src/renderer/onboarding/`. Components are functional React + TypeScript. State is local — no XState additions; the data we need is server-side. Routing reuses whatever the SPA's existing router is (check `src/renderer/main.tsx` for the pattern before starting).

### Task 21: useDepsStatus hook + api helpers

**Files:**
- Create: `src/renderer/onboarding/api.ts`
- Create: `src/renderer/onboarding/useDepsStatus.ts`

- [ ] **Step 1: Implement api.ts**

Create `src/renderer/onboarding/api.ts`:

```ts
// API helpers. Uses the same fetch pattern the rest of the SPA uses for
// server-mode calls — always relative URLs, never absolute.

export type Category = 'bundled' | 'optional' | 'model';

export interface DepStatus {
  id: string;
  category: Category;
  name: string;
  state: 'ready' | 'missing' | 'broken' | 'installed' | 'not_installed' | 'queued' | 'downloading' | 'verifying' | 'failed' | 'cancelled';
  version?: string;
  size_bytes?: number;
  path?: string;
  error?: string;
  detail?: any;
}

export async function fetchStatus(): Promise<DepStatus[]> {
  const r = await fetch('/api/deps/status');
  if (!r.ok) throw new Error(`status ${r.status}`);
  return r.json();
}

export async function startModelDownload(id: string): Promise<void> {
  const r = await fetch(`/api/deps/models/${encodeURIComponent(id)}/download`, { method: 'POST' });
  if (!r.ok && r.status !== 202) throw new Error(`download ${r.status}`);
}

export async function cancelModelDownload(id: string): Promise<void> {
  await fetch(`/api/deps/models/${encodeURIComponent(id)}/cancel`, { method: 'POST' });
}

export async function deleteModel(id: string): Promise<void> {
  await fetch(`/api/deps/models/${encodeURIComponent(id)}`, { method: 'DELETE' });
}
```

- [ ] **Step 2: Implement useDepsStatus.ts**

Create `src/renderer/onboarding/useDepsStatus.ts`:

```ts
import { useEffect, useRef, useState } from 'react';
import { fetchStatus, type DepStatus } from './api';

// useDepsStatus polls /api/deps/status every 4s and overlays SSE progress
// events from /api/deps/models/progress. Returns the current status list.
export function useDepsStatus(): { status: DepStatus[]; refresh: () => void; error: string | null } {
  const [status, setStatus] = useState<DepStatus[]>([]);
  const [error, setError] = useState<string | null>(null);
  const sseRef = useRef<EventSource | null>(null);

  const refresh = async () => {
    try {
      const s = await fetchStatus();
      setStatus(s);
      setError(null);
    } catch (e: any) {
      setError(e?.message ?? 'unknown error');
    }
  };

  useEffect(() => {
    refresh();
    const t = window.setInterval(refresh, 4000);

    const es = new EventSource('/api/deps/models/progress');
    sseRef.current = es;
    es.onmessage = (ev) => {
      try {
        const inst = JSON.parse(ev.data);
        setStatus((cur) =>
          cur.map((s) =>
            s.category === 'model' && s.id === inst.id
              ? { ...s, state: inst.state, detail: inst, error: inst.error }
              : s,
          ),
        );
      } catch {
        /* drop */
      }
    };
    es.onerror = () => {
      // EventSource retries on its own; we don't surface transient errors.
    };

    return () => {
      window.clearInterval(t);
      es.close();
    };
  }, []);

  return { status, refresh, error };
}
```

- [ ] **Step 3: Commit**

```bash
git add src/renderer/onboarding/api.ts src/renderer/onboarding/useDepsStatus.ts
git commit -m "feat(onboarding): useDepsStatus hook + fetch helpers

4s polling fallback overlaid with SSE progress events for model
installs. Hook returns current status, a refresh imperative, and the
last error."
```

### Task 22: BundledPanel + OptionalPanel components

**Files:**
- Create: `src/renderer/onboarding/BundledPanel.tsx`
- Create: `src/renderer/onboarding/OptionalPanel.tsx`
- Create: `src/renderer/onboarding/styles.module.css`

- [ ] **Step 1: Implement BundledPanel.tsx**

Create `src/renderer/onboarding/BundledPanel.tsx`:

```tsx
import React from 'react';
import type { DepStatus } from './api';
import styles from './styles.module.css';

interface Props { items: DepStatus[]; }

export const BundledPanel: React.FC<Props> = ({ items }) => {
  const bundled = items.filter((i) => i.category === 'bundled');
  const allReady = bundled.every((i) => i.state === 'ready');
  return (
    <section className={styles.panel}>
      <header>
        <h2>Bundled tools</h2>
        <p>These ship with the server and need no setup.</p>
      </header>
      <ul className={styles.list}>
        {bundled.map((i) => (
          <li key={i.id} className={styles[i.state] || styles.row}>
            <span className={styles.icon} aria-hidden>{i.state === 'ready' ? '✓' : i.state === 'missing' ? '✗' : '⚠'}</span>
            <span className={styles.name}>{i.name}</span>
            {i.version && <span className={styles.version}>{i.version}</span>}
            {i.state !== 'ready' && i.error && <span className={styles.error}>{i.error}</span>}
          </li>
        ))}
      </ul>
      {!allReady && (
        <p className={styles.warn}>
          Something is wrong with the server install. Please reinstall the server.
        </p>
      )}
    </section>
  );
};
```

- [ ] **Step 2: Implement OptionalPanel.tsx**

Create `src/renderer/onboarding/OptionalPanel.tsx`:

```tsx
import React, { useState } from 'react';
import type { DepStatus } from './api';
import styles from './styles.module.css';

interface Props { items: DepStatus[]; }

export const OptionalPanel: React.FC<Props> = ({ items }) => {
  const optional = items.filter((i) => i.category === 'optional');
  return (
    <section className={styles.panel}>
      <header>
        <h2>Optional tools</h2>
        <p>Install these yourself if you want the features they unlock. The server runs fine without them.</p>
      </header>
      <ul className={styles.list}>
        {optional.map((i) => <OptionalRow key={i.id} item={i} />)}
      </ul>
    </section>
  );
};

const OptionalRow: React.FC<{ item: DepStatus }> = ({ item }) => {
  const hint = item.detail || {};
  const [open, setOpen] = useState(false);
  const cmds: Array<{ os: string; label: string; command: string }> = hint.Commands || hint.commands || [];
  const desc: string = hint.Description || hint.description || '';
  const docsURL: string = hint.DocsURL || hint.docs_url || hint.docsURL || '';
  return (
    <li className={styles[item.state] || styles.row}>
      <div className={styles.head}>
        <span className={styles.icon} aria-hidden>{item.state === 'installed' ? '✓' : '○'}</span>
        <span className={styles.name}>{item.name}</span>
        {item.version && <span className={styles.version}>{item.version}</span>}
        <button type="button" className={styles.disclose} onClick={() => setOpen((o) => !o)}>
          {open ? 'Hide install commands' : 'Show install commands'}
        </button>
      </div>
      {open && (
        <div className={styles.detail}>
          {desc && <p>{desc}</p>}
          <ul className={styles.cmds}>
            {cmds.map((c) => (
              <li key={c.os + c.label}>
                <strong>{c.os} ({c.label}):</strong>
                <code>{c.command}</code>
              </li>
            ))}
          </ul>
          {docsURL && <a href={docsURL} target="_blank" rel="noreferrer">Documentation</a>}
        </div>
      )}
    </li>
  );
};
```

- [ ] **Step 3: Implement styles.module.css**

Create `src/renderer/onboarding/styles.module.css`:

```css
.panel { margin-block: 2rem; }
.panel header h2 { margin: 0 0 .25rem; }
.panel header p  { color: var(--muted, #888); margin: 0 0 1rem; }

.list { list-style: none; padding: 0; margin: 0; display: grid; gap: .5rem; }
.row, .ready, .missing, .broken, .installed, .not_installed,
.queued, .downloading, .verifying, .failed, .cancelled {
  display: flex; align-items: center; gap: .75rem;
  padding: .65rem .85rem;
  border-radius: .4rem;
  background: var(--row-bg, #1c1c20);
}
.ready, .installed { border-left: 3px solid #5ec27a; }
.missing, .failed, .broken { border-left: 3px solid #d35353; }
.not_installed { border-left: 3px solid #555; }
.downloading, .verifying, .queued { border-left: 3px solid #5b9bd5; }

.icon { font-weight: bold; min-width: 1.2rem; text-align: center; }
.name { flex: 1; }
.version { color: var(--muted, #888); font-size: .85em; font-family: monospace; }
.error { color: #d35353; font-size: .85em; }
.warn { color: #d35353; margin-top: 1rem; }

.head { display: flex; align-items: center; gap: .75rem; width: 100%; }
.disclose {
  margin-left: auto;
  background: transparent;
  border: 1px solid #555;
  color: inherit;
  padding: .25rem .6rem;
  border-radius: .25rem;
  cursor: pointer;
}

.detail { padding: .5rem 0 0 2rem; width: 100%; }
.cmds  { list-style: none; padding: 0; display: grid; gap: .25rem; }
.cmds code { background: #000; padding: 0 .35rem; border-radius: 3px; margin-left: .35rem; user-select: all; }

.progressBar {
  height: 6px; background: #333; border-radius: 3px; overflow: hidden; flex: 1;
}
.progressBarFill { height: 100%; background: #5b9bd5; transition: width 200ms; }

.actions { display: flex; gap: .5rem; }
.btn {
  border: 1px solid #555;
  background: #2a2a30;
  color: inherit;
  padding: .35rem .75rem;
  border-radius: .25rem;
  cursor: pointer;
}
.btn:hover { background: #34343b; }
.btnPrimary { background: #2b6cb0; border-color: #2b6cb0; }
.btnDanger  { background: #6b1f1f; border-color: #6b1f1f; }
```

- [ ] **Step 4: Commit**

```bash
git add src/renderer/onboarding/BundledPanel.tsx src/renderer/onboarding/OptionalPanel.tsx src/renderer/onboarding/styles.module.css
git commit -m "feat(onboarding): bundled + optional panels

BundledPanel renders a green/red-bordered status list. OptionalPanel
collapses install commands behind a per-row disclosure, showing OS,
package manager label, and a copy-paste command line."
```

### Task 23: ModelsPanel with download controls + progress

**Files:**
- Create: `src/renderer/onboarding/ModelsPanel.tsx`

- [ ] **Step 1: Implement ModelsPanel.tsx**

Create `src/renderer/onboarding/ModelsPanel.tsx`:

```tsx
import React from 'react';
import { cancelModelDownload, deleteModel, startModelDownload, type DepStatus } from './api';
import styles from './styles.module.css';

interface Props { items: DepStatus[]; onChange: () => void; }

export const ModelsPanel: React.FC<Props> = ({ items, onChange }) => {
  const models = items.filter((i) => i.category === 'model');
  return (
    <section className={styles.panel}>
      <header>
        <h2>AI models</h2>
        <p>Download what you want; you can come back anytime.</p>
      </header>
      <ul className={styles.list}>
        {models.map((m) => <ModelRow key={m.id} item={m} onChange={onChange} />)}
      </ul>
    </section>
  );
};

const fmtSize = (n?: number): string => {
  if (!n) return '';
  const mb = n / 1024 / 1024;
  if (mb < 1024) return `${mb.toFixed(0)} MB`;
  return `${(mb / 1024).toFixed(2)} GB`;
};

const ModelRow: React.FC<{ item: DepStatus; onChange: () => void }> = ({ item, onChange }) => {
  const inst = item.detail || {};
  const done: number = inst.bytes_done ?? 0;
  const total: number = inst.bytes_total ?? item.size_bytes ?? 0;
  const pct = total > 0 ? Math.min(100, Math.round((done / total) * 100)) : 0;

  const onDownload = async () => { await startModelDownload(item.id); onChange(); };
  const onCancel   = async () => { await cancelModelDownload(item.id); onChange(); };
  const onDelete   = async () => { await deleteModel(item.id); onChange(); };

  return (
    <li className={styles[item.state] || styles.row}>
      <div className={styles.head}>
        <span className={styles.icon} aria-hidden>{stateIcon(item.state)}</span>
        <span className={styles.name}>{item.name} <span className={styles.version}>{fmtSize(item.size_bytes)}</span></span>
        <div className={styles.actions}>
          {item.state === 'installed' && (
            <button type="button" className={`${styles.btn} ${styles.btnDanger}`} onClick={onDelete}>Delete</button>
          )}
          {(item.state === 'missing' || item.state === 'failed' || item.state === 'cancelled') && (
            <button type="button" className={`${styles.btn} ${styles.btnPrimary}`} onClick={onDownload}>Download</button>
          )}
          {(item.state === 'downloading' || item.state === 'queued' || item.state === 'verifying') && (
            <button type="button" className={styles.btn} onClick={onCancel}>Cancel</button>
          )}
        </div>
      </div>
      {(item.state === 'downloading' || item.state === 'queued' || item.state === 'verifying') && total > 0 && (
        <div className={styles.detail}>
          <div className={styles.progressBar}><div className={styles.progressBarFill} style={{ width: `${pct}%` }} /></div>
          <small>{fmtSize(done)} / {fmtSize(total)} ({pct}%) {inst.current_file && `· ${inst.current_file}`}</small>
        </div>
      )}
      {item.error && <div className={styles.error}>{item.error}</div>}
    </li>
  );
};

function stateIcon(s: string): string {
  switch (s) {
    case 'installed': return '✓';
    case 'downloading':
    case 'queued':
    case 'verifying': return '↓';
    case 'failed': return '✗';
    default: return '○';
  }
}
```

- [ ] **Step 2: Commit**

```bash
git add src/renderer/onboarding/ModelsPanel.tsx
git commit -m "feat(onboarding): ModelsPanel with progress + controls

Per-model card showing size, status, and a Download/Cancel/Delete
button set scoped to the current state. Progress bar appears during
download/verify; error text appears on failure."
```

### Task 24: OnboardingWizard + route registration

**Files:**
- Create: `src/renderer/onboarding/OnboardingWizard.tsx`
- Create: `src/renderer/onboarding/__tests__/OnboardingWizard.test.tsx`
- Modify: whatever file declares the SPA's routes (search `src/renderer/` for the existing router pattern)
- Create: `media-server/handlers/onboarding_handlers.go`

- [ ] **Step 1: Implement the wizard component**

Create `src/renderer/onboarding/OnboardingWizard.tsx`:

```tsx
import React, { useState } from 'react';
import { BundledPanel } from './BundledPanel';
import { ModelsPanel } from './ModelsPanel';
import { OptionalPanel } from './OptionalPanel';
import { useDepsStatus } from './useDepsStatus';
import styles from './styles.module.css';

interface Props {
  // Called when the user dismisses (skips or completes). Parent persists.
  onDismiss?: () => void;
  // Optional: render mode without the wizard chrome (skip button, headings).
  // Used by /settings/dependencies which embeds the same panels.
  embedded?: boolean;
}

export const OnboardingWizard: React.FC<Props> = ({ onDismiss, embedded }) => {
  const { status, refresh, error } = useDepsStatus();
  const [step, setStep] = useState(0);
  const steps = [
    { title: 'Welcome', render: () => <BundledPanel items={status} /> },
    { title: 'Optional tools', render: () => <OptionalPanel items={status} /> },
    { title: 'AI models', render: () => <ModelsPanel items={status} onChange={refresh} /> },
  ];
  const last = step === steps.length - 1;

  if (embedded) {
    return (
      <div>
        {error && <div className={styles.error}>Status error: {error}</div>}
        <BundledPanel items={status} />
        <OptionalPanel items={status} />
        <ModelsPanel items={status} onChange={refresh} />
      </div>
    );
  }

  return (
    <div className={styles.panel}>
      <header>
        <h1>Welcome to Lowkey Media Server</h1>
        <p>Step {step + 1} of {steps.length}: {steps[step].title}</p>
      </header>
      {error && <div className={styles.error}>Status error: {error}</div>}
      {steps[step].render()}
      <div className={styles.actions} style={{ marginTop: '1.5rem' }}>
        <button type="button" className={styles.btn} onClick={onDismiss}>Skip — I'll do this later</button>
        {step > 0 && <button type="button" className={styles.btn} onClick={() => setStep(step - 1)}>Back</button>}
        {!last && <button type="button" className={`${styles.btn} ${styles.btnPrimary}`} onClick={() => setStep(step + 1)}>Next</button>}
        {last && <button type="button" className={`${styles.btn} ${styles.btnPrimary}`} onClick={onDismiss}>Finish</button>}
      </div>
    </div>
  );
};
```

- [ ] **Step 2: Wire up the server side of onboarding-state**

Create `media-server/handlers/onboarding_handlers.go`:

```go
package handlers

import (
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/stevecastle/shrike/platform"
)

type onboardingState struct {
	Shown       bool       `json:"shown"`
	DismissedAt *time.Time `json:"dismissed_at,omitempty"`
}

var onbMu sync.Mutex

func onboardingPath() string {
	return filepath.Join(platform.GetDataDir(), "onboarding.json")
}

// GET /api/onboarding/state
func HandleOnboardingGet(w http.ResponseWriter, _ *http.Request) {
	onbMu.Lock()
	defer onbMu.Unlock()
	b, err := os.ReadFile(onboardingPath())
	if err != nil {
		writeJSON(w, http.StatusOK, onboardingState{Shown: false})
		return
	}
	var s onboardingState
	if err := json.Unmarshal(b, &s); err != nil {
		writeJSON(w, http.StatusOK, onboardingState{Shown: false})
		return
	}
	writeJSON(w, http.StatusOK, s)
}

// POST /api/onboarding/dismiss
func HandleOnboardingDismiss(w http.ResponseWriter, _ *http.Request) {
	onbMu.Lock()
	defer onbMu.Unlock()
	now := time.Now().UTC()
	s := onboardingState{Shown: true, DismissedAt: &now}
	b, _ := json.MarshalIndent(s, "", "  ")
	if err := os.MkdirAll(filepath.Dir(onboardingPath()), 0o755); err != nil {
		writeErr(w, http.StatusInternalServerError, err)
		return
	}
	if err := os.WriteFile(onboardingPath(), b, 0o644); err != nil {
		writeErr(w, http.StatusInternalServerError, err)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// POST /api/onboarding/reset
func HandleOnboardingReset(w http.ResponseWriter, _ *http.Request) {
	onbMu.Lock()
	defer onbMu.Unlock()
	_ = os.Remove(onboardingPath())
	w.WriteHeader(http.StatusNoContent)
}
```

- [ ] **Step 3: Add the routes**

Edit `media-server/routes_deps.go`, append to `RegisterDepsRoutes`:

```go
	mux.HandleFunc("GET /api/onboarding/state",     handlers.HandleOnboardingGet)
	mux.HandleFunc("POST /api/onboarding/dismiss",  handlers.HandleOnboardingDismiss)
	mux.HandleFunc("POST /api/onboarding/reset",    handlers.HandleOnboardingReset)
```

- [ ] **Step 4: Mount the wizard in the SPA**

Locate the SPA router (e.g. `src/renderer/main.tsx` or wherever React Router or the equivalent is configured). Add a route for `/onboarding` rendering `<OnboardingWizard onDismiss={...} />` and `/settings/dependencies` rendering `<OnboardingWizard embedded />`.

In the top-level layout, on first render, fetch `/api/onboarding/state`. If `shown === false`, navigate to `/onboarding`. The `onDismiss` callback calls `POST /api/onboarding/dismiss` then navigates to `/`.

Exact wiring depends on the existing SPA structure — investigate `src/renderer/main.tsx` and the existing route registration before coding. If the SPA does not have a router yet, mount the wizard at the top of the root component conditionally based on the fetched state.

- [ ] **Step 5: Write a basic component test**

Create `src/renderer/onboarding/__tests__/OnboardingWizard.test.tsx`:

```tsx
import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { OnboardingWizard } from '../OnboardingWizard';

beforeEach(() => {
  // @ts-ignore
  global.fetch = jest.fn(async (url: string) => {
    if (url === '/api/deps/status') {
      return new Response(JSON.stringify([
        { id: 'ffmpeg', category: 'bundled', name: 'FFmpeg', state: 'ready', version: '7.1' },
        { id: 'yt-dlp', category: 'optional', name: 'yt-dlp', state: 'not_installed', detail: { Commands: [{ os: 'darwin', label: 'brew', command: 'brew install yt-dlp' }] } },
        { id: 'wd-eva02-large-tagger-v3', category: 'model', name: 'WD EVA02', state: 'missing', size_bytes: 1257385984 },
      ]), { status: 200 });
    }
    return new Response('', { status: 200 });
  });
  // @ts-ignore — minimal EventSource stub
  global.EventSource = class { constructor() {} onmessage: any; onerror: any; close() {} };
});

afterEach(() => { jest.resetAllMocks(); });

test('renders bundled status on first step', async () => {
  render(<OnboardingWizard />);
  await waitFor(() => screen.getByText(/FFmpeg/));
  expect(screen.getByText(/7.1/)).toBeInTheDocument();
});
```

- [ ] **Step 6: Build and test the SPA**

```bash
cd C:\Users\steph\dev\loki
yarn build:web
npm test -- --testPathPattern=onboarding
```

Expected: build succeeds; test passes.

- [ ] **Step 7: Commit**

```bash
git add src/renderer/onboarding/OnboardingWizard.tsx src/renderer/onboarding/__tests__ media-server/handlers/onboarding_handlers.go media-server/routes_deps.go <spa-router-file>
git commit -m "feat(onboarding): wizard component + onboarding state API

Three skippable steps (bundled / optional / models) sharing the same
panel components used by /settings/dependencies in embedded mode.
Server tracks first-run dismissal in <dataDir>/onboarding.json."
```

---

## Phase 9: Integration verification

### Task 25: Rebuild SPA into media-server, full end-to-end smoke

The Go server embeds `loki-static/**` at compile time. After SPA changes, the renderer output must be copied into `media-server/loki-static/` before the binary will serve the new wizard.

- [ ] **Step 1: Rebuild SPA into server**

Run: `npm run build:server`
Expected: success; `media-server/loki-static/` updated; server binary rebuilt.

- [ ] **Step 2: Fetch bundled deps locally for your dev OS**

Pick the target matching your dev machine:

```bash
# macOS arm64 example:
media-server/scripts/fetch-bundled-deps.sh darwin-arm64 verify
mkdir -p media-server/bin
cp -R media-server/bin/darwin-arm64/* media-server/bin/
```

(or on Windows, run the .ps1 with target windows-amd64; copy files from media-server\bin\windows-amd64\ into media-server\bin\)

- [ ] **Step 3: Start the server and walk the wizard**

```bash
cd media-server && ./lowkeymediaserver &
sleep 2
open http://127.0.0.1:8060  # or xdg-open / start, per OS
```

Verify in the browser:
1. The wizard appears on first load.
2. Step 1 lists all bundled deps as ready (or correctly marks any missing).
3. Step 2 shows yt-dlp / gallery-dl / ollama. Toggle "Show install commands" — verify the right per-OS command appears.
4. Step 3 shows the WD tagger. Click "Download." Watch the progress bar fill via SSE. (If you don't want to actually download 1.2GB, click Cancel — verify it returns to the missing state.)
5. Skip/Finish — the wizard should not reappear on next page load.
6. Visit `/settings/dependencies` — the same content should render embedded (no wizard chrome).

- [ ] **Step 4: macOS-specific verification (if you have access to a Mac)**

This is the bug the user originally reported. Verify:

```bash
xattr /Applications/path/to/lowkeymediaserver/bin/ffmpeg
# Expected: no com.apple.quarantine entry after the first boot.

/Applications/.../lowkeymediaserver/bin/ffmpeg -version
# Expected: prints ffmpeg version info, not "killed" / "cannot be opened".
```

If quarantine is still present, check the server log for "deps: stripped quarantine from …" entries — and confirm `quarantine_darwin.go` is being compiled (build tag must be `//go:build darwin`).

- [ ] **Step 5: Run the full test suite one more time**

```bash
cd media-server && go test ./...
cd C:\Users\steph\dev\loki && npm test
```

Expected: all PASS.

- [ ] **Step 6: Final commit (only if anything actually changed)**

```bash
git status
# If anything was tweaked during smoke testing:
git add -A
git commit -m "chore: post-smoke fixups"
```

### Task 26: Update the project README

**Files:** `media-server/README.md`

- [ ] **Step 1: Replace the "Dependencies" section**

Open `media-server/README.md` and find the section describing the old `/dependencies` UI and Dependencies-download flow. Replace it with:

```markdown
## Dependencies

The server ships with everything it needs to run out of the box.

- **Bundled:** `ffmpeg`, `ffprobe`, `ffplay` (not on macOS), `exiftool`, `onnxtag`, `onnxruntime` are included in the release archive under `bin/`. No setup required.
- **Optional:** `yt-dlp`, `gallery-dl`, and `ollama` enable additional features. Install via your OS package manager — the server detects them on `PATH` and shows install commands in the welcome wizard.
- **AI models:** Downloaded on demand from the `/onboarding` or `/settings/dependencies` page. Manifest, SHA-256 checksums, atomic install, and resume support are built in.

The welcome wizard appears on first run and is reachable any time at `/onboarding`. Skipping it does not break anything.
```

- [ ] **Step 2: Commit**

```bash
git add media-server/README.md
git commit -m "docs: README — new dependency model

Describes the bundled / optional / models split that replaces the
old on-demand downloader."
```

---

## Self-review checklist

After implementing every task, verify:

- [ ] `go test ./...` passes across all three platforms (`go build` for darwin/linux from the dev OS).
- [ ] `npm test` passes for the SPA.
- [ ] `/api/deps/status` returns entries for every manifest item in all three categories.
- [ ] On macOS, the `com.apple.quarantine` xattr is absent after boot.
- [ ] The legacy `/dependencies`, `/dependencies/download`, `/setup` routes return 404.
- [ ] The legacy `dependencies.json` is renamed to `.bak` on first boot.
- [ ] No file in the repository imports `github.com/stevecastle/shrike/downloads`.
- [ ] No file in the repository references the deleted `deps.Dependency` / `deps.Register` / `deps.GetMetadataStore` symbols.
- [ ] The release archive built by `release-server.yml` contains both the server binary and a populated `bin/` directory.

End of plan.

