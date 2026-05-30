# Custom Description Prompt Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users optionally type a custom prompt in the React metadata
component that replaces `appconfig.DescribePrompt` for that single description-
generation job. Empty/absent prompt → uses the configured default exactly as
today.

**Architecture:**
- Extend `POST /create` with an optional `fields` map; the server appends each
  entry as `--<key> <value>` to the job's `Arguments` slice. The `metadata`
  task gains a new `prompt` `TaskOption` and threads `customPrompt` through to
  `callOllamaVision`, which falls back to the config default when empty.
- Add `GET /api/prompts/describe` so the React panel can display the current
  default as a placeholder.
- React `GenerateDescription` gains a collapsible "Custom prompt" panel; a
  small module-level singleton remembers the last typed value for the session.

**Tech Stack:** Go 1.x stdlib (`net/http`, `httptest`), React + TypeScript,
Jest with jsdom + ts-jest.

**Spec:** `docs/superpowers/specs/2026-05-30-custom-description-prompt-design.md`

---

## File Structure

**New files:**
- `media-server/createjob_args.go` — `appendFieldArgs` helper (no build tag).
- `media-server/createjob_args_test.go` — unit tests for the helper.
- `media-server/handlers_prompts.go` — `describePromptHandler` (no build tag).
- `media-server/handlers_prompts_test.go` — handler test using `httptest`.
- `media-server/tasks/metadata_ops_test.go` — `resolveDescribePrompt` test.
- `src/renderer/components/metadata/customPromptStore.ts` — module-level
  session memory for the last typed prompt and cached default fetch.
- `src/__tests__/customPromptStore.test.ts` — unit tests for the store.

**Modified files:**
- `media-server/main.go` (`//go:build windows`): `CreateJobHandlerRequest`
  gains `Fields`; `createJobHandler` calls `appendFieldArgs`; mux registers
  `/api/prompts/describe`.
- `media-server/main_darwin.go` and `media-server/main_linux.go`: same three
  changes, mirrored.
- `media-server/tasks/options.go` — **unchanged**, used as-is.
- `media-server/tasks/media_metadata.go` — new `prompt` option in
  `metadataOptions`; pass `customPrompt` to `processDescriptionForFile`.
- `media-server/tasks/metadata_ops.go` — `processDescriptionForFile`,
  `describeFileWithOllama`, `callOllamaVision` each take a new
  `customPrompt string` parameter; new `resolveDescribePrompt` helper.
- `media-server/tasks/lora_dataset.go` — pass `""` at the
  `describeFileWithOllama` call site.
- `media-server/tasks/options_test.go` — add a case for `--prompt`.
- `src/renderer/components/metadata/generate-description.tsx` — render panel,
  fetch default, read/write `customPromptStore`, include `fields.prompt` in
  POST.
- `src/renderer/components/metadata/generate-description.css` — styles for the
  toggle link, panel, textarea.

---

## Task 1: `appendFieldArgs` helper (Go, TDD)

**Files:**
- Create: `media-server/createjob_args.go`
- Create: `media-server/createjob_args_test.go`

- [ ] **Step 1: Write the failing tests**

Create `media-server/createjob_args_test.go`:

```go
package main

import (
	"reflect"
	"testing"
)

func TestAppendFieldArgs_NilFields(t *testing.T) {
	args := []string{"--type", "description"}
	got := appendFieldArgs(args, nil)
	want := []string{"--type", "description"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestAppendFieldArgs_EmptyFields(t *testing.T) {
	args := []string{"--type", "description"}
	got := appendFieldArgs(args, map[string]string{})
	want := []string{"--type", "description"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestAppendFieldArgs_SingleField(t *testing.T) {
	args := []string{"--type", "description"}
	got := appendFieldArgs(args, map[string]string{"prompt": "describe this"})
	want := []string{"--type", "description", "--prompt", "describe this"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestAppendFieldArgs_SkipsEmptyValue(t *testing.T) {
	args := []string{"--type", "description"}
	got := appendFieldArgs(args, map[string]string{"prompt": ""})
	want := []string{"--type", "description"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestAppendFieldArgs_SkipsEmptyKey(t *testing.T) {
	args := []string{"--type", "description"}
	got := appendFieldArgs(args, map[string]string{"": "ignored"})
	want := []string{"--type", "description"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestAppendFieldArgs_PreservesArbitraryText(t *testing.T) {
	// Quotes, newlines, equals signs must pass through unchanged because
	// the helper appends them as separate slice elements (not a shell string).
	args := []string{"--type", "description"}
	value := `weird "quoted" text with
newlines and = signs`
	got := appendFieldArgs(args, map[string]string{"prompt": value})
	want := []string{"--type", "description", "--prompt", value}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run from the repo root:

```
cd media-server && go test -run TestAppendFieldArgs -count=1 .
```

Expected: build failure — `undefined: appendFieldArgs`.

- [ ] **Step 3: Write the helper**

Create `media-server/createjob_args.go`:

```go
package main

// appendFieldArgs appends each non-empty key/value pair from fields onto args
// as two slice elements: "--<key>" then "<value>". The values are passed
// through verbatim (no quoting, no escaping) because ParseOptions consumes
// the resulting slice directly — never via the shell-style ParseCommand
// splitter. Callers can therefore safely include arbitrary text such as
// embedded quotes or newlines in field values.
func appendFieldArgs(args []string, fields map[string]string) []string {
	if len(fields) == 0 {
		return args
	}
	for k, v := range fields {
		if k == "" || v == "" {
			continue
		}
		args = append(args, "--"+k, v)
	}
	return args
}
```

- [ ] **Step 4: Run tests to verify they pass**

```
cd media-server && go test -run TestAppendFieldArgs -count=1 .
```

Expected: all six tests PASS.

- [ ] **Step 5: Commit**

```
git add media-server/createjob_args.go media-server/createjob_args_test.go
git commit -m "feat(media-server): appendFieldArgs helper for /create"
```

---

## Task 2: Wire `Fields` into the `/create` handler (Go, three platform files)

**Files:**
- Modify: `media-server/main.go:381-410` (`CreateJobHandlerRequest`, `createJobHandler`)
- Modify: `media-server/main_darwin.go` (same struct + handler)
- Modify: `media-server/main_linux.go` (same struct + handler)

This task is mechanical wiring across all three platform-tagged copies of the
handler. The behavior is covered by `appendFieldArgs` tests from Task 1 — no
new tests in this task. Hand-verification happens in Task 7.

- [ ] **Step 1: Locate the struct and handler in each file**

In each of `main.go`, `main_darwin.go`, `main_linux.go`, find the block:

```go
type CreateJobHandlerRequest struct {
	Input string `json:"input"`
}

func createJobHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Use POST", http.StatusMethodNotAllowed)
			return
		}

		var req CreateJobHandlerRequest
		if err := readJSONBody(r, &req); err != nil {
			http.Error(w, "bad json", http.StatusBadRequest)
			return
		}

		args := ParseCommand(req.Input)
		if len(args) == 0 {
			http.Error(w, "Invalid input", http.StatusBadRequest)
			return
		}

		cmd, input := args[0], ""
		if len(args) > 1 {
			input = args[len(args)-1]
			args = args[1 : len(args)-1]
		} else {
			args = nil
		}
		// ...
```

- [ ] **Step 2: Modify the struct (in all three files)**

Change `CreateJobHandlerRequest` to:

```go
type CreateJobHandlerRequest struct {
	Input  string            `json:"input"`
	Fields map[string]string `json:"fields,omitempty"`
}
```

- [ ] **Step 3: Modify the handler (in all three files)**

After the `cmd, input := args[0], ""` block (which finishes with the
`args = args[1 : len(args)-1]` slicing or `args = nil`), insert a single line
that merges fields into args:

```go
		cmd, input := args[0], ""
		if len(args) > 1 {
			input = args[len(args)-1]
			args = args[1 : len(args)-1]
		} else {
			args = nil
		}

		args = appendFieldArgs(args, req.Fields)

		id, err := deps.Queue.AddWorkflow(jobqueue.Workflow{
			Tasks: []jobqueue.WorkflowTask{
				{
					Command:   cmd,
					Arguments: args,
					Input:     input,
				},
			},
		})
```

Apply this change to all three platform files. The line is identical because
`appendFieldArgs` is in a no-build-tag file.

- [ ] **Step 4: Build the active-platform binary to catch typos**

On Windows:

```
cd media-server && go build .
```

Expected: build succeeds. (On macOS / Linux dev boxes, the equivalent build
covers `main_darwin.go` / `main_linux.go` instead — the developer typically
only builds their own platform; CI covers the others.)

- [ ] **Step 5: Re-run the helper tests**

```
cd media-server && go test -run TestAppendFieldArgs -count=1 .
```

Expected: still PASS.

- [ ] **Step 6: Commit**

```
git add media-server/main.go media-server/main_darwin.go media-server/main_linux.go
git commit -m "feat(media-server): accept optional fields map on /create"
```

---

## Task 3: Add `prompt` option to the `metadata` task (Go, TDD)

**Files:**
- Modify: `media-server/tasks/options_test.go` (add a case)
- Modify: `media-server/tasks/media_metadata.go:14-19` (add option, read it,
  pass it through)
- Modify: `media-server/tasks/metadata_ops.go` (params, helper)
- Modify: `media-server/tasks/lora_dataset.go:210` (pass `""`)
- Create: `media-server/tasks/metadata_ops_test.go` (helper test)

- [ ] **Step 1: Write the ParseOptions test for `--prompt`**

Append to `media-server/tasks/options_test.go`:

```go
func TestParseOptionsPromptValue(t *testing.T) {
	options := []TaskOption{
		{Name: "prompt", Type: "string"},
	}
	value := `describe this "thing"
on two lines`
	j := &jobqueue.Job{Arguments: []string{"--prompt", value}}
	result := ParseOptions(j, options)

	if result["prompt"] != value {
		t.Errorf("prompt: got %q, want %q", result["prompt"], value)
	}
}
```

(Note: `ParseOptions` already handles `--key value` for string options. This
test just locks the expected behavior in for arbitrary multi-line text values
that the new field carries.)

- [ ] **Step 2: Write the failing test for `resolveDescribePrompt`**

Create `media-server/tasks/metadata_ops_test.go`:

```go
package tasks

import (
	"testing"

	"github.com/stevecastle/shrike/appconfig"
)

func TestResolveDescribePromptUsesCustomWhenProvided(t *testing.T) {
	got := resolveDescribePrompt("custom override")
	if got != "custom override" {
		t.Errorf("got %q, want %q", got, "custom override")
	}
}

func TestResolveDescribePromptTrimsWhitespace(t *testing.T) {
	got := resolveDescribePrompt("   custom   ")
	if got != "custom" {
		t.Errorf("got %q, want %q", got, "custom")
	}
}

func TestResolveDescribePromptFallsBackToConfigWhenEmpty(t *testing.T) {
	old := appconfig.Get()
	cfg := old
	cfg.DescribePrompt = "DEFAULT-FROM-CONFIG"
	appconfig.Set(cfg)
	t.Cleanup(func() { appconfig.Set(old) })

	got := resolveDescribePrompt("")
	if got != "DEFAULT-FROM-CONFIG" {
		t.Errorf("got %q, want %q", got, "DEFAULT-FROM-CONFIG")
	}
}

func TestResolveDescribePromptFallsBackOnAllWhitespace(t *testing.T) {
	old := appconfig.Get()
	cfg := old
	cfg.DescribePrompt = "DEFAULT-FROM-CONFIG"
	appconfig.Set(cfg)
	t.Cleanup(func() { appconfig.Set(old) })

	got := resolveDescribePrompt("   \n\t  ")
	if got != "DEFAULT-FROM-CONFIG" {
		t.Errorf("got %q, want %q", got, "DEFAULT-FROM-CONFIG")
	}
}
```

- [ ] **Step 3: Confirm `appconfig.Set` exists; add it if not**

Run:

```
cd media-server && grep -n "func Set(" appconfig/config.go
```

If `func Set(c Config)` already exists, skip the next sub-step. If not, add it
to `media-server/appconfig/config.go` (just below `func Get()`):

```go
// Set replaces the in-memory config snapshot. Intended for tests that need
// to swap a specific field temporarily — production code paths use Load /
// Save, which also persist to disk.
func Set(c Config) {
	cfgMu.Lock()
	defer cfgMu.Unlock()
	cfg = c
}
```

If `Set` is missing, also confirm `Get` already locks `cfgMu` — if it doesn't,
do not add locking in this PR (out of scope; the existing helper is assumed
correct).

- [ ] **Step 4: Run the new tests to verify they fail**

```
cd media-server && go test -run "TestParseOptionsPromptValue|TestResolveDescribePrompt" -count=1 ./tasks/...
```

Expected: build failure — `undefined: resolveDescribePrompt`. The
`TestParseOptionsPromptValue` test should compile cleanly; it will be
overshadowed by the build failure but will run after the helper exists.

- [ ] **Step 5: Add the `prompt` option to the metadata task**

In `media-server/tasks/media_metadata.go`, modify `metadataOptions` (lines 14-19):

```go
var metadataOptions = []TaskOption{
	{Name: "type", Label: "Metadata Types", Type: "multi-enum", Choices: []string{"description", "transcript", "hash", "dimensions", "autotag"}, Default: "description,hash,dimensions", Description: "Comma-separated list of metadata types to generate"},
	{Name: "overwrite", Label: "Overwrite Existing", Type: "bool", Description: "Overwrite existing metadata values"},
	{Name: "apply", Label: "Apply Scope", Type: "enum", Choices: []string{"new", "all"}, Default: "new", Description: "Apply to new items only or all items"},
	{Name: "model", Label: "Ollama Model", Type: "string", Description: "Ollama model to use for descriptions"},
	{Name: "prompt", Label: "Custom Description Prompt", Type: "string", Description: "Override the configured describe prompt for this run"},
}
```

In the same file, inside `metadataTask`, just after the existing
`ollamaModel, _ := opts["model"].(string)` block, read the new option:

```go
	ollamaModel, _ := opts["model"].(string)
	if ollamaModel == "" {
		ollamaModel = appconfig.Get().OllamaModel
	}
	customDescribePrompt, _ := opts["prompt"].(string)
```

Then update the `switch mType` in the per-file loop to pass it through:

```go
			case "description":
				opErr = processDescriptionForFile(ctx, q, j.ID, filePath, overwrite, ollamaModel, customDescribePrompt, fromQuery)
```

- [ ] **Step 6: Thread `customPrompt` through `metadata_ops.go`**

In `media-server/tasks/metadata_ops.go`, change three signatures and add one
helper. (Line numbers below match the current HEAD; verify by reading the
file.)

a) Change `processDescriptionForFile` (around line 726):

```go
func processDescriptionForFile(ctx context.Context, q *jobqueue.Queue, jobID string, filePath string, overwrite bool, model string, customPrompt string, fromQuery bool) error {
	if !fromQuery {
		exists, err := fileExistsInDatabase(q.Db, filePath)
		if err != nil {
			return fmt.Errorf("error checking database: %w", err)
		}
		if !exists {
			return nil
		}
	}

	if !overwrite {
		hasDescription, err := hasExistingMetadata(q.Db, filePath, "description")
		if err != nil {
			return fmt.Errorf("error checking existing description: %w", err)
		}
		if hasDescription {
			return nil
		}
	}

	description, err := describeFileWithOllama(ctx, filePath, model, customPrompt)
	if err != nil {
		return fmt.Errorf("failed to describe: %w", err)
	}
	if err := updateMediaMetadata(q.Db, filePath, "description", description); err != nil {
		return fmt.Errorf("failed to update: %w", err)
	}
	q.PushJobStdout(jobID, fmt.Sprintf("  description: generated"))
	return nil
}
```

b) Change `describeFileWithOllama` (around line 441):

```go
func describeFileWithOllama(ctx context.Context, mediaPath, model, customPrompt string) (string, error) {
	ext := strings.ToLower(filepath.Ext(mediaPath))
	var tempImagePath string
	var cleanupPaths []string
	if ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".webp" {
		tempImagePath = mediaPath
	} else {
		screenshotPath, err := extractVideoFrame(ctx, mediaPath, "")
		if err != nil {
			return "", fmt.Errorf("failed to extract video frame: %w", err)
		}
		cleanupPaths = append(cleanupPaths, screenshotPath)
		tempImagePath = screenshotPath
	}
	resizedPath, err := resizeImageIfNeeded(tempImagePath)
	if err != nil {
		for _, p := range cleanupPaths {
			_ = os.Remove(p)
		}
		return "", fmt.Errorf("failed to resize image: %w", err)
	}
	if resizedPath != tempImagePath {
		cleanupPaths = append(cleanupPaths, resizedPath)
	}
	description, err := callOllamaVision(ctx, resizedPath, model, customPrompt)
	if err != nil {
		for _, p := range cleanupPaths {
			_ = os.Remove(p)
		}
		return "", fmt.Errorf("ollama call failed: %w", err)
	}
	for _, p := range cleanupPaths {
		_ = os.Remove(p)
	}
	return description, nil
}
```

c) Change `callOllamaVision` (around line 535) and add `resolveDescribePrompt`
just above it:

```go
// resolveDescribePrompt returns the custom prompt when it has non-whitespace
// content, otherwise falls back to the prompt stored in app config. Extracted
// so the fallback rule can be unit-tested without a live vision backend.
func resolveDescribePrompt(custom string) string {
	if trimmed := strings.TrimSpace(custom); trimmed != "" {
		return trimmed
	}
	return appconfig.Get().DescribePrompt
}

// callOllamaVision routes to either RunPod or the local Ollama HTTP API for
// image description. The 10-minute deadline preserves the upper bound that
// used to live on the per-request http.Client. If customPrompt is non-empty
// (after trimming) it replaces the configured DescribePrompt for this call
// only.
func callOllamaVision(ctx context.Context, imagePath, _ string, customPrompt string) (string, error) {
	timeoutCtx, cancel := context.WithTimeout(ctx, 600*time.Second)
	defer cancel()
	return callVisionLLM(timeoutCtx, imagePath, resolveDescribePrompt(customPrompt))
}
```

d) Update the dead-but-still-compiled `generateDescriptions` call site (around
line 69) to pass `""`:

```go
		description, err := describeFileWithOllama(ctx, filePath, model, "")
```

- [ ] **Step 7: Update the LoRA caller**

In `media-server/tasks/lora_dataset.go:210`:

```go
	generatedDesc, err := describeFileWithOllama(ctx, filePath, model, "")
```

- [ ] **Step 8: Run the new tests to verify they pass**

```
cd media-server && go test -run "TestParseOptionsPromptValue|TestResolveDescribePrompt" -count=1 ./tasks/...
```

Expected: all four `TestResolveDescribePrompt*` tests plus
`TestParseOptionsPromptValue` PASS.

- [ ] **Step 9: Run the full tasks test suite to catch regressions**

```
cd media-server && go test -count=1 ./tasks/...
```

Expected: all tests PASS. (Several tests in this package exercise registry,
parsing, and host resolution — they should be unaffected.)

- [ ] **Step 10: Commit**

```
git add media-server/tasks/media_metadata.go media-server/tasks/metadata_ops.go media-server/tasks/lora_dataset.go media-server/tasks/options_test.go media-server/tasks/metadata_ops_test.go
# If appconfig.Set was added in Step 3:
git add media-server/appconfig/config.go
git commit -m "feat(media-server): per-job describe prompt override on metadata task"
```

---

## Task 4: `GET /api/prompts/describe` endpoint (Go, TDD)

**Files:**
- Create: `media-server/handlers_prompts.go` (no build tag)
- Create: `media-server/handlers_prompts_test.go` (no build tag)
- Modify: `media-server/main.go` (mux registration only)
- Modify: `media-server/main_darwin.go` (mux registration only)
- Modify: `media-server/main_linux.go` (mux registration only)

- [ ] **Step 1: Write the failing test**

Create `media-server/handlers_prompts_test.go`:

```go
package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stevecastle/shrike/appconfig"
)

func TestDescribePromptHandlerReturnsCurrentDefault(t *testing.T) {
	old := appconfig.Get()
	cfg := old
	cfg.DescribePrompt = "the-current-default"
	appconfig.Set(cfg)
	t.Cleanup(func() { appconfig.Set(old) })

	req := httptest.NewRequest(http.MethodGet, "/api/prompts/describe", nil)
	rec := httptest.NewRecorder()

	describePromptHandler(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	if ct := rec.Header().Get("Content-Type"); ct != "application/json" {
		t.Errorf("Content-Type = %q, want application/json", ct)
	}

	var body struct {
		Prompt string `json:"prompt"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("response body not valid json: %v", err)
	}
	if body.Prompt != "the-current-default" {
		t.Errorf("prompt = %q, want %q", body.Prompt, "the-current-default")
	}
}

func TestDescribePromptHandlerRejectsNonGet(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/api/prompts/describe", nil)
	rec := httptest.NewRecorder()

	describePromptHandler(rec, req)

	if rec.Code != http.StatusMethodNotAllowed {
		t.Errorf("status = %d, want 405", rec.Code)
	}
}
```

- [ ] **Step 2: Run the test to verify it fails**

```
cd media-server && go test -run TestDescribePromptHandler -count=1 .
```

Expected: build failure — `undefined: describePromptHandler`.

- [ ] **Step 3: Implement the handler**

Create `media-server/handlers_prompts.go`:

```go
package main

import (
	"encoding/json"
	"net/http"

	"github.com/stevecastle/shrike/appconfig"
)

// describePromptHandler returns the currently configured describe prompt as
// JSON. The React metadata UI calls this to render the default text as a
// placeholder inside the optional custom-prompt panel. Read-only and tiny;
// safe to call on every component mount.
func describePromptHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Use GET", http.StatusMethodNotAllowed)
		return
	}
	resp := struct {
		Prompt string `json:"prompt"`
	}{
		Prompt: appconfig.Get().DescribePrompt,
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}
```

- [ ] **Step 4: Run the test to verify it passes**

```
cd media-server && go test -run TestDescribePromptHandler -count=1 .
```

Expected: both tests PASS.

- [ ] **Step 5: Register the route in all three platform files**

In each of `main.go`, `main_darwin.go`, `main_linux.go`, locate the
`mux.HandleFunc("/config", ...)` registration. Just above (or below) it, add:

```go
	mux.HandleFunc("/api/prompts/describe", renderer.ApplyMiddlewares(describePromptHandler, renderer.RoleAdmin))
```

(Use the same `renderer.RoleAdmin` middleware as `/config` so auth behavior is
consistent — the React UI already sends the bearer token for `/create` calls.)

- [ ] **Step 6: Build on the current platform**

```
cd media-server && go build .
```

Expected: build succeeds.

- [ ] **Step 7: Commit**

```
git add media-server/handlers_prompts.go media-server/handlers_prompts_test.go media-server/main.go media-server/main_darwin.go media-server/main_linux.go
git commit -m "feat(media-server): GET /api/prompts/describe returns current describe prompt"
```

---

## Task 5: React `customPromptStore` (TypeScript, TDD)

**Files:**
- Create: `src/renderer/components/metadata/customPromptStore.ts`
- Create: `src/__tests__/customPromptStore.test.ts`

- [ ] **Step 1: Write the failing tests**

Create `src/__tests__/customPromptStore.test.ts`:

```ts
import {
  getLastCustomPrompt,
  setLastCustomPrompt,
  getCachedDefaultPrompt,
  setCachedDefaultPrompt,
  __resetCustomPromptStoreForTests,
} from '../renderer/components/metadata/customPromptStore';

describe('customPromptStore', () => {
  beforeEach(() => {
    __resetCustomPromptStoreForTests();
  });

  it('returns empty string for last prompt by default', () => {
    expect(getLastCustomPrompt()).toBe('');
  });

  it('persists last prompt across reads within the session', () => {
    setLastCustomPrompt('hello world');
    expect(getLastCustomPrompt()).toBe('hello world');
  });

  it('ignores empty / whitespace-only writes to last prompt', () => {
    setLastCustomPrompt('keep me');
    setLastCustomPrompt('');
    setLastCustomPrompt('   ');
    expect(getLastCustomPrompt()).toBe('keep me');
  });

  it('returns null for cached default until set', () => {
    expect(getCachedDefaultPrompt()).toBeNull();
  });

  it('caches the fetched default prompt', () => {
    setCachedDefaultPrompt('default text');
    expect(getCachedDefaultPrompt()).toBe('default text');
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

```
npm run build && npx jest customPromptStore
```

(Jest config lives inline in `package.json`'s `"jest"` key, so no `-c` flag
is needed. The `npm test` script always runs the full build first via
`check-build-exists.ts`; once that build has succeeded at least once in a
session, `npx jest <pattern>` is a faster iteration loop.)

Expected: test fails because the module doesn't exist.

- [ ] **Step 3: Implement the module**

Create `src/renderer/components/metadata/customPromptStore.ts`:

```ts
// Session-only memory for the per-call custom description prompt.
//
// "Session-only" means in-memory for the lifetime of the renderer window —
// cleared on reload / app restart, and never persisted to disk. A simple
// module-level singleton fits the contract; no React state, no XState, no
// localStorage.

let lastCustomPrompt = '';
let cachedDefaultPrompt: string | null = null;

export function getLastCustomPrompt(): string {
  return lastCustomPrompt;
}

export function setLastCustomPrompt(value: string): void {
  // Only remember non-empty submissions so that clearing the textarea
  // doesn't wipe the previously-used prompt.
  if (value.trim() === '') return;
  lastCustomPrompt = value;
}

export function getCachedDefaultPrompt(): string | null {
  return cachedDefaultPrompt;
}

export function setCachedDefaultPrompt(value: string): void {
  cachedDefaultPrompt = value;
}

// Test-only reset. Exported under a deliberately ugly name to discourage
// production callers.
export function __resetCustomPromptStoreForTests(): void {
  lastCustomPrompt = '';
  cachedDefaultPrompt = null;
}
```

- [ ] **Step 4: Run the test to verify it passes**

```
npx jest customPromptStore
```

Expected: all five tests PASS.

- [ ] **Step 5: Commit**

```
git add src/renderer/components/metadata/customPromptStore.ts src/__tests__/customPromptStore.test.ts
git commit -m "feat(renderer): customPromptStore for per-session describe-prompt memory"
```

---

## Task 6: Panel UI inside `GenerateDescription` (React, no automated test)

**Files:**
- Modify: `src/renderer/components/metadata/generate-description.tsx` (full
  rewrite of the file — current contents are short)
- Modify: `src/renderer/components/metadata/generate-description.css` (append)

The component lives behind a heavy XState context and authToken selector, so a
focused unit test would need substantial mock plumbing. The store logic (the
only non-trivial part) is already covered by Task 5. UI behavior is verified
manually in Task 7.

- [ ] **Step 1: Rewrite the component**

Replace the contents of
`src/renderer/components/metadata/generate-description.tsx` with:

```tsx
import { useContext, useEffect, useState } from 'react';
import { useSelector } from '@xstate/react';
import { GlobalStateContext } from '../../state';
import './generate-description.css';
import {
  getCachedDefaultPrompt,
  setCachedDefaultPrompt,
  getLastCustomPrompt,
  setLastCustomPrompt,
} from './customPromptStore';

type Props = {
  path: string;
  label?: string;
  variant?: 'centered' | 'inline';
};

const FALLBACK_PLACEHOLDER =
  'Describe this image, focusing on people, clothing, items, text, and actions.';

export default function GenerateDescription({
  path,
  label,
  variant = 'centered',
}: Props) {
  const { libraryService } = useContext(GlobalStateContext);
  const authToken = useSelector(
    libraryService,
    (state) => state.context.authToken
  );
  const [jobServerAvailable, setJobServerAvailable] = useState<boolean | null>(
    null
  );
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [panelOpen, setPanelOpen] = useState<boolean>(false);
  const [promptDraft, setPromptDraft] = useState<string>(() =>
    getLastCustomPrompt()
  );
  const [defaultPrompt, setDefaultPrompt] = useState<string | null>(() =>
    getCachedDefaultPrompt()
  );

  useEffect(() => {
    const checkJobServer = async () => {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 3000);

        const headers: HeadersInit = {};
        if (authToken) {
          headers['Authorization'] = `Bearer ${authToken}`;
        }

        const response = await fetch('http://localhost:8090/health', {
          method: 'GET',
          headers,
          signal: controller.signal,
        });
        clearTimeout(timeoutId);
        setJobServerAvailable(response.ok);
      } catch (error) {
        setJobServerAvailable(false);
      }
    };

    checkJobServer();
  }, [authToken]);

  // Lazily fetch the default prompt the first time the panel is opened, then
  // cache it module-wide so subsequent renders (and other component instances)
  // hit memory instead of the network.
  useEffect(() => {
    if (!panelOpen) return;
    if (defaultPrompt !== null) return;
    let cancelled = false;
    const load = async () => {
      try {
        const headers: HeadersInit = {};
        if (authToken) {
          headers['Authorization'] = `Bearer ${authToken}`;
        }
        const response = await fetch(
          'http://localhost:8090/api/prompts/describe',
          { headers }
        );
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const body = (await response.json()) as { prompt?: string };
        const prompt = body.prompt ?? FALLBACK_PLACEHOLDER;
        if (cancelled) return;
        setCachedDefaultPrompt(prompt);
        setDefaultPrompt(prompt);
      } catch (error) {
        if (cancelled) return;
        setCachedDefaultPrompt(FALLBACK_PLACEHOLDER);
        setDefaultPrompt(FALLBACK_PLACEHOLDER);
      }
    };
    load();
    return () => {
      cancelled = true;
    };
  }, [panelOpen, defaultPrompt, authToken]);

  const handleGenerateDescription = async () => {
    try {
      setIsSubmitting(true);
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000);
      const headers: HeadersInit = {
        'Content-Type': 'application/json',
      };

      if (authToken) {
        headers['Authorization'] = `Bearer ${authToken}`;
      }

      const trimmed = promptDraft.trim();
      const body: { input: string; fields?: { prompt: string } } = {
        input: `metadata --type description --apply all --overwrite "${path}"`,
      };
      if (trimmed !== '') {
        body.fields = { prompt: trimmed };
        setLastCustomPrompt(trimmed);
      }

      const response = await fetch('http://localhost:8090/create', {
        method: 'POST',
        headers,
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // Let the ToastSystem show job lifecycle
    } catch (error) {
      console.error('Failed to create description job:', error);
      libraryService.send({
        type: 'ADD_TOAST',
        data: {
          type: 'error',
          title: 'Failed to Create Job',
          message: 'Could not communicate with job service',
        },
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  if (jobServerAvailable === null) {
    return (
      <div className={`GenerateDescription ${variant}`}>
        <div className="checking-server">Checking job service...</div>
      </div>
    );
  }

  if (jobServerAvailable === false) {
    return (
      <div className={`GenerateDescription ${variant}`}>
        <div className="server-unavailable">
          <div className="icon">⚠️</div>
          <div className="message">
            <strong>Job Service Required</strong>
            <p>
              To generate descriptions and run other long-running tasks, you
              need to install and run the Lowkey Media Server job service.
            </p>
            <p>
              Start the service at <code>localhost:8090</code> to enable this
              feature.
            </p>
          </div>
        </div>
      </div>
    );
  }

  const placeholder =
    defaultPrompt ?? 'Loading default prompt…';

  return (
    <div className={`GenerateDescription ${variant}`}>
      <div className="generate-row">
        <button
          className="generate"
          onClick={handleGenerateDescription}
          disabled={isSubmitting}
        >
          {label || 'Generate Description'}
        </button>
        <button
          type="button"
          className="prompt-toggle"
          onClick={() => setPanelOpen((v) => !v)}
          aria-expanded={panelOpen}
        >
          {panelOpen ? 'Hide prompt' : 'Customize prompt'}
        </button>
      </div>
      {panelOpen && (
        <div className="prompt-panel">
          <textarea
            className="prompt-textarea"
            value={promptDraft}
            placeholder={placeholder}
            onChange={(e) => setPromptDraft(e.target.value)}
            onKeyDown={(e) => e.stopPropagation()}
            onKeyUp={(e) => e.stopPropagation()}
            rows={4}
          />
          <div className="prompt-hint">
            Leave empty to use the configured default. Uses this prompt for the
            next generation only — your global config is unchanged.
          </div>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Append styles**

Append to `src/renderer/components/metadata/generate-description.css`:

```css
.GenerateDescription {
  flex-direction: column;
  gap: 8px;
}

.GenerateDescription .generate-row {
  display: flex;
  align-items: center;
  gap: 12px;
}

.GenerateDescription.centered .generate-row {
  justify-content: center;
}

.GenerateDescription.inline .generate-row {
  justify-content: flex-start;
}

.GenerateDescription .prompt-toggle {
  background: none;
  border: none;
  color: #888;
  font-size: 11px;
  text-decoration: underline;
  cursor: pointer;
  padding: 2px 4px;
}

.GenerateDescription .prompt-toggle:hover {
  color: #ccc;
}

.GenerateDescription .prompt-panel {
  width: 100%;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.GenerateDescription .prompt-textarea {
  width: 100%;
  min-height: 80px;
  background-color: rgba(255, 255, 255, 0.03);
  color: #ddd;
  border: 1px solid #555;
  border-radius: 4px;
  padding: 8px;
  font-size: 12px;
  font-family: inherit;
  resize: vertical;
  box-sizing: border-box;
}

.GenerateDescription .prompt-textarea:focus {
  outline: none;
  border-color: #888;
}

.GenerateDescription .prompt-textarea::placeholder {
  color: #666;
  font-style: italic;
}

.GenerateDescription .prompt-hint {
  font-size: 10px;
  color: #777;
  line-height: 1.3;
}
```

Note: the new `.GenerateDescription { flex-direction: column; gap: 8px; }`
overrides the existing `display: flex; align-items: center` rule on the same
selector. That's intentional — the row layout now lives on `.generate-row`,
so the outer container stacks vertically when the panel is open.

- [ ] **Step 3: Type-check and lint**

```
yarn lint
```

Expected: no new errors. Pre-existing TypeScript errors documented in
`CLAUDE.md` will still appear in `tsc --noEmit`; do not chase them.

- [ ] **Step 4: Run the full Jest suite to catch regressions**

```
npm test
```

Expected: existing tests still PASS; the new `customPromptStore` test from
Task 5 PASSes; no other tests broken.

- [ ] **Step 5: Commit**

```
git add src/renderer/components/metadata/generate-description.tsx src/renderer/components/metadata/generate-description.css
git commit -m "feat(renderer): custom-prompt panel on GenerateDescription"
```

---

## Task 7: Manual end-to-end verification

**Files:** none — runtime check only.

- [ ] **Step 1: Build the embedded SPA and the server**

```
npm run build:server
```

Expected: builds the renderer, copies output into
`media-server/loki-static/`, then builds `media-server/media-server.exe` (or
the platform equivalent).

- [ ] **Step 2: Start the media server**

```
cd media-server && ./media-server
```

(On Windows: `./media-server.exe`. Pick a free port via env / config if the
default 8090 is taken.)

- [ ] **Step 3: Open the web UI and navigate to a media file with metadata**

In a browser, go to `http://localhost:8090/` and pick any media file whose
metadata panel you can see.

- [ ] **Step 4: Verify the default-path behavior is unchanged**

- Confirm a "Generate Description" button is visible.
- Click it without expanding the panel.
- A job should appear in `http://localhost:8090/jobs`, the job's stdout should
  scroll past "Description: 1 files to process" → "Description 1/1: <name>".
- After it completes, the file's `description` field should be populated, and
  the inference provider's logs (Ollama / RunPod) should show the unchanged
  configured prompt was used.

- [ ] **Step 5: Verify the panel-open custom-prompt path**

- Click "Customize prompt".
- The textarea should appear with the current configured `describePrompt` as
  greyed placeholder text.
- Type a distinctive custom prompt, e.g.
  `Reply with only the word "OVERRIDE" and nothing else.`
- Click "Generate Description".
- Open `http://localhost:8090/jobs`, find the new job, click into it.
- The job's `Arguments` (visible in the detail view or by inspecting the job
  record) should contain `--prompt Reply with only the word "OVERRIDE" and nothing else.`.
- After completion, the file's `description` field should contain whatever
  the provider returned for that custom prompt (typically a literal
  `OVERRIDE` for a vision-capable model, or a refusal / restated phrase).

- [ ] **Step 6: Verify session memory**

- Reload the metadata view (without restarting the server / browser).
- Click "Customize prompt" again.
- The textarea should be pre-filled with the prompt you typed in step 5.

- [ ] **Step 7: Verify the global config was not changed**

- Visit `http://localhost:8090/config`.
- The "Describe prompt" field should still hold the original default text —
  not the custom prompt you typed.

- [ ] **Step 8: Verify quote-safety**

- Open the panel again.
- Type a prompt containing double quotes and a newline:
  ```
  Describe the "main" subject.
  Use one sentence.
  ```
- Generate. The job should run without a 400/500 from `/create` — Task 1's
  helper test covers this code path, but a live confirmation rules out any
  serialization surprise from JSON → handler.

- [ ] **Step 9: If everything passes, no commit needed**

If any step fails, fix the underlying bug (don't paper over with a UI tweak)
and re-run the affected tests + this verification.

---

## Self-Review

Coverage check against the spec:

| Spec section / requirement                                            | Covered by      |
| --------------------------------------------------------------------- | --------------- |
| React: collapsible "Custom prompt" panel                              | Task 6          |
| React: panel in both empty-state and Regenerate contexts              | Task 6 (single component is used in both spots) |
| React: empty textarea, default as placeholder                         | Task 6          |
| React: session-memory of last-used prompt                             | Tasks 5, 6      |
| React: fetch default for placeholder                                  | Task 6 (uses Task 4 endpoint, cached via Task 5) |
| HTTP: optional `fields` map on `/create`                              | Tasks 1, 2      |
| HTTP: `GET /api/prompts/describe`                                     | Task 4          |
| Server task: new `prompt` `TaskOption`                                | Task 3          |
| Server task: thread `customPrompt` through to `callVisionLLM`         | Task 3          |
| Server: fall back to config default when override is empty/whitespace | Task 3 (`resolveDescribePrompt`) |
| `media_ingest.go` continues to use default                            | Task 3 (no change to that file) |
| `lora_dataset.go` continues to use default                            | Task 3 Step 7   |
| No DB schema change                                                   | Confirmed — no migration tasks |
| No XState change                                                      | Confirmed — `customPromptStore` is module-level |
| Quote/newline-safe transport                                          | Task 1 `TestAppendFieldArgs_PreservesArbitraryText` + Task 7 Step 8 |

Placeholder scan: no TBDs, no "implement later", no "add tests for the above"
without code, no "similar to Task N" references.

Type consistency: function signatures match across tasks
(`processDescriptionForFile(... customPrompt string, fromQuery bool)`,
`describeFileWithOllama(... model, customPrompt string)`,
`callOllamaVision(... model string, customPrompt string)`,
`resolveDescribePrompt(custom string) string`,
`appendFieldArgs(args []string, fields map[string]string) []string`,
TS store functions all match between Tasks 5 and 6).
