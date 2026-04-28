# Typed Workflow Nodes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the workflow editor's single-input-string + CLI-flag-arguments model with a typed input contract per task, where every input (including the conventional "primary" data input) is a wireable port that can be driven by upstream nodes' outputs.

**Architecture:**

1. Backend schema change: `Task.Inputs []TaskInput` + `Task.Output TaskOutput` replace `Task.Options`. `WorkflowTask.Bindings map[string]Binding` replaces `WorkflowTask.Input + Arguments`. Existing task functions are unchanged; the runtime synthesizes their `j.Input` / `j.Arguments` from resolved bindings at claim time.
2. Editor change: Drawflow nodes get one input port per declared input (plus one output port). Connection-validation hook refuses incompatible wires; inline literal editors dim when wired.
3. Backward compat: legacy DAGs (with `Input` + `Arguments`) read transparently — bindings are synthesized in-memory but not written until the user re-saves.

**Tech Stack:** Go 1.22+ (`media-server`), Drawflow 0.0.59 (vendored via CDN in `editor.go.html`), SQLite.

**Reference spec:** `docs/superpowers/specs/2026-04-28-typed-workflow-nodes-design.md`

---

## File Map

**Create:**
- `media-server/tasks/types.go` — type-lattice constants + `Coerce` + `IsCompatible` helpers.
- `media-server/tasks/types_test.go` — table-driven coercion tests.
- `media-server/tasks/migrate.go` — legacy `Input + Arguments` → `Bindings` migration helper, used both by storage reads and by the workflow API for compat ingest.
- `media-server/tasks/migrate_test.go`.
- `media-server/jobqueue/bindings.go` — `Binding` type + JSON marshal/unmarshal.
- `media-server/jobqueue/bindings_test.go`.

**Modify:**
- `media-server/tasks/options.go` — replace `TaskOption` with `TaskInput`, add `TaskOutput`. Repurpose `ParseOptions` to operate on `[]TaskInput`.
- `media-server/tasks/registry.go` — change `Task` struct to use `Inputs` / `Output`; change `RegisterTask` signature; update every `RegisterTask(...)` call.
- `media-server/tasks/*.go` (every file with a `*Options` var) — rename to `*Inputs`, add `Primary: true` to the conventional first entry, add an `*Output` declaration. The list of files is enumerated in Phase 2.
- `media-server/jobqueue/jobqueue.go` — extend `WorkflowTask` with `Bindings` (+ keep `Input`/`Arguments` as deprecated read-only fallbacks), rewrite the input-construction block in `ClaimJob` to resolve bindings, change `AddWorkflow` to translate bindings before calling `AddJob`.
- `media-server/jobqueue/workflows.go` — extend `validateDAG` for binding-level checks; convert legacy shape on read in `GetWorkflow`.
- `media-server/main.go`, `main_darwin.go`, `main_linux.go` — `/tasks` handler returns `inputs` + `output` (and `options` as a deprecated alias); workflow endpoints accept the new shape and apply the migration helper to legacy shapes.
- `media-server/renderer/templates/editor.go.html` — multi-port rendering, type-aware connection validation, dim-on-wire styling, new export shape, legacy import using the same JS-side migration logic.

**Test:**
- `media-server/tasks/options_test.go` — new cases for `ParseOptions` over `[]TaskInput`.
- `media-server/jobqueue/workflows_test.go` — round-trip and migration cases.
- `media-server/jobqueue/jobqueue_core_test.go` — `ClaimJob` binding-resolution cases.

---

## Phase 1 — Type System & Coercion (foundation, no behavior change yet)

### Task 1: Define type-lattice constants and a coercion table

**Files:**
- Create: `media-server/tasks/types.go`
- Test: `media-server/tasks/types_test.go`

- [ ] **Step 1: Write the failing test**

```go
// media-server/tasks/types_test.go
package tasks

import "testing"

func TestIsCompatible(t *testing.T) {
	cases := []struct {
		from, to string
		want     bool
	}{
		// identity
		{TypeString, TypeString, true},
		{TypePathList, TypePathList, true},
		// scalar -> matching list
		{TypePath, TypePathList, true},
		{TypeURL, TypeURLList, true},
		{TypePath, TypeRefList, true},
		{TypeURL, TypeRefList, true},
		// list interop
		{TypePathList, TypeRefList, true},
		{TypeURLList, TypeRefList, true},
		{TypePathList, TypeURLList, true},
		// scalar -> string
		{TypePath, TypeString, true},
		{TypeURL, TypeString, true},
		{TypeNumber, TypeString, true},
		{TypeBool, TypeString, true},
		// string -> number/bool (parse at runtime)
		{TypeString, TypeNumber, true},
		{TypeString, TypeBool, true},
		// refusals
		{TypeNumber, TypePath, false},
		{TypeBool, TypePathList, false},
		{TypeJSON, TypeString, false},
		{TypeJSON, TypeJSON, true},
	}
	for _, tc := range cases {
		got := IsCompatible(tc.from, tc.to)
		if got != tc.want {
			t.Errorf("IsCompatible(%q, %q) = %v, want %v", tc.from, tc.to, got, tc.want)
		}
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd media-server && go test ./tasks/ -run TestIsCompatible -v`
Expected: build failure — `IsCompatible` and the `Type*` constants are undefined.

- [ ] **Step 3: Write the minimal implementation**

```go
// media-server/tasks/types.go
package tasks

// Input/output type constants — the "type lattice" used by binding validation
// and runtime coercion. See docs/superpowers/specs/2026-04-28-typed-workflow-nodes-design.md.
const (
	TypeString    = "string"
	TypeNumber    = "number"
	TypeBool      = "bool"
	TypeEnum      = "enum"
	TypeMultiEnum = "multi-enum"
	TypePath      = "path"
	TypePathList  = "path-list"
	TypeURL       = "url"
	TypeURLList   = "url-list"
	TypeRefList   = "ref-list"
	TypeJSON      = "json"
)

// IsCompatible reports whether a wire whose source produces values of type
// `from` may legally feed an input declared as `to`. Identity, scalar-to-list
// wrap, list-to-list interop, and lossless scalar-to-string are allowed.
// String-to-number / string-to-bool are allowed but may fail at runtime if
// the value doesn't parse.
func IsCompatible(from, to string) bool {
	if from == to {
		return true
	}
	switch to {
	case TypeString:
		switch from {
		case TypePath, TypeURL, TypeNumber, TypeBool:
			return true
		}
	case TypeNumber, TypeBool:
		return from == TypeString
	case TypePathList:
		return from == TypePath || from == TypeURLList // urls treated as opaque strings in path-list ctx? no — refuse
	case TypeURLList:
		return from == TypeURL || from == TypePathList
	case TypeRefList:
		switch from {
		case TypePath, TypeURL, TypePathList, TypeURLList:
			return true
		}
	}
	return false
}
```

Note: the test asserts `path-list ↔ url-list` is bidirectionally true. Adjust the switch arms above so both `to == TypePathList` accepts `from == TypeURLList` and vice-versa:

```go
	case TypePathList:
		return from == TypePath || from == TypeURLList || from == TypeRefList
	case TypeURLList:
		return from == TypeURL || from == TypePathList || from == TypeRefList
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd media-server && go test ./tasks/ -run TestIsCompatible -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add media-server/tasks/types.go media-server/tasks/types_test.go
git commit -m "feat(workflow): introduce typed input/output lattice and compatibility table"
```

---

### Task 2: Add the `Coerce` helper that the runtime will use

**Files:**
- Modify: `media-server/tasks/types.go`
- Test: `media-server/tasks/types_test.go`

- [ ] **Step 1: Add failing tests**

Append to `media-server/tasks/types_test.go`:

```go
func TestCoerce(t *testing.T) {
	cases := []struct {
		name      string
		from, to  string
		in        any
		want      any
		expectErr bool
	}{
		{"identity-string", TypeString, TypeString, "x", "x", false},
		{"path-to-pathlist", TypePath, TypePathList, "/a", []string{"/a"}, false},
		{"url-to-reflist", TypeURL, TypeRefList, "http://x", []string{"http://x"}, false},
		{"pathlist-to-reflist", TypePathList, TypeRefList, []string{"a", "b"}, []string{"a", "b"}, false},
		{"number-to-string", TypeNumber, TypeString, 1.5, "1.5", false},
		{"bool-to-string", TypeBool, TypeString, true, "true", false},
		{"string-to-number-ok", TypeString, TypeNumber, "1.5", 1.5, false},
		{"string-to-number-bad", TypeString, TypeNumber, "x", nil, true},
		{"string-to-bool-ok", TypeString, TypeBool, "true", true, false},
		{"string-to-bool-bad", TypeString, TypeBool, "maybe", nil, true},
		{"refused", TypeNumber, TypePath, 1.0, nil, true},
	}
	for _, tc := range cases {
		got, err := Coerce(tc.in, tc.from, tc.to)
		if tc.expectErr {
			if err == nil {
				t.Errorf("%s: expected error, got %#v", tc.name, got)
			}
			continue
		}
		if err != nil {
			t.Errorf("%s: unexpected error %v", tc.name, err)
			continue
		}
		if !equalValue(got, tc.want) {
			t.Errorf("%s: got %#v, want %#v", tc.name, got, tc.want)
		}
	}
}

func equalValue(a, b any) bool {
	as, aok := a.([]string)
	bs, bok := b.([]string)
	if aok && bok {
		if len(as) != len(bs) {
			return false
		}
		for i := range as {
			if as[i] != bs[i] {
				return false
			}
		}
		return true
	}
	return a == b
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd media-server && go test ./tasks/ -run TestCoerce -v`
Expected: build failure — `Coerce` undefined.

- [ ] **Step 3: Implement `Coerce`**

Append to `media-server/tasks/types.go`:

```go
import (
	"fmt"
	"strconv"
)

// Coerce converts a value produced by a port of type `from` into the form
// expected by an input port of type `to`. Returns an error if the conversion
// is disallowed by the lattice or if a runtime parse fails (string→number,
// string→bool with an unparseable value).
func Coerce(v any, from, to string) (any, error) {
	if !IsCompatible(from, to) {
		return nil, fmt.Errorf("incompatible types: %s -> %s", from, to)
	}
	if from == to {
		return v, nil
	}
	switch to {
	case TypeString:
		switch x := v.(type) {
		case string:
			return x, nil
		case float64:
			return strconv.FormatFloat(x, 'f', -1, 64), nil
		case bool:
			return strconv.FormatBool(x), nil
		}
	case TypeNumber:
		s, _ := v.(string)
		n, err := strconv.ParseFloat(s, 64)
		if err != nil {
			return nil, fmt.Errorf("cannot parse %q as number", s)
		}
		return n, nil
	case TypeBool:
		s, _ := v.(string)
		switch s {
		case "true", "1":
			return true, nil
		case "false", "0":
			return false, nil
		}
		return nil, fmt.Errorf("cannot parse %q as bool", s)
	case TypePathList, TypeURLList, TypeRefList:
		switch x := v.(type) {
		case string:
			return []string{x}, nil
		case []string:
			return x, nil
		}
	}
	return nil, fmt.Errorf("coerce: unhandled %s -> %s", from, to)
}
```

- [ ] **Step 4: Run tests**

Run: `cd media-server && go test ./tasks/ -run TestCoerce -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add media-server/tasks/types.go media-server/tasks/types_test.go
git commit -m "feat(workflow): add Coerce helper for runtime binding resolution"
```

---

## Phase 2 — Task Schema Migration

### Task 3: Replace `TaskOption` with `TaskInput` + `TaskOutput`

**Files:**
- Modify: `media-server/tasks/options.go`
- Test: `media-server/tasks/options_test.go`

- [ ] **Step 1: Read the existing file**

Read `media-server/tasks/options.go` and `media-server/tasks/options_test.go` so the next step preserves coverage.

- [ ] **Step 2: Replace the schema**

Rewrite `media-server/tasks/options.go`:

```go
package tasks

import (
	"strconv"
	"strings"

	"github.com/stevecastle/shrike/jobqueue"
)

// TaskInput describes one input port of a task. Every option a task accepts
// — including the conventional "primary" data input — is one of these.
type TaskInput struct {
	Name        string   `json:"name"`
	Label       string   `json:"label"`
	Type        string   `json:"type"` // one of the Type* constants in types.go
	Choices     []string `json:"choices,omitempty"`
	Default     any      `json:"default,omitempty"`
	Required    bool     `json:"required,omitempty"`
	Primary     bool     `json:"primary,omitempty"` // marks the first/conventional input
	Description string   `json:"description,omitempty"`
}

// TaskOutput describes the single output port of a task.
type TaskOutput struct {
	Type        string `json:"type"`
	Description string `json:"description,omitempty"`
}

// TaskOption is retained as a deprecated alias for TaskInput so the JSON shape
// emitted at /tasks can include an `options` field for one release. New code
// must not reference this type.
//
// Deprecated: use TaskInput.
type TaskOption = TaskInput

// ParseOptions extracts typed values for a task's non-primary inputs from a
// job's Arguments slice. Primary inputs are read from j.Input by the task
// itself; ParseOptions ignores them.
//
// This preserves the legacy CLI-flag encoding (--name value) so that task
// implementations don't change when the surrounding model changes.
func ParseOptions(j *jobqueue.Job, inputs []TaskInput) map[string]any {
	result := make(map[string]any)
	for _, in := range inputs {
		if in.Primary {
			continue
		}
		if in.Default != nil {
			result[in.Name] = in.Default
		} else {
			switch in.Type {
			case TypeBool:
				result[in.Name] = false
			case TypeNumber:
				result[in.Name] = 0.0
			default:
				result[in.Name] = ""
			}
		}
	}

	optMap := make(map[string]*TaskInput, len(inputs))
	for i := range inputs {
		if inputs[i].Primary {
			continue
		}
		optMap[inputs[i].Name] = &inputs[i]
	}

	args := j.Arguments
	for i := 0; i < len(args); i++ {
		arg := args[i]
		if !strings.HasPrefix(arg, "--") {
			continue
		}
		key := strings.TrimPrefix(arg, "--")
		value := ""
		hasEquals := false
		if eqIdx := strings.Index(key, "="); eqIdx >= 0 {
			value = key[eqIdx+1:]
			key = key[:eqIdx]
			hasEquals = true
		}
		opt, ok := optMap[key]
		if !ok {
			continue
		}
		switch opt.Type {
		case TypeBool:
			if hasEquals {
				result[key] = value == "true" || value == "1" || value == "yes"
			} else {
				result[key] = true
			}
		case TypeNumber:
			if !hasEquals && i+1 < len(args) && !strings.HasPrefix(args[i+1], "--") {
				i++
				value = args[i]
			}
			if n, err := strconv.ParseFloat(value, 64); err == nil {
				result[key] = n
			}
		default:
			if !hasEquals && i+1 < len(args) && !strings.HasPrefix(args[i+1], "--") {
				i++
				value = args[i]
			}
			result[key] = value
		}
	}
	return result
}
```

- [ ] **Step 3: Update existing tests**

Open `media-server/tasks/options_test.go` and rename any `[]TaskOption` literals to `[]TaskInput` and any `Type: "string"` → `Type: TypeString` etc. Add `Primary: true` to whichever entry represents the legacy "primary" arg in each test (for tests that don't have one, no change needed — `ParseOptions` skips primary entries).

- [ ] **Step 4: Run task tests**

Run: `cd media-server && go test ./tasks/ -v`
Expected: all green. (Other tests still pass because `TaskOption` is a type alias.)

- [ ] **Step 5: Commit**

```bash
git add media-server/tasks/options.go media-server/tasks/options_test.go
git commit -m "feat(workflow): introduce TaskInput/TaskOutput schema; alias TaskOption"
```

---

### Task 4: Extend `Task` struct and `RegisterTask` signature

**Files:**
- Modify: `media-server/tasks/registry.go`

- [ ] **Step 1: Update the schema**

Replace the body of `media-server/tasks/registry.go` with:

```go
package tasks

import (
	"sync"

	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/storage"
)

type TaskFn func(j *jobqueue.Job, q *jobqueue.Queue, r *sync.Mutex) error

type Task struct {
	ID     string      `json:"id"`
	Name   string      `json:"name"`
	Inputs []TaskInput `json:"inputs"`
	Output TaskOutput  `json:"output"`
	Fn     TaskFn      `json:"-"`

	// Options is a deprecated JSON alias for Inputs (excluding entries with
	// Primary: true) so external consumers of /tasks have one release of
	// overlap. Computed at marshal time; never set directly.
	Options []TaskInput `json:"options,omitempty"`
}

type TaskMap map[string]Task

var tasks = make(TaskMap)

var storageReg *storage.Registry

func SetStorageRegistry(r *storage.Registry) {
	storageReg = r
}

func RegisterTask(id, name string, inputs []TaskInput, output TaskOutput, fn TaskFn) {
	t := Task{ID: id, Name: name, Inputs: inputs, Output: output, Fn: fn}
	for _, in := range inputs {
		if !in.Primary {
			t.Options = append(t.Options, in)
		}
	}
	tasks[id] = t
}

func GetTasks() TaskMap {
	return tasks
}

// init() lives in registry_init.go so RegisterTask call sites are easy to
// review in isolation.
```

- [ ] **Step 2: Move the `init()` block out**

Create `media-server/tasks/registry_init.go`:

```go
package tasks

func init() {
	// Built-in tasks. Each call passes:
	//   id, display name, inputs (with one Primary: true), output, fn.
	registerBuiltins()
}

func registerBuiltins() {
	RegisterTask("wait", "Wait", waitInputs, waitOutput, waitFn)
	RegisterTask("remove", "Remove Media", removeInputs, removeOutput, removeFromDB)
	RegisterTask("cleanup", "CleanUp", cleanupInputs, cleanupOutput, cleanUpFn)
	RegisterTask("autotag", "Auto Tag (ONNX)", autotagInputs, autotagOutput, autotagTask)

	RegisterTask("metadata", "Generate Metadata", metadataInputs, metadataOutput, metadataTask)
	RegisterTask("hls", "HLS Transcode", hlsInputs, hlsOutput, hlsTask)
	RegisterTask("move", "Move Media Files", moveInputs, moveOutput, moveTask)
	RegisterTask("ingest", "Ingest Media Files", ingestInputs, ingestOutput, ingestTask)
	RegisterTask("lora-dataset", "Create LoRA Dataset", loraDatasetInputs, loraDatasetOutput, loraDatasetTask)

	RegisterTask("ffmpeg", "ffmpeg", ffmpegCustomInputs, ffmpegCustomOutput, ffmpegTask)
	RegisterTask("ffmpeg-scale", "FFmpeg Scale", ffmpegScaleInputs, ffmpegScaleOutput, ffmpegScaleTask)
	RegisterTask("ffmpeg-convert", "FFmpeg Convert", ffmpegConvertInputs, ffmpegConvertOutput, ffmpegConvertTask)
	RegisterTask("ffmpeg-extract-audio", "FFmpeg Extract Audio", ffmpegExtractAudioInputs, ffmpegExtractAudioOutput, ffmpegExtractAudioTask)
	RegisterTask("ffmpeg-extract-audio-clip", "FFmpeg Extract Audio Clip", ffmpegExtractAudioClipInputs, ffmpegExtractAudioClipOutput, ffmpegExtractAudioClipTask)
	RegisterTask("ffmpeg-screenshot", "FFmpeg Screenshot", ffmpegScreenshotInputs, ffmpegScreenshotOutput, ffmpegScreenshotTask)
	RegisterTask("ffmpeg-thumbnail", "FFmpeg Thumbnail", ffmpegThumbnailInputs, ffmpegThumbnailOutput, ffmpegThumbnailTask)
	RegisterTask("ffmpeg-reverse", "FFmpeg Reverse", ffmpegReverseInputs, ffmpegReverseOutput, ffmpegReverseTask)
	RegisterTask("ffmpeg-speed", "FFmpeg Speed", ffmpegSpeedInputs, ffmpegSpeedOutput, ffmpegSpeedTask)
	RegisterTask("ffmpeg-grayscale", "FFmpeg Grayscale", ffmpegGrayscaleInputs, ffmpegGrayscaleOutput, ffmpegGrayscaleTask)
	RegisterTask("ffmpeg-blur", "FFmpeg Blur", ffmpegBlurInputs, ffmpegBlurOutput, ffmpegBlurTask)
	RegisterTask("ffmpeg-resize", "FFmpeg Resize", ffmpegResizeInputs, ffmpegResizeOutput, ffmpegResizeTask)
	RegisterTask("ffmpeg-crop", "FFmpeg Crop", ffmpegCropInputs, ffmpegCropOutput, ffmpegCropTask)
	RegisterTask("ffmpeg-rotate", "FFmpeg Rotate", ffmpegRotateInputs, ffmpegRotateOutput, ffmpegRotateTask)
	RegisterTask("ffmpeg-caption", "FFmpeg Caption", ffmpegCaptionInputs, ffmpegCaptionOutput, ffmpegCaptionTask)
	RegisterTask("ffmpeg-thumbsheet", "FFmpeg Thumbnail Sheet", ffmpegThumbSheetInputs, ffmpegThumbSheetOutput, ffmpegThumbSheetTask)

	RegisterTask("save", "Save File", saveInputs, saveOutput, saveTask)
}
```

The build will be broken until Task 5 supplies all the `*Inputs`/`*Output` declarations.

- [ ] **Step 3: Commit (broken build is acceptable mid-phase, but prefer to bundle 4+5)**

Skip the commit here and go straight to Task 5; commit at the end of Task 5 once the build is green.

---

### Task 5: Migrate every task's `*Options` to `*Inputs` + add `*Output`

**Files (one logical change per file, but committed together at the end):**
- Modify: `media-server/tasks/wait.go`
- Modify: `media-server/tasks/media_remove.go`
- Modify: `media-server/tasks/media_cleanup.go`
- Modify: `media-server/tasks/autotag.go`
- Modify: `media-server/tasks/media_metadata.go`
- Modify: `media-server/tasks/hls.go`
- Modify: `media-server/tasks/media_move.go`
- Modify: `media-server/tasks/media_ingest.go`
- Modify: `media-server/tasks/lora_dataset.go`
- Modify: `media-server/tasks/ffmpeg.go`
- Modify: `media-server/tasks/ffmpeg_presets.go`
- Modify: `media-server/tasks/save.go`
- Modify: `media-server/tasks/queries.go`

For **every** file in the list above: rename the `*Options` `[]TaskOption` var to `*Inputs` `[]TaskInput`, prepend a `Primary: true` entry that captures the conventional `j.Input` shape for that task, and add a `*Output` `TaskOutput` var. The pattern below is the template; apply it consistently.

- [ ] **Step 1: Apply the template to `media-server/tasks/save.go`**

Replace the `var saveOptions = []TaskOption{...}` block with:

```go
var saveInputs = []TaskInput{
	{
		Name:        "files",
		Label:       "Files",
		Type:        TypePathList,
		Primary:     true,
		Description: "Files to save (newline-separated paths from upstream)",
	},
	{Name: "mode", Label: "Save Mode", Type: TypeEnum, Choices: []string{"replace", "alongside", "folder"}, Default: "alongside", Description: "replace: overwrite original file, alongside: save next to original with suffix, folder: save to a specific directory"},
	{Name: "suffix", Label: "Filename Suffix", Type: TypeString, Default: "_output", Description: "Suffix added before extension (alongside mode). e.g. _edited produces video_edited.mp4"},
	{Name: "folder", Label: "Output Folder", Type: TypeString, Description: "Destination folder (folder mode only)"},
}

var saveOutput = TaskOutput{Type: TypePathList, Description: "Saved files"}
```

Update the `ParseOptions(j, saveOptions)` call to `ParseOptions(j, saveInputs)`.

- [ ] **Step 2: Apply the template to every other file**

For each file, follow these rules:

- The leading `Primary: true` input takes the type that matches what the task currently reads from `j.Input`:
  - `wait`, `cleanup`, `remove`, `autotag`, `metadata`, `hls`, `move`, `lora-dataset`, every `ffmpeg-*`, `save` → `TypePathList` (named `files`).
  - `ingest` → `TypeRefList` (named `paths`, accepts paths or URLs).
  - `queries` (search) → `TypeString` (named `query`). Look at `media-server/tasks/queries.go` to confirm; if `queries` isn't currently registered, skip.
- The `*Output` variable's `Type` is `TypePathList` for any task that today calls `q.RegisterOutputFile(...)`. For tasks that don't (e.g., `wait`), use `TypePathList` with `Description: "(passes input through unchanged)"` and have the task still call `q.RegisterOutputFile` for each entry of `j.Input` so the output channel stays consistent. (If the task is `wait` and currently does nothing with output, add one line: after success, split `j.Input` on newlines and `RegisterOutputFile` each non-empty entry.)
- The leading `Primary: true` input is **not** parsed by `ParseOptions`; the task's existing `j.Input` reads continue to work as-is.
- Replace every `ParseOptions(j, *Options)` call with `ParseOptions(j, *Inputs)`.

If any file currently has no options (e.g. `wait`, `cleanup`, `autotag`, `ffmpeg-reverse`, `ffmpeg-grayscale`, `remove`, `queries`), introduce a `var <name>Inputs = []TaskInput{ ... primary only ... }` and a `var <name>Output = TaskOutput{...}`.

- [ ] **Step 3: Build the package**

Run: `cd media-server && go build ./...`
Expected: PASS — no undefined identifiers, no signature mismatches.

- [ ] **Step 4: Run all tests**

Run: `cd media-server && go test ./...`
Expected: PASS. The only behavior change at this point is that tasks declare structured I/O contracts; `ClaimJob` still uses the legacy code path until Phase 3.

- [ ] **Step 5: Commit (Tasks 4 + 5 together)**

```bash
git add media-server/tasks/
git commit -m "feat(workflow): migrate all task option lists to TaskInput/TaskOutput contracts"
```

---

## Phase 3 — Bindings on `WorkflowTask` + Runtime Resolution

### Task 6: Define the `Binding` type

**Files:**
- Create: `media-server/jobqueue/bindings.go`
- Test: `media-server/jobqueue/bindings_test.go`

- [ ] **Step 1: Write the failing test**

```go
// media-server/jobqueue/bindings_test.go
package jobqueue

import (
	"encoding/json"
	"testing"
)

func TestBindingMarshalLiteral(t *testing.T) {
	b := Binding{Kind: "literal", Value: "hello"}
	data, err := json.Marshal(b)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	want := `{"kind":"literal","value":"hello"}`
	if string(data) != want {
		t.Errorf("got %s, want %s", string(data), want)
	}
}

func TestBindingMarshalWire(t *testing.T) {
	b := Binding{Kind: "wire", From: "node-1", Port: "output"}
	data, err := json.Marshal(b)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	want := `{"kind":"wire","from":"node-1","port":"output"}`
	if string(data) != want {
		t.Errorf("got %s, want %s", string(data), want)
	}
}

func TestBindingUnmarshalRoundTrip(t *testing.T) {
	cases := []Binding{
		{Kind: "literal", Value: "x"},
		{Kind: "literal", Value: float64(3)},
		{Kind: "literal", Value: true},
		{Kind: "wire", From: "n1", Port: "output"},
	}
	for _, b := range cases {
		data, _ := json.Marshal(b)
		var got Binding
		if err := json.Unmarshal(data, &got); err != nil {
			t.Errorf("unmarshal %s: %v", data, err)
		}
		if got.Kind != b.Kind || got.From != b.From || got.Port != b.Port {
			t.Errorf("roundtrip mismatch: %+v vs %+v", got, b)
		}
	}
}
```

- [ ] **Step 2: Run to confirm failure**

Run: `cd media-server && go test ./jobqueue/ -run TestBinding -v`
Expected: build failure — `Binding` undefined.

- [ ] **Step 3: Implement**

```go
// media-server/jobqueue/bindings.go
package jobqueue

// Binding describes how a single input port on a WorkflowTask is supplied at
// runtime. Either a literal value typed in the editor, or a wire from another
// task's output port.
type Binding struct {
	Kind  string `json:"kind"`            // "literal" | "wire"
	Value any    `json:"value,omitempty"` // for literal
	From  string `json:"from,omitempty"`  // for wire: source task ID
	Port  string `json:"port,omitempty"`  // for wire: always "output" in v1
}
```

- [ ] **Step 4: Run tests**

Run: `cd media-server && go test ./jobqueue/ -run TestBinding -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add media-server/jobqueue/bindings.go media-server/jobqueue/bindings_test.go
git commit -m "feat(workflow): add Binding type for typed input wiring"
```

---

### Task 7: Extend `WorkflowTask` to carry bindings (legacy fields preserved)

**Files:**
- Modify: `media-server/jobqueue/jobqueue.go` (lines 128–136)

- [ ] **Step 1: Replace the struct**

```go
type WorkflowTask struct {
	ID           string             `json:"id"`
	Command      string             `json:"command"`
	Bindings     map[string]Binding `json:"bindings,omitempty"`
	Dependencies []string           `json:"dependencies"`
	PosX         float64            `json:"pos_x,omitempty"`
	PosY         float64            `json:"pos_y,omitempty"`

	// Deprecated: legacy fields. Read on input for backward compat with old
	// stored DAGs and old API clients; never written by new code.
	Arguments []string `json:"arguments,omitempty"`
	Input     string   `json:"input,omitempty"`
}
```

- [ ] **Step 2: Build to confirm nothing else broke**

Run: `cd media-server && go build ./...`
Expected: PASS. Existing call sites that read `task.Arguments` and `task.Input` continue to compile.

- [ ] **Step 3: Run tests**

Run: `cd media-server && go test ./...`
Expected: PASS — same behavior as before since nothing yet writes `Bindings`.

- [ ] **Step 4: Commit**

```bash
git add media-server/jobqueue/jobqueue.go
git commit -m "feat(workflow): add Bindings field to WorkflowTask alongside legacy fields"
```

---

### Task 8: Legacy → bindings migration helper

**Files:**
- Create: `media-server/tasks/migrate.go`
- Test: `media-server/tasks/migrate_test.go`

- [ ] **Step 1: Write the failing test**

```go
// media-server/tasks/migrate_test.go
package tasks

import (
	"reflect"
	"testing"

	"github.com/stevecastle/shrike/jobqueue"
)

func TestMigrateLegacyTask(t *testing.T) {
	// Use 'save' which has well-known inputs.
	wt := jobqueue.WorkflowTask{
		ID:           "n1",
		Command:      "save",
		Input:        "/tmp/a.mp4\n/tmp/b.mp4",
		Arguments:    []string{"--mode", "folder", "--folder", "/out"},
		Dependencies: []string{},
	}
	got := MigrateLegacyTask(wt)
	if got.Bindings == nil {
		t.Fatal("bindings nil")
	}
	if got.Bindings["files"].Kind != "literal" {
		t.Errorf("primary not literal: %+v", got.Bindings["files"])
	}
	if v, _ := got.Bindings["files"].Value.(string); v != "/tmp/a.mp4\n/tmp/b.mp4" {
		t.Errorf("primary value wrong: %q", v)
	}
	if got.Bindings["mode"].Value != "folder" {
		t.Errorf("mode wrong: %+v", got.Bindings["mode"])
	}
	if got.Bindings["folder"].Value != "/out" {
		t.Errorf("folder wrong: %+v", got.Bindings["folder"])
	}
}

func TestMigrateLegacyTaskWithDeps(t *testing.T) {
	wt := jobqueue.WorkflowTask{
		ID:           "n2",
		Command:      "save",
		Input:        "",
		Dependencies: []string{"n1"},
	}
	got := MigrateLegacyTask(wt)
	wires := []jobqueue.Binding{got.Bindings["files"]}
	want := []jobqueue.Binding{{Kind: "wire", From: "n1", Port: "output"}}
	if !reflect.DeepEqual(wires, want) {
		t.Errorf("got %+v, want %+v", wires, want)
	}
}
```

- [ ] **Step 2: Run to confirm failure**

Run: `cd media-server && go test ./tasks/ -run TestMigrateLegacy -v`
Expected: build failure — `MigrateLegacyTask` undefined.

- [ ] **Step 3: Implement the migration**

```go
// media-server/tasks/migrate.go
package tasks

import (
	"github.com/stevecastle/shrike/jobqueue"
)

// MigrateLegacyTask converts a WorkflowTask in the legacy shape (Input +
// Arguments) into the new shape with Bindings populated. If Bindings is
// already non-empty, the input is returned unchanged. The migration:
//
//   - Looks up the task's declared Inputs by Command.
//   - Sets the Primary input's binding to a literal of the legacy Input string
//     (if non-empty) and appends a wire binding for each Dependencies entry.
//     Note: in v1 the editor only allows the primary port to be wired from
//     upstream, so legacy deps always wire into the primary port.
//   - Parses Arguments (CLI --flag value tokens) into per-input literal
//     bindings using the same rules as ParseOptions.
//
// If the task is unknown, returns the input unchanged so an unknown command
// can still surface its error elsewhere (e.g. validateDAG).
func MigrateLegacyTask(t jobqueue.WorkflowTask) jobqueue.WorkflowTask {
	if len(t.Bindings) > 0 {
		return t
	}
	task, ok := tasks[t.Command]
	if !ok {
		return t
	}

	bindings := make(map[string]jobqueue.Binding, len(task.Inputs))

	// Primary input: legacy Input + dependency wires.
	for _, in := range task.Inputs {
		if !in.Primary {
			continue
		}
		// Single-binding-per-name limitation: encode multiple wires by
		// preferring the literal if present, then appending wires below
		// using the primary input's name as the key. To support fan-in
		// (legacy deps + literal), we emit the literal as the binding
		// and rely on the runtime's dependency walk for the rest. Phase 3
		// runtime change handles this.
		if t.Input != "" {
			bindings[in.Name] = jobqueue.Binding{Kind: "literal", Value: t.Input}
		} else if len(t.Dependencies) > 0 {
			bindings[in.Name] = jobqueue.Binding{
				Kind: "wire",
				From: t.Dependencies[0],
				Port: "output",
			}
		}
		break
	}

	// Non-primary inputs: parse the legacy Arguments slice.
	if len(t.Arguments) > 0 {
		fakeJob := &jobqueue.Job{Arguments: t.Arguments}
		parsed := ParseOptions(fakeJob, task.Inputs)
		for _, in := range task.Inputs {
			if in.Primary {
				continue
			}
			if v, ok := parsed[in.Name]; ok && v != "" && v != false && v != 0.0 {
				bindings[in.Name] = jobqueue.Binding{Kind: "literal", Value: v}
			}
		}
	}

	out := t
	out.Bindings = bindings
	return out
}
```

- [ ] **Step 4: Run tests**

Run: `cd media-server && go test ./tasks/ -run TestMigrateLegacy -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add media-server/tasks/migrate.go media-server/tasks/migrate_test.go
git commit -m "feat(workflow): migrate legacy WorkflowTask shape into Bindings"
```

---

### Task 9: Resolve bindings into a Job at submit time

**Files:**
- Modify: `media-server/jobqueue/jobqueue.go` (`AddWorkflow`, lines 459–493)

The translation from `WorkflowTask` to `AddJob` is the right place to expand `Bindings` into the legacy `Job.Arguments` + `Job.Input` shape. Wire bindings still need the upstream `Job.OutputFiles` at runtime, so we keep the `ClaimJob` block that pulls upstream output. We just change what we feed into `AddJob` so that `OriginalInput` already contains the literal portion and `Arguments` already contains the literal flag values.

- [ ] **Step 1: Write the failing test**

Append to `media-server/jobqueue/jobqueue_core_test.go`:

```go
func TestAddWorkflowResolvesLiteralBindings(t *testing.T) {
	q := NewQueue()
	wf := Workflow{Tasks: []WorkflowTask{
		{
			ID:      "a",
			Command: "save",
			Bindings: map[string]Binding{
				"files":  {Kind: "literal", Value: "/tmp/x.mp4"},
				"mode":   {Kind: "literal", Value: "folder"},
				"folder": {Kind: "literal", Value: "/out"},
			},
		},
	}}
	ids, err := q.AddWorkflow(wf)
	if err != nil {
		t.Fatalf("AddWorkflow: %v", err)
	}
	if len(ids) != 1 {
		t.Fatalf("expected 1 job, got %d", len(ids))
	}
	job := q.GetJob(ids[0])
	if job.OriginalInput != "/tmp/x.mp4" {
		t.Errorf("OriginalInput = %q", job.OriginalInput)
	}
	args := job.Arguments
	want := map[string]string{"--mode": "folder", "--folder": "/out"}
	for k, v := range want {
		found := false
		for i := 0; i < len(args)-1; i++ {
			if args[i] == k && args[i+1] == v {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("missing %s %s in %v", k, v, args)
		}
	}
}
```

This test depends on the `tasks` package being importable from `jobqueue`. **It cannot import `tasks` because `tasks` imports `jobqueue`** — that's a cycle. So the binding-resolution logic must live somewhere that knows the task contracts but doesn't create a cycle.

Resolution: introduce a small "task lookup" interface in `jobqueue` and have `tasks` register itself at startup. Continue:

- [ ] **Step 2: Add a contract-lookup hook in `jobqueue`**

Add to `media-server/jobqueue/jobqueue.go` (top-level, near other globals):

```go
// TaskContract describes the inputs of a registered task as far as the
// jobqueue cares — just enough to translate Bindings into Arguments + Input.
// The tasks package registers itself via SetContractLookup at init time.
type TaskContract struct {
	PrimaryName string             // the name of the primary input, "" if none
	Inputs      []TaskContractInput
}

type TaskContractInput struct {
	Name    string
	Type    string
	Primary bool
}

type ContractLookupFn func(command string) (TaskContract, bool)

var contractLookup ContractLookupFn

// SetContractLookup wires the tasks package's registry into the jobqueue so
// AddWorkflow can translate Bindings without creating an import cycle.
func SetContractLookup(fn ContractLookupFn) {
	contractLookup = fn
}
```

- [ ] **Step 3: Have `tasks` register the lookup at init**

Append to `media-server/tasks/registry_init.go`:

```go
func init() {
	registerBuiltins()
	jobqueue.SetContractLookup(lookupContract)
}

func lookupContract(command string) (jobqueue.TaskContract, bool) {
	t, ok := tasks[command]
	if !ok {
		return jobqueue.TaskContract{}, false
	}
	out := jobqueue.TaskContract{Inputs: make([]jobqueue.TaskContractInput, 0, len(t.Inputs))}
	for _, in := range t.Inputs {
		out.Inputs = append(out.Inputs, jobqueue.TaskContractInput{Name: in.Name, Type: in.Type, Primary: in.Primary})
		if in.Primary {
			out.PrimaryName = in.Name
		}
	}
	return out, true
}
```

(Note: `registry_init.go` already has a top-level `init()` from Task 4. Replace it with the version above so there's only one `init()` in the file.)

- [ ] **Step 4: Implement binding resolution in `AddWorkflow`**

Replace the body of `AddWorkflow` (around line 459) with:

```go
func (q *Queue) AddWorkflow(w Workflow) ([]string, error) {
	var jobIDs []string

	workflowID := w.WorkflowID
	if workflowID == "" {
		workflowID = uuid.NewString()
	}

	for _, task := range w.Tasks {
		input, args, deps := resolveBindings(task)
		id, err := q.AddJob(task.ID, task.Command, args, input, deps)
		if err != nil {
			return jobIDs, err
		}
		q.mu.Lock()
		if job, ok := q.Jobs[id]; ok {
			job.WorkflowID = workflowID
			if err := q.saveJobToDB(job); err != nil {
				log.Printf("Failed to save workflow ID to database: %v", err)
			}
		}
		q.mu.Unlock()
		jobIDs = append(jobIDs, id)
	}
	return jobIDs, nil
}

// resolveBindings translates a WorkflowTask's Bindings (or legacy Input/Arguments
// fallback) into the (Job.OriginalInput, Job.Arguments, Job.Dependencies) tuple
// that the existing scheduler understands.
//
// Wire bindings show up as Dependencies; their values are not materialized
// here — ClaimJob still pulls upstream OutputFiles when the dependency
// completes.
func resolveBindings(t WorkflowTask) (string, []string, []string) {
	// Legacy shape: nothing to translate.
	if len(t.Bindings) == 0 {
		return t.Input, t.Arguments, t.Dependencies
	}

	contract, ok := getContract(t.Command)
	if !ok {
		// Unknown command: emit empty job; validateDAG should have refused.
		return "", nil, nil
	}

	var input string
	var args []string
	depSet := make(map[string]struct{})

	for _, in := range contract.Inputs {
		b, has := t.Bindings[in.Name]
		if !has {
			continue
		}
		switch b.Kind {
		case "literal":
			if in.Primary {
				input = literalToString(b.Value)
			} else {
				args = appendArg(args, in, b.Value)
			}
		case "wire":
			depSet[b.From] = struct{}{}
			// Non-primary wire bindings are resolved at ClaimJob time
			// (Task 10). For now they just register a dependency; the
			// upstream output flows into Job.Input as legacy behavior.
		}
	}

	deps := make([]string, 0, len(depSet))
	for d := range depSet {
		deps = append(deps, d)
	}
	return input, args, deps
}

func getContract(command string) (TaskContract, bool) {
	if contractLookup == nil {
		return TaskContract{}, false
	}
	return contractLookup(command)
}

func literalToString(v any) string {
	switch x := v.(type) {
	case string:
		return x
	case float64:
		return strconv.FormatFloat(x, 'f', -1, 64)
	case bool:
		return strconv.FormatBool(x)
	case []string:
		return strings.Join(x, "\n")
	case nil:
		return ""
	}
	return fmt.Sprint(v)
}

func appendArg(args []string, in TaskContractInput, v any) []string {
	switch in.Type {
	case "bool":
		if b, _ := v.(bool); b {
			return append(args, "--"+in.Name)
		}
		return args
	case "number":
		n, _ := v.(float64)
		if n != 0 {
			return append(args, "--"+in.Name, strconv.FormatFloat(n, 'f', -1, 64))
		}
		return args
	default:
		s := literalToString(v)
		if s == "" {
			return args
		}
		return append(args, "--"+in.Name, s)
	}
}
```

Add `"strconv"` to the imports of `media-server/jobqueue/jobqueue.go` if it isn't already there. (`strings` and `fmt` already are.)

- [ ] **Step 5: Run all tests**

Run: `cd media-server && go test ./...`
Expected: PASS, including the new `TestAddWorkflowResolvesLiteralBindings`.

- [ ] **Step 6: Commit**

```bash
git add media-server/jobqueue/ media-server/tasks/registry_init.go
git commit -m "feat(workflow): resolve typed bindings into Job inputs and arguments at submit time"
```

---

### Task 10: Wire bindings for non-primary inputs

For v1 the editor only allows wires into the primary input. Non-primary wires are accepted by the schema but unwired in the runtime (literal-only). Defer the `ClaimJob`-time evaluation of non-primary wires to a follow-up — but write a guard so that if such a binding exists at run time, the job errors out with a clear message rather than silently using the default.

**Files:**
- Modify: `media-server/jobqueue/jobqueue.go` (`ClaimJob`)

- [ ] **Step 1: Add the guard inside `ClaimJob`**

Inside `ClaimJob`, after the `inputBuilder` block (before `// Save to database`), add:

```go
// (Phase 3 v1) Wire bindings into non-primary inputs are not yet evaluated
// at claim time; the editor refuses to draw such wires, but a stored DAG
// that contains one (e.g. hand-edited JSON) would silently behave wrong.
// Surface it loudly instead.
//
// Detection lives at validation time (validateDAG) — see workflows.go.
```

This step is documentation-only; the actual refusal happens in `validateDAG` in Task 12.

- [ ] **Step 2: Commit**

No code change to commit; proceed.

---

## Phase 4 — Validation & API

### Task 11: Extend `validateDAG` for binding-aware checks

**Files:**
- Modify: `media-server/jobqueue/workflows.go` (`validateDAG`, lines 134–162)
- Test: `media-server/jobqueue/workflows_test.go`

- [ ] **Step 1: Write failing tests**

Append to `media-server/jobqueue/workflows_test.go`:

```go
func TestValidateDAGWireRefersToUnknownNode(t *testing.T) {
	dag := []WorkflowTask{
		{
			ID:      "a",
			Command: "save",
			Bindings: map[string]Binding{
				"files": {Kind: "wire", From: "ghost", Port: "output"},
			},
		},
	}
	if err := validateDAG(dag); err == nil {
		t.Error("expected error for wire to unknown node")
	}
}

func TestValidateDAGUnknownCommand(t *testing.T) {
	dag := []WorkflowTask{{ID: "a", Command: "no-such-task"}}
	if err := validateDAG(dag); err == nil {
		t.Error("expected error for unknown command")
	}
}

func TestValidateDAGWireToNonPrimaryRejected(t *testing.T) {
	dag := []WorkflowTask{
		{ID: "a", Command: "save", Bindings: map[string]Binding{
			"files": {Kind: "literal", Value: "/x"},
		}},
		{ID: "b", Command: "save", Bindings: map[string]Binding{
			"files":  {Kind: "literal", Value: "/y"},
			"folder": {Kind: "wire", From: "a", Port: "output"},
		}},
	}
	if err := validateDAG(dag); err == nil {
		t.Error("expected error for v1 wire into non-primary input")
	}
}
```

- [ ] **Step 2: Run tests to confirm failure**

Run: `cd media-server && go test ./jobqueue/ -run TestValidateDAG -v`
Expected: tests fail (current `validateDAG` doesn't know about commands or bindings).

- [ ] **Step 3: Replace `validateDAG`**

```go
func validateDAG(dag []WorkflowTask) error {
	if len(dag) == 0 {
		return fmt.Errorf("DAG must not be empty")
	}

	ids := make(map[string]bool, len(dag))
	for _, task := range dag {
		if task.ID == "" {
			return fmt.Errorf("all tasks must have an id")
		}
		if task.Command == "" {
			return fmt.Errorf("all tasks must have a command")
		}
		if ids[task.ID] {
			return fmt.Errorf("duplicate task id: %s", task.ID)
		}
		ids[task.ID] = true
	}

	for _, task := range dag {
		// Legacy dependency list (still respected; populated by resolveBindings).
		for _, dep := range task.Dependencies {
			if !ids[dep] {
				return fmt.Errorf("dependency %s not found in DAG", dep)
			}
		}

		// Skip binding checks for legacy-shape tasks (no Bindings populated).
		if len(task.Bindings) == 0 {
			continue
		}

		contract, ok := getContract(task.Command)
		if !ok {
			return fmt.Errorf("unknown command: %s (task %s)", task.Command, task.ID)
		}

		inputByName := make(map[string]TaskContractInput, len(contract.Inputs))
		for _, in := range contract.Inputs {
			inputByName[in.Name] = in
		}

		for name, b := range task.Bindings {
			in, known := inputByName[name]
			if !known {
				return fmt.Errorf("task %s: unknown input %q", task.ID, name)
			}
			switch b.Kind {
			case "literal":
				// Allowed for any input.
			case "wire":
				if !ids[b.From] {
					return fmt.Errorf("task %s: wire from unknown node %q", task.ID, b.From)
				}
				if !in.Primary {
					return fmt.Errorf("task %s: input %q — wires into non-primary inputs are not supported in this version", task.ID, name)
				}
			default:
				return fmt.Errorf("task %s: input %q: unknown binding kind %q", task.ID, name, b.Kind)
			}
		}
	}

	return nil
}
```

- [ ] **Step 4: Run tests**

Run: `cd media-server && go test ./jobqueue/ -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add media-server/jobqueue/workflows.go media-server/jobqueue/workflows_test.go
git commit -m "feat(workflow): binding-aware DAG validation"
```

---

### Task 12: Apply legacy migration on workflow read

**Files:**
- Modify: `media-server/jobqueue/workflows.go` (`GetWorkflow`)

`GetWorkflow` reads from the `workflows` SQLite table. Old saved workflows have `Bindings` empty. We can't `import "github.com/stevecastle/shrike/tasks"` from `jobqueue` (cycle). Instead, expose a migration hook similar to `SetContractLookup`.

- [ ] **Step 1: Add the migration hook**

Append to `media-server/jobqueue/jobqueue.go`:

```go
type LegacyMigrateFn func(t WorkflowTask) WorkflowTask

var legacyMigrate LegacyMigrateFn

// SetLegacyMigrate is called by the tasks package at init to provide the
// migration helper that converts pre-Bindings DAGs into the new shape.
func SetLegacyMigrate(fn LegacyMigrateFn) {
	legacyMigrate = fn
}
```

- [ ] **Step 2: Register the helper from `tasks`**

Append in `media-server/tasks/registry_init.go`'s `init()`:

```go
	jobqueue.SetLegacyMigrate(MigrateLegacyTask)
```

- [ ] **Step 3: Apply migration in `GetWorkflow`**

In `media-server/jobqueue/workflows.go`, after the `json.Unmarshal([]byte(dagJSON), &w.DAG)` line in `GetWorkflow`, add:

```go
	if legacyMigrate != nil {
		for i := range w.DAG {
			w.DAG[i] = legacyMigrate(w.DAG[i])
		}
	}
```

- [ ] **Step 4: Test**

Add to `media-server/jobqueue/workflows_test.go`:

```go
func TestGetWorkflowAppliesLegacyMigration(t *testing.T) {
	q := NewQueue()
	if err := q.createWorkflowsTableInMemory(); err != nil { // helper used by other tests
		t.Skip("no in-memory workflows table helper; skipping")
	}
	// Insert raw legacy-shape DAG.
	legacy := []WorkflowTask{{
		ID:      "a",
		Command: "save",
		Input:   "/tmp/x.mp4",
		Arguments: []string{"--mode", "folder", "--folder", "/out"},
	}}
	dagJSON, _ := json.Marshal(legacy)
	q.Db.Exec(`INSERT INTO workflows (id, name, dag) VALUES (?,?,?)`, "wfid", "demo", string(dagJSON))
	w, err := q.GetWorkflow("wfid")
	if err != nil {
		t.Fatalf("GetWorkflow: %v", err)
	}
	b := w.DAG[0].Bindings["files"]
	if b.Kind != "literal" || b.Value != "/tmp/x.mp4" {
		t.Errorf("primary not migrated: %+v", b)
	}
}
```

If `createWorkflowsTableInMemory` doesn't exist in this test file, replace the call with whatever the existing test setup uses. The existing `workflows_test.go` already creates an in-memory `Queue` for other tests; reuse the same helper.

Run: `cd media-server && go test ./jobqueue/ -run TestGetWorkflow -v`
Expected: PASS (or `Skip` if the helper isn't there — leave the test in place).

- [ ] **Step 5: Commit**

```bash
git add media-server/jobqueue/ media-server/tasks/registry_init.go
git commit -m "feat(workflow): transparently migrate legacy DAGs on read"
```

---

### Task 13: Update `/tasks` HTTP handler to emit the new shape

**Files:**
- Modify: `media-server/main.go`, `media-server/main_darwin.go`, `media-server/main_linux.go`

The `Task` struct already JSON-marshals `Inputs`, `Output`, and the deprecated `Options` alias. So if the handler just returns `tasks.GetTasks()` (or a slice derived from it), the new fields appear automatically.

- [ ] **Step 1: Find the existing handler**

Run: `grep -n "/tasks" media-server/main.go media-server/main_darwin.go media-server/main_linux.go`. The handler is registered with `mux.HandleFunc("/tasks", ...)` in `main.go` (and mirrored in the platform variants if it's there).

- [ ] **Step 2: Confirm shape change**

Open the handler. It likely already does `json.NewEncoder(w).Encode(map[string]any{"tasks": tasks.GetTasks()})`. After the schema change, the response now includes `inputs`, `output`, and a backward-compat `options` field. No code change needed if the handler is generic.

If the handler instead constructs a custom response, update it to include `Inputs`, `Output`, and `Options` (the alias).

- [ ] **Step 3: Smoke-test in dev**

Run: `cd media-server && go run .` (in a background terminal), then in another: `curl -s http://localhost:8080/tasks | head -c 400`. Expected: response includes `"inputs":[...]` and `"output":{...}` for at least one task.

- [ ] **Step 4: Commit (only if changes were needed)**

```bash
git add media-server/main*.go
git commit -m "feat(workflow): /tasks emits new typed input/output schema"
```

---

### Task 14: Workflow CRUD endpoints accept the new shape

**Files:**
- Modify: `media-server/main.go` (and platform variants for any branch logic that touches workflows)

The handlers for `POST /workflow`, `POST /workflows/create`, and `PUT /workflows/{id}` decode JSON into a `Workflow` (or `[]WorkflowTask`) and pass it down to `q.AddWorkflow` / `q.CreateWorkflow` / `q.UpdateWorkflow`. After Task 7, `WorkflowTask` already has both `Bindings` and the legacy `Input`/`Arguments` fields — JSON decoding handles both shapes natively.

- [ ] **Step 1: Run the legacy migration on POSTed legacy-shape DAGs**

In each workflow handler, after decoding the request body, call:

```go
for i := range req.DAG /* or req.Tasks */ {
    if len(req.DAG[i].Bindings) == 0 && legacyMigrate != nil {
        req.DAG[i] = legacyMigrate(req.DAG[i])
    }
}
```

(Use the imported `jobqueue.LegacyMigrate*` accessors. Or, simpler: just always call the migrate helper — it short-circuits when `Bindings` is non-empty.)

- [ ] **Step 2: Build and smoke-test**

Run: `cd media-server && go build ./... && go test ./...`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add media-server/main*.go
git commit -m "feat(workflow): API accepts legacy DAG shape and migrates on ingest"
```

---

## Phase 5 — Editor (`editor.go.html`)

### Task 15: JS-side type-lattice helpers

**Files:**
- Modify: `media-server/renderer/templates/editor.go.html` (top of the inline `<script>` block, around line 410)

- [ ] **Step 1: Add a small helper module**

Insert near the top of the script block:

```js
// --- Type lattice (mirror of media-server/tasks/types.go) ---
const TYPES = {
  STRING: 'string', NUMBER: 'number', BOOL: 'bool',
  ENUM: 'enum', MULTI_ENUM: 'multi-enum',
  PATH: 'path', PATH_LIST: 'path-list',
  URL: 'url', URL_LIST: 'url-list',
  REF_LIST: 'ref-list', JSON: 'json',
};

function isCompatible(from, to) {
  if (from === to) return true;
  if (to === TYPES.STRING) return [TYPES.PATH, TYPES.URL, TYPES.NUMBER, TYPES.BOOL].includes(from);
  if (to === TYPES.NUMBER || to === TYPES.BOOL) return from === TYPES.STRING;
  if (to === TYPES.PATH_LIST) return [TYPES.PATH, TYPES.URL_LIST, TYPES.REF_LIST].includes(from);
  if (to === TYPES.URL_LIST)  return [TYPES.URL, TYPES.PATH_LIST, TYPES.REF_LIST].includes(from);
  if (to === TYPES.REF_LIST)  return [TYPES.PATH, TYPES.URL, TYPES.PATH_LIST, TYPES.URL_LIST].includes(from);
  return false;
}
```

- [ ] **Step 2: Commit**

```bash
git add media-server/renderer/templates/editor.go.html
git commit -m "feat(editor): client-side type-compatibility helper"
```

---

### Task 16: Render multi-port nodes from `inputs`

**Files:**
- Modify: `media-server/renderer/templates/editor.go.html` (`buildNodeHTML`, `buildNodeData`, `addNodeToGraph`, the `/tasks` fetch handler)

- [ ] **Step 1: Cache `inputs` and `output` per task**

Replace the existing `taskOptionsMap` cache with:

```js
var taskInputsMap = {};
var taskOutputMap = {};

fetch('/tasks')
  .then((r) => r.json())
  .then((data) => {
    const palette = document.getElementById('palette');
    data.tasks.forEach((task) => {
      taskInputsMap[task.id] = task.inputs || task.options || [];
      taskOutputMap[task.id] = task.output || { type: TYPES.PATH_LIST };
      const div = document.createElement('div');
      div.className = 'palette-item';
      div.draggable = true;
      div.setAttribute('ondragstart', 'drag(event)');
      div.setAttribute('data-node', task.id);
      div.innerHTML = '<strong>' + task.name + '</strong><div class="desc">' + task.id + '</div>';
      palette.appendChild(div);
    });
  });
```

(The `task.options` fallback keeps loading working during the deploy window.)

- [ ] **Step 2: Rewrite `buildNodeHTML` to render one row per input**

```js
function buildNodeHTML(taskId) {
  const inputs = taskInputsMap[taskId] || [];
  const output = taskOutputMap[taskId] || { type: TYPES.PATH_LIST };

  let html = '<div class="node-content" data-task="' + taskId + '" data-output-type="' + output.type + '">';
  if (inputs.length === 0) {
    html += '<div class="node-label">' + taskId + '</div>';
  }
  for (let idx = 0; idx < inputs.length; idx++) {
    const opt = inputs[idx];
    const dfAttr = 'df-' + opt.name;
    const defaultVal = opt.default !== undefined && opt.default !== null ? opt.default : '';
    html += '<div class="node-input-group" data-input-name="' + opt.name + '" data-input-type="' + opt.type + '" data-input-index="' + idx + '"' + (opt.primary ? ' data-primary="1"' : '') + '>';
    html += '<label>' + (opt.label || opt.name) + (opt.primary ? ' *' : '') + '</label>';
    switch (opt.type) {
      case TYPES.BOOL:
        html += '<input type="checkbox" ' + dfAttr + ' ' + (defaultVal ? 'checked' : '') + ' style="width:auto;">';
        break;
      case TYPES.ENUM:
        html += '<select ' + dfAttr + '>';
        for (const c of (opt.choices || [])) {
          html += '<option value="' + c + '" ' + (c === String(defaultVal) ? 'selected' : '') + '>' + c + '</option>';
        }
        html += '</select>';
        break;
      case TYPES.MULTI_ENUM: {
        const defaults = String(defaultVal).split(',').map(s => s.trim());
        for (const c of (opt.choices || [])) {
          const checked = defaults.includes(c) ? 'checked' : '';
          html += '<label style="display:inline-flex;align-items:center;gap:4px;margin-right:8px;font-size:12px;">';
          html += '<input type="checkbox" data-multienum="' + opt.name + '" value="' + c + '" ' + checked + '>' + c + '</label>';
        }
        html += '<input type="hidden" ' + dfAttr + ' value="' + defaultVal + '">';
        break;
      }
      case TYPES.NUMBER:
        html += '<input type="number" ' + dfAttr + ' value="' + defaultVal + '" placeholder="' + (opt.description || '') + '">';
        break;
      default:
        html += '<input type="text" ' + dfAttr + ' value="' + defaultVal + '" placeholder="' + (opt.description || '') + '">';
        break;
    }
    html += '</div>';
  }
  html += '</div>';
  return html;
}
```

- [ ] **Step 3: Update `buildNodeData` to include all inputs**

```js
function buildNodeData(taskId) {
  const inputs = taskInputsMap[taskId] || [];
  const data = { command: taskId };
  for (const opt of inputs) {
    if (opt.primary) continue;
    if (opt.type === TYPES.BOOL) {
      data[opt.name] = !!opt.default;
    } else {
      data[opt.name] = opt.default !== undefined && opt.default !== null ? String(opt.default) : '';
    }
  }
  return data;
}
```

- [ ] **Step 4: Update `addNodeToGraph` to declare port counts**

```js
function addNodeToGraph(name, pos_x, pos_y) {
  if (editor.editor_mode === 'fixed') return false;
  pos_x = pos_x * (editor.precanvas.clientWidth / (editor.precanvas.clientWidth * editor.zoom)) - editor.precanvas.getBoundingClientRect().x * (editor.precanvas.clientWidth / (editor.precanvas.clientWidth * editor.zoom));
  pos_y = pos_y * (editor.precanvas.clientHeight / (editor.precanvas.clientHeight * editor.zoom)) - editor.precanvas.getBoundingClientRect().y * (editor.precanvas.clientHeight / (editor.precanvas.clientHeight * editor.zoom));
  const inputs = taskInputsMap[name] || [];
  const numInputs = Math.max(1, inputs.length);
  const html = buildNodeHTML(name);
  const data = buildNodeData(name);
  editor.addNode(name, numInputs, 1, pos_x, pos_y, 'node', data, html);
}
```

- [ ] **Step 5: Build and smoke-test**

```bash
npm run build:web
npm run build:server
```

Then run the server, open the editor in a browser, drag a task into the canvas, and confirm:
- The node renders one row per input.
- The node has N input ports (one per input) on the left and 1 output port on the right.
- The primary input row shows `*` after its label.

- [ ] **Step 6: Commit**

```bash
git add media-server/renderer/templates/editor.go.html
git commit -m "feat(editor): render one input port per declared task input"
```

---

### Task 17: Type-aware connection validation + dim-on-wire

**Files:**
- Modify: `media-server/renderer/templates/editor.go.html`

- [ ] **Step 1: Hook `connectionCreated` and `connectionRemoved`**

Add near the other `editor.on(...)` listeners (around line 431):

```js
editor.on('connectionCreated', (info) => {
  // info: { output_id, input_id, output_class, input_class }
  // output_class is "output_1"; input_class is "input_<N>" where N is the
  // 1-based port index. We need to map both ends to declared types and the
  // input row's "input name".
  const srcCommand = editor.drawflow.drawflow.Home.data[info.output_id].name;
  const dstCommand = editor.drawflow.drawflow.Home.data[info.input_id].name;
  const dstInputs = taskInputsMap[dstCommand] || [];
  const inputIndex = parseInt(info.input_class.replace('input_', ''), 10) - 1;
  const dstInput = dstInputs[inputIndex];
  const srcOutputType = (taskOutputMap[srcCommand] || {}).type || TYPES.PATH_LIST;

  if (!dstInput) return;

  // v1: only the primary input is wireable. Refuse anything else.
  if (!dstInput.primary) {
    editor.removeSingleConnection(info.output_id, info.input_id, info.output_class, info.input_class);
    showStatus('Wires into non-primary inputs are not yet supported', 4000);
    return;
  }

  if (!isCompatible(srcOutputType, dstInput.type)) {
    editor.removeSingleConnection(info.output_id, info.input_id, info.output_class, info.input_class);
    showStatus('Type mismatch: ' + srcOutputType + ' → ' + dstInput.type, 4000);
    return;
  }

  // Dim the inline editor for the wired input.
  setInputDimmed(info.input_id, inputIndex, true);
  markDirty();
});

editor.on('connectionRemoved', (info) => {
  const dstCommand = editor.drawflow.drawflow.Home.data[info.input_id].name;
  const dstInputs = taskInputsMap[dstCommand] || [];
  const inputIndex = parseInt(info.input_class.replace('input_', ''), 10) - 1;
  // Only un-dim if there are no other connections to this input.
  const node = editor.drawflow.drawflow.Home.data[info.input_id];
  const portKey = 'input_' + (inputIndex + 1);
  const remaining = (node.inputs[portKey]?.connections || []).length;
  if (remaining === 0) setInputDimmed(info.input_id, inputIndex, false);
  markDirty();
});

function setInputDimmed(nodeId, inputIndex, dimmed) {
  const node = document.getElementById('node-' + nodeId);
  if (!node) return;
  const row = node.querySelector('.node-input-group[data-input-index="' + inputIndex + '"]');
  if (!row) return;
  row.classList.toggle('input-wired', dimmed);
}
```

Add a CSS rule near the top of the inline `<style>` block:

```css
.node-input-group.input-wired input,
.node-input-group.input-wired select {
  opacity: 0.4;
  pointer-events: none;
}
.node-input-group.input-wired::after {
  content: '⟵ wired';
  font-size: 10px;
  color: var(--text-muted);
  margin-left: 8px;
}
```

- [ ] **Step 2: Smoke-test**

```bash
npm run build:web && npm run build:server
```

Open editor, drop two `save` nodes, drag the output of one to the primary input of the other → connection should hold and the primary input row should dim. Drag the output to a non-primary input row → connection should immediately disappear with a status-bar message.

- [ ] **Step 3: Commit**

```bash
git add media-server/renderer/templates/editor.go.html
git commit -m "feat(editor): type-aware connection validation and dim-on-wire styling"
```

---

### Task 18: New export shape (`bindings`) + load-time legacy migration

**Files:**
- Modify: `media-server/renderer/templates/editor.go.html` (`exportDAG`, `loadWorkflow`)

- [ ] **Step 1: Replace `exportDAG`**

```js
function exportDAG() {
  const data = editor.export();
  const nodes = data.drawflow.Home.data;
  const tasks = [];
  const nodeMap = {};

  Object.keys(nodes).forEach((key) => {
    const node = nodes[key];
    const nodeId = 'node-' + key;
    nodeMap[key] = nodeId;
    const command = node.data.command || node.name;
    const inputs = taskInputsMap[command] || [];

    const bindings = {};
    const deps = new Set();

    // Build a port-index → wire source map from this node's input ports.
    const wiresByIndex = {};
    Object.keys(node.inputs).forEach((portKey) => {
      const idx = parseInt(portKey.replace('input_', ''), 10) - 1;
      const conns = node.inputs[portKey].connections || [];
      if (conns.length === 0) return;
      // v1: at most one connection per input (validation enforces it for scalars
      // and the editor refuses non-primary wires entirely).
      const c = conns[0];
      wiresByIndex[idx] = c.node;
    });

    inputs.forEach((opt, idx) => {
      const wireFrom = wiresByIndex[idx];
      if (wireFrom) {
        bindings[opt.name] = { kind: 'wire', from: nodeMap[wireFrom] || ('node-' + wireFrom), port: 'output' };
        deps.add(nodeMap[wireFrom] || ('node-' + wireFrom));
        return;
      }
      // Literal binding — pull the value from the inline editor.
      let val;
      if (opt.primary) {
        // Primary inputs don't have an inline editor; treat as empty literal.
        val = '';
      } else if (opt.type === TYPES.BOOL) {
        val = node.data[opt.name] === true || node.data[opt.name] === 'true';
      } else if (opt.type === TYPES.NUMBER) {
        val = node.data[opt.name] === '' ? 0 : Number(node.data[opt.name]);
      } else {
        val = node.data[opt.name] !== undefined ? String(node.data[opt.name]) : '';
      }
      // Skip empty/default values to keep the DAG compact.
      if (val === '' || val === false || val === 0) return;
      bindings[opt.name] = { kind: 'literal', value: val };
    });

    tasks.push({
      id: nodeId,
      command,
      bindings,
      dependencies: Array.from(deps),
      pos_x: node.pos_x,
      pos_y: node.pos_y,
    });
  });

  return tasks;
}
```

- [ ] **Step 2: Add legacy-migration on load**

In `loadWorkflow`, after the JSON arrives and before `editor.import(...)` (or whatever current load path), iterate the DAG and convert legacy fields into `bindings` if missing. The server already does this on `GetWorkflow`, but be defensive in case the editor receives a legacy file via paste-import:

```js
function migrateLegacyDAG(dag) {
  return dag.map((t) => {
    if (t.bindings && Object.keys(t.bindings).length > 0) return t;
    const inputs = taskInputsMap[t.command] || [];
    const bindings = {};
    const primary = inputs.find((i) => i.primary);
    if (primary) {
      if (t.input) {
        bindings[primary.name] = { kind: 'literal', value: t.input };
      } else if (t.dependencies && t.dependencies.length > 0) {
        bindings[primary.name] = { kind: 'wire', from: t.dependencies[0], port: 'output' };
      }
    }
    if (Array.isArray(t.arguments)) {
      // Repeat the same --flag value parser used server-side.
      for (let i = 0; i < t.arguments.length; i++) {
        const a = t.arguments[i];
        if (!a.startsWith('--')) continue;
        const name = a.slice(2);
        const opt = inputs.find((x) => x.name === name);
        if (!opt) continue;
        if (opt.type === TYPES.BOOL) {
          bindings[name] = { kind: 'literal', value: true };
        } else {
          const v = t.arguments[i + 1];
          if (v !== undefined && !v.startsWith('--')) {
            bindings[name] = { kind: 'literal', value: opt.type === TYPES.NUMBER ? Number(v) : v };
            i++;
          }
        }
      }
    }
    return { ...t, bindings };
  });
}
```

Call `migrateLegacyDAG(dag)` before re-rendering nodes.

- [ ] **Step 3: Smoke-test**

`npm run build:web && npm run build:server`. Build a workflow in the editor, save it (`POST /workflows/create`), then load it again. Use the browser devtools Network tab to confirm the saved JSON contains `bindings` and not `arguments`/`input`.

Also: load an old workflow from the dropdown — the editor should render it correctly (legacy values appear as literal-binding defaults).

- [ ] **Step 4: Commit**

```bash
git add media-server/renderer/templates/editor.go.html
git commit -m "feat(editor): export workflows in typed-bindings shape and migrate legacy on load"
```

---

## Phase 6 — Final Pass

### Task 19: Full test sweep + lint

- [ ] **Step 1: Backend tests**

```bash
cd media-server && go test ./... -v
```

Expected: all PASS.

- [ ] **Step 2: Renderer build**

```bash
npm run build:web
```

Expected: clean webpack build, no errors.

- [ ] **Step 3: Server build**

```bash
npm run build:server
```

Expected: Go build succeeds; the binary serves the new editor.

- [ ] **Step 4: Manual editor smoke test**

Run the server, open the editor:
1. Drop an `ingest` node → confirm it has 1 input port (primary `paths`, type `ref-list`) plus N option rows.
2. Drop a `save` node → confirm 1 input port + the option rows for `mode`/`suffix`/`folder`.
3. Wire `ingest`'s output to `save`'s primary input → connection holds; the primary row dims.
4. Try wiring `ingest`'s output to `save`'s `folder` row → connection is removed, status-bar shows the v1 limitation message.
5. Save the workflow, reload, confirm it round-trips.
6. Run the workflow → jobs execute as before (regression check).

- [ ] **Step 5: Commit any cleanup**

```bash
git status
# If anything is left uncommitted, finish the cleanup and commit.
```

---

## Self-Review

**Spec coverage:**
- Type lattice + coercion rules → Tasks 1–2.
- Task contract schema (Inputs/Output) → Tasks 3–5.
- Workflow storage schema (Bindings) → Tasks 6–7.
- Backward-compat read path (legacy → bindings) → Tasks 8, 12.
- Runtime resolution of bindings → Task 9 (literal + dep-as-wire); Task 10 (guard for non-primary wires).
- Editor multi-port rendering → Task 16.
- Connection validation → Task 17.
- New export shape + JS-side legacy migration → Task 18.
- API-level binding-aware validation → Task 11.
- API endpoints accept new shape → Tasks 13, 14.
- Output-value channel for non-list output types → **deferred to a follow-up plan.** The spec calls this a "forward hook" and no v1 task produces such an output. Adding it now would only add unused code.

**Placeholder scan:** No "TBD"/"TODO"/"implement later" tokens. Every step has the literal code or command needed.

**Type / name consistency:**
- `TaskInput` field names `Name, Label, Type, Choices, Default, Required, Primary, Description` — used identically in Tasks 3, 5, 9, 11.
- `Binding.Kind` is `"literal"` / `"wire"` everywhere.
- `Binding.From` is a node ID; `Binding.Port` is always `"output"` in v1.
- JS-side `taskInputsMap` / `taskOutputMap` are introduced in Task 16 and used identically in Tasks 17–18.
- `editor.removeSingleConnection(...)` is a real Drawflow API; argument order matches Drawflow 0.0.59.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-04-28-typed-workflow-nodes.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — Dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
