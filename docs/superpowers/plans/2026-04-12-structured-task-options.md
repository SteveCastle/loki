# Structured Task Options Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace raw string arguments with structured option definitions for all tasks, render dynamic widgets in the Drawflow editor, and split ffmpeg into focused preset tasks.

**Architecture:** `TaskOption` type defines each task's options as typed metadata. `ParseOptions` parses `j.Arguments` against the schema. Each task registers with its options. The `/tasks` API includes options in the response. The editor generates form widgets dynamically from the schema.

**Tech Stack:** Go (tasks package), HTML/JS (Drawflow editor)

---

## File Structure

| File | Responsibility |
|------|---------------|
| `media-server/tasks/options.go` | **Create** — TaskOption type, ParseOptions function |
| `media-server/tasks/options_test.go` | **Create** — Tests for ParseOptions |
| `media-server/tasks/registry.go` | **Modify** — Add Options to Task, update RegisterTask signature |
| `media-server/tasks/ffmpeg.go` | **Modify** — Extract shared runFFmpegOnFiles helper, keep custom task |
| `media-server/tasks/ffmpeg_presets.go` | **Create** — Scale, convert, extract-audio, screenshot, thumbnail tasks |
| `media-server/tasks/media_metadata.go` | **Modify** — Switch to ParseOptions |
| `media-server/tasks/hls.go` | **Modify** — Switch to ParseOptions |
| `media-server/tasks/media_move.go` | **Modify** — Switch to ParseOptions |
| `media-server/tasks/media_ingest.go` | **Modify** — Switch to ParseOptions |
| `media-server/tasks/lora_dataset.go` | **Modify** — Switch to ParseOptions |
| `media-server/main.go` | **Modify** — Update TaskInfo and tasksHandler to include options |
| `media-server/renderer/templates/editor.go.html` | **Modify** — Dynamic node rendering with option widgets |

---

### Task 1: TaskOption type and ParseOptions

**Files:**
- Create: `media-server/tasks/options.go`
- Create: `media-server/tasks/options_test.go`

- [ ] **Step 1: Create the TaskOption type and ParseOptions function**

Create `media-server/tasks/options.go`:

```go
package tasks

import (
	"strconv"
	"strings"

	"github.com/stevecastle/shrike/jobqueue"
)

// TaskOption describes a single configurable option for a task.
type TaskOption struct {
	Name        string   `json:"name"`
	Label       string   `json:"label"`
	Type        string   `json:"type"` // "string", "bool", "enum", "multi-enum", "number"
	Choices     []string `json:"choices,omitempty"`
	Default     any      `json:"default,omitempty"`
	Required    bool     `json:"required,omitempty"`
	Description string   `json:"description,omitempty"`
}

// ParseOptions parses j.Arguments against the option schema and returns a typed map.
// Handles --flag value, --flag=value, and bare --flag (for bools).
// Missing optional values get their defaults.
func ParseOptions(j *jobqueue.Job, options []TaskOption) map[string]any {
	result := make(map[string]any)

	// Apply defaults first
	for _, opt := range options {
		if opt.Default != nil {
			result[opt.Name] = opt.Default
		} else {
			switch opt.Type {
			case "bool":
				result[opt.Name] = false
			case "number":
				result[opt.Name] = 0.0
			case "string", "enum":
				result[opt.Name] = ""
			case "multi-enum":
				result[opt.Name] = ""
			}
		}
	}

	// Build a lookup map for option names
	optMap := make(map[string]*TaskOption, len(options))
	for i := range options {
		optMap[options[i].Name] = &options[i]
	}

	args := j.Arguments
	for i := 0; i < len(args); i++ {
		arg := args[i]
		if !strings.HasPrefix(arg, "--") {
			continue
		}

		// Handle --flag=value
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
		case "bool":
			if hasEquals {
				result[key] = value == "true" || value == "1" || value == "yes"
			} else {
				result[key] = true
			}
		case "number":
			if !hasEquals && i+1 < len(args) && !strings.HasPrefix(args[i+1], "--") {
				i++
				value = args[i]
			}
			if n, err := strconv.ParseFloat(value, 64); err == nil {
				result[key] = n
			}
		default: // string, enum, multi-enum
			if !hasEquals && i+1 < len(args) && !strings.HasPrefix(args[i+1], "--") {
				i++
				value = args[i]
			}
			result[key] = value
		}
	}

	return result
}

// OptionsToArgs converts a parsed options map back to a []string argument list.
// Used when tasks need to pass options to sub-tasks.
func OptionsToArgs(opts map[string]any, schema []TaskOption) []string {
	var args []string
	for _, opt := range schema {
		v, ok := opts[opt.Name]
		if !ok {
			continue
		}
		switch opt.Type {
		case "bool":
			if b, ok := v.(bool); ok && b {
				args = append(args, "--"+opt.Name)
			}
		case "number":
			switch n := v.(type) {
			case float64:
				if n != 0 {
					args = append(args, "--"+opt.Name, strconv.FormatFloat(n, 'f', -1, 64))
				}
			}
		default:
			if s, ok := v.(string); ok && s != "" {
				args = append(args, "--"+opt.Name, s)
			}
		}
	}
	return args
}
```

- [ ] **Step 2: Write tests**

Create `media-server/tasks/options_test.go`:

```go
package tasks

import (
	"testing"

	"github.com/stevecastle/shrike/jobqueue"
)

func TestParseOptionsDefaults(t *testing.T) {
	opts := []TaskOption{
		{Name: "overwrite", Type: "bool", Default: false},
		{Name: "type", Type: "multi-enum", Default: "description,hash"},
		{Name: "width", Type: "number", Default: 1280.0},
	}
	j := &jobqueue.Job{Arguments: []string{}}
	result := ParseOptions(j, opts)

	if result["overwrite"] != false {
		t.Errorf("overwrite = %v, want false", result["overwrite"])
	}
	if result["type"] != "description,hash" {
		t.Errorf("type = %v, want description,hash", result["type"])
	}
	if result["width"] != 1280.0 {
		t.Errorf("width = %v, want 1280", result["width"])
	}
}

func TestParseOptionsBoolFlag(t *testing.T) {
	opts := []TaskOption{
		{Name: "overwrite", Type: "bool"},
		{Name: "recursive", Type: "bool"},
	}
	j := &jobqueue.Job{Arguments: []string{"--overwrite"}}
	result := ParseOptions(j, opts)

	if result["overwrite"] != true {
		t.Errorf("overwrite = %v, want true", result["overwrite"])
	}
	if result["recursive"] != false {
		t.Errorf("recursive = %v, want false", result["recursive"])
	}
}

func TestParseOptionsKeyValue(t *testing.T) {
	opts := []TaskOption{
		{Name: "type", Type: "multi-enum"},
		{Name: "apply", Type: "enum"},
	}
	j := &jobqueue.Job{Arguments: []string{"--type", "transcript,hash", "--apply", "all"}}
	result := ParseOptions(j, opts)

	if result["type"] != "transcript,hash" {
		t.Errorf("type = %v", result["type"])
	}
	if result["apply"] != "all" {
		t.Errorf("apply = %v", result["apply"])
	}
}

func TestParseOptionsEquals(t *testing.T) {
	opts := []TaskOption{
		{Name: "format", Type: "enum"},
		{Name: "overwrite", Type: "bool"},
	}
	j := &jobqueue.Job{Arguments: []string{"--format=webm", "--overwrite=true"}}
	result := ParseOptions(j, opts)

	if result["format"] != "webm" {
		t.Errorf("format = %v", result["format"])
	}
	if result["overwrite"] != true {
		t.Errorf("overwrite = %v", result["overwrite"])
	}
}

func TestParseOptionsNumber(t *testing.T) {
	opts := []TaskOption{
		{Name: "width", Type: "number"},
	}
	j := &jobqueue.Job{Arguments: []string{"--width", "1920"}}
	result := ParseOptions(j, opts)

	if result["width"] != 1920.0 {
		t.Errorf("width = %v", result["width"])
	}
}

func TestParseOptionsUnknownFlagsIgnored(t *testing.T) {
	opts := []TaskOption{
		{Name: "type", Type: "string"},
	}
	j := &jobqueue.Job{Arguments: []string{"--unknown", "foo", "--type", "hash"}}
	result := ParseOptions(j, opts)

	if result["type"] != "hash" {
		t.Errorf("type = %v", result["type"])
	}
	if _, ok := result["unknown"]; ok {
		t.Error("unknown flag should not be in result")
	}
}

func TestOptionsToArgs(t *testing.T) {
	schema := []TaskOption{
		{Name: "overwrite", Type: "bool"},
		{Name: "type", Type: "multi-enum"},
		{Name: "width", Type: "number"},
		{Name: "target", Type: "string"},
	}
	opts := map[string]any{
		"overwrite": true,
		"type":      "transcript,hash",
		"width":     1280.0,
		"target":    "/out",
	}
	args := OptionsToArgs(opts, schema)

	// Should contain --overwrite, --type transcript,hash, --width 1280, --target /out
	found := map[string]bool{}
	for i := 0; i < len(args); i++ {
		if args[i] == "--overwrite" {
			found["overwrite"] = true
		}
		if args[i] == "--type" && i+1 < len(args) && args[i+1] == "transcript,hash" {
			found["type"] = true
		}
		if args[i] == "--width" && i+1 < len(args) && args[i+1] == "1280" {
			found["width"] = true
		}
		if args[i] == "--target" && i+1 < len(args) && args[i+1] == "/out" {
			found["target"] = true
		}
	}
	for _, k := range []string{"overwrite", "type", "width", "target"} {
		if !found[k] {
			t.Errorf("missing %s in args: %v", k, args)
		}
	}
}
```

- [ ] **Step 3: Run tests**

Run: `cd media-server && go test ./tasks/ -run TestParseOptions -v && go test ./tasks/ -run TestOptionsToArgs -v`
Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add media-server/tasks/options.go media-server/tasks/options_test.go
git commit -m "feat: add TaskOption type and ParseOptions function"
```

---

### Task 2: Registry refactor and task re-registration

**Files:**
- Modify: `media-server/tasks/registry.go`

- [ ] **Step 1: Update Task struct and RegisterTask**

Replace the contents of `media-server/tasks/registry.go` with:

```go
package tasks

import (
	"sync"

	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/storage"
)

// TaskFn is the signature for task execution functions.
type TaskFn func(j *jobqueue.Job, q *jobqueue.Queue, r *sync.Mutex) error

// Task represents a runnable unit bound to the jobqueue.
type Task struct {
	ID      string       `json:"id"`
	Name    string       `json:"name"`
	Options []TaskOption `json:"options"`
	Fn      TaskFn       `json:"-"`
}

type TaskMap map[string]Task

var tasks = make(TaskMap)

var storageReg *storage.Registry

func SetStorageRegistry(r *storage.Registry) {
	storageReg = r
}

func init() {
	// Tasks with no options
	RegisterTask("wait", "Wait", nil, waitFn)
	RegisterTask("remove", "Remove Media", nil, removeFromDB)
	RegisterTask("cleanup", "CleanUp", nil, cleanUpFn)
	RegisterTask("autotag", "Auto Tag (ONNX)", nil, autotagTask)

	// Tasks with options (defined inline for readability)
	RegisterTask("metadata", "Generate Metadata", metadataOptions, metadataTask)
	RegisterTask("hls", "HLS Transcode", hlsOptions, hlsTask)
	RegisterTask("move", "Move Media Files", moveOptions, moveTask)
	RegisterTask("ingest", "Ingest Media Files", ingestOptions, ingestTask)
	RegisterTask("lora-dataset", "Create LoRA Dataset", loraDatasetOptions, loraDatasetTask)

	// FFmpeg tasks
	RegisterTask("ffmpeg", "FFmpeg (Custom)", ffmpegCustomOptions, ffmpegTask)
	RegisterTask("ffmpeg-scale", "Scale Media", ffmpegScaleOptions, ffmpegScaleTask)
	RegisterTask("ffmpeg-convert", "Convert Format", ffmpegConvertOptions, ffmpegConvertTask)
	RegisterTask("ffmpeg-extract-audio", "Extract Audio", ffmpegExtractAudioOptions, ffmpegExtractAudioTask)
	RegisterTask("ffmpeg-screenshot", "Screenshot", ffmpegScreenshotOptions, ffmpegScreenshotTask)
	RegisterTask("ffmpeg-thumbnail", "Create Thumbnail", ffmpegThumbnailOptions, ffmpegThumbnailTask)
}

func RegisterTask(id, name string, options []TaskOption, fn TaskFn) {
	tasks[id] = Task{
		ID:      id,
		Name:    name,
		Options: options,
		Fn:      fn,
	}
}

func GetTasks() TaskMap {
	return tasks
}
```

Note: The `gallery-dl`, `dce`, `yt-dlp` registrations are removed. The option variables (`metadataOptions`, `hlsOptions`, etc.) are defined in their respective task files — they'll be created in subsequent tasks. For now, this file won't compile until all option vars are defined. That's expected.

- [ ] **Step 2: Commit (partial — won't compile yet)**

```bash
git add media-server/tasks/registry.go
git commit -m "feat: update registry with Options field and new task registrations"
```

---

### Task 3: Define option variables for existing tasks

**Files:**
- Modify: `media-server/tasks/media_metadata.go`
- Modify: `media-server/tasks/hls.go`
- Modify: `media-server/tasks/media_move.go`
- Modify: `media-server/tasks/media_ingest.go`
- Modify: `media-server/tasks/lora_dataset.go`

- [ ] **Step 1: Add metadataOptions and switch to ParseOptions**

At the top of `media-server/tasks/media_metadata.go` (after the imports), add the options variable:

```go
var metadataOptions = []TaskOption{
	{Name: "type", Label: "Metadata Types", Type: "multi-enum", Choices: []string{"description", "transcript", "hash", "dimensions", "autotag"}, Default: "description,hash,dimensions", Description: "Metadata types to generate"},
	{Name: "overwrite", Label: "Overwrite", Type: "bool", Default: false, Description: "Overwrite existing metadata"},
	{Name: "apply", Label: "Apply Scope", Type: "enum", Choices: []string{"new", "all"}, Default: "new", Description: "Which files to process"},
	{Name: "model", Label: "Ollama Model", Type: "string", Description: "Ollama model name (default from config)"},
}
```

Then replace the manual argument parsing block in `metadataTask` (the `for i, arg := range j.Arguments` loop, approximately lines 23-43) with:

```go
	opts := ParseOptions(j, metadataOptions)

	metadataTypesStr, _ := opts["type"].(string)
	var metadataTypes []string
	if metadataTypesStr != "" {
		for _, t := range strings.Split(metadataTypesStr, ",") {
			metadataTypes = append(metadataTypes, strings.TrimSpace(t))
		}
	}
	overwrite, _ := opts["overwrite"].(bool)
	applyScope, _ := opts["apply"].(string)
	if applyScope == "" {
		applyScope = "new"
	}
	ollamaModel, _ := opts["model"].(string)
	if ollamaModel == "" {
		ollamaModel = appconfig.Get().OllamaModel
	}
```

Remove the old `var metadataTypes []string`, `var overwrite bool`, `var applyScope string = "new"`, and `var ollamaModel string = appconfig.Get().OllamaModel` declarations that preceded the for loop.

- [ ] **Step 2: Add hlsOptions and switch to ParseOptions**

At the top of `media-server/tasks/hls.go` (after the imports), add:

```go
var hlsOptions = []TaskOption{
	{Name: "preset", Label: "Preset Mode", Type: "enum", Choices: []string{"passthrough", "adaptive"}, Default: "passthrough", Description: "HLS preset mode"},
	{Name: "presets", Label: "Quality Tiers", Type: "multi-enum", Choices: []string{"480p", "720p", "1080p"}, Description: "Explicit quality tiers (used with adaptive)"},
}
```

Replace the manual `--preset`/`--presets` parsing loop in `hlsTask` (approximately lines 48-86) with:

```go
	opts := ParseOptions(j, hlsOptions)
	presetMode, _ := opts["preset"].(string)
	if presetMode == "" {
		presetMode = "passthrough"
	}
	var requestedPresets []string
	if presetsStr, _ := opts["presets"].(string); presetsStr != "" {
		for _, p := range strings.Split(presetsStr, ",") {
			p = strings.TrimSpace(p)
			if p != "" {
				requestedPresets = append(requestedPresets, p)
			}
		}
	}
```

Remove the old `presetMode := "passthrough"` and `var requestedPresets []string` declarations and the entire for loop that parsed `j.Arguments`.

- [ ] **Step 3: Add moveOptions and switch to ParseOptions**

At the top of `media-server/tasks/media_move.go` (after the imports), add:

```go
var moveOptions = []TaskOption{
	{Name: "target", Label: "Target Directory", Type: "string", Required: true, Description: "Directory to move files to"},
	{Name: "prefix", Label: "Path Prefix", Type: "string", Description: "Prefix to strip from source paths (auto-detected if omitted)"},
}
```

Replace the manual argument parsing in `moveTask` (approximately lines 17-27) with:

```go
	opts := ParseOptions(j, moveOptions)
	targetDir, _ := opts["target"].(string)
	specifiedPrefix, _ := opts["prefix"].(string)
	if targetDir == "" {
		// Fallback: check for positional arg in Arguments for backwards compat
		for _, arg := range j.Arguments {
			if !strings.HasPrefix(arg, "-") && targetDir == "" {
				targetDir = strings.TrimSpace(arg)
			}
		}
	}
```

Remove the old `var targetDir string` and `var specifiedPrefix string` declarations and the for loop.

- [ ] **Step 4: Add ingestOptions and switch to ParseOptions**

At the top of `media-server/tasks/media_ingest.go` (after the imports, before `IngestOptions`), add:

```go
var ingestOptions = []TaskOption{
	{Name: "recursive", Label: "Recursive", Type: "bool", Default: false, Description: "Scan directories recursively"},
	{Name: "transcript", Label: "Generate Transcript", Type: "bool", Default: false, Description: "Queue transcript generation"},
	{Name: "description", Label: "Generate Description", Type: "bool", Default: false, Description: "Queue description generation"},
	{Name: "filemeta", Label: "File Metadata", Type: "bool", Default: false, Description: "Queue hash and dimensions"},
	{Name: "autotag", Label: "Auto Tag", Type: "bool", Default: false, Description: "Queue ONNX auto-tagging"},
}
```

Then update `parseIngestOptions` to use `ParseOptions` internally:

```go
func parseIngestOptions(args []string) (IngestOptions, []string) {
	// Create a temporary job to use ParseOptions
	tempJob := &jobqueue.Job{Arguments: args}
	parsed := ParseOptions(tempJob, ingestOptions)

	opts := IngestOptions{
		Recursive:   boolOpt(parsed, "recursive"),
		Transcript:  boolOpt(parsed, "transcript"),
		Description: boolOpt(parsed, "description"),
		FileMeta:    boolOpt(parsed, "filemeta"),
		AutoTag:     boolOpt(parsed, "autotag"),
	}

	// Extract --tag= args and collect remaining non-option args
	var remaining []string
	for _, arg := range args {
		lower := strings.ToLower(arg)
		if strings.HasPrefix(lower, "--tag=") {
			value := arg[len("--tag="):]
			label, category := parseTagArg(value)
			if label != "" {
				opts.Tags = append(opts.Tags, TagInfo{Label: label, Category: category})
			}
		} else if !strings.HasPrefix(arg, "--") && !strings.HasPrefix(arg, "-") {
			remaining = append(remaining, arg)
		}
	}

	return opts, remaining
}

func boolOpt(m map[string]any, key string) bool {
	v, _ := m[key].(bool)
	return v
}
```

- [ ] **Step 5: Add loraDatasetOptions and switch to ParseOptions**

At the top of `media-server/tasks/lora_dataset.go` (after the imports), add:

```go
var loraDatasetOptions = []TaskOption{
	{Name: "target", Label: "Target Directory", Type: "string", Required: true, Description: "Output directory for dataset"},
	{Name: "name", Label: "Dataset Name", Type: "string", Required: true, Description: "Name for the LoRA dataset"},
	{Name: "prefix", Label: "Concept Prefix", Type: "string", Description: "Prefix prepended to descriptions"},
	{Name: "model", Label: "Ollama Model", Type: "string", Description: "Ollama model for descriptions (default from config)"},
}
```

Replace the manual argument parsing in `loraDatasetTask` (the `for i, arg := range j.Arguments` loop) with:

```go
	opts := ParseOptions(j, loraDatasetOptions)
	targetDir, _ := opts["target"].(string)
	loraName, _ := opts["name"].(string)
	conceptPrefix, _ := opts["prefix"].(string)
	ollamaModel, _ := opts["model"].(string)
	if ollamaModel == "" {
		ollamaModel = appconfig.Get().OllamaModel
	}
```

Remove the old variable declarations and the for loop.

- [ ] **Step 6: Verify compilation**

Run: `cd media-server && go build ./...`
Expected: Compiles. (The ffmpeg preset tasks aren't defined yet but they're referenced in registry.go — if this doesn't compile, temporarily comment out the ffmpeg-* registrations and uncomment in Task 4.)

- [ ] **Step 7: Run existing tests**

Run: `cd media-server && go test ./tasks/ -v`
Expected: All existing tests still pass (ParseOptions is backwards-compatible with the same argument format).

- [ ] **Step 8: Commit**

```bash
git add media-server/tasks/media_metadata.go media-server/tasks/hls.go media-server/tasks/media_move.go media-server/tasks/media_ingest.go media-server/tasks/lora_dataset.go
git commit -m "feat: migrate existing tasks to structured ParseOptions"
```

---

### Task 4: FFmpeg refactor — shared helper and preset tasks

**Files:**
- Modify: `media-server/tasks/ffmpeg.go`
- Create: `media-server/tasks/ffmpeg_presets.go`

- [ ] **Step 1: Extract shared helper and add custom options**

Replace `media-server/tasks/ffmpeg.go` with:

```go
package tasks

import (
	"bufio"
	"fmt"
	"path/filepath"
	"strconv"
	"strings"
	"sync"

	"github.com/stevecastle/shrike/deps"
	"github.com/stevecastle/shrike/jobqueue"
)

var ffmpegCustomOptions = []TaskOption{
	{Name: "arguments", Label: "FFmpeg Arguments", Type: "string", Required: true, Description: "Raw ffmpeg args. Templates: {input}, {dir}, {base}, {name}, {ext}, {idx}"},
}

// runFFmpegOnFiles is the shared execution loop for all ffmpeg-based tasks.
// buildArgs receives the absolute source path and returns ffmpeg arguments (without -i input).
// The function handles file gathering, cancellation, progress, and output chaining.
func runFFmpegOnFiles(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex, buildArgs func(src, dir, name, ext string) (args []string, outputPath string)) error {
	ctx := j.Ctx

	var files []string
	if qstr, ok := extractQueryFromJob(j); ok {
		q.PushJobStdout(j.ID, fmt.Sprintf("ffmpeg: using query: %s", qstr))
		mediaPaths, err := getMediaPathsByQueryFast(q.Db, qstr)
		if err != nil {
			q.PushJobStdout(j.ID, "ffmpeg: query failed: "+err.Error())
			q.ErrorJob(j.ID)
			return err
		}
		files = mediaPaths
	} else {
		raw := strings.TrimSpace(j.Input)
		if raw == "" {
			q.PushJobStdout(j.ID, "ffmpeg: no input")
			q.CompleteJob(j.ID)
			return nil
		}
		files = parseInputPaths(raw)
	}

	if len(files) == 0 {
		q.PushJobStdout(j.ID, "ffmpeg: no files to process")
		q.CompleteJob(j.ID)
		return nil
	}

	for _, src := range files {
		select {
		case <-ctx.Done():
			q.PushJobStdout(j.ID, "ffmpeg: canceled")
			q.ErrorJob(j.ID)
			return ctx.Err()
		default:
		}

		abs := src
		if a, err := filepath.Abs(src); err == nil {
			abs = filepath.FromSlash(a)
		}
		dir := filepath.Dir(abs)
		base := filepath.Base(abs)
		ext := filepath.Ext(abs)
		name := strings.TrimSuffix(base, ext)

		taskArgs, outputPath := buildArgs(abs, dir, name, ext)
		finalArgs := append([]string{"-i", abs}, taskArgs...)

		q.PushJobStdout(j.ID, fmt.Sprintf("ffmpeg: %s -> %s", base, filepath.Base(outputPath)))

		cmd, err := deps.GetExec(ctx, "ffmpeg", "ffmpeg", finalArgs...)
		if err != nil {
			q.PushJobStdout(j.ID, "ffmpeg: prepare failed: "+err.Error())
			q.ErrorJob(j.ID)
			return err
		}

		stdout, _ := cmd.StdoutPipe()
		stderr, _ := cmd.StderrPipe()

		doneErr := make(chan struct{})
		go func() {
			s := bufio.NewScanner(stderr)
			for s.Scan() {
				_ = q.PushJobStdout(j.ID, "ffmpeg: "+s.Text())
			}
			close(doneErr)
		}()

		if err := cmd.Start(); err != nil {
			q.PushJobStdout(j.ID, "ffmpeg: start failed: "+err.Error())
			q.ErrorJob(j.ID)
			return err
		}

		scan := bufio.NewScanner(stdout)
		for scan.Scan() {
			_ = q.PushJobStdout(j.ID, scan.Text())
		}
		_ = cmd.Wait()
		<-doneErr

		q.PushJobStdout(j.ID, "ffmpeg: completed "+base)
		q.PushJobStdout(j.ID, outputPath)
	}

	q.CompleteJob(j.ID)
	return nil
}

// ffmpegTask is the custom/advanced ffmpeg task with freeform template arguments.
func ffmpegTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	opts := ParseOptions(j, ffmpegCustomOptions)
	templateArgsStr, _ := opts["arguments"].(string)
	if templateArgsStr == "" {
		q.PushJobStdout(j.ID, "ffmpeg: no arguments provided")
		q.CompleteJob(j.ID)
		return nil
	}

	// Tokenize the template args string
	templateArgs := strings.Fields(templateArgsStr)

	return runFFmpegOnFiles(j, q, mu, func(abs, dir, name, ext string) ([]string, string) {
		base := filepath.Base(abs)
		idx := "1" // custom mode doesn't batch with idx

		expanded := make([]string, len(templateArgs))
		for i, ta := range templateArgs {
			s := ta
			s = strings.ReplaceAll(s, "{input}", abs)
			s = strings.ReplaceAll(s, "{dir}", dir)
			s = strings.ReplaceAll(s, "{base}", base)
			s = strings.ReplaceAll(s, "{name}", name)
			s = strings.ReplaceAll(s, "{ext}", ext)
			s = strings.ReplaceAll(s, "{idx}", idx)
			expanded[i] = s
		}

		// Determine output path
		outputPath := filepath.Join(dir, name+"_output"+ext)
		if len(expanded) > 0 {
			last := expanded[len(expanded)-1]
			if !strings.HasPrefix(last, "-") && (strings.Contains(last, string(filepath.Separator)) || strings.Contains(last, "/") || strings.Contains(last, ".")) {
				outputPath = last
			} else {
				expanded = append(expanded, outputPath)
			}
		}

		// Remove -i if present (runFFmpegOnFiles adds it)
		var cleaned []string
		for i := 0; i < len(expanded); i++ {
			if expanded[i] == "-i" && i+1 < len(expanded) {
				i++ // skip -i and its value
				continue
			}
			cleaned = append(cleaned, expanded[i])
		}

		return cleaned, outputPath
	})
}
```

- [ ] **Step 2: Create ffmpeg preset tasks**

Create `media-server/tasks/ffmpeg_presets.go`:

```go
package tasks

import (
	"fmt"
	"path/filepath"
	"strings"
	"sync"

	"github.com/stevecastle/shrike/jobqueue"
)

var ffmpegScaleOptions = []TaskOption{
	{Name: "width", Label: "Width", Type: "number", Default: 1280.0, Required: true, Description: "Target width (height auto-calculated)"},
}

var ffmpegConvertOptions = []TaskOption{
	{Name: "format", Label: "Output Format", Type: "enum", Choices: []string{"mp4", "webm", "mkv", "mov", "gif", "mp3", "wav"}, Default: "mp4", Required: true, Description: "Target format"},
}

var ffmpegExtractAudioOptions = []TaskOption{
	{Name: "format", Label: "Audio Format", Type: "enum", Choices: []string{"mp3", "wav", "aac", "flac", "ogg"}, Default: "mp3", Description: "Audio output format"},
}

var ffmpegScreenshotOptions = []TaskOption{
	{Name: "timestamp", Label: "Timestamp", Type: "string", Default: "00:00:01", Description: "Time position (HH:MM:SS)"},
	{Name: "format", Label: "Image Format", Type: "enum", Choices: []string{"jpg", "png", "webp"}, Default: "jpg", Description: "Output image format"},
}

var ffmpegThumbnailOptions = []TaskOption{
	{Name: "timestamp", Label: "Timestamp", Type: "string", Default: "00:00:01", Description: "Time position (HH:MM:SS)"},
	{Name: "width", Label: "Width", Type: "number", Default: 600.0, Description: "Thumbnail width"},
	{Name: "format", Label: "Image Format", Type: "enum", Choices: []string{"jpg", "png", "webp"}, Default: "jpg", Description: "Output image format"},
}

func ffmpegScaleTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	opts := ParseOptions(j, ffmpegScaleOptions)
	width := int(opts["width"].(float64))
	if width <= 0 {
		width = 1280
	}

	return runFFmpegOnFiles(j, q, mu, func(abs, dir, name, ext string) ([]string, string) {
		out := filepath.Join(dir, name+"_scaled"+ext)
		return []string{"-vf", fmt.Sprintf("scale=%d:-1", width), out}, out
	})
}

func ffmpegConvertTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	opts := ParseOptions(j, ffmpegConvertOptions)
	format, _ := opts["format"].(string)
	if format == "" {
		format = "mp4"
	}

	return runFFmpegOnFiles(j, q, mu, func(abs, dir, name, ext string) ([]string, string) {
		out := filepath.Join(dir, name+"."+format)
		return []string{out}, out
	})
}

func ffmpegExtractAudioTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	opts := ParseOptions(j, ffmpegExtractAudioOptions)
	format, _ := opts["format"].(string)
	if format == "" {
		format = "mp3"
	}

	codecMap := map[string]string{
		"mp3":  "libmp3lame",
		"wav":  "pcm_s16le",
		"aac":  "aac",
		"flac": "flac",
		"ogg":  "libvorbis",
	}
	codec := codecMap[format]
	if codec == "" {
		codec = "libmp3lame"
	}

	return runFFmpegOnFiles(j, q, mu, func(abs, dir, name, ext string) ([]string, string) {
		out := filepath.Join(dir, name+"."+format)
		return []string{"-vn", "-acodec", codec, out}, out
	})
}

func ffmpegScreenshotTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	opts := ParseOptions(j, ffmpegScreenshotOptions)
	timestamp, _ := opts["timestamp"].(string)
	if timestamp == "" {
		timestamp = "00:00:01"
	}
	format, _ := opts["format"].(string)
	if format == "" {
		format = "jpg"
	}

	return runFFmpegOnFiles(j, q, mu, func(abs, dir, name, ext string) ([]string, string) {
		out := filepath.Join(dir, name+"_screenshot."+format)
		// -ss before -i is faster (seeks before decoding) but runFFmpegOnFiles adds -i first,
		// so we use -ss after -i which is frame-accurate
		return []string{"-ss", timestamp, "-frames:v", "1", out}, out
	})
}

func ffmpegThumbnailTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	opts := ParseOptions(j, ffmpegThumbnailOptions)
	timestamp, _ := opts["timestamp"].(string)
	if timestamp == "" {
		timestamp = "00:00:01"
	}
	width := int(opts["width"].(float64))
	if width <= 0 {
		width = 600
	}
	format, _ := opts["format"].(string)
	if format == "" {
		format = "jpg"
	}

	return runFFmpegOnFiles(j, q, mu, func(abs, dir, name, ext string) ([]string, string) {
		out := filepath.Join(dir, name+"_thumb."+format)
		return []string{"-ss", timestamp, "-frames:v", "1", "-vf", fmt.Sprintf("scale=%d:-1", width), out}, out
	})
}

// audioFormatToExt converts format names to file extensions.
// Most are identical but this centralizes the mapping.
func audioFormatToExt(format string) string {
	return strings.ToLower(format)
}
```

- [ ] **Step 3: Verify compilation**

Run: `cd media-server && go build ./...`
Expected: Compiles without errors.

- [ ] **Step 4: Run all tests**

Run: `cd media-server && go test ./... 2>&1 | tail -20`
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add media-server/tasks/ffmpeg.go media-server/tasks/ffmpeg_presets.go
git commit -m "feat: split ffmpeg into preset tasks with shared execution helper"
```

---

### Task 5: Update /tasks API to include options

**Files:**
- Modify: `media-server/main.go`

- [ ] **Step 1: Update TaskInfo and tasksHandler**

In `media-server/main.go`, replace the `TaskInfo` struct (around line 1029) and the `tasksHandler` function:

Replace:
```go
type TaskInfo struct {
	ID   string `json:"id"`
	Name string `json:"name"`
}
```

With:
```go
type TaskInfo struct {
	ID      string             `json:"id"`
	Name    string             `json:"name"`
	Options []tasks.TaskOption `json:"options"`
}
```

In the `tasksHandler` function, update the loop that builds `taskList` to include options:

Replace:
```go
		for _, t := range taskMap {
			taskList = append(taskList, TaskInfo{
				ID:   t.ID,
				Name: t.Name,
			})
		}
```

With:
```go
		for _, t := range taskMap {
			opts := t.Options
			if opts == nil {
				opts = []tasks.TaskOption{}
			}
			taskList = append(taskList, TaskInfo{
				ID:      t.ID,
				Name:    t.Name,
				Options: opts,
			})
		}
```

- [ ] **Step 2: Verify compilation and test the endpoint**

Run: `cd media-server && go build ./...`
Expected: Compiles.

- [ ] **Step 3: Commit**

```bash
git add media-server/main.go
git commit -m "feat: include task options in /tasks API response"
```

---

### Task 6: Editor — dynamic node rendering with option widgets

**Files:**
- Modify: `media-server/renderer/templates/editor.go.html`

- [ ] **Step 1: Update the task loading to cache options**

In `editor.go.html`, replace the task loading block (the `fetch('/tasks')` call around line 339) with:

```javascript
      // Cache task options by ID for node rendering and DAG import
      var taskOptionsMap = {};

      fetch('/tasks')
        .then((r) => r.json())
        .then((data) => {
          const palette = document.getElementById('palette');
          data.tasks.forEach((task) => {
            taskOptionsMap[task.id] = task.options || [];
            const div = document.createElement('div');
            div.className = 'palette-item';
            div.draggable = true;
            div.setAttribute('ondragstart', 'drag(event)');
            div.setAttribute('data-node', task.id);
            div.innerHTML = `<strong>${task.name}</strong><div class="desc">${task.id}</div>`;
            palette.appendChild(div);
          });
        });
```

- [ ] **Step 2: Replace addNodeToGraph with dynamic widget rendering**

Replace the `addNodeToGraph` function (around lines 368-416) with:

```javascript
      function buildNodeHTML(taskId) {
        const options = taskOptionsMap[taskId] || [];
        if (options.length === 0) {
          return `<div class="node-content"><div class="node-label">${taskId}</div></div>`;
        }

        let html = '<div class="node-content">';
        for (const opt of options) {
          html += '<div class="node-input-group">';
          html += `<label>${opt.label || opt.name}</label>`;

          const dfAttr = `df-${opt.name}`;
          const defaultVal = opt.default !== undefined && opt.default !== null ? opt.default : '';

          switch (opt.type) {
            case 'bool':
              html += `<input type="checkbox" ${dfAttr} ${defaultVal ? 'checked' : ''} style="width:auto;">`;
              break;
            case 'enum':
              html += `<select ${dfAttr}>`;
              for (const c of (opt.choices || [])) {
                html += `<option value="${c}" ${c === String(defaultVal) ? 'selected' : ''}>${c}</option>`;
              }
              html += '</select>';
              break;
            case 'multi-enum': {
              const defaults = String(defaultVal).split(',').map(s => s.trim());
              for (const c of (opt.choices || [])) {
                const checked = defaults.includes(c) ? 'checked' : '';
                html += `<label style="display:inline-flex;align-items:center;gap:4px;margin-right:8px;font-size:12px;">`;
                html += `<input type="checkbox" data-multienum="${opt.name}" value="${c}" ${checked}>${c}</label>`;
              }
              html += `<input type="hidden" ${dfAttr} value="${defaultVal}">`;
              break;
            }
            case 'number':
              html += `<input type="number" ${dfAttr} value="${defaultVal}" placeholder="${opt.description || ''}">`;
              break;
            default: // string
              html += `<input type="text" ${dfAttr} value="${defaultVal}" placeholder="${opt.description || ''}">`;
              break;
          }
          html += '</div>';
        }
        html += '</div>';
        return html;
      }

      function buildNodeData(taskId) {
        const options = taskOptionsMap[taskId] || [];
        const data = { command: taskId };
        for (const opt of options) {
          if (opt.type === 'bool') {
            data[opt.name] = opt.default ? true : false;
          } else {
            data[opt.name] = opt.default !== undefined && opt.default !== null ? String(opt.default) : '';
          }
        }
        return data;
      }

      function addNodeToGraph(name, pos_x, pos_y) {
        if (editor.editor_mode === 'fixed') return false;

        pos_x =
          pos_x *
            (editor.precanvas.clientWidth /
              (editor.precanvas.clientWidth * editor.zoom)) -
          editor.precanvas.getBoundingClientRect().x *
            (editor.precanvas.clientWidth /
              (editor.precanvas.clientWidth * editor.zoom));
        pos_y =
          pos_y *
            (editor.precanvas.clientHeight /
              (editor.precanvas.clientHeight * editor.zoom)) -
          editor.precanvas.getBoundingClientRect().y *
            (editor.precanvas.clientHeight /
              (editor.precanvas.clientHeight * editor.zoom));

        const html = buildNodeHTML(name);
        const data = buildNodeData(name);
        editor.addNode(name, 1, 1, pos_x, pos_y, 'node', data, html);
      }
```

- [ ] **Step 3: Add multi-enum change listener**

After the `editor.start()` call, add a delegated event handler for multi-enum checkboxes:

```javascript
      // Sync multi-enum checkboxes to hidden input
      document.getElementById('drawflow').addEventListener('change', function(e) {
        if (e.target.dataset.multienum) {
          const name = e.target.dataset.multienum;
          const container = e.target.closest('.node-input-group');
          const checkboxes = container.querySelectorAll(`input[data-multienum="${name}"]`);
          const values = [];
          checkboxes.forEach(cb => { if (cb.checked) values.push(cb.value); });
          const hidden = container.querySelector(`input[df-${name}]`);
          if (hidden) {
            hidden.value = values.join(',');
            // Trigger Drawflow data update
            hidden.dispatchEvent(new Event('input', { bubbles: true }));
          }
        }
      });
```

- [ ] **Step 4: Update exportDAG to serialize options as arguments**

Replace the `exportDAG` function with:

```javascript
      function exportDAG() {
        const data = editor.export();
        const nodes = data.drawflow.Home.data;
        const tasks = [];
        const nodeMap = {};

        Object.keys(nodes).forEach(key => {
          const node = nodes[key];
          const id = 'node-' + key;
          nodeMap[key] = id;

          const command = node.data.command || node.name;
          const options = taskOptionsMap[command] || [];

          // Convert node data to arguments array
          const args = [];
          for (const opt of options) {
            const val = node.data[opt.name];
            if (val === undefined || val === null || val === '') continue;

            switch (opt.type) {
              case 'bool':
                if (val === true || val === 'true') {
                  args.push('--' + opt.name);
                }
                break;
              case 'number': {
                const n = Number(val);
                if (n !== 0 || opt.required) {
                  args.push('--' + opt.name, String(n));
                }
                break;
              }
              default: // string, enum, multi-enum
                if (String(val).trim() !== '') {
                  args.push('--' + opt.name, String(val));
                }
                break;
            }
          }

          tasks.push({
            id: id,
            drawflowId: key,
            command: command,
            arguments: args,
            input: '',
            dependencies: [],
          });
        });

        // Link dependencies
        tasks.forEach(task => {
          const node = nodes[task.drawflowId];
          Object.keys(node.inputs).forEach(inputKey => {
            node.inputs[inputKey].connections.forEach(conn => {
              const parentId = nodeMap[conn.node];
              if (parentId) task.dependencies.push(parentId);
            });
          });
          delete task.drawflowId;
        });

        return tasks;
      }
```

- [ ] **Step 5: Update loadWorkflow to populate widgets from arguments**

In the `loadWorkflow` function, replace the node creation section (the `dag.forEach` loop that creates nodes) with:

```javascript
            dag.forEach((task, idx) => {
              const command = task.command;
              const options = taskOptionsMap[command] || [];

              // Parse arguments back into option values
              const nodeData = { command: command };
              const args = task.arguments || [];
              for (const opt of options) {
                // Set default
                if (opt.type === 'bool') {
                  nodeData[opt.name] = false;
                } else {
                  nodeData[opt.name] = opt.default !== undefined ? String(opt.default) : '';
                }
              }
              // Parse --flag value pairs
              for (let i = 0; i < args.length; i++) {
                const arg = args[i];
                if (!arg.startsWith('--')) continue;
                let key = arg.slice(2);
                let value = '';
                const eqIdx = key.indexOf('=');
                if (eqIdx >= 0) {
                  value = key.slice(eqIdx + 1);
                  key = key.slice(0, eqIdx);
                }
                const opt = options.find(o => o.name === key);
                if (!opt) continue;
                if (opt.type === 'bool') {
                  nodeData[key] = value ? (value === 'true') : true;
                } else if (!value && i + 1 < args.length && !args[i + 1].startsWith('--')) {
                  i++;
                  nodeData[key] = args[i];
                } else {
                  nodeData[key] = value;
                }
              }

              const html = buildNodeHTML(command);
              const drawflowId = editor.addNode(
                command,
                1, 1,
                x + (idx % 3) * 300,
                y + Math.floor(idx / 3) * 200,
                'node',
                nodeData,
                html
              );
              idToDrawflowId[task.id] = drawflowId;

              // After adding node, update widget states from nodeData
              setTimeout(() => {
                const nodeEl = document.querySelector(`#node-${drawflowId}`);
                if (!nodeEl) return;
                // Update checkboxes for bools
                for (const opt of options) {
                  if (opt.type === 'bool') {
                    const cb = nodeEl.querySelector(`input[df-${opt.name}]`);
                    if (cb) cb.checked = !!nodeData[opt.name];
                  }
                  if (opt.type === 'multi-enum') {
                    const vals = String(nodeData[opt.name] || '').split(',');
                    nodeEl.querySelectorAll(`input[data-multienum="${opt.name}"]`).forEach(cb => {
                      cb.checked = vals.includes(cb.value);
                    });
                  }
                  if (opt.type === 'enum') {
                    const sel = nodeEl.querySelector(`select[df-${opt.name}]`);
                    if (sel) sel.value = nodeData[opt.name] || '';
                  }
                  if (opt.type === 'number' || opt.type === 'string') {
                    const inp = nodeEl.querySelector(`input[df-${opt.name}]`);
                    if (inp) inp.value = nodeData[opt.name] || '';
                  }
                }
              }, 50);
            });
```

- [ ] **Step 6: Add CSS for node widgets**

Add these styles inside the `<style>` block in the editor template:

```css
      .node-label {
        padding: 8px;
        font-weight: 600;
        text-align: center;
        color: var(--text-primary);
      }

      .node-content select {
        width: 100%;
        background: var(--bg-surface);
        color: var(--text-primary);
        border: 1px solid var(--border-subtle);
        padding: 4px 6px;
        border-radius: 3px;
        font-size: 12px;
      }

      .node-content input[type="checkbox"] {
        accent-color: var(--accent-primary);
      }

      .node-content input[type="number"] {
        width: 100%;
      }
```

- [ ] **Step 7: Commit**

```bash
git add media-server/main.go media-server/renderer/templates/editor.go.html
git commit -m "feat: dynamic option widgets in Drawflow editor"
```

---

### Task 7: Manual testing

- [ ] **Step 1: Test /tasks API returns options**

```bash
curl http://localhost:8090/tasks -H "Authorization: Bearer <token>" | python -m json.tool
```

Verify: each task has an `options` array. `metadata` has 4 options, `ffmpeg-scale` has 1, `wait` has 0.

- [ ] **Step 2: Test editor renders widgets**

1. Open `/editor` in browser
2. Drag "Generate Metadata" node — verify dropdown for type, checkbox for overwrite, dropdown for apply, text for model
3. Drag "Scale Media" node — verify number input for width
4. Drag "Screenshot" node — verify text for timestamp, dropdown for format
5. Drag "Wait" node — verify just a label, no inputs

- [ ] **Step 3: Test save and load preserves option values**

1. Create a metadata node, set type to "transcript", check overwrite
2. Save the workflow
3. Clear, load it back
4. Verify the widgets show the correct state (transcript selected, overwrite checked)

- [ ] **Step 4: Test running a workflow with structured options**

1. Create a workflow: autotag → metadata (type: transcript)
2. Run it against a library query
3. Verify the jobs are created with correct `--type transcript` arguments

- [ ] **Step 5: Verify external command tasks are removed**

Confirm `gallery-dl`, `dce`, `yt-dlp` no longer appear in `/tasks` response or editor palette.

- [ ] **Step 6: Commit any fixes**

```bash
git add -A
git commit -m "fix: structured task options polish from manual testing"
```
