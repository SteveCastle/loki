# Structured Task Options Design

## Overview

Replace the raw string arguments model for tasks with structured option definitions. Each task declares its options as typed metadata, the server parses arguments against the schema, and the Drawflow editor renders appropriate widgets per option type. FFmpeg is split into focused preset tasks. The external command pass-through tasks (gallery-dl, yt-dlp, dce) are removed since `ingest` already abstracts over them.

## TaskOption Schema

```go
type TaskOption struct {
    Name        string   `json:"name"`
    Label       string   `json:"label"`
    Type        string   `json:"type"`     // "string", "bool", "enum", "multi-enum", "number"
    Choices     []string `json:"choices,omitempty"`
    Default     any      `json:"default,omitempty"`
    Required    bool     `json:"required,omitempty"`
    Description string   `json:"description,omitempty"`
}
```

## Option Parsing

A shared `ParseOptions(j *jobqueue.Job, options []TaskOption) map[string]any` function parses `j.Arguments` against the declared schema, applies defaults for missing values, and returns a typed map. Handles `--flag value`, `--flag=value`, and bare `--flag` (for bools). Each task switches from manual string parsing to this function.

The wire format stays the same — `arguments` is still a `[]string` in the job/workflow JSON. The editor serializes structured values back to this format:
- Bools: `--overwrite` (present if true, omitted if false)
- Enums/strings: `--type transcript`
- Multi-enums: `--type description,hash,dimensions` (comma-joined)
- Numbers: `--width 1280`

## Task Registration Changes

`RegisterTask` expands to include options:

```go
func RegisterTask(id, name string, options []TaskOption, fn TaskFn)
```

The `Task` struct gains `Options []TaskOption`. The `/tasks` API endpoint now includes `options` in each task's JSON response.

**Removed tasks:** `gallery-dl`, `dce`, `yt-dlp` — `ingest` handles routing to these tools internally.

## Per-Task Option Definitions

### autotag
No options. Reads input paths, uses app config for ONNX settings.

### metadata
| Option | Type | Choices | Default | Required | Description |
|--------|------|---------|---------|----------|-------------|
| type | multi-enum | description, transcript, hash, dimensions, autotag | description,hash,dimensions | no | Metadata types to generate |
| overwrite | bool | | false | no | Overwrite existing metadata |
| apply | enum | new, all | new | no | Apply scope |
| model | string | | (from config) | no | Ollama model name |

### hls
| Option | Type | Choices | Default | Required | Description |
|--------|------|---------|---------|----------|-------------|
| preset | enum | passthrough, adaptive | passthrough | no | HLS preset mode |
| presets | multi-enum | 480p, 720p, 1080p | | no | Explicit quality tiers |

### move
| Option | Type | Default | Required | Description |
|--------|------|---------|----------|-------------|
| target | string | | yes | Target directory path |
| prefix | string | (auto) | no | Prefix to strip from source paths |

### ingest
| Option | Type | Default | Required | Description |
|--------|------|---------|----------|-------------|
| recursive | bool | false | no | Scan directories recursively |
| transcript | bool | false | no | Queue transcript generation |
| description | bool | false | no | Queue description generation |
| filemeta | bool | false | no | Queue hash and dimensions |
| autotag | bool | false | no | Queue ONNX auto-tagging |

### lora-dataset
| Option | Type | Default | Required | Description |
|--------|------|---------|----------|-------------|
| target | string | | yes | Target directory for dataset |
| name | string | | yes | Dataset name |
| prefix | string | | no | Concept prefix for descriptions |
| model | string | (from config) | no | Ollama model name |

### ffmpeg (custom/advanced)
| Option | Type | Default | Required | Description |
|--------|------|---------|----------|-------------|
| arguments | string | | yes | Raw ffmpeg arguments with template variable expansion |

Template variables: `{input}`, `{dir}`, `{base}`, `{name}`, `{ext}`, `{idx}`

### ffmpeg-scale
| Option | Type | Default | Required | Description |
|--------|------|---------|----------|-------------|
| width | number | 1280 | yes | Target width (height auto-calculated) |

Builds: `-vf scale={width}:-1`

### ffmpeg-convert
| Option | Type | Choices | Default | Required | Description |
|--------|------|---------|---------|----------|-------------|
| format | enum | mp4, webm, mkv, mov, gif, mp3, wav | mp4 | yes | Output format |

Builds: output path with new extension.

### ffmpeg-extract-audio
| Option | Type | Choices | Default | Required | Description |
|--------|------|---------|---------|----------|-------------|
| format | enum | mp3, wav, aac, flac, ogg | mp3 | no | Audio output format |

Builds: `-vn -acodec {codec}` where codec is derived from format.

### ffmpeg-screenshot
| Option | Type | Default | Required | Description |
|--------|------|---------|----------|-------------|
| timestamp | string | 00:00:01 | no | Time position to capture |
| format | enum (jpg, png, webp) | jpg | no | Image output format |

Builds: `-ss {timestamp} -frames:v 1`

### ffmpeg-thumbnail
| Option | Type | Default | Required | Description |
|--------|------|---------|----------|-------------|
| timestamp | string | 00:00:01 | no | Time position to capture |
| width | number | 600 | no | Thumbnail width |
| format | enum (jpg, png, webp) | jpg | no | Image output format |

Builds: `-ss {timestamp} -frames:v 1 -vf scale={width}:-1`

### remove, cleanup, wait
No options.

## FFmpeg Refactoring

The current `ffmpegTask` is split:

1. Extract shared `runFFmpegOnFiles(j, q, mu, files []string, buildArgs func(src string) []string) error` helper containing the file iteration loop, query resolution, cancellation, progress reporting, and output chaining logic.

2. Each ffmpeg variant registers as its own task and implements a `buildArgs` function:
   - `ffmpegTask` (custom) — uses template variable expansion on freeform arguments, as today
   - `ffmpegScaleTask` — builds scale filter args from `width` option
   - `ffmpegConvertTask` — builds format conversion args from `format` option
   - `ffmpegExtractAudioTask` — builds audio extraction args from `format` option
   - `ffmpegScreenshotTask` — builds frame extraction args from `timestamp` and `format` options
   - `ffmpegThumbnailTask` — builds scaled frame extraction args from `timestamp`, `width`, and `format` options

All variants share output chaining (push output path to stdout) and auto-output-path generation.

## Drawflow Editor Changes

### Node rendering

`addNodeToGraph` generates form HTML dynamically from the task's options array:

| Option Type | Widget |
|-------------|--------|
| bool | Checkbox |
| enum | Select dropdown |
| multi-enum | Checkbox group |
| number | Number input |
| string | Text input |

Each widget uses Drawflow's `df-{name}` attribute for data binding. Default values are pre-filled from the schema. Tasks with no options render as a simple label-only node.

### Task options cache

The `/tasks` response is fetched once on page load and stored in a `taskOptionsMap` keyed by task ID. Both `addNodeToGraph` and `loadWorkflow` reference this map.

### DAG export

`exportDAG()` reads Drawflow node data and converts option values to `arguments[]`:
- Bools: `--{name}` (present if true, omitted if false)
- Enums/strings: `--{name} {value}`
- Multi-enums: `--{name} {value1},{value2}` (comma-joined)
- Numbers: `--{name} {value}`

### DAG import

`loadWorkflow()` parses `arguments[]` back into option values using the task's schema from `taskOptionsMap`, then sets node data so widgets display the correct state.

## Files Modified/Created

### Server (Go)
- **Create:** `media-server/tasks/options.go` — `TaskOption` type, `ParseOptions` function
- **Modify:** `media-server/tasks/registry.go` — add `Options` to `Task`, update `RegisterTask` signature
- **Modify:** `media-server/tasks/ffmpeg.go` — extract shared helper, keep custom task, register with options
- **Create:** `media-server/tasks/ffmpeg_presets.go` — scale, convert, extract-audio, screenshot, thumbnail task functions
- **Modify:** `media-server/tasks/media_metadata.go` — switch to `ParseOptions`
- **Modify:** `media-server/tasks/hls.go` — switch to `ParseOptions`
- **Modify:** `media-server/tasks/media_move.go` — switch to `ParseOptions`
- **Modify:** `media-server/tasks/media_ingest.go` — switch to `ParseOptions`
- **Modify:** `media-server/tasks/lora_dataset.go` — switch to `ParseOptions`
- **Modify:** `media-server/tasks/command.go` — remove or keep as internal-only (not registered)
- **Modify:** `media-server/main.go` — update `/tasks` handler to include options in JSON response

### Editor (HTML/JS)
- **Modify:** `media-server/renderer/templates/editor.go.html` — dynamic node rendering, DAG export/import with structured options
