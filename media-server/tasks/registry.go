package tasks

import (
	"sync"

	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/media"
	"github.com/stevecastle/shrike/storage"
)

// TaskFn is the function signature for task implementations.
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

// storageReg holds a reference to the storage registry so tasks can upload to
// the default backend. Set once at startup via SetStorageRegistry.
var storageReg *storage.Registry

// SetStorageRegistry provides the storage registry to the tasks package.
// Must be called before any task that needs storage access runs.
func SetStorageRegistry(r *storage.Registry) {
	storageReg = r
}

func init() {
	// Whenever media rows are deleted (cleanup task, remove task, or any other
	// RemoveItemsFromDB caller), evict the paths from the live vector index so
	// similarity search stops returning deleted items immediately instead of
	// after the next index rebuild.
	media.SetMediaRemovalHook(func(paths []string) {
		for _, p := range paths {
			IndexDelete(p)
			FaceIndexDeletePath(p)
		}
	})

	// Per-item operations: each is a standalone task AND composable with the
	// others into a single per-file pass via the "process" task. All of them
	// share one contract — query/path-list inputs, --overwrite semantics,
	// done/total progress, pause/resume with per-item durability.
	registerBuiltinItemOps()

	// Register built-in tasks
	RegisterTask("wait", "Wait", nil, waitFn)
	RegisterTask("remove", "Remove Media", nil, removeFromDB)
	RegisterTask("cleanup", "CleanUp", nil, cleanUpFn)
	RegisterTask("autotag", "Auto Tag (ONNX)", itemOpTaskOptions("autotag"), makeItemOpTaskFn("autotag"))
	RegisterTask("embed", "Visual Embedding (ONNX)", itemOpTaskOptions("embed"), makeItemOpTaskFn("embed"))
	RegisterTask("describe", "Generate Descriptions", itemOpTaskOptions("describe"), makeItemOpTaskFn("describe"))
	RegisterTask("transcribe", "Generate Transcripts", itemOpTaskOptions("transcribe"), makeItemOpTaskFn("transcribe"))
	RegisterTask("hash", "Generate Hashes", itemOpTaskOptions("hash"), makeItemOpTaskFn("hash"))
	RegisterTask("dimensions", "Generate Dimensions", itemOpTaskOptions("dimensions"), makeItemOpTaskFn("dimensions"))
	RegisterTask("process", "Process Media (Combined Ops)", processTaskOptions(), processTask)
	RegisterTask("faces", "Detect Faces (ONNX)", itemOpTaskOptions("faces"), makeItemOpTaskFn("faces"))
	RegisterTask("faces-cluster", "Cluster Faces into People", nil, facesClusterTask)

	// Legacy alias: maps --type onto the split-out ops above.
	RegisterTask("metadata", "Generate Metadata (Legacy)", metadataOptions, metadataTask)
	RegisterTask("hls", "HLS Transcode", hlsOptions, hlsTask)
	RegisterTask("move", "Move Media Files", moveOptions, moveTask)
	RegisterTask("ingest", "Ingest Media Files", ingestOptions, ingestTask)
	RegisterTask("lora-dataset", "Create LoRA Dataset", loraDatasetOptions, loraDatasetTask)

	// Host resolvers. Each entry maps a task command to its concurrency
	// bucket so jobqueue can rate-limit work per resource. Tasks without
	// an entry fall through to ResolveHost's "localhost" default. Adding a
	// new vision-using task is one line: route it through InferenceHost.
	visionHost := func(string) string { return InferenceHost() }
	// Auto-tagging is a local ONNX task with its own concurrency bucket — like
	// embed, it parallelizes internally and must not share the LLM cap.
	RegisterHostResolver("autotag", func(string) string { return HostBucketAutotag })
	RegisterHostResolver("metadata", visionHost)
	// The split-out LLM-vision ops share the inference cap, exactly as their
	// former metadata-task selves did. Transcription also historically ran
	// under the metadata task's inference bucket, so it keeps that behavior.
	RegisterHostResolver("describe", visionHost)
	RegisterHostResolver("transcribe", visionHost)
	// A combined job may include LLM ops, so it conservatively takes the
	// inference bucket (a hash-only combined run parking there is harmless).
	RegisterHostResolver("process", visionHost)
	// Embedding is a local ONNX task with its own concurrency bucket — it must
	// not share the LLM inference cap (it parallelizes internally instead).
	RegisterHostResolver("embed", func(string) string { return HostBucketEmbed })
	// Face scanning is a local ONNX task with its own concurrency bucket — like
	// embed/autotag, it parallelizes internally via its worker pool. Clustering
	// shares the bucket so a scan and a recluster never run concurrently.
	RegisterHostResolver("faces", func(string) string { return HostBucketFaces })
	RegisterHostResolver("faces-cluster", func(string) string { return HostBucketFaces })
	RegisterHostResolver("ingest", urlHostResolver)

	RegisterTask("ffmpeg", "ffmpeg", ffmpegCustomOptions, ffmpegTask)
	RegisterTask("ffmpeg-scale", "FFmpeg Scale", ffmpegScaleOptions, ffmpegScaleTask)
	RegisterTask("ffmpeg-convert", "FFmpeg Convert", ffmpegConvertOptions, ffmpegConvertTask)
	RegisterTask("ffmpeg-extract-audio", "FFmpeg Extract Audio", ffmpegExtractAudioOptions, ffmpegExtractAudioTask)
	RegisterTask("ffmpeg-extract-audio-clip", "FFmpeg Extract Audio Clip", ffmpegExtractAudioClipOptions, ffmpegExtractAudioClipTask)
	RegisterTask("ffmpeg-screenshot", "FFmpeg Screenshot", ffmpegScreenshotOptions, ffmpegScreenshotTask)
	RegisterTask("ffmpeg-thumbnail", "FFmpeg Thumbnail", ffmpegThumbnailOptions, ffmpegThumbnailTask)
	RegisterTask("ffmpeg-reverse", "FFmpeg Reverse", nil, ffmpegReverseTask)
	RegisterTask("ffmpeg-speed", "FFmpeg Speed", ffmpegSpeedOptions, ffmpegSpeedTask)
	RegisterTask("ffmpeg-grayscale", "FFmpeg Grayscale", nil, ffmpegGrayscaleTask)
	RegisterTask("ffmpeg-blur", "FFmpeg Blur", ffmpegBlurOptions, ffmpegBlurTask)
	RegisterTask("ffmpeg-resize", "FFmpeg Resize", ffmpegResizeOptions, ffmpegResizeTask)
	RegisterTask("ffmpeg-crop", "FFmpeg Crop", ffmpegCropOptions, ffmpegCropTask)
	RegisterTask("ffmpeg-rotate", "FFmpeg Rotate", ffmpegRotateOptions, ffmpegRotateTask)
	RegisterTask("ffmpeg-caption", "FFmpeg Caption", ffmpegCaptionOptions, ffmpegCaptionTask)
	RegisterTask("ffmpeg-thumbsheet", "FFmpeg Thumbnail Sheet", ffmpegThumbSheetOptions, ffmpegThumbSheetTask)

	RegisterTask("save", "Save File", saveOptions, saveTask)
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
