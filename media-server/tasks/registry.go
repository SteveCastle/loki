package tasks

import (
	"sync"

	"github.com/stevecastle/shrike/jobqueue"
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
	// Register built-in tasks
	RegisterTask("wait", "Wait", nil, waitFn)
	RegisterTask("remove", "Remove Media", nil, removeFromDB)
	RegisterTask("cleanup", "CleanUp", nil, cleanUpFn)
	RegisterTask("autotag", "Auto Tag (ONNX)", nil, autotagTask)

	RegisterTask("metadata", "Generate Metadata", metadataOptions, metadataTask)
	RegisterTask("hls", "HLS Transcode", hlsOptions, hlsTask)
	RegisterTask("move", "Move Media Files", moveOptions, moveTask)
	RegisterTask("ingest", "Ingest Media Files", ingestOptions, ingestTask)
	RegisterTask("lora-dataset", "Create LoRA Dataset", loraDatasetOptions, loraDatasetTask)

	RegisterTask("ffmpeg", "ffmpeg", ffmpegCustomOptions, ffmpegTask)
	RegisterTask("ffmpeg-scale", "FFmpeg Scale", ffmpegScaleOptions, ffmpegScaleTask)
	RegisterTask("ffmpeg-convert", "FFmpeg Convert", ffmpegConvertOptions, ffmpegConvertTask)
	RegisterTask("ffmpeg-extract-audio", "FFmpeg Extract Audio", ffmpegExtractAudioOptions, ffmpegExtractAudioTask)
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
