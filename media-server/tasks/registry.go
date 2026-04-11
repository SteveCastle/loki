package tasks

import (
	"sync"

	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/storage"
)

// Task represents a runnable unit bound to the jobqueue.
type Task struct {
	ID   string                                                        `json:"id"`
	Name string                                                        `json:"name"`
	Fn   func(j *jobqueue.Job, q *jobqueue.Queue, r *sync.Mutex) error `json:"-"`
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
	RegisterTask("wait", "Wait", waitFn)
	RegisterTask("gallery-dl", "gallery-dl", executeCommand)
	RegisterTask("dce", "dce", executeCommand)
	RegisterTask("yt-dlp", "yt-dlp", executeCommand)
	RegisterTask("ffmpeg", "ffmpeg", ffmpegTask)
	RegisterTask("remove", "Remove Media", removeFromDB)
	RegisterTask("cleanup", "CleanUp", cleanUpFn)
	RegisterTask("ingest", "Ingest Media Files", ingestTask)
	RegisterTask("metadata", "Generate Metadata", metadataTask)
	RegisterTask("move", "Move Media Files", moveTask)
	RegisterTask("autotag", "Auto Tag (ONNX)", autotagTask)
	RegisterTask("lora-dataset", "Create LoRA Dataset", loraDatasetTask)
	RegisterTask("hls", "HLS Transcode", hlsTask)
}

func RegisterTask(id, name string, fn func(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error) {
	tasks[id] = Task{
		ID:   id,
		Name: name,
		Fn:   fn,
	}
}

func GetTasks() TaskMap {
	return tasks
}
