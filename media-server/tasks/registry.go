package tasks

import (
	"sync"

	"github.com/stevecastle/shrike/jobqueue"
)

// Task represents a runnable unit bound to the jobqueue.
type Task struct {
	ID   string                                                        `json:"id"`
	Name string                                                        `json:"name"`
	Fn   func(j *jobqueue.Job, q *jobqueue.Queue, r *sync.Mutex) error `json:"-"`
}

type TaskMap map[string]Task

var tasks = make(TaskMap)

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
