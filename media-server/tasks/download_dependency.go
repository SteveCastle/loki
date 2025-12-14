package tasks

import (
	"fmt"
	"strings"
	"sync"

	"github.com/stevecastle/shrike/deps"
	"github.com/stevecastle/shrike/jobqueue"
)

func init() {
	// Register the download-dependency task
	RegisterTask("download-dependency", "Download Dependency", downloadDependencyTask)
}

// downloadDependencyTask handles downloading a dependency as a job.
func downloadDependencyTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	depID := strings.TrimSpace(j.Input)

	q.PushJobStdout(j.ID, fmt.Sprintf("Downloading dependency: %s", depID))

	dep, ok := deps.Get(depID)
	if !ok {
		q.PushJobStdout(j.ID, fmt.Sprintf("Unknown dependency: %s", depID))
		q.ErrorJob(j.ID)
		return fmt.Errorf("unknown dependency: %s", depID)
	}

	q.PushJobStdout(j.ID, fmt.Sprintf("Dependency: %s (%s)", dep.Name, dep.Description))

	// Update metadata status to downloading
	metadata := deps.GetMetadataStore()
	metadata.UpdateStatus(depID, deps.StatusDownloading)
	metadata.Save()

	// Call the dependency's download function
	if err := dep.Download(j, q, mu); err != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("Download failed: %v", err))
		metadata.UpdateStatus(depID, deps.StatusNotInstalled)
		metadata.ClearJobID(depID)
		metadata.Save()
		q.ErrorJob(j.ID)
		return err
	}

	q.PushJobStdout(j.ID, fmt.Sprintf("Successfully downloaded %s", dep.Name))
	metadata.UpdateStatus(depID, deps.StatusInstalled)
	metadata.ClearJobID(depID)
	metadata.Save()
	q.CompleteJob(j.ID)
	return nil
}
