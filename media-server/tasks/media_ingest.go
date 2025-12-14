package tasks

import (
	"strings"
	"sync"

	"github.com/stevecastle/shrike/jobqueue"
)

// IngestOptions holds the optional follow-up task flags for ingestion
type IngestOptions struct {
	Recursive   bool // For local ingestion: scan directories recursively
	Transcript  bool // Create transcript metadata task for each file
	Description bool // Create description metadata task for each file
	FileMeta    bool // Create file metadata (hash, dimensions) task for each file
	AutoTag     bool // Create ONNX autotag task for each file
}

// parseIngestOptions parses arguments to extract ingest options
// Returns the options and any remaining arguments that weren't option flags
func parseIngestOptions(args []string) (IngestOptions, []string) {
	var opts IngestOptions
	var remaining []string

	for _, arg := range args {
		switch strings.ToLower(arg) {
		case "-r", "--recursive":
			opts.Recursive = true
		case "--transcript":
			opts.Transcript = true
		case "--description":
			opts.Description = true
		case "--filemeta", "--file-meta":
			opts.FileMeta = true
		case "--autotag", "--auto-tag":
			opts.AutoTag = true
		default:
			remaining = append(remaining, arg)
		}
	}

	return opts, remaining
}

// queueFollowUpTasks creates follow-up tasks for each ingested file based on options
func queueFollowUpTasks(q *jobqueue.Queue, jobID string, files []string, opts IngestOptions) {
	if len(files) == 0 {
		return
	}

	for _, filePath := range files {
		// Queue transcript metadata task
		if opts.Transcript {
			_, err := q.AddJob("metadata", []string{"-t", "transcript", "-a", "all"}, filePath, nil)
			if err != nil {
				q.PushJobStdout(jobID, "Warning: failed to queue transcript task for "+filePath+": "+err.Error())
			} else {
				q.PushJobStdout(jobID, "Queued transcript task for: "+filePath)
			}
		}

		// Queue description metadata task
		if opts.Description {
			_, err := q.AddJob("metadata", []string{"-t", "description", "-a", "all"}, filePath, nil)
			if err != nil {
				q.PushJobStdout(jobID, "Warning: failed to queue description task for "+filePath+": "+err.Error())
			} else {
				q.PushJobStdout(jobID, "Queued description task for: "+filePath)
			}
		}

		// Queue file metadata (hash, dimensions) task
		if opts.FileMeta {
			_, err := q.AddJob("metadata", []string{"-t", "hash,dimensions", "-a", "all"}, filePath, nil)
			if err != nil {
				q.PushJobStdout(jobID, "Warning: failed to queue file metadata task for "+filePath+": "+err.Error())
			} else {
				q.PushJobStdout(jobID, "Queued file metadata task for: "+filePath)
			}
		}

		// Queue ONNX autotag task
		if opts.AutoTag {
			_, err := q.AddJob("autotag", nil, filePath, nil)
			if err != nil {
				q.PushJobStdout(jobID, "Warning: failed to queue autotag task for "+filePath+": "+err.Error())
			} else {
				q.PushJobStdout(jobID, "Queued autotag task for: "+filePath)
			}
		}
	}
}

// ingestTask is the main dispatcher for media ingestion
// It routes to the appropriate handler based on input type:
// - Local file paths: scans directories for media files
// - YouTube URLs: uses yt-dlp to download
// - Other HTTP URLs: uses gallery-dl to download
//
// Supported arguments:
//   - -r, --recursive: Scan directories recursively (local only)
//   - --transcript: Queue transcript metadata task for each ingested file
//   - --description: Queue description metadata task for each ingested file
//   - --filemeta, --file-meta: Queue file metadata (hash, dimensions) task for each file
//   - --autotag, --auto-tag: Queue ONNX autotag task for each ingested file
func ingestTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	input := strings.TrimSpace(j.Input)

	// Parse ingest options from arguments
	opts, remainingArgs := parseIngestOptions(j.Arguments)

	// Store remaining args back (for passthrough to underlying handlers)
	j.Arguments = remainingArgs

	// Determine the input type and route accordingly
	switch {
	case isHTTPURL(input):
		if isYouTubeURL(input) {
			return ingestYouTubeTaskWithOptions(j, q, mu, opts)
		}
		return ingestGalleryTaskWithOptions(j, q, mu, opts)
	default:
		// Treat as local path
		return ingestLocalTaskWithOptions(j, q, mu, opts)
	}
}

// isHTTPURL checks if the input looks like an HTTP(S) URL
func isHTTPURL(input string) bool {
	lower := strings.ToLower(input)
	return strings.HasPrefix(lower, "http://") || strings.HasPrefix(lower, "https://")
}
