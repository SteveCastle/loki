package tasks

import (
	"database/sql"
	"fmt"
	"net/url"
	"strings"
	"sync"

	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/media"
)

// IngestOptions holds the optional follow-up task flags for ingestion
type IngestOptions struct {
	Recursive   bool      // For local ingestion: scan directories recursively
	Transcript  bool      // Create transcript metadata task for each file
	Description bool      // Create description metadata task for each file
	FileMeta    bool      // Create file metadata (hash, dimensions) task for each file
	AutoTag     bool      // Create ONNX autotag task for each file
	Tags        []TagInfo // Tags to apply to each ingested file
}

// parseIngestOptions parses arguments to extract ingest options
// Returns the options and any remaining arguments that weren't option flags
func parseIngestOptions(args []string) (IngestOptions, []string) {
	var opts IngestOptions
	var remaining []string

	for _, arg := range args {
		lower := strings.ToLower(arg)
		switch {
		case lower == "-r" || lower == "--recursive":
			opts.Recursive = true
		case lower == "--transcript":
			opts.Transcript = true
		case lower == "--description":
			opts.Description = true
		case lower == "--filemeta" || lower == "--file-meta":
			opts.FileMeta = true
		case lower == "--autotag" || lower == "--auto-tag":
			opts.AutoTag = true
		case strings.HasPrefix(lower, "--tag="):
			value := arg[len("--tag="):]
			label, category := parseTagArg(value)
			if label != "" {
				opts.Tags = append(opts.Tags, TagInfo{Label: label, Category: category})
			}
		default:
			remaining = append(remaining, arg)
		}
	}

	return opts, remaining
}

// parseTagArg parses a tag argument value in the form "label:category" or just "label".
// Both parts are URL-decoded. The split is on the first colon.
func parseTagArg(value string) (label, category string) {
	parts := strings.SplitN(value, ":", 2)
	label = parts[0]
	if len(parts) == 2 {
		category = parts[1]
	}
	if decoded, err := url.QueryUnescape(label); err == nil {
		label = decoded
	}
	if decoded, err := url.QueryUnescape(category); err == nil {
		category = decoded
	}
	return label, category
}

// queueFollowUpTasks creates follow-up tasks for each ingested file based on options
func queueFollowUpTasks(q *jobqueue.Queue, jobID string, files []string, opts IngestOptions) {
	if len(files) == 0 {
		return
	}

	for _, filePath := range files {
		// Queue transcript metadata task
		if opts.Transcript {
			_, err := q.AddJob("", "metadata", []string{"-t", "transcript", "-a", "all"}, filePath, nil)
			if err != nil {
				q.PushJobStdout(jobID, "Warning: failed to queue transcript task for "+filePath+": "+err.Error())
			} else {
				q.PushJobStdout(jobID, "Queued transcript task for: "+filePath)
			}
		}

		// Queue description metadata task
		if opts.Description {
			_, err := q.AddJob("", "metadata", []string{"-t", "description", "-a", "all"}, filePath, nil)
			if err != nil {
				q.PushJobStdout(jobID, "Warning: failed to queue description task for "+filePath+": "+err.Error())
			} else {
				q.PushJobStdout(jobID, "Queued description task for: "+filePath)
			}
		}

		// Queue file metadata (hash, dimensions) task
		if opts.FileMeta {
			_, err := q.AddJob("", "metadata", []string{"-t", "hash,dimensions", "-a", "all"}, filePath, nil)
			if err != nil {
				q.PushJobStdout(jobID, "Warning: failed to queue file metadata task for "+filePath+": "+err.Error())
			} else {
				q.PushJobStdout(jobID, "Queued file metadata task for: "+filePath)
			}
		}

		// Queue ONNX autotag task
		if opts.AutoTag {
			_, err := q.AddJob("", "autotag", nil, filePath, nil)
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
		if isDiscordURL(input) {
			return ingestDiscordTaskWithOptions(j, q, mu, opts)
		}
		return ingestGalleryTaskWithOptions(j, q, mu, opts)
	default:
		// Treat as local path
		return ingestLocalTaskWithOptions(j, q, mu, opts)
	}
}

// applyIngestTags resolves tag categories and applies tags to every ingested file
func applyIngestTags(db *sql.DB, jobID string, q *jobqueue.Queue, files []string, tags []TagInfo) {
	resolved := resolveTagCategories(db, tags)
	for _, filePath := range files {
		for _, tag := range resolved {
			if err := media.AddTag(db, filePath, tag.Label, tag.Category); err != nil {
				q.PushJobStdout(jobID, fmt.Sprintf("Warning: failed to add tag %s:%s to %s: %v", tag.Label, tag.Category, filePath, err))
			}
		}
	}
	q.PushJobStdout(jobID, fmt.Sprintf("Applied %d tag(s) to %d file(s)", len(resolved), len(files)))
}

// isHTTPURL checks if the input looks like an HTTP(S) URL
func isHTTPURL(input string) bool {
	lower := strings.ToLower(input)
	return strings.HasPrefix(lower, "http://") || strings.HasPrefix(lower, "https://")
}
