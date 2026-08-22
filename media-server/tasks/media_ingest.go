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

var ingestOptions = []TaskOption{
	{Name: "recursive", Label: "Recursive", Type: "bool", Description: "Scan directories recursively"},
	{Name: "transcript", Label: "Transcript", Type: "bool", Description: "Queue transcript metadata task for each file"},
	{Name: "description", Label: "Description", Type: "bool", Description: "Queue description metadata task for each file"},
	{Name: "filemeta", Label: "File Metadata", Type: "bool", Description: "Queue file metadata (hash, dimensions) task for each file"},
	{Name: "autotag", Label: "Auto Tag", Type: "bool", Description: "Queue ONNX autotag task for each file"},
}

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
	// Use ParseOptions for the structured flags
	j := &jobqueue.Job{Arguments: args}
	parsed := ParseOptions(j, ingestOptions)

	var opts IngestOptions
	opts.Recursive, _ = parsed["recursive"].(bool)
	opts.Transcript, _ = parsed["transcript"].(bool)
	opts.Description, _ = parsed["description"].(bool)
	opts.FileMeta, _ = parsed["filemeta"].(bool)
	opts.AutoTag, _ = parsed["autotag"].(bool)

	// Also support legacy short flags for recursive
	for _, arg := range args {
		lower := strings.ToLower(arg)
		if lower == "-r" {
			opts.Recursive = true
		}
	}

	// Parse --tag= args (not a simple option type) and collect remaining args
	var remaining []string
	knownFlags := map[string]bool{
		"--recursive": true, "-r": true,
		"--transcript": true, "--description": true,
		"--filemeta": true, "--file-meta": true,
		"--autotag": true, "--auto-tag": true,
	}
	for _, arg := range args {
		lower := strings.ToLower(arg)
		if strings.HasPrefix(lower, "--tag=") {
			value := arg[len("--tag="):]
			label, category := parseTagArg(value)
			if label != "" {
				opts.Tags = append(opts.Tags, TagInfo{Label: label, Category: category})
			}
		} else if !knownFlags[lower] {
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

// queueFollowUpTasks creates follow-up tasks for each ingested file based on
// options. Per-file success lines are deliberately NOT pushed to stdout —
// every PushJobStdout rewrites the whole job row, which is quadratic over a
// large ingest — so progress goes through the job progress bar and one
// summary line per task type; only failures get individual lines.
func queueFollowUpTasks(q *jobqueue.Queue, jobID string, files []string, opts IngestOptions) {
	if len(files) == 0 {
		return
	}

	type followUp struct {
		enabled bool
		name    string
		task    string
		args    []string
	}
	all := []followUp{
		{opts.Transcript, "transcript", "metadata", []string{"-t", "transcript", "-a", "all"}},
		{opts.Description, "description", "metadata", []string{"-t", "description", "-a", "all"}},
		{opts.FileMeta, "file metadata", "metadata", []string{"-t", "hash,dimensions", "-a", "all"}},
		{opts.AutoTag, "autotag", "autotag", nil},
	}
	var active []followUp
	for _, fu := range all {
		if fu.enabled {
			active = append(active, fu)
		}
	}
	if len(active) == 0 {
		return
	}

	total := len(files) * len(active)
	q.PushJobStdout(jobID, fmt.Sprintf("Queueing %d follow-up task(s)...", total))
	_ = q.SetJobProgress(jobID, 0, total)
	done := 0
	for _, fu := range active {
		queued := 0
		for _, filePath := range files {
			if _, err := q.AddJob("", fu.task, fu.args, filePath, nil); err != nil {
				q.PushJobStdout(jobID, "Warning: failed to queue "+fu.name+" task for "+filePath+": "+err.Error())
			} else {
				queued++
			}
			done++
			_ = q.SetJobProgress(jobID, done, total)
		}
		q.PushJobStdout(jobID, fmt.Sprintf("Queued %s task for %d file(s)", fu.name, queued))
	}
}

// ingestTask is the main dispatcher for media ingestion
// It routes to the appropriate handler based on input type:
//   - Local file paths: scans directories for media files
//   - YouTube URLs: uses yt-dlp to download
//   - URLs matching a registered native extractor: downloaded in Go with no
//     external dependency (see ingest_media.go)
//   - Other HTTP URLs: uses gallery-dl to download
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
		if ext := findMediaExtractor(input); ext != nil {
			return ingestMediaTaskWithOptions(j, q, mu, opts, ext)
		}
		return ingestGalleryTaskWithOptions(j, q, mu, opts)
	default:
		// Storage-backend paths (s3://... or relative keys like "uploads/"
		// when the default root is S3) scan through the backend; everything
		// else is treated as a local filesystem path.
		if backend := backendForIngest(input); backend != nil {
			return ingestBackendTaskWithOptions(j, q, mu, opts, backend)
		}
		return ingestLocalTaskWithOptions(j, q, mu, opts)
	}
}

// applyIngestTags resolves tag categories and applies tags to every ingested
// file, reporting per-file progress through the job progress bar.
func applyIngestTags(db *sql.DB, jobID string, q *jobqueue.Queue, files []string, tags []TagInfo) {
	resolved := resolveTagCategories(db, tags)
	if len(resolved) > 0 && len(files) > 0 {
		q.PushJobStdout(jobID, fmt.Sprintf("Applying %d tag(s) to %d file(s)...", len(resolved), len(files)))
		_ = q.SetJobProgress(jobID, 0, len(files))
	}
	for i, filePath := range files {
		for _, tag := range resolved {
			if err := media.AddTag(db, filePath, tag.Label, tag.Category); err != nil {
				q.PushJobStdout(jobID, fmt.Sprintf("Warning: failed to add tag %s:%s to %s: %v", tag.Label, tag.Category, filePath, err))
			}
		}
		_ = q.SetJobProgress(jobID, i+1, len(files))
	}
	q.PushJobStdout(jobID, fmt.Sprintf("Applied %d tag(s) to %d file(s)", len(resolved), len(files)))
}

// isHTTPURL checks if the input looks like an HTTP(S) URL
func isHTTPURL(input string) bool {
	lower := strings.ToLower(input)
	return strings.HasPrefix(lower, "http://") || strings.HasPrefix(lower, "https://")
}
