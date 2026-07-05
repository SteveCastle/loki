package tasks

import (
	"database/sql"
	"fmt"
	"path/filepath"
	"strings"
	"sync"

	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/storage"
)

// backendForIngest resolves the storage backend that should handle a non-URL
// ingest input, or nil when the input should go through the local-filesystem
// scanner. Two input shapes route through a backend:
//   - full storage paths owned by a non-local root (e.g. "s3://media/uploads/")
//   - relative keys (e.g. "uploads/" from the upload handler) when the
//     default backend is non-local
func backendForIngest(input string) storage.Backend {
	if storageReg == nil {
		return nil
	}
	if b := storageReg.BackendFor(input); b != nil {
		if b.Root().Type != "local" {
			return b
		}
		return nil // absolute path inside a local root — local scanner handles it
	}
	if !filepath.IsAbs(input) && !strings.Contains(input, "://") {
		if b := defaultBackend(); b != nil && b.Root().Type != "local" {
			return b
		}
	}
	return nil
}

// backendFullPath converts an ingest input into the full storage path form
// used in the media table (e.g. "uploads/" → "s3://media/uploads/").
func backendFullPath(root storage.Entry, input string) string {
	if strings.Contains(input, "://") {
		return input
	}
	return strings.TrimSuffix(root.Path, "/") + "/" + strings.TrimPrefix(input, "/")
}

// ingestBackendTaskWithOptions scans a storage backend (S3 etc.) for media
// files under the job input prefix and adds them to the database. It mirrors
// ingestLocalTaskWithOptions but goes through the storage abstraction, so
// uploads into an S3 default root can be ingested (the upload handler queues
// exactly this shape of job with input "uploads/").
func ingestBackendTaskWithOptions(j *jobqueue.Job, q *jobqueue.Queue, _ *sync.Mutex, opts IngestOptions, backend storage.Backend) error {
	ctx := j.Ctx
	input := strings.TrimSpace(j.Input)
	recursive := opts.Recursive
	for _, arg := range j.Arguments {
		switch strings.ToLower(arg) {
		case "-r", "--recursive":
			recursive = true
		}
	}

	if err := ensureMediaTableSchema(q.Db); err != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("Error setting up database schema: %v", err))
		q.ErrorJob(j.ID)
		return err
	}

	root := backend.Root()
	q.PushJobStdout(j.ID, fmt.Sprintf("Starting media ingestion from storage root %q: %s", root.Name, input))
	if recursive {
		q.PushJobStdout(j.ID, "Scanning recursively...")
	}

	files, err := backend.Scan(ctx, input, recursive)
	if err != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("Error scanning storage: %v", err))
		q.ErrorJob(j.ID)
		return err
	}

	q.PushJobStdout(j.ID, fmt.Sprintf("Found %d media files", len(files)))
	if len(files) == 0 {
		q.PushJobStdout(j.ID, "No media files found to ingest")
		q.CompleteJob(j.ID)
		return nil
	}

	existingPaths, err := getExistingMediaPathsByPrefix(q.Db, backendFullPath(root, input))
	if err != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("Error loading existing database entries: %v", err))
		q.ErrorJob(j.ID)
		return err
	}

	var allPaths, newFiles []string
	for _, f := range files {
		allPaths = append(allPaths, f.Path)
		if _, ok := existingPaths[f.Path]; !ok {
			newFiles = append(newFiles, f.Path)
		}
	}

	q.PushJobStdout(j.ID, fmt.Sprintf("Found %d new files to ingest", len(newFiles)))
	if len(newFiles) == 0 && len(opts.Tags) == 0 {
		q.PushJobStdout(j.ID, "All files already exist in database")
		q.CompleteJob(j.ID)
		return nil
	}

	var insertedFiles []string
	for i, filePath := range newFiles {
		select {
		case <-ctx.Done():
			q.PushJobStdout(j.ID, "Task was canceled")
			_ = q.CancelJob(j.ID)
			return ctx.Err()
		default:
		}

		// Object size isn't available from Scan; the metadata task fills it in.
		if err := insertMediaRecord(q.Db, filePath, 0); err != nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("Warning: failed to insert %s: %v", filePath, err))
			continue
		}
		insertedFiles = append(insertedFiles, filePath)
		q.RegisterOutputFile(j.ID, filePath)
		if (i+1)%100 == 0 || i == len(newFiles)-1 {
			q.PushJobStdout(j.ID, fmt.Sprintf("Progress: %d/%d files ingested", i+1, len(newFiles)))
		}
	}

	q.PushJobStdout(j.ID, fmt.Sprintf("Ingestion completed: %d files added to database", len(insertedFiles)))

	// Apply tags to ALL scanned media files (both new and existing)
	if len(opts.Tags) > 0 {
		applyIngestTags(q.Db, j.ID, q, allPaths, opts.Tags)
	}

	// Queue follow-up tasks for each inserted file
	queueFollowUpTasks(q, j.ID, insertedFiles, opts)

	select {
	case <-ctx.Done():
		q.PushJobStdout(j.ID, "Task was canceled")
		_ = q.CancelJob(j.ID)
		return ctx.Err()
	default:
	}
	q.CompleteJob(j.ID)
	return nil
}

// getExistingMediaPathsByPrefix returns media paths that start with prefix,
// matched literally against the stored path (no filepath.Abs normalization —
// storage paths like "s3://bucket/key" must not be touched).
func getExistingMediaPathsByPrefix(db *sql.DB, prefix string) (map[string]struct{}, error) {
	rows, err := db.Query(`SELECT path FROM media WHERE path LIKE ?`, prefix+"%")
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	result := make(map[string]struct{})
	for rows.Next() {
		var p string
		if err := rows.Scan(&p); err != nil {
			return nil, err
		}
		result[p] = struct{}{}
	}
	return result, rows.Err()
}
