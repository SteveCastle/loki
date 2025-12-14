package tasks

import (
	"database/sql"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/stevecastle/shrike/jobqueue"
)

// ingestLocalTask scans local directories for media files and adds them to the database
// This is the legacy entry point; prefer ingestLocalTaskWithOptions for new code.
func ingestLocalTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	return ingestLocalTaskWithOptions(j, q, mu, IngestOptions{})
}

// ingestLocalTaskWithOptions scans local directories for media files and adds them to the database
// It supports optional follow-up tasks based on the provided IngestOptions.
func ingestLocalTaskWithOptions(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex, opts IngestOptions) error {
	ctx := j.Ctx

	var dirPath string
	recursive := opts.Recursive
	if j.Input != "" {
		dirPath = strings.TrimSpace(j.Input)
	} else {
		dirPath = "."
	}
	for _, arg := range j.Arguments {
		switch strings.ToLower(arg) {
		case "-r", "--recursive":
			recursive = true
		}
		if !strings.HasPrefix(arg, "-") && arg != "" {
			dirPath = arg
		}
	}

	if err := ensureMediaTableSchema(q.Db); err != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("Error setting up database schema: %v", err))
		q.ErrorJob(j.ID)
		return err
	}

	q.PushJobStdout(j.ID, fmt.Sprintf("Starting media file ingestion from: %s", dirPath))
	if recursive {
		q.PushJobStdout(j.ID, "Scanning recursively...")
	}

	mediaFiles, err := scanMediaFiles(dirPath, recursive)
	if err != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("Error scanning directory: %v", err))
		q.ErrorJob(j.ID)
		return err
	}

	q.PushJobStdout(j.ID, fmt.Sprintf("Found %d media files", len(mediaFiles)))
	if len(mediaFiles) == 0 {
		q.PushJobStdout(j.ID, "No media files found to ingest")
		q.CompleteJob(j.ID)
		return nil
	}

	existingPaths, err := getExistingMediaPaths(q.Db, dirPath)
	if err != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("Error loading existing database entries: %v", err))
		q.ErrorJob(j.ID)
		return err
	}

	var newFiles []string
	for _, f := range mediaFiles {
		if _, ok := existingPaths[f]; !ok {
			newFiles = append(newFiles, f)
		}
	}

	q.PushJobStdout(j.ID, fmt.Sprintf("Found %d new files to ingest", len(newFiles)))
	if len(newFiles) == 0 {
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

		var size int64
		if fi, err := os.Stat(filePath); err == nil {
			size = fi.Size()
		}
		if err := insertMediaRecord(q.Db, filePath, size); err != nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("Warning: failed to insert %s: %v", filePath, err))
			continue
		}
		insertedFiles = append(insertedFiles, filePath)
		if (i+1)%100 == 0 || i == len(newFiles)-1 {
			q.PushJobStdout(j.ID, fmt.Sprintf("Progress: %d/%d files ingested", i+1, len(newFiles)))
		}
	}

	q.PushJobStdout(j.ID, fmt.Sprintf("Ingestion completed: %d files added to database", len(insertedFiles)))

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

// ensureMediaTableSchema ensures the media table has all required columns
func ensureMediaTableSchema(db *sql.DB) error {
	createTableSQL := `
	CREATE TABLE IF NOT EXISTS media (
		path TEXT PRIMARY KEY,
		description TEXT,
		transcript TEXT,
		hash TEXT,
		size INTEGER,
		width INTEGER,
		height INTEGER
	);`
	if _, err := db.Exec(createTableSQL); err != nil {
		return fmt.Errorf("failed to create media table: %w", err)
	}
	_, _ = db.Exec(`ALTER TABLE media ADD COLUMN width INTEGER;`)
	_, _ = db.Exec(`ALTER TABLE media ADD COLUMN height INTEGER;`)
	return nil
}

// scanMediaFiles scans the directory (recursively if specified) for media files
func scanMediaFiles(dir string, recursive bool) ([]string, error) {
	var files []string
	isMedia := func(path string) bool {
		ext := strings.ToLower(filepath.Ext(path))
		switch ext {
		case ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".heic", ".tif", ".tiff",
			".mp4", ".mov", ".avi", ".mkv", ".webm", ".wmv":
			return true
		}
		return false
	}
	walkFn := func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() && !recursive && path != dir {
			return filepath.SkipDir
		}
		if !info.IsDir() && isMedia(path) {
			absPath, err := filepath.Abs(path)
			if err == nil {
				files = append(files, filepath.FromSlash(absPath))
			} else {
				files = append(files, path)
			}
		}
		return nil
	}
	if err := filepath.Walk(dir, walkFn); err != nil {
		return nil, err
	}
	return files, nil
}

// getExistingMediaPaths loads existing media paths from the database
func getExistingMediaPaths(db *sql.DB, dirPath string) (map[string]struct{}, error) {
	query := `SELECT path FROM media`
	var args []interface{}
	if dirPath != "" && dirPath != "." {
		if absDir, err := filepath.Abs(dirPath); err == nil {
			dirPath = filepath.FromSlash(absDir)
		}
		query += ` WHERE path LIKE ?`
		args = append(args, dirPath+"%")
	}
	rows, err := db.Query(query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	result := make(map[string]struct{})
	for rows.Next() {
		var path string
		if err := rows.Scan(&path); err != nil {
			return nil, err
		}
		result[path] = struct{}{}
	}
	return result, nil
}

// insertMediaRecord inserts a basic media record into the database
func insertMediaRecord(db *sql.DB, path string, size int64) error {
	stmt := `INSERT OR IGNORE INTO media (path, size) VALUES (?, ?)`
	_, err := db.Exec(stmt, path, size)
	return err
}
