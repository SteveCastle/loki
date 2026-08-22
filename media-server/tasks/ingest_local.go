package tasks

import (
	"context"
	"database/sql"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/media"
	"github.com/stevecastle/shrike/mediaext"
)

// ingestLocalTask scans local directories for media files and adds them to the database
// This is the legacy entry point; prefer ingestLocalTaskWithOptions for new code.
func ingestLocalTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	return ingestLocalTaskWithOptions(j, q, mu, IngestOptions{})
}

// scannedFile is one media file emitted by the streaming scanner.
type scannedFile struct {
	path string
	size int64
}

// scanState is shared between the walker goroutine and the ingest loop so
// heartbeats can report activity even through long stretches of directories
// that contain no media at all.
type scanState struct {
	entries atomic.Int64 // every directory entry visited
	dir     atomic.Value // string: directory currently being walked
}

func (s *scanState) currentDir() string {
	v, _ := s.dir.Load().(string)
	return v
}

// scanHeartbeatInterval paces the "still scanning" stdout lines and the
// time-based batch flush. Each stdout push rewrites the job row, so this must
// stay coarse.
const scanHeartbeatInterval = 2 * time.Second

// maxWalkWarnings caps per-entry skip warnings in stdout; the remainder is
// summarized in one line at the end of the scan.
const maxWalkWarnings = 20

// ingestLocalTaskWithOptions scans local directories for media files and adds
// them to the database. Discovery streams: files are inserted (in short
// batched transactions) while the walk is still running, the job progress bar
// grows as files are found, and heartbeat lines show scan activity even when
// no media has turned up yet. It supports optional follow-up tasks based on
// the provided IngestOptions.
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

	// One query up front so each discovered file can be classified (and
	// inserted) the moment the walker finds it.
	existingPaths, err := getExistingMediaPaths(q.Db, dirPath)
	if err != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("Error loading existing database entries: %v", err))
		q.ErrorJob(j.ID)
		return err
	}

	scan := &scanState{}
	var walkWarnings atomic.Int64
	warn := func(msg string) {
		if walkWarnings.Add(1) <= maxWalkWarnings {
			q.PushJobStdout(j.ID, msg)
		}
	}

	filesCh := make(chan scannedFile, 512)
	walkDone := make(chan error, 1)
	go func() {
		err := streamMediaFiles(ctx, dirPath, recursive, filesCh, scan, warn)
		close(filesCh)
		walkDone <- err
	}()

	batch := newMediaInsertBatch(q.Db)
	defer batch.Discard()

	var (
		allFiles       []string // every media file scanned (for tag application)
		insertedFiles  []string
		existingCount  int
		flushedInserts int
	)
	discovered := func() int { return len(allFiles) }
	reportProgress := func() {
		if n := discovered(); n > 0 {
			_ = q.SetJobProgress(j.ID, existingCount+flushedInserts, n)
		}
	}
	flush := func() error {
		n, err := batch.Flush()
		if err != nil {
			return err
		}
		flushedInserts += n
		reportProgress()
		return nil
	}
	// finishPartial makes whatever streamed in before a cancel/error durable
	// and visible to downstream workflow steps.
	finishPartial := func() {
		_, _ = batch.Flush()
		if len(insertedFiles) > 0 {
			_ = q.RegisterOutputFiles(j.ID, insertedFiles)
		}
	}
	// finishPartial runs first: while the batch transaction is open it holds
	// the SQLite write lock, so job-row writes (stdout/state) must not race it.
	fail := func(what string, err error) error {
		finishPartial()
		q.PushJobStdout(j.ID, fmt.Sprintf("%s: %v", what, err))
		q.ErrorJob(j.ID)
		return err
	}

	ticker := time.NewTicker(scanHeartbeatInterval)
	defer ticker.Stop()

	var lastBeatEntries int64 = -1
scanLoop:
	for {
		select {
		case <-ctx.Done():
			finishPartial()
			<-walkDone
			q.PushJobStdout(j.ID, fmt.Sprintf("Task was canceled — %d file(s) added before cancel", len(insertedFiles)))
			_ = q.CancelJob(j.ID)
			return ctx.Err()
		case <-ticker.C:
			if err := flush(); err != nil {
				return fail("Error writing to database", err)
			}
			// Heartbeat only when the walk moved; a genuinely stuck walk
			// (dead network share) would otherwise repeat the same line.
			if entries := scan.entries.Load(); entries != lastBeatEntries {
				lastBeatEntries = entries
				q.PushJobStdout(j.ID, fmt.Sprintf(
					"Scanning: %d entries examined, %d media files found (%d new) — %s",
					entries, discovered(), len(insertedFiles), scan.currentDir()))
			}
		case f, ok := <-filesCh:
			if !ok {
				break scanLoop
			}
			allFiles = append(allFiles, f.path)
			if _, exists := existingPaths[f.path]; exists {
				existingCount++
				continue
			}
			if err := batch.Add(f.path, f.size); err != nil {
				return fail(fmt.Sprintf("Error inserting %s", f.path), err)
			}
			insertedFiles = append(insertedFiles, f.path)
			if batch.pending >= mediaInsertBatchSize {
				if err := flush(); err != nil {
					return fail("Error writing to database", err)
				}
			}
		}
	}
	if err := flush(); err != nil {
		return fail("Error writing to database", err)
	}

	if extra := walkWarnings.Load() - maxWalkWarnings; extra > 0 {
		q.PushJobStdout(j.ID, fmt.Sprintf("...and %d more unreadable entries skipped", extra))
	}
	if walkErr := <-walkDone; walkErr != nil {
		if ctx.Err() != nil {
			finishPartial()
			q.PushJobStdout(j.ID, "Task was canceled")
			_ = q.CancelJob(j.ID)
			return ctx.Err()
		}
		return fail("Error scanning directory", walkErr)
	}

	if n := discovered(); n > 0 {
		_ = q.SetJobProgress(j.ID, n, n)
	}
	q.PushJobStdout(j.ID, fmt.Sprintf("Scan complete: %d media files found (%d new, %d already in database)",
		discovered(), len(insertedFiles), existingCount))

	if discovered() == 0 {
		q.PushJobStdout(j.ID, "No media files found to ingest")
		q.CompleteJob(j.ID)
		return nil
	}
	if len(insertedFiles) == 0 && len(opts.Tags) == 0 {
		q.PushJobStdout(j.ID, "All files already exist in database")
		q.CompleteJob(j.ID)
		return nil
	}

	q.PushJobStdout(j.ID, fmt.Sprintf("Ingestion completed: %d files added to database", len(insertedFiles)))
	if len(insertedFiles) > 0 {
		_ = q.RegisterOutputFiles(j.ID, insertedFiles)
	}

	// Apply tags to ALL scanned media files (both new and existing)
	if len(opts.Tags) > 0 {
		applyIngestTags(q.Db, j.ID, q, allFiles, opts.Tags)
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

// streamMediaFiles walks dir (recursively if specified) and sends each media
// file on out as soon as it is found. Unreadable entries below the root are
// reported through warn and skipped; only a failure on the root itself (bad
// path, no permission) fails the scan. WalkDir is used over Walk so
// non-media entries cost no stat call, and sizes come from the DirEntry
// (free on Windows, one lstat elsewhere) instead of a second os.Stat.
func streamMediaFiles(ctx context.Context, dir string, recursive bool, out chan<- scannedFile, scan *scanState, warn func(string)) error {
	isMedia := mediaext.IsMedia
	return filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			if path == dir {
				return err
			}
			warn(fmt.Sprintf("Warning: skipping %s: %v", path, err))
			return nil
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		scan.entries.Add(1)
		if d.IsDir() {
			if !recursive && path != dir {
				return filepath.SkipDir
			}
			scan.dir.Store(path)
			return nil
		}
		if !isMedia(path) {
			return nil
		}
		p := path
		if abs, absErr := filepath.Abs(path); absErr == nil {
			p = filepath.FromSlash(abs)
		}
		var size int64
		if d.Type()&fs.ModeSymlink != 0 {
			if fi, statErr := os.Stat(path); statErr == nil {
				size = fi.Size()
			}
		} else if fi, infoErr := d.Info(); infoErr == nil {
			size = fi.Size()
		}
		select {
		case out <- scannedFile{path: p, size: size}:
			return nil
		case <-ctx.Done():
			return ctx.Err()
		}
	})
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

// mediaInsertSQL is the shared upsert for media rows. Existing rows are left
// alone except for size, which is backfilled when a real size is now known
// and the stored one is NULL/0 (older S3 ingests never set it).
const mediaInsertSQL = `INSERT INTO media (path, size) VALUES (?, ?)
	ON CONFLICT(path) DO UPDATE SET size = excluded.size
	WHERE excluded.size > 0 AND (media.size IS NULL OR media.size = 0)`

// mediaInsertBatchSize bounds how many rows share one transaction. Small
// enough that a batch commits well inside the heartbeat interval (so streamed
// files become visible quickly and other writers aren't starved), large
// enough to amortize the per-transaction fsync that made one-row-per-insert
// ingestion crawl.
const mediaInsertBatchSize = 400

// mediaInsertBatch groups media inserts into short transactions.
type mediaInsertBatch struct {
	db       *sql.DB
	tx       *sql.Tx
	stmt     *sql.Stmt
	pending  int
	affected int64
}

func newMediaInsertBatch(db *sql.DB) *mediaInsertBatch {
	return &mediaInsertBatch{db: db}
}

func (b *mediaInsertBatch) Add(path string, size int64) error {
	if b.tx == nil {
		tx, err := b.db.Begin()
		if err != nil {
			return err
		}
		stmt, err := tx.Prepare(mediaInsertSQL)
		if err != nil {
			_ = tx.Rollback()
			return err
		}
		b.tx, b.stmt = tx, stmt
	}
	res, err := b.stmt.Exec(path, size)
	if err != nil {
		return err
	}
	if n, raErr := res.RowsAffected(); raErr == nil {
		b.affected += n
	}
	b.pending++
	return nil
}

// Flush commits the open transaction and returns how many rows it carried.
// New media rows enter the swipe pool (the sampler universe is the media
// table), so the sample cache is invalidated once per batch, not per row.
func (b *mediaInsertBatch) Flush() (int, error) {
	if b.tx == nil {
		return 0, nil
	}
	n, affected := b.pending, b.affected
	_ = b.stmt.Close()
	err := b.tx.Commit()
	b.tx, b.stmt, b.pending, b.affected = nil, nil, 0, 0
	if err != nil {
		return 0, err
	}
	if affected > 0 {
		media.InvalidateRandomSampleCache()
	}
	return n, nil
}

// Discard rolls back any open transaction. Safe to call when nothing is open.
func (b *mediaInsertBatch) Discard() {
	if b.tx == nil {
		return
	}
	_ = b.stmt.Close()
	_ = b.tx.Rollback()
	b.tx, b.stmt, b.pending, b.affected = nil, nil, 0, 0
}

// insertMediaRecord inserts a basic media record into the database in its own
// transaction. Batch callers should use mediaInsertBatch instead.
func insertMediaRecord(db *sql.DB, path string, size int64) error {
	res, err := db.Exec(mediaInsertSQL, path, size)
	if err == nil {
		// A new media row enters the swipe pool (the sampler universe is the
		// media table). Cheap flag set; skipped for no-op upserts.
		if n, raErr := res.RowsAffected(); raErr == nil && n > 0 {
			media.InvalidateRandomSampleCache()
		}
	}
	return err
}
