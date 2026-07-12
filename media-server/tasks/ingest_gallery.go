package tasks

import (
	"bufio"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"

	"github.com/stevecastle/shrike/deps"
	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/platform"
)

// ingestGalleryTask downloads media using gallery-dl and adds to database
// Each line of stdout from gallery-dl is the path of a written file
// This is the legacy entry point; prefer ingestGalleryTaskWithOptions for new code.
func ingestGalleryTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	return ingestGalleryTaskWithOptions(j, q, mu, IngestOptions{})
}

// ingestGalleryTaskWithOptions downloads media using gallery-dl and adds to database
// It supports optional follow-up tasks based on the provided IngestOptions.
func ingestGalleryTaskWithOptions(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex, opts IngestOptions) error {
	ctx := j.Ctx
	url := strings.TrimSpace(j.Input)

	if err := ensureMediaTableSchema(q.Db); err != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("Error setting up database schema: %v", err))
		q.ErrorJob(j.ID)
		return err
	}

	// Resolve where gallery-dl writes: straight into the final local location
	// (so it can detect and skip already-downloaded files) or a temp staging
	// dir for S3 / no-backend setups.
	target, err := resolveIngestDir(j.ID, "downloads/")
	if err != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("Error resolving download directory: %v", err))
		q.ErrorJob(j.ID)
		return err
	}
	defer target.cleanup()

	q.PushJobStdout(j.ID, fmt.Sprintf("Starting gallery-dl download: %s", url))
	if target.direct {
		q.PushJobStdout(j.ID, fmt.Sprintf("Download directory (direct): %s", target.dir))
	} else {
		q.PushJobStdout(j.ID, fmt.Sprintf("Staging directory: %s", target.dir))
	}

	// Build gallery-dl arguments
	// -d sets the base download directory
	// --write-log outputs the downloaded file paths
	args := []string{
		"-d", target.dir,
		"--write-metadata",
		url,
	}

	// Add any extra arguments from job
	for _, arg := range j.Arguments {
		args = append(args, arg)
	}

	status, _ := deps.DetectOptional("gallery-dl")
	if !status.Installed {
		msg := "gallery-dl is not installed. Install it (e.g. pipx install gallery-dl) and try again. See /onboarding for per-OS instructions."
		q.PushJobStdout(j.ID, msg)
		q.ErrorJob(j.ID)
		return fmt.Errorf("gallery-dl not installed")
	}
	cmd := exec.CommandContext(ctx, status.Path, args...)
	platform.HideSubprocessWindow(cmd)

	// Handle cancellation
	go func() {
		<-ctx.Done()
		if cmd.Process != nil {
			if runtime.GOOS == "windows" {
				tk := exec.Command("taskkill", "/F", "/T", "/PID", fmt.Sprintf("%d", cmd.Process.Pid))
				platform.HideSubprocessWindow(tk)
				_ = tk.Run()
			} else {
				_ = cmd.Process.Kill()
			}
		}
	}()

	stdoutPipe, err := cmd.StdoutPipe()
	if err != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("Error getting stdout pipe: %s", err))
		q.ErrorJob(j.ID)
		return fmt.Errorf("stdout pipe: %w", err)
	}
	stderrPipe, err := cmd.StderrPipe()
	if err != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("Error getting stderr pipe: %s", err))
		q.ErrorJob(j.ID)
		return fmt.Errorf("stderr pipe: %w", err)
	}

	if err := cmd.Start(); err != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("Error starting command: %s", err))
		q.ErrorJob(j.ID)
		return fmt.Errorf("start: %w", err)
	}

	var downloadedFiles []string
	doneReading := make(chan struct{})
	totalReaders := 2
	doneCount := 0

	// Read stdout - gallery-dl outputs file paths line by line
	go func() {
		s := bufio.NewScanner(stdoutPipe)
		for s.Scan() {
			line := strings.TrimSpace(s.Text())
			q.PushJobStdout(j.ID, line)

			// gallery-dl outputs the file path for each downloaded file
			// Check if the line looks like a valid file path
			if line != "" && isDownloadedFilePath(line, target.dir) {
				mu.Lock()
				downloadedFiles = append(downloadedFiles, line)
				mu.Unlock()
			}
		}
		mu.Lock()
		doneCount++
		if doneCount == totalReaders {
			close(doneReading)
		}
		mu.Unlock()
	}()

	// Read stderr
	go func() {
		s := bufio.NewScanner(stderrPipe)
		for s.Scan() {
			q.PushJobStdout(j.ID, s.Text())
		}
		mu.Lock()
		doneCount++
		if doneCount == totalReaders {
			close(doneReading)
		}
		mu.Unlock()
	}()

	galleryErr := cmd.Wait()
	<-doneReading

	select {
	case <-ctx.Done():
		q.PushJobStdout(j.ID, "Task was canceled")
		_ = q.CancelJob(j.ID)
		return ctx.Err()
	default:
	}

	// On gallery-dl failure, still ingest anything that was successfully
	// downloaded before the error so partial work isn't thrown away. We
	// fall through to the upload/insert/tag/follow-up flow and mark the
	// job as errored at the end to surface the underlying failure.
	if galleryErr != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("gallery-dl error: %s", galleryErr))
		if len(downloadedFiles) == 0 {
			q.ErrorJob(j.ID)
			return galleryErr
		}
		q.PushJobStdout(j.ID, fmt.Sprintf("Continuing to ingest %d file(s) downloaded before the failure", len(downloadedFiles)))
	}

	// Collect gallery-dl `--write-metadata` sidecars (`<media>.json`) so they
	// can be copied alongside their media file. Sidecars are filtered out of
	// `downloadedFiles` by isDownloadedFilePath, so look them up explicitly.
	var metadataFiles []string
	for _, mediaPath := range downloadedFiles {
		sidecar := mediaPath + ".json"
		if _, err := os.Stat(sidecar); err == nil {
			metadataFiles = append(metadataFiles, sidecar)
		}
	}

	// In direct mode the downloaded paths are already final; otherwise upload
	// the staged files (and their sidecars) to the default storage backend.
	var finalFiles []storedFile
	if target.direct {
		finalFiles = localStoredFiles(downloadedFiles)
	} else {
		finalFiles = uploadStagedFiles(ctx, q, j.ID, downloadedFiles, target.dir, "downloads/")

		// Upload metadata sidecars to the same destination so they sit next to
		// the media. Not inserted into the media table — they are sidecars only.
		if len(metadataFiles) > 0 {
			uploadStagedFiles(ctx, q, j.ID, metadataFiles, target.dir, "downloads/")
		}
	}

	// Add final files to database
	var insertedFiles []string
	for _, f := range finalFiles {
		if err := insertMediaRecord(q.Db, f.Path, f.Size); err != nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("Warning: failed to insert %s: %v", f.Path, err))
			continue
		}
		insertedFiles = append(insertedFiles, f.Path)
		q.PushJobStdout(j.ID, fmt.Sprintf("Added to database: %s", f.Path))
		q.RegisterOutputFile(j.ID, f.Path)
	}

	q.PushJobStdout(j.ID, fmt.Sprintf("Download completed: %d files added to database", len(insertedFiles)))

	// Tag inserted files using the tags found in their JSON sidecars (if any).
	// The sidecars only exist on local disk under stagingPath until the deferred
	// cleanupStaging runs, so this must happen before the function returns.
	if len(insertedFiles) > 0 {
		insertedSet := make(map[string]struct{}, len(insertedFiles))
		for _, p := range insertedFiles {
			insertedSet[p] = struct{}{}
		}
		finalToSidecar := make(map[string]string)
		for _, mediaStaging := range downloadedFiles {
			// In direct mode the downloaded path is already final; otherwise
			// predict where the staged file landed in the backend.
			predictedFinal := mediaStaging
			if !target.direct {
				predictedFinal = stagedToFinalPath(mediaStaging, target.dir, "downloads/")
			}
			if _, ok := insertedSet[predictedFinal]; !ok {
				continue
			}
			sidecar := mediaStaging + ".json"
			if _, err := os.Stat(sidecar); err != nil {
				continue
			}
			finalToSidecar[predictedFinal] = sidecar
		}
		applySidecarTagsToMedia(q.Db, q, j.ID, finalToSidecar)
	}

	// Apply tags to downloaded files
	if len(opts.Tags) > 0 {
		applyIngestTags(q.Db, j.ID, q, insertedFiles, opts.Tags)
	}

	// Queue follow-up tasks for each inserted file
	queueFollowUpTasks(q, j.ID, insertedFiles, opts)

	if galleryErr != nil {
		q.ErrorJob(j.ID)
		return galleryErr
	}

	q.CompleteJob(j.ID)
	return nil
}

// isDownloadedFilePath checks if a line looks like a downloaded file path
// gallery-dl outputs the path of each written file to stdout
func isDownloadedFilePath(line string, downloadPath string) bool {
	// gallery-dl prefixes a path with '#' when the file already exists and is
	// being skipped (not re-downloaded). These are not new files, so they must
	// not be ingested.
	if strings.HasPrefix(line, "#") {
		return false
	}

	// Check if it's an absolute path or starts with download path
	if filepath.IsAbs(line) {
		// Verify it's a media file
		if isMediaFile(line) {
			return true
		}
		// Also accept other file types that gallery-dl might download
		ext := strings.ToLower(filepath.Ext(line))
		switch ext {
		case ".json", ".txt", ".html":
			// Skip metadata files
			return false
		}
		// Accept if it exists in download path
		if strings.HasPrefix(filepath.Clean(line), filepath.Clean(downloadPath)) {
			return true
		}
	}

	// Check if line contains the download path ('#'-prefixed skip lines are
	// already rejected above).
	if strings.Contains(line, downloadPath) {
		if isMediaFile(line) {
			return true
		}
	}

	return false
}
