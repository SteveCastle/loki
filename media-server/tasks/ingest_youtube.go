package tasks

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"

	"github.com/stevecastle/shrike/appconfig"
	"github.com/stevecastle/shrike/embedexec"
	"github.com/stevecastle/shrike/jobqueue"
)

// ingestYouTubeTask downloads media from YouTube using yt-dlp and adds to database
// This is the legacy entry point; prefer ingestYouTubeTaskWithOptions for new code.
func ingestYouTubeTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	return ingestYouTubeTaskWithOptions(j, q, mu, IngestOptions{})
}

// ingestYouTubeTaskWithOptions downloads media from YouTube using yt-dlp and adds to database
// It supports optional follow-up tasks based on the provided IngestOptions.
func ingestYouTubeTaskWithOptions(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex, opts IngestOptions) error {
	ctx := j.Ctx
	url := strings.TrimSpace(j.Input)

	if err := ensureMediaTableSchema(q.Db); err != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("Error setting up database schema: %v", err))
		q.ErrorJob(j.ID)
		return err
	}

	// Get download path from config
	cfg := appconfig.Get()
	downloadPath := cfg.DownloadPath
	if downloadPath == "" {
		downloadPath = filepath.Join(os.Getenv("USERPROFILE"), "media")
	}

	// Ensure download directory exists
	if err := os.MkdirAll(downloadPath, 0755); err != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("Error creating download directory: %v", err))
		q.ErrorJob(j.ID)
		return err
	}

	q.PushJobStdout(j.ID, fmt.Sprintf("Starting YouTube download: %s", url))
	q.PushJobStdout(j.ID, fmt.Sprintf("Output directory: %s", downloadPath))

	// Build yt-dlp arguments
	// Use --print to get the final filename after download
	outputTemplate := filepath.Join(downloadPath, "%(title)s [%(id)s].%(ext)s")
	args := []string{
		"-o", outputTemplate,
		"--print", "after_move:filepath", // Print the final file path after download
		url,
	}

	// Add any extra arguments from job
	for _, arg := range j.Arguments {
		args = append(args, arg)
	}

	cmd, cleanup, err := embedexec.GetExec(ctx, "yt-dlp", args...)
	if err != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("Error starting yt-dlp: %s", err))
		q.ErrorJob(j.ID)
		return fmt.Errorf("start yt-dlp: %w", err)
	}
	if cleanup != nil {
		defer cleanup()
	}

	// Handle cancellation
	go func() {
		<-ctx.Done()
		if cmd.Process != nil {
			if runtime.GOOS == "windows" {
				_ = exec.Command("taskkill", "/F", "/T", "/PID", fmt.Sprintf("%d", cmd.Process.Pid)).Run()
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

	// Read stdout - capture file paths from --print output
	go func() {
		s := bufio.NewScanner(stdoutPipe)
		for s.Scan() {
			line := s.Text()
			q.PushJobStdout(j.ID, line)

			// Check if this line is a file path (from --print after_move:filepath)
			// It will be an absolute path to a media file
			if isMediaFile(line) && filepath.IsAbs(line) {
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

	err = cmd.Wait()
	<-doneReading

	select {
	case <-ctx.Done():
		q.PushJobStdout(j.ID, "Task was canceled")
		_ = q.CancelJob(j.ID)
		return ctx.Err()
	default:
	}

	if err != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("yt-dlp error: %s", err))
		q.ErrorJob(j.ID)
		return err
	}

	// Add downloaded files to database
	var insertedFiles []string
	for _, filePath := range downloadedFiles {
		var size int64
		if fi, statErr := os.Stat(filePath); statErr == nil {
			size = fi.Size()
		}
		if err := insertMediaRecord(q.Db, filePath, size); err != nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("Warning: failed to insert %s: %v", filePath, err))
			continue
		}
		insertedFiles = append(insertedFiles, filePath)
		q.PushJobStdout(j.ID, fmt.Sprintf("Added to database: %s", filePath))
	}

	q.PushJobStdout(j.ID, fmt.Sprintf("Download completed: %d files added to database", len(insertedFiles)))

	// Queue follow-up tasks for each inserted file
	queueFollowUpTasks(q, j.ID, insertedFiles, opts)

	q.CompleteJob(j.ID)
	return nil
}

// isYouTubeURL checks if the URL is a YouTube domain
func isYouTubeURL(url string) bool {
	lowerURL := strings.ToLower(url)
	return strings.Contains(lowerURL, "youtube.com") ||
		strings.Contains(lowerURL, "youtu.be") ||
		strings.Contains(lowerURL, "youtube-nocookie.com")
}

// scanYouTubeOutput scans output for file paths (helper for streaming output)
func scanYouTubeOutput(pipe io.ReadCloser, q *jobqueue.Queue, jobID string, files *[]string, mu *sync.Mutex) {
	s := bufio.NewScanner(pipe)
	for s.Scan() {
		line := s.Text()
		q.PushJobStdout(jobID, line)

		// Check if this line is a file path
		if isMediaFile(line) && filepath.IsAbs(line) {
			mu.Lock()
			*files = append(*files, line)
			mu.Unlock()
		}
	}
}
