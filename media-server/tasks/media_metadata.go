package tasks

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/stevecastle/shrike/appconfig"
	"github.com/stevecastle/shrike/jobqueue"
)

// metadataTask generates various metadata for media files
func metadataTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	ctx := j.Ctx

	var metadataTypes []string
	var overwrite bool
	var applyScope string = "new"
	var ollamaModel string = appconfig.Get().OllamaModel

	for i, arg := range j.Arguments {
		switch strings.ToLower(arg) {
		case "--type", "-t":
			if i+1 < len(j.Arguments) {
				metadataTypes = strings.Split(j.Arguments[i+1], ",")
				for idx, t := range metadataTypes {
					metadataTypes[idx] = strings.TrimSpace(t)
				}
			}
		case "--overwrite", "-o":
			overwrite = true
		case "--apply", "-a":
			if i+1 < len(j.Arguments) {
				applyScope = strings.ToLower(strings.TrimSpace(j.Arguments[i+1]))
			}
		case "--model", "-m":
			if i+1 < len(j.Arguments) {
				ollamaModel = strings.TrimSpace(j.Arguments[i+1])
			}
		}
	}

	if len(metadataTypes) == 0 {
		metadataTypes = []string{"description", "hash", "dimensions"}
	}

	// Normalize metadata types to lowercase
	for idx, t := range metadataTypes {
		metadataTypes[idx] = strings.ToLower(t)
	}

	validTypes := map[string]bool{
		"description": true,
		"transcript":  true,
		"hash":        true,
		"dimensions":  true,
		"autotag":     true,
	}
	for _, mType := range metadataTypes {
		if !validTypes[mType] {
			q.PushJobStdout(j.ID, fmt.Sprintf("Warning: unknown metadata type '%s' - valid types are: description, transcript, hash, dimensions, autotag", mType))
		}
	}

	q.PushJobStdout(j.ID, fmt.Sprintf("Starting metadata generation for types: %s", strings.Join(metadataTypes, ", ")))
	q.PushJobStdout(j.ID, fmt.Sprintf("Apply scope: %s", applyScope))
	q.PushJobStdout(j.ID, fmt.Sprintf("Overwrite existing: %t", overwrite))

	// Gather file paths without checking existence upfront
	var filesToProcess []string
	var err error
	fromQuery := false // Track if files came from database query (skip DB existence checks)
	if qstr, ok := extractQueryFromJob(j); ok {
		q.PushJobStdout(j.ID, fmt.Sprintf("Using query to select files: %s", qstr))
		filesToProcess, err = getMediaPathsByQueryFast(q.Db, qstr)
		if err != nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("Error loading media paths for query: %v", err))
			q.ErrorJob(j.ID)
			return err
		}
		q.PushJobStdout(j.ID, fmt.Sprintf("Query matched %d items", len(filesToProcess)))
		fromQuery = true
	} else if strings.TrimSpace(j.Input) != "" {
		raw := strings.TrimSpace(j.Input)
		inputPaths := parseInputPaths(raw)
		q.PushJobStdout(j.ID, fmt.Sprintf("Processing %d files from input list", len(inputPaths)))
		for _, p := range inputPaths {
			absPath, err := filepath.Abs(p)
			if err == nil {
				p = filepath.FromSlash(absPath)
			}
			filesToProcess = append(filesToProcess, p)
		}
	}

	if len(filesToProcess) == 0 {
		q.PushJobStdout(j.ID, "No files found to process")
		q.CompleteJob(j.ID)
		return nil
	}

	q.PushJobStdout(j.ID, fmt.Sprintf("Total files to process: %d", len(filesToProcess)))

	// Pre-fetch available tags if autotag is requested
	var availableTags []TagInfo
	hasAutotag := false
	for _, mType := range metadataTypes {
		if mType == "autotag" {
			hasAutotag = true
			break
		}
	}
	if hasAutotag {
		availableTags, err = getAllAvailableTags(q.Db)
		if err != nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("Warning: failed to fetch available tags: %v", err))
		} else {
			q.PushJobStdout(j.ID, fmt.Sprintf("Found %d available tags for auto-tagging", len(availableTags)))
		}
	}

	// Process each file once, applying all selected operations
	processed := 0
	skipped := 0
	errors := 0
	for i, filePath := range filesToProcess {
		select {
		case <-ctx.Done():
			q.PushJobStdout(j.ID, "Task was canceled")
			_ = q.CancelJob(j.ID)
			return ctx.Err()
		default:
		}

		// Check if file exists
		if _, err := os.Stat(filePath); os.IsNotExist(err) {
			q.PushJobStdout(j.ID, fmt.Sprintf("Skipping (not found): %s", filePath))
			skipped++
			continue
		}

		// Check if file is a supported media file (skip for query sources - already filtered)
		if !fromQuery && !isMediaFile(filePath) {
			q.PushJobStdout(j.ID, fmt.Sprintf("Skipping (not media): %s", filePath))
			skipped++
			continue
		}

		q.PushJobStdout(j.ID, fmt.Sprintf("Processing %d/%d: %s", i+1, len(filesToProcess), filepath.Base(filePath)))

		// Apply each selected operation to this file
		fileProcessed := false
		for _, mType := range metadataTypes {
			var opErr error
			switch mType {
			case "description":
				opErr = processDescriptionForFile(ctx, q, j.ID, filePath, overwrite, ollamaModel, fromQuery)
			case "transcript":
				opErr = processTranscriptForFile(ctx, q, j.ID, filePath, overwrite, fromQuery)
			case "hash":
				opErr = processHashForFile(ctx, q, j.ID, filePath, overwrite, fromQuery)
			case "dimensions":
				opErr = processDimensionsForFile(ctx, q, j.ID, filePath, overwrite, fromQuery)
			case "autotag":
				opErr = processAutotagForFile(ctx, q, j.ID, filePath, overwrite, ollamaModel, availableTags, fromQuery)
			}
			if opErr != nil {
				q.PushJobStdout(j.ID, fmt.Sprintf("  %s failed: %v", mType, opErr))
			} else {
				fileProcessed = true
			}
		}
		if fileProcessed {
			processed++
		}
	}

	q.PushJobStdout(j.ID, fmt.Sprintf("Metadata generation completed: %d processed, %d skipped, %d errors", processed, skipped, errors))
	q.CompleteJob(j.ID)
	return nil
}
