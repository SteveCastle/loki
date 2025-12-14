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

	validTypes := map[string]bool{
		"description": true,
		"transcript":  true,
		"hash":        true,
		"dimensions":  true,
		"autotag":     true,
	}
	for _, mType := range metadataTypes {
		if !validTypes[strings.ToLower(mType)] {
			q.PushJobStdout(j.ID, fmt.Sprintf("Warning: unknown metadata type '%s' - valid types are: description, transcript, hash, dimensions, autotag", mType))
		}
	}

	q.PushJobStdout(j.ID, fmt.Sprintf("Starting metadata generation for types: %s", strings.Join(metadataTypes, ", ")))
	q.PushJobStdout(j.ID, fmt.Sprintf("Apply scope: %s", applyScope))
	q.PushJobStdout(j.ID, fmt.Sprintf("Overwrite existing: %t", overwrite))

	var filesToProcess []string
	var err error
	if qstr, ok := extractQueryFromJob(j); ok {
		q.PushJobStdout(j.ID, fmt.Sprintf("Using query to select files: %s", qstr))
		filesToProcess, err = getMediaPathsByQuery(q.Db, qstr)
		if err != nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("Error loading media paths for query: %v", err))
			q.ErrorJob(j.ID)
			return err
		}
		q.PushJobStdout(j.ID, fmt.Sprintf("Query matched %d items before filtering", len(filesToProcess)))
	} else if strings.TrimSpace(j.Input) != "" {
		raw := strings.TrimSpace(j.Input)
		if raw == "" {
			q.PushJobStdout(j.ID, "No input paths provided and no query; expecting comma/newline-separated file paths")
			q.CompleteJob(j.ID)
			return nil
		}
		inputPaths := parseInputPaths(raw)
		q.PushJobStdout(j.ID, fmt.Sprintf("Processing %d files from input list", len(inputPaths)))
		q.PushJobStdout(j.ID, fmt.Sprintf("Raw: %s", raw))
		for _, p := range inputPaths {
			absPath, err := filepath.Abs(p)
			if err == nil {
				p = filepath.FromSlash(absPath)
			}
			if _, err := os.Stat(p); os.IsNotExist(err) {
				q.PushJobStdout(j.ID, fmt.Sprintf("Warning: file does not exist: %s", p))
				continue
			}
			if !isMediaFile(p) {
				q.PushJobStdout(j.ID, fmt.Sprintf("Warning: not a supported media file: %s", p))
				continue
			}
			filesToProcess = append(filesToProcess, p)
		}
		q.PushJobStdout(j.ID, "Processing files from input list")
	}

	if len(filesToProcess) == 0 {
		q.PushJobStdout(j.ID, "No valid files found to process")
		q.CompleteJob(j.ID)
		return nil
	}

	if applyScope == "new" {
		var newFiles []string
		for _, filePath := range filesToProcess {
			exists, err := fileExistsInDatabase(q.Db, filePath)
			if err != nil {
				q.PushJobStdout(j.ID, fmt.Sprintf("Warning: error checking database for %s: %v", filePath, err))
				continue
			}
			if !exists {
				newFiles = append(newFiles, filePath)
			}
		}
		filesToProcess = newFiles
		q.PushJobStdout(j.ID, fmt.Sprintf("After filtering for new files: %d files to process", len(filesToProcess)))
	} else {
		q.PushJobStdout(j.ID, fmt.Sprintf("Processing all specified files: %d files", len(filesToProcess)))
	}

	if len(filesToProcess) == 0 {
		q.PushJobStdout(j.ID, "No files to process after filtering")
		q.CompleteJob(j.ID)
		return nil
	}

	// Announce the final total to process before any specific metadata type is run
	q.PushJobStdout(j.ID, fmt.Sprintf("Total files to process: %d", len(filesToProcess)))

	for _, metadataType := range metadataTypes {
		mType := strings.ToLower(metadataType)
		q.PushJobStdout(j.ID, fmt.Sprintf("Generating %s metadata...", mType))
		switch mType {
		case "description":
			err = generateDescriptions(ctx, q, j.ID, filesToProcess, overwrite, ollamaModel)
		case "transcript":
			err = generateTranscripts(ctx, q, j.ID, filesToProcess, overwrite)
		case "hash":
			err = generateHashes(ctx, q, j.ID, filesToProcess, overwrite)
		case "dimensions":
			err = generateDimensions(ctx, q, j.ID, filesToProcess, overwrite)
		case "autotag":
			err = generateAutoTags(ctx, q, j.ID, filesToProcess, overwrite, ollamaModel)
		default:
			q.PushJobStdout(j.ID, fmt.Sprintf("Skipping unknown metadata type: %s", mType))
			continue
		}
		if err != nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("Error generating %s: %v", mType, err))
			q.ErrorJob(j.ID)
			return err
		}
		select {
		case <-ctx.Done():
			q.PushJobStdout(j.ID, "Task was canceled")
			_ = q.CancelJob(j.ID)
			return ctx.Err()
		default:
		}
	}

	q.PushJobStdout(j.ID, "Metadata generation completed successfully")
	q.CompleteJob(j.ID)
	return nil
}
