package tasks

import (
	"context"
	"database/sql"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"

	"github.com/google/uuid"
	"github.com/stevecastle/shrike/appconfig"
	"github.com/stevecastle/shrike/embedexec"
	"github.com/stevecastle/shrike/jobqueue"
)

// loraDatasetTask creates a LoRA training dataset from selected media files.
// It copies images to a new folder, converts them to JPG with lowercase UUID names,
// and creates accompanying text files with descriptions.
func loraDatasetTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	ctx := j.Ctx

	var targetDir string
	var loraName string
	var conceptPrefix string
	var ollamaModel string = appconfig.Get().OllamaModel

	// Parse arguments
	for i, arg := range j.Arguments {
		switch strings.ToLower(arg) {
		case "--target", "-t":
			if i+1 < len(j.Arguments) {
				targetDir = strings.TrimSpace(j.Arguments[i+1])
			}
		case "--name", "-n":
			if i+1 < len(j.Arguments) {
				loraName = strings.TrimSpace(j.Arguments[i+1])
			}
		case "--prefix", "-p":
			if i+1 < len(j.Arguments) {
				conceptPrefix = strings.TrimSpace(j.Arguments[i+1])
			}
		case "--model", "-m":
			if i+1 < len(j.Arguments) {
				ollamaModel = strings.TrimSpace(j.Arguments[i+1])
			}
		}
	}

	// Validate required arguments
	if targetDir == "" {
		q.PushJobStdout(j.ID, "Error: No target directory specified. Use --target or -t")
		q.ErrorJob(j.ID)
		return fmt.Errorf("no target directory specified")
	}
	if loraName == "" {
		q.PushJobStdout(j.ID, "Error: No LoRA name specified. Use --name or -n")
		q.ErrorJob(j.ID)
		return fmt.Errorf("no lora name specified")
	}

	// Create the output directory based on lora name
	outputDir := filepath.Join(targetDir, loraName)
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("Error creating output directory: %v", err))
		q.ErrorJob(j.ID)
		return err
	}
	q.PushJobStdout(j.ID, fmt.Sprintf("Created output directory: %s", outputDir))

	// Get files to process from input or query
	var filesToProcess []string
	if qstr, ok := extractQueryFromJob(j); ok {
		q.PushJobStdout(j.ID, fmt.Sprintf("Using query to select files: %s", qstr))
		mediaPaths, err := getMediaPathsByQuery(q.Db, qstr)
		if err != nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("Error loading media paths for query: %v", err))
			q.ErrorJob(j.ID)
			return err
		}
		filesToProcess = mediaPaths
		q.PushJobStdout(j.ID, fmt.Sprintf("Query matched %d items", len(filesToProcess)))
	} else if strings.TrimSpace(j.Input) != "" {
		raw := strings.TrimSpace(j.Input)
		inputPaths := parseInputPaths(raw)
		q.PushJobStdout(j.ID, fmt.Sprintf("Processing %d files from input list", len(inputPaths)))
		for _, p := range inputPaths {
			absPath, err := filepath.Abs(p)
			if err == nil {
				p = filepath.FromSlash(absPath)
			}
			if _, err := os.Stat(p); os.IsNotExist(err) {
				q.PushJobStdout(j.ID, fmt.Sprintf("Warning: file does not exist: %s", p))
				continue
			}
			if !isImageFile(p) {
				q.PushJobStdout(j.ID, fmt.Sprintf("Warning: not a supported image file: %s", p))
				continue
			}
			filesToProcess = append(filesToProcess, p)
		}
	}

	// Filter to only image files
	var imageFiles []string
	for _, p := range filesToProcess {
		if isImageFile(p) {
			imageFiles = append(imageFiles, p)
		}
	}
	filesToProcess = imageFiles

	if len(filesToProcess) == 0 {
		q.PushJobStdout(j.ID, "No valid image files found to process")
		q.CompleteJob(j.ID)
		return nil
	}

	q.PushJobStdout(j.ID, fmt.Sprintf("Processing %d image files for LoRA dataset", len(filesToProcess)))

	processedCount := 0
	for i, srcPath := range filesToProcess {
		select {
		case <-ctx.Done():
			q.PushJobStdout(j.ID, "Task was canceled")
			_ = q.CancelJob(j.ID)
			return ctx.Err()
		default:
		}

		// Generate lowercase UUID for the filename
		newUUID := strings.ToLower(uuid.New().String())
		jpgName := newUUID + ".jpg"
		txtName := newUUID + ".txt"

		jpgPath := filepath.Join(outputDir, jpgName)
		txtPath := filepath.Join(outputDir, txtName)

		// Convert image to JPG using ffmpeg
		if err := convertToJPG(ctx, srcPath, jpgPath); err != nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("Warning: failed to convert %s to JPG: %v", filepath.Base(srcPath), err))
			continue
		}

		// Get or generate description
		description, err := getOrGenerateDescription(ctx, q.Db, srcPath, ollamaModel)
		if err != nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("Warning: failed to get/generate description for %s: %v", filepath.Base(srcPath), err))
			// Still create the file but with an empty description or placeholder
			description = ""
		}

		// Prepend concept prefix if provided
		finalDescription := description
		if conceptPrefix != "" {
			if description != "" {
				finalDescription = conceptPrefix + " " + description
			} else {
				finalDescription = conceptPrefix
			}
		}

		// Write the text file with description
		if err := os.WriteFile(txtPath, []byte(finalDescription), 0644); err != nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("Warning: failed to write text file for %s: %v", filepath.Base(srcPath), err))
			continue
		}

		processedCount++
		q.PushJobStdout(j.ID, fmt.Sprintf("Progress %d/%d: %s -> %s", i+1, len(filesToProcess), filepath.Base(srcPath), jpgName))
	}

	q.PushJobStdout(j.ID, fmt.Sprintf("LoRA dataset creation completed: %d files processed, saved to %s", processedCount, outputDir))
	q.CompleteJob(j.ID)
	return nil
}

// isImageFile checks if a file is a supported image file based on its extension
func isImageFile(path string) bool {
	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".heic", ".tif", ".tiff":
		return true
	}
	return false
}

// convertToJPG converts an image file to JPG format using ffmpeg
func convertToJPG(ctx context.Context, srcPath, dstPath string) error {
	// Use ffmpeg to convert to JPG with quality 2 (high quality)
	cmd, cleanup, err := embedexec.GetExec(ctx, "ffmpeg", "-i", srcPath, "-q:v", "2", "-y", dstPath)
	if err != nil {
		// Try using system ffmpeg if embedded is not available
		cmd = exec.CommandContext(ctx, "ffmpeg", "-i", srcPath, "-q:v", "2", "-y", dstPath)
	} else if cleanup != nil {
		defer cleanup()
	}

	if err := cmd.Run(); err != nil {
		return fmt.Errorf("ffmpeg conversion failed: %w", err)
	}

	return nil
}

// getOrGenerateDescription retrieves the description from the database,
// or generates one using Ollama if it doesn't exist
func getOrGenerateDescription(ctx context.Context, db *sql.DB, filePath string, model string) (string, error) {
	// First, check if description exists in database
	var description sql.NullString
	err := db.QueryRow(`SELECT description FROM media WHERE path = ?`, filePath).Scan(&description)
	if err != nil && err != sql.ErrNoRows {
		return "", fmt.Errorf("database query failed: %w", err)
	}

	// If description exists and is not empty, return it
	if description.Valid && strings.TrimSpace(description.String) != "" {
		return description.String, nil
	}

	// Generate description using Ollama
	generatedDesc, err := describeFileWithOllama(ctx, filePath, model)
	if err != nil {
		return "", fmt.Errorf("failed to generate description: %w", err)
	}

	// Update the database with the generated description
	if err := updateMediaMetadata(db, filePath, "description", generatedDesc); err != nil {
		// Log but don't fail - we still have the description
		// The description will be written to the text file regardless
	}

	return generatedDesc, nil
}
