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
	"github.com/stevecastle/shrike/deps"
	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/platform"
)

var loraDatasetOptions = []TaskOption{
	{Name: "target", Label: "Target Directory", Type: "string", Required: true, Description: "Base directory for the LoRA dataset"},
	{Name: "name", Label: "LoRA Name", Type: "string", Required: true, Description: "Name of the LoRA dataset (used as subfolder name)"},
	{Name: "prefix", Label: "Concept Prefix", Type: "string", Description: "Prefix prepended to each description caption"},
	{Name: "model", Label: "Ollama Model", Type: "string", Description: "Ollama model to use for generating descriptions"},
}

// loraDatasetTask creates a LoRA training dataset from selected media files.
// It copies images to a new folder, converts them to JPG with lowercase UUID names,
// and creates accompanying text files with descriptions.
func loraDatasetTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	ctx := j.Ctx

	opts := ParseOptions(j, loraDatasetOptions)
	targetDir, _ := opts["target"].(string)
	loraName, _ := opts["name"].(string)
	conceptPrefix, _ := opts["prefix"].(string)
	ollamaModel, _ := opts["model"].(string)
	if ollamaModel == "" {
		ollamaModel = appconfig.Get().OllamaModel
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
	res, rerr := resolveJobItems(j, q)
	if rerr != nil {
		q.PushJobStdout(j.ID, "Failed to resolve input: "+rerr.Error())
		q.ErrorJob(j.ID)
		return rerr
	}
	if res.FromQuery {
		q.PushJobStdout(j.ID, fmt.Sprintf("Query: %s", res.Query))
	}

	// Filter to only image files
	var filesToProcess []string
	for _, p := range res.Paths {
		if isImageFile(p) {
			filesToProcess = append(filesToProcess, p)
		}
	}

	if len(filesToProcess) == 0 {
		q.PushJobStdout(j.ID, "No valid image files found to process")
		q.CompleteJob(j.ID)
		return nil
	}

	q.PushJobStdout(j.ID, fmt.Sprintf("Processing %d image files for LoRA dataset", len(filesToProcess)))
	_ = q.SetJobProgress(j.ID, 0, len(filesToProcess))

	processedCount := 0
	for i, srcPath := range filesToProcess {
		select {
		case <-ctx.Done():
			q.PushJobStdout(j.ID, "Task was canceled")
			_ = q.CancelJob(j.ID)
			return ctx.Err()
		default:
		}
		if q.PauseRequested(j.ID) {
			q.PushJobStdout(j.ID, fmt.Sprintf("Paused at %d/%d - resume to continue", i, len(filesToProcess)))
			return jobqueue.ErrPaused
		}
		_ = q.SetJobProgress(j.ID, i, len(filesToProcess))

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
		q.RegisterOutputFile(j.ID, jpgPath)
	}

	_ = q.SetJobProgress(j.ID, len(filesToProcess), len(filesToProcess))
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
	cmd := exec.CommandContext(ctx, deps.MustBundled("ffmpeg"), "-i", srcPath, "-q:v", "2", "-y", dstPath)
	platform.HideSubprocessWindow(cmd)

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
	generatedDesc, err := describeFileWithOllama(ctx, nil, "", filePath, model, "")
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
