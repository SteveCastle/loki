package tasks

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/stevecastle/shrike/appconfig"
	"github.com/stevecastle/shrike/jobqueue"
)

// generateAutoTags generates automatic tags for media files using vision model
func generateAutoTags(ctx context.Context, q *jobqueue.Queue, jobID string, filePaths []string, overwrite bool, model string) error {
	availableTags, err := getAllAvailableTags(q.Db)
	if err != nil {
		return fmt.Errorf("failed to fetch available tags: %w", err)
	}
	if len(availableTags) == 0 {
		q.PushJobStdout(jobID, "No tags available in database for auto-tagging")
		return nil
	}
	q.PushJobStdout(jobID, fmt.Sprintf("Found %d available tags in database", len(availableTags)))

	// Pre-filter to compute exact candidates and total for progress
	var candidates []string
	for _, filePath := range filePaths {
		if _, err := os.Stat(filePath); os.IsNotExist(err) {
			q.PushJobStdout(jobID, fmt.Sprintf("Warning: file does not exist: %s", filePath))
			continue
		}
		ext := strings.ToLower(filepath.Ext(filePath))
		switch ext {
		case ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff", ".heic":
		default:
			continue
		}
		if !overwrite {
			existingTags, err := getExistingTagsForFile(q.Db, filePath)
			if err != nil {
				q.PushJobStdout(jobID, fmt.Sprintf("Warning: failed to check existing tags for %s: %v", filePath, err))
				continue
			}
			if len(existingTags) > 0 {
				continue
			}
		}
		candidates = append(candidates, filePath)
	}
	if len(candidates) == 0 {
		q.PushJobStdout(jobID, "Auto-tag: 0 image files to process")
		return nil
	}
	q.PushJobStdout(jobID, fmt.Sprintf("Auto-tag: %d image files to process", len(candidates)))

	processed := 0
	for i, filePath := range candidates {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		selectedTags, err := generateAutoTagsWithVision(ctx, filePath, availableTags, model)
		if err != nil {
			q.PushJobStdout(jobID, fmt.Sprintf("Warning: failed to auto-tag %s: %v", filePath, err))
			continue
		}
		if len(selectedTags) == 0 {
			q.PushJobStdout(jobID, fmt.Sprintf("No tags selected for: %s", filePath))
			continue
		}
		if overwrite {
			if err := removeExistingTagsForFile(q.Db, filePath); err != nil {
				q.PushJobStdout(jobID, fmt.Sprintf("Warning: failed to remove existing tags for %s: %v", filePath, err))
				continue
			}
		}
		if err := insertTagsForFile(q.Db, filePath, selectedTags); err != nil {
			q.PushJobStdout(jobID, fmt.Sprintf("Warning: failed to insert tags for %s: %v", filePath, err))
			continue
		}
		processed++
		tagLabels := make([]string, len(selectedTags))
		for i, tag := range selectedTags {
			tagLabels[i] = tag.Label
		}
		q.PushJobStdout(jobID, fmt.Sprintf("Auto-tagged %s with: %s", filepath.Base(filePath), strings.Join(tagLabels, ", ")))
		q.PushJobStdout(jobID, fmt.Sprintf("Auto-tag %d/%d: %s", i+1, len(candidates), filepath.Base(filePath)))
	}
	q.PushJobStdout(jobID, fmt.Sprintf("Generated auto-tags for %d image files", processed))
	return nil
}

// generateAutoTagsWithVision uses the vision model to select appropriate tags from available options
func generateAutoTagsWithVision(ctx context.Context, mediaPath string, availableTags []TagInfo, model string) ([]TagInfo, error) {
	ext := strings.ToLower(filepath.Ext(mediaPath))
	var tempImagePath string
	var cleanupPaths []string
	if ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".webp" {
		tempImagePath = mediaPath
	} else {
		screenshotPath, err := extractVideoFrame(ctx, mediaPath, "")
		if err != nil {
			return nil, fmt.Errorf("failed to extract video frame: %w", err)
		}
		cleanupPaths = append(cleanupPaths, screenshotPath)
		tempImagePath = screenshotPath
	}
	resizedPath, err := resizeImageIfNeeded(tempImagePath)
	if err != nil {
		for _, p := range cleanupPaths {
			_ = os.Remove(p)
		}
		return nil, fmt.Errorf("failed to resize image: %w", err)
	}
	if resizedPath != tempImagePath {
		cleanupPaths = append(cleanupPaths, resizedPath)
	}
	selectedTags, err := callOllamaVisionForTags(ctx, resizedPath, availableTags, model)
	if err != nil {
		for _, p := range cleanupPaths {
			_ = os.Remove(p)
		}
		return nil, fmt.Errorf("ollama auto-tag call failed: %w", err)
	}
	for _, p := range cleanupPaths {
		_ = os.Remove(p)
	}
	return selectedTags, nil
}

// callOllamaVisionForTags asks the vision LLM (RunPod when configured, else
// Ollama) to pick appropriate tags for an image from a constrained list.
// The unused model arg is preserved for callsite signature parity; the
// underlying client picks its own model from config.
func callOllamaVisionForTags(ctx context.Context, imagePath string, availableTags []TagInfo, _ string) ([]TagInfo, error) {
	var tagOptions strings.Builder
	tagOptions.WriteString("Available tags by category:\n")
	categoryMap := make(map[string][]string)
	for _, tag := range availableTags {
		categoryMap[tag.Category] = append(categoryMap[tag.Category], tag.Label)
	}
	for category, labels := range categoryMap {
		tagOptions.WriteString(fmt.Sprintf("- %s: %s\n", category, strings.Join(labels, ", ")))
	}
	prompt := fmt.Sprintf(appconfig.Get().AutotagPrompt, tagOptions.String())
	log.Printf("AutoTag Vision Prompt for %s:\n%s", imagePath, prompt)

	// 60s timeout matched the prior http.Client literal; preserved here as
	// a context deadline so the RunPod async path can also honor it.
	timeoutCtx, cancel := context.WithTimeout(ctx, 60*time.Second)
	defer cancel()
	response, err := callVisionLLM(timeoutCtx, imagePath, prompt)
	if err != nil {
		return nil, err
	}
	log.Printf("AutoTag Vision Raw Response for %s:\n%s", imagePath, response)
	selectedTags, err := parseTagsFromResponse(response, availableTags)
	if err != nil {
		return nil, fmt.Errorf("failed to parse tags from response: %w", err)
	}
	return selectedTags, nil
}

// parseTagsFromResponse extracts valid tags from the Ollama response
func parseTagsFromResponse(response string, availableTags []TagInfo) ([]TagInfo, error) {
	response = strings.TrimSpace(response)
	start := strings.Index(response, "[")
	end := strings.LastIndex(response, "]")
	if start == -1 || end == -1 || start >= end {
		log.Printf("AutoTag Parse: No valid JSON array found in response (start=%d, end=%d)", start, end)
		return []TagInfo{}, nil
	}
	jsonStr := response[start : end+1]
	log.Printf("AutoTag Parse: Extracted JSON string: %s", jsonStr)
	var rawTags []map[string]interface{}
	if err := json.Unmarshal([]byte(jsonStr), &rawTags); err != nil {
		log.Printf("AutoTag Parse: JSON unmarshal failed: %v", err)
		return []TagInfo{}, nil
	}
	log.Printf("AutoTag Parse: Successfully parsed %d raw tags from JSON", len(rawTags))
	lookup := make(map[string]TagInfo)
	for _, t := range availableTags {
		lookup[strings.ToLower(t.Category)+":"+strings.ToLower(t.Label)] = t
	}
	var selected []TagInfo
	for i, raw := range rawTags {
		labelInterface, okL := raw["label"]
		categoryInterface, okC := raw["category"]
		if !okL || !okC {
			log.Printf("AutoTag Parse: Raw tag %d missing label or category fields", i)
			continue
		}
		label, ok1 := labelInterface.(string)
		category, ok2 := categoryInterface.(string)
		if !ok1 || !ok2 {
			log.Printf("AutoTag Parse: Raw tag %d has non-string label or category", i)
			continue
		}
		key := strings.ToLower(category) + ":" + strings.ToLower(label)
		if vt, exists := lookup[key]; exists {
			selected = append(selected, vt)
			log.Printf("AutoTag Parse: Validated tag %d: %s/%s", i, category, label)
		} else {
			log.Printf("AutoTag Parse: Tag %d not found in available tags: %s/%s (key: %s)", i, category, label, key)
		}
	}
	return selected, nil
}

// processAutotagForFile generates auto-tags for a single image file
func processAutotagForFile(ctx context.Context, q *jobqueue.Queue, jobID string, filePath string, overwrite bool, model string, availableTags []TagInfo, fromQuery bool) error {
	// Check if it's an image file
	ext := strings.ToLower(filepath.Ext(filePath))
	switch ext {
	case ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff", ".heic":
		// Valid image file
	default:
		return nil // Not an image file, skip silently
	}

	if len(availableTags) == 0 {
		return nil // No tags available, skip silently
	}

	// If not from query, check if file exists in database first
	if !fromQuery {
		exists, err := fileExistsInDatabase(q.Db, filePath)
		if err != nil {
			return fmt.Errorf("error checking database: %w", err)
		}
		if !exists {
			return nil // File not in database, skip
		}
	}

	if !overwrite {
		existingTags, err := getExistingTagsForFile(q.Db, filePath)
		if err != nil {
			return fmt.Errorf("failed to check existing tags: %w", err)
		}
		if len(existingTags) > 0 {
			return nil // Skip, already has tags
		}
	}

	selectedTags, err := generateAutoTagsWithVision(ctx, filePath, availableTags, model)
	if err != nil {
		return fmt.Errorf("failed to auto-tag: %w", err)
	}
	if len(selectedTags) == 0 {
		q.PushJobStdout(jobID, fmt.Sprintf("  autotag: no tags selected"))
		return nil
	}

	if overwrite {
		if err := removeExistingTagsForFile(q.Db, filePath); err != nil {
			return fmt.Errorf("failed to remove existing tags: %w", err)
		}
	}
	if err := insertTagsForFile(q.Db, filePath, selectedTags); err != nil {
		return fmt.Errorf("failed to insert tags: %w", err)
	}

	tagLabels := make([]string, len(selectedTags))
	for i, tag := range selectedTags {
		tagLabels[i] = tag.Label
	}
	q.PushJobStdout(jobID, fmt.Sprintf("  autotag: %s", strings.Join(tagLabels, ", ")))
	return nil
}
