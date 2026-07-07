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
)

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

	// 5 minutes covers a RunPod cold start (observed up to ~2 min) plus
	// inference time with comfortable headroom. Generous for the local
	// Ollama path too — only matters as a ceiling on hung requests.
	timeoutCtx, cancel := context.WithTimeout(ctx, 5*time.Minute)
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

