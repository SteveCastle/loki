package tasks

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
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

// callOllamaVisionForTags calls Ollama API to select appropriate tags for an image
func callOllamaVisionForTags(ctx context.Context, imagePath string, availableTags []TagInfo, model string) ([]TagInfo, error) {
	data, err := os.ReadFile(imagePath)
	if err != nil {
		return nil, fmt.Errorf("could not read image for Ollama: %w", err)
	}
	b64 := base64.StdEncoding.EncodeToString(data)
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
	reqJSON := fmt.Sprintf(`{"model":"%s","stream":false,"prompt":%s,"images":["%s"]}`,
		model, strconv.Quote(prompt), b64)
	base := strings.TrimRight(appconfig.Get().OllamaBaseURL, "/")
	req, err := http.NewRequestWithContext(ctx, "POST", base+"/api/generate", strings.NewReader(reqJSON))
	if err != nil {
		return nil, fmt.Errorf("failed to build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("ollama request failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ollama error: status=%d, body=%s", resp.StatusCode, string(body))
	}
	var response struct {
		Response string `json:"response"`
	}
	respData, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading response body failed: %w", err)
	}
	if err := json.Unmarshal(respData, &response); err != nil {
		return nil, fmt.Errorf("could not unmarshal Ollama response: %w", err)
	}
	log.Printf("AutoTag Vision Raw Response for %s:\n%s", imagePath, response.Response)
	selectedTags, err := parseTagsFromResponse(response.Response, availableTags)
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
