package tasks

import (
	"context"
	"database/sql"
	"encoding/base64"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"

	"github.com/stevecastle/shrike/deps"
	"github.com/stevecastle/shrike/embedvec"
	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/media"
	"github.com/stevecastle/shrike/platform"
)

// EmbedModelID is the active embedding model; vectors are stored keyed by it.
const EmbedModelID = "siglip-base-patch16-224"

// EmbedDim is the SigLIP base embedding dimension. Confirm against the chosen
// ONNX export (Task 7) and update if different.
const EmbedDim = 768

func shouldSkipEmbed(db *sql.DB, path, model string) bool {
	ok, err := media.HasEmbedding(db, path, model)
	return err == nil && ok
}

// runEmbedSubprocess invokes embed.exe for one image and returns the decoded,
// already-L2-normalized vector.
func runEmbedSubprocess(ctx context.Context, embedBin, model, ortLib, imagePath string, dim int) ([]float32, error) {
	args := []string{
		"--model=" + model,
		"--image=" + imagePath,
		fmt.Sprintf("--dim=%d", dim),
	}
	if ortLib != "" {
		args = append(args, "--ort="+ortLib)
	}
	cmd := exec.CommandContext(ctx, embedBin, args...)
	platform.HideSubprocessWindow(cmd)
	out, err := cmd.Output()
	if err != nil {
		return nil, err
	}
	line := strings.TrimSpace(string(out))
	raw, err := base64.StdEncoding.DecodeString(line)
	if err != nil {
		return nil, fmt.Errorf("decode base64 vector: %w", err)
	}
	return embedvec.Decode(raw)
}

func embedTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	ctx := j.Ctx

	var paths []string
	fromQuery := false
	if qstr, ok := extractQueryFromJob(j); ok {
		q.PushJobStdout(j.ID, fmt.Sprintf("Using query: %s", qstr))
		mediaPaths, err := getMediaPathsByQueryFast(q.Db, qstr)
		if err != nil {
			q.PushJobStdout(j.ID, "Failed to load paths from query: "+err.Error())
			q.ErrorJob(j.ID)
			return err
		}
		paths = mediaPaths
		fromQuery = true
		q.PushJobStdout(j.ID, fmt.Sprintf("Query matched %d items", len(paths)))
	} else {
		raw := strings.TrimSpace(j.Input)
		if raw == "" {
			q.PushJobStdout(j.ID, "No image path provided")
			q.CompleteJob(j.ID)
			return nil
		}
		paths = parseInputPaths(raw)
		q.PushJobStdout(j.ID, fmt.Sprintf("Processing %d files from input", len(paths)))
	}
	if len(paths) == 0 {
		q.PushJobStdout(j.ID, "No files to process")
		q.CompleteJob(j.ID)
		return nil
	}

	// Resolve model + runtime + binary (deps first, like autotag).
	imageModel, _ := deps.ModelPath(EmbedModelID, "image_model.onnx")
	if imageModel == "" {
		q.PushJobStdout(j.ID, "SigLIP model not installed; install it from Dependencies")
		q.ErrorJob(j.ID)
		return fmt.Errorf("model %s not installed", EmbedModelID)
	}
	ortLib := deps.BundledOrEmpty("onnxruntime")
	embedBin := deps.BundledOrEmpty("embed")
	if embedBin == "" {
		q.PushJobStdout(j.ID, "embed binary not installed; install it from Dependencies")
		q.ErrorJob(j.ID)
		return fmt.Errorf("embed binary not installed")
	}

	processed, skipped := 0, 0
	for idx, mediaPath := range paths {
		select {
		case <-ctx.Done():
			q.PushJobStdout(j.ID, "Task canceled")
			_ = q.CancelJob(j.ID)
			return ctx.Err()
		default:
		}

		if shouldSkipEmbed(q.Db, mediaPath, EmbedModelID) {
			skipped++
			continue
		}
		if !fromQuery {
			if _, err := os.Stat(mediaPath); os.IsNotExist(err) {
				q.PushJobStdout(j.ID, fmt.Sprintf("Skipping (not found): %s", filepath.Base(mediaPath)))
				skipped++
				continue
			}
		}

		imagePath := mediaPath
		var tempFramePath string
		switch strings.ToLower(filepath.Ext(mediaPath)) {
		case ".mp4", ".mov", ".avi", ".mkv", ".webm", ".wmv", ".gif":
			framePath, err := extractVideoFrame(ctx, mediaPath, "")
			if err != nil {
				q.PushJobStdout(j.ID, fmt.Sprintf("  Failed to extract frame: %v", err))
				skipped++
				continue
			}
			tempFramePath = framePath
			imagePath = framePath
		}

		q.PushJobStdout(j.ID, fmt.Sprintf("[%d/%d] Embedding: %s", idx+1, len(paths), filepath.Base(mediaPath)))
		vec, err := runEmbedSubprocess(ctx, embedBin, imageModel, ortLib, imagePath, EmbedDim)
		if tempFramePath != "" {
			_ = os.Remove(tempFramePath)
		}
		if err != nil {
			q.PushJobStdout(j.ID, "  embed failed: "+err.Error())
			skipped++
			continue
		}
		if err := media.UpsertEmbedding(q.Db, mediaPath, EmbedModelID, vec, 0); err != nil {
			q.PushJobStdout(j.ID, "  Failed to store embedding: "+err.Error())
			q.ErrorJob(j.ID)
			return err
		}
		processed++
		q.RegisterOutputFile(j.ID, mediaPath)
	}

	q.PushJobStdout(j.ID, fmt.Sprintf("Completed: %d embedded, %d skipped", processed, skipped))
	q.CompleteJob(j.ID)
	return nil
}
