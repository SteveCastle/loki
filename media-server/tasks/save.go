package tasks

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/stream"
)

var saveOptions = []TaskOption{
	{Name: "mode", Label: "Save Mode", Type: "enum", Choices: []string{"replace", "alongside", "folder"}, Default: "alongside", Description: "replace: overwrite original file, alongside: save next to original with suffix, folder: save to a specific directory"},
	{Name: "suffix", Label: "Filename Suffix", Type: "string", Default: "_output", Description: "Suffix added before extension (alongside mode). e.g. _edited produces video_edited.mp4"},
	{Name: "folder", Label: "Output Folder", Type: "string", Description: "Destination folder (folder mode only)"},
}

func saveTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	opts := ParseOptions(j, saveOptions)
	mode, _ := opts["mode"].(string)
	if mode == "" {
		mode = "alongside"
	}
	suffix, _ := opts["suffix"].(string)
	if suffix == "" && mode == "alongside" {
		suffix = "_output"
	}
	folder, _ := opts["folder"].(string)

	if mode == "folder" && folder == "" {
		q.PushJobStdout(j.ID, "save: no output folder specified")
		q.ErrorJob(j.ID)
		return fmt.Errorf("output folder required for folder mode")
	}

	raw := strings.TrimSpace(j.Input)
	if raw == "" {
		q.PushJobStdout(j.ID, "save: no input files")
		q.CompleteJob(j.ID)
		return nil
	}
	files := parseInputPaths(raw)

	if len(files) == 0 {
		q.PushJobStdout(j.ID, "save: no files to save")
		q.CompleteJob(j.ID)
		return nil
	}

	// Build a map from output path -> original source path so we can
	// recover the original filename for "replace" mode.
	sourceMap := q.GetSourceMap(j.ID)

	saved := 0
	skipped := 0
	var savedPaths []string
	for _, src := range files {
		select {
		case <-j.Ctx.Done():
			q.PushJobStdout(j.ID, "save: task canceled")
			q.ErrorJob(j.ID)
			return j.Ctx.Err()
		default:
		}

		if _, err := os.Stat(src); os.IsNotExist(err) {
			q.PushJobStdout(j.ID, fmt.Sprintf("save: file not found, skipping: %s", src))
			skipped++
			continue
		}

		// Recover the original source path. The sourceMap tracks which
		// original file produced each output through the chain.
		originalPath := sourceMap[src]
		if originalPath == "" {
			// No source tracking — derive original dir from temp path
			originalPath = src
		}
		originalDir := stripLokiTemp(originalPath)
		originalExt := filepath.Ext(originalPath)
		originalName := strings.TrimSuffix(filepath.Base(originalPath), originalExt)

		// The processed file may have a different extension (e.g. converted .webm -> .mp4)
		processedExt := filepath.Ext(src)

		var destPath string
		switch mode {
		case "replace":
			// Overwrite the original file with the processed version.
			destPath = filepath.Join(originalDir, originalName+processedExt)

		case "folder":
			// Save into the specified folder, using the original name + suffix.
			if err := os.MkdirAll(folder, 0755); err != nil {
				q.PushJobStdout(j.ID, fmt.Sprintf("save: failed to create folder %s: %v", folder, err))
				continue
			}
			destPath = filepath.Join(folder, originalName+suffix+processedExt)

		default: // "alongside"
			// Save next to the original with a suffix.
			destPath = filepath.Join(originalDir, originalName+suffix+processedExt)
		}

		if err := os.MkdirAll(filepath.Dir(destPath), 0755); err != nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("save: failed to create directory: %v", err))
			continue
		}

		// For alongside and folder modes, avoid clobbering existing files.
		if mode != "replace" {
			if _, err := os.Stat(destPath); err == nil {
				destPath = resolveConflict(destPath)
			}
		}

		if err := moveFile(src, destPath); err != nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("save: failed %s -> %s: %v", filepath.Base(src), destPath, err))
			continue
		}

		q.PushJobStdout(j.ID, fmt.Sprintf("save: %s -> %s", filepath.Base(src), destPath))
		q.RegisterOutputFile(j.ID, destPath)
		savedPaths = append(savedPaths, destPath)
		saved++
	}

	if j.WorkflowID != "" {
		cleanupWorkflowTempDirs(q, j.WorkflowID)
	}

	// Notify clients about saved files.
	if mode == "replace" {
		broadcastMediaUpdated(savedPaths)
	} else {
		broadcastMediaCreated(savedPaths)
	}

	q.PushJobStdout(j.ID, fmt.Sprintf("save: done — %d saved, %d skipped", saved, skipped))
	q.CompleteJob(j.ID)
	return nil
}

// broadcastMediaUpdated sends an SSE event so clients can bust their
// cache for the given file paths (e.g. after overwriting originals).
func broadcastMediaUpdated(paths []string) {
	if len(paths) == 0 {
		return
	}
	payload, _ := json.Marshal(map[string]any{"paths": paths})
	stream.Broadcast(stream.Message{Type: "media-updated", Msg: string(payload)})
}

// broadcastMediaCreated sends an SSE event so clients can add newly
// created files to the library (e.g. after saving alongside/folder).
func broadcastMediaCreated(paths []string) {
	if len(paths) == 0 {
		return
	}
	payload, _ := json.Marshal(map[string]any{"paths": paths})
	stream.Broadcast(stream.Message{Type: "media-created", Msg: string(payload)})
}

// stripLokiTemp removes the .loki-temp/<jobID>/ segment from a path,
// returning the original source directory.
func stripLokiTemp(path string) string {
	dir := filepath.Dir(path)
	parts := strings.Split(filepath.ToSlash(dir), "/")
	for i, part := range parts {
		if part == ".loki-temp" && i+1 < len(parts) {
			cleaned := strings.Join(append(parts[:i], parts[i+2:]...), "/")
			return filepath.FromSlash(cleaned)
		}
	}
	return dir
}

// resolveConflict appends _1, _2, etc. to avoid overwriting an existing file.
func resolveConflict(path string) string {
	dir := filepath.Dir(path)
	ext := filepath.Ext(path)
	name := strings.TrimSuffix(filepath.Base(path), ext)
	for i := 1; i < 1000; i++ {
		candidate := filepath.Join(dir, fmt.Sprintf("%s_%d%s", name, i, ext))
		if _, err := os.Stat(candidate); os.IsNotExist(err) {
			return candidate
		}
	}
	return path
}

// moveFile tries os.Rename first (fast, same-drive). Falls back to copy+delete.
func moveFile(src, dst string) error {
	if err := os.Rename(src, dst); err == nil {
		return nil
	}
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()

	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer out.Close()

	if _, err := io.Copy(out, in); err != nil {
		os.Remove(dst)
		return err
	}
	if err := out.Close(); err != nil {
		os.Remove(dst)
		return err
	}
	in.Close()
	return os.Remove(src)
}

// cleanupWorkflowTempDirs removes .loki-temp directories for all jobs
// in the given workflow.
func cleanupWorkflowTempDirs(q *jobqueue.Queue, workflowID string) {
	paths := q.GetWorkflowOutputFiles(workflowID)

	seen := make(map[string]bool)
	for _, f := range paths {
		dir := filepath.Dir(f)
		parts := strings.Split(filepath.ToSlash(dir), "/")
		for i, part := range parts {
			if part == ".loki-temp" && i+1 < len(parts) {
				tempRoot := filepath.FromSlash(strings.Join(parts[:i+2], "/"))
				if !seen[tempRoot] {
					seen[tempRoot] = true
					os.RemoveAll(tempRoot)
				}
				break
			}
		}
	}

	for dir := range seen {
		parent := filepath.Dir(dir)
		os.Remove(parent) // Only succeeds if empty
	}
}
