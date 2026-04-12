package tasks

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/stevecastle/shrike/jobqueue"
)

var saveOptions = []TaskOption{
	{Name: "destination", Label: "Destination", Type: "enum", Choices: []string{"original", "directory"}, Default: "original", Description: "Where to save: original directory or a specific directory"},
	{Name: "directory", Label: "Target Directory", Type: "string", Description: "Target directory (only used when destination=directory)"},
	{Name: "conflict", Label: "Conflict Resolution", Type: "enum", Choices: []string{"suffix", "overwrite", "skip"}, Default: "suffix", Description: "How to handle existing files"},
	{Name: "suffix", Label: "Custom Suffix", Type: "string", Description: "Custom suffix for output filename (e.g. _edited). Empty keeps processing name"},
	{Name: "flatten", Label: "Flatten", Type: "bool", Description: "Flatten all files into target directory (ignore relative paths)"},
}

func saveTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	opts := ParseOptions(j, saveOptions)
	destination, _ := opts["destination"].(string)
	if destination == "" {
		destination = "original"
	}
	directory, _ := opts["directory"].(string)
	conflict, _ := opts["conflict"].(string)
	if conflict == "" {
		conflict = "suffix"
	}
	suffix, _ := opts["suffix"].(string)
	flatten, _ := opts["flatten"].(bool)
	_ = flatten // reserved for future use

	if destination == "directory" && directory == "" {
		q.PushJobStdout(j.ID, "save: no target directory specified")
		q.ErrorJob(j.ID)
		return fmt.Errorf("target directory required when destination=directory")
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

	saved := 0
	skipped := 0
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

		ext := filepath.Ext(src)
		name := strings.TrimSuffix(filepath.Base(src), ext)

		outName := buildSaveFilename(name, suffix, ext)

		var outDir string
		switch destination {
		case "directory":
			outDir = directory
		default: // "original"
			outDir = stripLokiTemp(src)
		}

		if err := os.MkdirAll(outDir, 0755); err != nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("save: failed to create directory %s: %v", outDir, err))
			continue
		}

		destPath := filepath.Join(outDir, outName)

		if _, err := os.Stat(destPath); err == nil {
			destPath = resolveConflict(destPath, conflict)
			if destPath == "" {
				q.PushJobStdout(j.ID, fmt.Sprintf("save: skipping (exists): %s", filepath.Join(outDir, outName)))
				skipped++
				continue
			}
		}

		if err := moveFile(src, destPath); err != nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("save: failed to save %s -> %s: %v", filepath.Base(src), destPath, err))
			continue
		}

		q.PushJobStdout(j.ID, fmt.Sprintf("save: %s -> %s", filepath.Base(src), destPath))
		q.RegisterOutputFile(j.ID, destPath)
		saved++
	}

	// Clean up .loki-temp directories for this workflow
	if j.WorkflowID != "" {
		cleanupWorkflowTempDirs(q, j.WorkflowID)
	}

	q.PushJobStdout(j.ID, fmt.Sprintf("save: completed — %d saved, %d skipped", saved, skipped))
	q.CompleteJob(j.ID)
	return nil
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

// buildSaveFilename constructs the output filename.
// If suffix is non-empty, it appends the custom suffix to the name.
// If suffix is empty, the name is used as-is.
func buildSaveFilename(name, suffix, ext string) string {
	if suffix == "" {
		return name + ext
	}
	return name + suffix + ext
}

// resolveConflict determines the final path when a file already exists.
// Returns empty string for "skip" mode.
func resolveConflict(path, mode string) string {
	switch mode {
	case "overwrite":
		return path
	case "skip":
		return ""
	default: // "suffix"
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
}

// moveFile tries os.Rename first (fast, same-drive). Falls back to copy+delete
// for cross-drive moves.
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

	// Try to remove the .loki-temp parent dirs if empty
	for dir := range seen {
		parent := filepath.Dir(dir)
		os.Remove(parent) // Only succeeds if empty
	}
}
