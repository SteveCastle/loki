package tasks

import (
	"database/sql"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/stevecastle/shrike/jobqueue"
)

func moveTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	ctx := j.Ctx

	var targetDir string
	var specifiedPrefix string
	for i, arg := range j.Arguments {
		if arg == "--prefix" || arg == "-p" {
			if i+1 < len(j.Arguments) {
				specifiedPrefix = strings.TrimSpace(j.Arguments[i+1])
			}
		} else if !strings.HasPrefix(arg, "-") && targetDir == "" {
			targetDir = strings.TrimSpace(arg)
		}
	}
	if targetDir == "" {
		q.PushJobStdout(j.ID, "Error: No target directory specified in arguments")
		q.ErrorJob(j.ID)
		return fmt.Errorf("no target directory specified")
	}

	var cleanedPaths []string
	if qstr, ok := extractQueryFromJob(j); ok {
		q.PushJobStdout(j.ID, fmt.Sprintf("Using query to select files: %s", qstr))
		mediaPaths, err := getMediaPathsByQuery(q.Db, qstr)
		if err != nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("Error loading media paths for query: %v", err))
			q.ErrorJob(j.ID)
			return err
		}
		cleanedPaths = mediaPaths
	} else {
		pathsStr := strings.TrimSpace(j.Input)
		if pathsStr == "" {
			q.PushJobStdout(j.ID, "No paths provided for moving")
			q.CompleteJob(j.ID)
			return nil
		}
		rawPaths := strings.Split(pathsStr, "\n")
		for _, rawPath := range rawPaths {
			cleanPath := strings.TrimSpace(rawPath)
			if cleanPath == "" {
				continue
			}
			cleanPath = strings.Trim(cleanPath, `"'`)
			if cleanPath != "" {
				cleanedPaths = append(cleanedPaths, cleanPath)
			}
		}
	}
	if len(cleanedPaths) == 0 {
		q.PushJobStdout(j.ID, "No valid paths found after parsing input")
		q.CompleteJob(j.ID)
		return nil
	}

	q.PushJobStdout(j.ID, fmt.Sprintf("Starting move operation to target directory: %s", targetDir))
	q.PushJobStdout(j.ID, fmt.Sprintf("Parsed %d paths from input", len(cleanedPaths)))

	var validPaths []string
	for _, p := range cleanedPaths {
		abs, err := filepath.Abs(p)
		if err != nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("Warning: could not resolve absolute path for %s: %v", p, err))
			continue
		}
		cp := filepath.FromSlash(abs)
		if _, err := os.Stat(cp); os.IsNotExist(err) {
			q.PushJobStdout(j.ID, fmt.Sprintf("Warning: file does not exist: %s", cp))
			continue
		}
		validPaths = append(validPaths, cp)
	}
	if len(validPaths) == 0 {
		q.PushJobStdout(j.ID, "No valid files found to move")
		q.CompleteJob(j.ID)
		return nil
	}

	if err := os.MkdirAll(targetDir, 0755); err != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("Error creating target directory: %v", err))
		q.ErrorJob(j.ID)
		return err
	}

	var prefixToUse string
	if specifiedPrefix != "" {
		if absPrefix, err := filepath.Abs(specifiedPrefix); err == nil {
			prefixToUse = filepath.FromSlash(absPrefix)
		} else {
			prefixToUse = filepath.Clean(specifiedPrefix)
		}
		q.PushJobStdout(j.ID, fmt.Sprintf("Using specified prefix: %s", prefixToUse))
	} else {
		prefixToUse = findCommonPrefix(validPaths)
		if prefixToUse == "" {
			q.PushJobStdout(j.ID, "Warning: No common prefix found, files will be moved to target directory root")
		} else {
			q.PushJobStdout(j.ID, fmt.Sprintf("Using calculated common prefix: %s", prefixToUse))
		}
	}

	moveCount := 0
	updateCount := 0
	for i, srcPath := range validPaths {
		select {
		case <-ctx.Done():
			q.PushJobStdout(j.ID, "Task was canceled")
			_ = q.CancelJob(j.ID)
			return ctx.Err()
		default:
		}

		var relativePath string
		if prefixToUse != "" && strings.HasPrefix(srcPath, prefixToUse) {
			relativePath = strings.TrimPrefix(srcPath, prefixToUse)
			relativePath = strings.TrimPrefix(relativePath, string(filepath.Separator))
		} else {
			relativePath = filepath.Base(srcPath)
		}
		if relativePath == "" {
			relativePath = filepath.Base(srcPath)
		}
		destPath := filepath.Join(targetDir, relativePath)

		if _, err := os.Stat(destPath); err == nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("Warning: destination already exists, skipping: %s", destPath))
			continue
		}
		destDir := filepath.Dir(destPath)
		if err := os.MkdirAll(destDir, 0755); err != nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("Warning: failed to create destination directory %s: %v", destDir, err))
			continue
		}
		if err := os.Rename(srcPath, destPath); err != nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("Warning: failed to move %s to %s: %v", srcPath, destPath, err))
			continue
		}
		moveCount++
		q.PushJobStdout(j.ID, fmt.Sprintf("Moved: %s -> %s", srcPath, destPath))

		if err := updateMediaPathInDatabase(q.Db, srcPath, destPath); err != nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("Warning: failed to update database for %s: %v", srcPath, err))
		} else {
			updateCount++
		}
		if (i+1)%10 == 0 || i == len(validPaths)-1 {
			q.PushJobStdout(j.ID, fmt.Sprintf("Progress: %d/%d files processed", i+1, len(validPaths)))
		}
	}

	q.PushJobStdout(j.ID, fmt.Sprintf("Move operation completed: %d files moved, %d database entries updated", moveCount, updateCount))
	select {
	case <-ctx.Done():
		q.PushJobStdout(j.ID, "Task was canceled")
		q.ErrorJob(j.ID)
		return ctx.Err()
	default:
	}
	q.CompleteJob(j.ID)
	return nil
}

// findCommonPrefix finds the common directory prefix among a list of file paths
func findCommonPrefix(paths []string) string {
	if len(paths) == 0 {
		return ""
	}
	if len(paths) == 1 {
		return filepath.Dir(paths[0])
	}
	prefix := filepath.Dir(paths[0])
	if prefix == "." || prefix == "/" || (len(prefix) == 3 && prefix[1] == ':' && prefix[2] == '\\') {
		return ""
	}
	for _, p := range paths[1:] {
		pd := filepath.Dir(p)
		if pd == "." || pd == "/" || (len(pd) == 3 && pd[1] == ':' && pd[2] == '\\') {
			return ""
		}
		newPrefix := ""
		pp := strings.Split(filepath.Clean(prefix), string(filepath.Separator))
		pq := strings.Split(filepath.Clean(pd), string(filepath.Separator))
		minLen := len(pp)
		if len(pq) < minLen {
			minLen = len(pq)
		}
		for i := 0; i < minLen; i++ {
			if pp[i] == pq[i] {
				if newPrefix == "" {
					newPrefix = pp[i]
				} else {
					newPrefix = filepath.Join(newPrefix, pp[i])
				}
			} else {
				break
			}
		}
		prefix = newPrefix
		if prefix == "" {
			break
		}
	}
	if prefix != "" {
		prefix = filepath.Clean(prefix)
		if !strings.HasSuffix(prefix, string(filepath.Separator)) {
			prefix += string(filepath.Separator)
		}
	}
	return prefix
}

// updateMediaPathInDatabase updates the path references in both media and media_tag_by_category tables
func updateMediaPathInDatabase(db *sql.DB, oldPath, newPath string) error {
	tx, err := db.Begin()
	if err != nil {
		return fmt.Errorf("failed to start transaction: %w", err)
	}
	defer tx.Rollback()
	if _, err = tx.Exec(`UPDATE media SET path = ? WHERE path = ?`, newPath, oldPath); err != nil {
		return fmt.Errorf("failed to update media table: %w", err)
	}
	if _, err = tx.Exec(`UPDATE media_tag_by_category SET media_path = ? WHERE media_path = ?`, newPath, oldPath); err != nil {
		return fmt.Errorf("failed to update media_tag_by_category table: %w", err)
	}
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("failed to commit transaction: %w", err)
	}
	return nil
}
