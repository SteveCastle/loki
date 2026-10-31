package tasks

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/media"
)

var moveOptions = []TaskOption{
	{Name: "target", Label: "Target Directory", Type: "string", Required: true, Description: "Destination directory to move files into"},
	{Name: "prefix", Label: "Path Prefix", Type: "string", Description: "Common prefix to strip when computing relative paths"},
}

func moveTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	ctx := j.Ctx

	opts := ParseOptions(j, moveOptions)
	targetDir, _ := opts["target"].(string)
	specifiedPrefix, _ := opts["prefix"].(string)

	// Backwards compatibility: support positional target argument
	if targetDir == "" {
		for _, arg := range j.Arguments {
			if !strings.HasPrefix(arg, "-") {
				targetDir = strings.TrimSpace(arg)
				break
			}
		}
	}
	if targetDir == "" {
		q.PushJobStdout(j.ID, "Error: No target directory specified in arguments")
		q.ErrorJob(j.ID)
		return fmt.Errorf("no target directory specified")
	}

	res, rerr := resolveJobItemsRaw(j, q)
	if rerr != nil {
		q.PushJobStdout(j.ID, "Failed to resolve input: "+rerr.Error())
		q.ErrorJob(j.ID)
		return rerr
	}
	if res.FromQuery {
		q.PushJobStdout(j.ID, fmt.Sprintf("Query: %s", res.Query))
	}
	cleanedPaths := res.Paths
	if len(cleanedPaths) == 0 {
		q.PushJobStdout(j.ID, "No valid paths found after parsing input")
		q.CompleteJob(j.ID)
		return nil
	}

	q.PushJobStdout(j.ID, fmt.Sprintf("Starting move operation to target directory: %s", targetDir))
	q.PushJobStdout(j.ID, fmt.Sprintf("Parsed %d paths from input", len(cleanedPaths)))

	absTarget, _ := filepath.Abs(targetDir)
	var validPaths, dirRoots []string
	looseFiles := false
	for _, p := range cleanedPaths {
		abs, err := filepath.Abs(p)
		if err != nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("Warning: could not resolve absolute path for %s: %v", p, err))
			continue
		}
		cp := filepath.FromSlash(abs)
		st, err := os.Stat(cp)
		if os.IsNotExist(err) {
			q.PushJobStdout(j.ID, fmt.Sprintf("Warning: file does not exist: %s", cp))
			continue
		}
		if err == nil && st.IsDir() {
			// A folder is moved item by item: renaming it whole would re-point
			// nothing (a folder is not a library row) and strand every item
			// under it on its old path.
			files, skipped, werr := expandMoveDir(cp, absTarget)
			if werr != nil {
				q.PushJobStdout(j.ID, fmt.Sprintf("Warning: could not read folder %s: %v", cp, werr))
				continue
			}
			q.PushJobStdout(j.ID, fmt.Sprintf("Folder %s: %d media files (%d other files left in place)", cp, len(files), skipped))
			dirRoots = append(dirRoots, cp)
			validPaths = append(validPaths, files...)
			continue
		}
		looseFiles = true
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
	} else if len(dirRoots) == 1 && !looseFiles {
		// One folder: its contents land in the target, sub-folders kept.
		prefixToUse = strings.TrimRight(dirRoots[0], `/\`) + string(filepath.Separator)
		q.PushJobStdout(j.ID, fmt.Sprintf("Moving the contents of %s", dirRoots[0]))
	} else if len(dirRoots) > 1 && !looseFiles {
		// Several folders: keep each folder's name under the target so their
		// contents can't collide (findCommonPrefix works on parents).
		prefixToUse = findCommonPrefix(dirRoots)
		q.PushJobStdout(j.ID, fmt.Sprintf("Using calculated common prefix: %s", prefixToUse))
	} else {
		prefixToUse = findCommonPrefix(validPaths)
		if prefixToUse == "" {
			q.PushJobStdout(j.ID, "Warning: No common prefix found, files will be moved to target directory root")
		} else {
			q.PushJobStdout(j.ID, fmt.Sprintf("Using calculated common prefix: %s", prefixToUse))
		}
	}

	_ = q.SetJobProgress(j.ID, 0, len(validPaths))
	moveCount, updateCount, inPlace := 0, 0, 0
	var failures []string
	fail := func(msg string) {
		failures = append(failures, msg)
		q.PushJobStdout(j.ID, "FAILED: "+msg)
	}
	peopleChanged := false
	for i, srcPath := range validPaths {
		select {
		case <-ctx.Done():
			q.PushJobStdout(j.ID, "Task was canceled")
			_ = q.CancelJob(j.ID)
			return ctx.Err()
		default:
		}
		if q.PauseRequested(j.ID) {
			q.PushJobStdout(j.ID, fmt.Sprintf("Paused at %d/%d - resume to continue", i, len(validPaths)))
			return jobqueue.ErrPaused
		}
		_ = q.SetJobProgress(j.ID, i, len(validPaths))

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

		res, err := moveMediaItem(ctx, q.Db, srcPath, destPath)
		switch {
		case errors.Is(err, errAlreadyInPlace):
			// Re-running a move over a query that still matches what an
			// earlier run moved: nothing to do, and nothing went wrong.
			inPlace++
			q.RegisterOutputFile(j.ID, destPath)
		case err != nil:
			fail(fmt.Sprintf("%s -> %s: %v", srcPath, destPath, err))
		default:
			moveCount++
			if res != nil && res.Items > 0 {
				updateCount++
				if res.Rows["face.media_path"] > 0 {
					peopleChanged = true
				}
			}
			q.PushJobStdout(j.ID, fmt.Sprintf("Moved: %s -> %s", srcPath, destPath))
			q.RegisterOutputFile(j.ID, destPath)
		}
		if (i+1)%10 == 0 || i == len(validPaths)-1 {
			q.PushJobStdout(j.ID, fmt.Sprintf("Progress: %d/%d files processed", i+1, len(validPaths)))
		}
	}

	_ = q.SetJobProgress(j.ID, len(validPaths), len(validPaths))
	for _, d := range dirRoots {
		removeEmptyDirs(d)
	}
	if peopleChanged {
		broadcastPeopleUpdated([]string{})
	}
	q.PushJobStdout(j.ID, fmt.Sprintf("Move operation completed: %d files moved, %d library items re-pointed, %d already in place, %d failed",
		moveCount, updateCount, inPlace, len(failures)))
	select {
	case <-ctx.Done():
		q.PushJobStdout(j.ID, "Task was canceled")
		q.ErrorJob(j.ID)
		return ctx.Err()
	default:
	}
	if len(failures) > 0 {
		// A partial move must not read as "completed" in the queue: the
		// failed items are still where they were, and re-running the job
		// picks them up (moved ones count as already in place).
		q.PushJobStdout(j.ID, fmt.Sprintf("%d of %d items were NOT moved:", len(failures), len(validPaths)))
		for _, f := range failures {
			q.PushJobStdout(j.ID, "  "+f)
		}
		q.ErrorJob(j.ID)
		return fmt.Errorf("%d of %d items failed to move", len(failures), len(validPaths))
	}
	q.CompleteJob(j.ID)
	return nil
}

// expandMoveDir lists the media files under dir, recursively, for a folder
// move. Non-media files are counted and left where they are (transcript
// sidecars travel with their video instead). The target folder is skipped
// when it sits inside dir, so a re-run never re-moves what it already moved.
func expandMoveDir(dir, target string) (files []string, skipped int, err error) {
	err = filepath.WalkDir(dir, func(p string, d os.DirEntry, werr error) error {
		if werr != nil {
			return werr
		}
		if d.IsDir() {
			if target != "" && p != dir && strings.EqualFold(filepath.Clean(p), filepath.Clean(target)) {
				return filepath.SkipDir
			}
			return nil
		}
		if !d.Type().IsRegular() {
			return nil
		}
		if isMediaFile(p) {
			files = append(files, p)
		} else {
			skipped++
		}
		return nil
	})
	return files, skipped, err
}

// removeEmptyDirs deletes folders under (and including) root that a folder
// move left empty, deepest first. Anything still holding a file stays.
func removeEmptyDirs(root string) {
	var dirs []string
	_ = filepath.WalkDir(root, func(p string, d os.DirEntry, err error) error {
		if err == nil && d.IsDir() {
			dirs = append(dirs, p)
		}
		return nil
	})
	for i := len(dirs) - 1; i >= 0; i-- {
		_ = os.Remove(dirs[i]) // fails, harmlessly, on a non-empty folder
	}
}

// errAlreadyInPlace marks an item whose computed destination is its current
// path — not a failure.
var errAlreadyInPlace = errors.New("already at destination")

// moveMediaItem moves one file on disk and then re-points every reference to
// it: all the media-path columns media.MovePath knows (tags, embeddings,
// faces, scan markers, battles, duplicates), the live vector and face indexes,
// and a .vtt transcript sidecar. If the library refuses the new path (another
// item already owns it) the file is moved back, so disk and library never
// disagree. The returned result is nil when the file wasn't in the library.
func moveMediaItem(ctx context.Context, db *sql.DB, src, dst string) (*media.MoveResult, error) {
	if filepath.Clean(src) == filepath.Clean(dst) {
		return nil, errAlreadyInPlace
	}
	if _, err := os.Stat(dst); err == nil {
		return nil, fmt.Errorf("destination already exists on disk")
	}
	if err := os.MkdirAll(filepath.Dir(dst), 0755); err != nil {
		return nil, fmt.Errorf("creating destination directory: %w", err)
	}
	if err := moveFile(src, dst); err != nil {
		return nil, err
	}

	var res *media.MoveResult
	if db != nil {
		var err error
		res, err = media.MovePath(ctx, db, src, dst, media.MoveOptions{})
		if err != nil {
			if undoErr := moveFile(dst, src); undoErr != nil {
				return nil, fmt.Errorf("library update failed (%v) and the file could not be moved back from %s: %w", err, dst, undoErr)
			}
			return nil, fmt.Errorf("library update failed, file left in place: %w", err)
		}
		if res.Items > 0 {
			IndexRenamePath(db, src, dst)
			FaceIndexRenamePath(src, dst)
		}
	}

	// The transcript travels with its video. Best effort: the media itself
	// has moved and is consistent; a stranded .vtt only loses captions.
	srcVtts, dstVtts := media.VttSidecarCandidates(src), media.VttSidecarCandidates(dst)
	for k, sv := range srcVtts {
		if st, err := os.Stat(sv); err != nil || st.IsDir() || k >= len(dstVtts) {
			continue
		}
		if _, err := os.Stat(dstVtts[k]); err == nil {
			continue
		}
		_ = moveFile(sv, dstVtts[k])
	}
	return res, nil
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
