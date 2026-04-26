package tasks

import (
	"context"
	"fmt"
	"io"
	"log"
	"mime"
	"os"
	"path/filepath"
	"strings"

	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/platform"
	"github.com/stevecastle/shrike/storage"
)

// stagingDir returns a per-job temporary directory for CLI tools that must
// write to local disk. The caller is responsible for cleaning up via os.RemoveAll.
func stagingDir(jobID string) (string, error) {
	dir := filepath.Join(platform.GetTempDir(), "staging", jobID)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return "", fmt.Errorf("create staging dir: %w", err)
	}
	return dir, nil
}

// uploadStagedFiles copies every file from stagingDir into the default storage
// backend under destPrefix (e.g. "downloads/"). It returns the list of final
// paths as they exist in the backend (for database insertion).
//
// If no default backend is configured the staged files are left in place and
// their local paths are returned — this preserves backwards-compatible behaviour
// for setups that have no storage roots yet.
func uploadStagedFiles(ctx context.Context, q *jobqueue.Queue, jobID string, stagedFiles []string, stagingBase string, destPrefix string) []string {
	backend := defaultBackend()

	// DEBUG: trace backend resolution
	cwd, _ := os.Getwd()
	log.Printf("[uploadStagedFiles] cwd=%s stagingBase=%s destPrefix=%s stagedFiles=%v", cwd, stagingBase, destPrefix, stagedFiles)

	if backend == nil {
		// No storage backend — return local paths as-is (legacy behaviour)
		log.Printf("[uploadStagedFiles] WARNING: defaultBackend() returned nil — returning raw local paths")
		q.PushJobStdout(jobID, "DEBUG: no storage backend configured, returning local staged paths as-is")
		return stagedFiles
	}

	// The backend root tells us whether we need to resolve relative paths
	// into absolute ones for the database. Local backends have an absolute
	// filesystem root (e.g. "X:\"); S3 backends have an "s3://" root and
	// store relative keys.
	root := backend.Root()
	log.Printf("[uploadStagedFiles] backend root: Type=%s Path=%s Name=%s", root.Type, root.Path, root.Name)
	q.PushJobStdout(jobID, fmt.Sprintf("DEBUG: backend root Type=%s Path=%s Name=%s", root.Type, root.Path, root.Name))

	var finalPaths []string
	for _, localPath := range stagedFiles {
		destKey := stagedDestKey(localPath, stagingBase, destPrefix)
		log.Printf("[uploadStagedFiles] localPath=%s destKey=%s", localPath, destKey)

		f, err := os.Open(localPath)
		if err != nil {
			q.PushJobStdout(jobID, fmt.Sprintf("Warning: cannot open staged file %s: %v", localPath, err))
			continue
		}

		ct := contentTypeFromPath(localPath)
		if err := backend.Upload(ctx, destKey, f, ct); err != nil {
			f.Close()
			q.PushJobStdout(jobID, fmt.Sprintf("Warning: upload to backend failed for %s: %v", destKey, err))
			continue
		}
		f.Close()

		storePath := storePathForKey(root, destKey)
		log.Printf("[uploadStagedFiles] storePath=%s (root.Type=%s root.Path=%s destKey=%s)", storePath, root.Type, root.Path, destKey)
		finalPaths = append(finalPaths, storePath)
		q.PushJobStdout(jobID, fmt.Sprintf("Uploaded to storage: %s", storePath))
	}

	log.Printf("[uploadStagedFiles] final paths: %v", finalPaths)
	return finalPaths
}

// defaultBackend is a small helper so callers don't need to nil-check the registry.
func defaultBackend() storage.Backend {
	if storageReg == nil {
		return nil
	}
	return storageReg.DefaultBackend()
}

// stagedDestKey computes the relative storage key for a staged file:
// destPrefix joined with the path of localPath relative to stagingBase, in
// forward-slash form. Falls back to the basename if the relative path can't
// be derived.
func stagedDestKey(localPath, stagingBase, destPrefix string) string {
	rel, err := filepath.Rel(stagingBase, localPath)
	if err != nil {
		log.Printf("[stagedDestKey] filepath.Rel(%s, %s) error: %v — falling back to Base", stagingBase, localPath, err)
		rel = filepath.Base(localPath)
	}
	return destPrefix + filepath.ToSlash(rel)
}

// storePathForKey converts a destKey into the path that should be persisted
// in the database for the given backend root. Local backends store absolute
// filesystem paths; S3 backends store the relative key as-is.
func storePathForKey(root storage.Entry, destKey string) string {
	if root.Type == "local" {
		return filepath.Join(root.Path, filepath.FromSlash(destKey))
	}
	return destKey
}

// stagedToFinalPath predicts the final stored path for a staged file using
// the default backend, mirroring the behaviour of uploadStagedFiles. When
// no backend is configured it returns localPath unchanged.
func stagedToFinalPath(localPath, stagingBase, destPrefix string) string {
	backend := defaultBackend()
	if backend == nil {
		return localPath
	}
	return storePathForKey(backend.Root(), stagedDestKey(localPath, stagingBase, destPrefix))
}

// cleanupStaging removes the staging directory, logging but not failing on error.
func cleanupStaging(dir string) {
	if err := os.RemoveAll(dir); err != nil {
		log.Printf("Warning: failed to clean staging dir %s: %v", dir, err)
	}
}

// contentTypeFromPath guesses a MIME type from the file extension.
func contentTypeFromPath(path string) string {
	ext := strings.ToLower(filepath.Ext(path))
	if ct := mime.TypeByExtension(ext); ct != "" {
		return ct
	}
	return "application/octet-stream"
}

// fileSizeOrZero returns the file size or 0 if stat fails.
func fileSizeOrZero(path string) int64 {
	fi, err := os.Stat(path)
	if err != nil {
		return 0
	}
	return fi.Size()
}

// scanStagingFiles walks a staging directory and returns all media file paths.
func scanStagingFiles(dir string) ([]string, error) {
	var files []string
	err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() && isMediaFile(path) {
			files = append(files, path)
		}
		return nil
	})
	return files, err
}

// downloadFromBackend downloads a file from the backend to a local path.
// Used when CLI tools need local files from S3 backends.
func downloadFromBackend(ctx context.Context, backend storage.Backend, remotePath, localPath string) error {
	rc, err := backend.Download(ctx, remotePath)
	if err != nil {
		return fmt.Errorf("download %s: %w", remotePath, err)
	}
	defer rc.Close()

	if err := os.MkdirAll(filepath.Dir(localPath), 0755); err != nil {
		return err
	}

	f, err := os.Create(localPath)
	if err != nil {
		return err
	}
	defer f.Close()

	_, err = io.Copy(f, rc)
	return err
}
