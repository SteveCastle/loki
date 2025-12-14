package deps

import (
	"context"
	"fmt"
	"path/filepath"
	"sync"

	"github.com/stevecastle/shrike/jobqueue"
)

// DependencyStatus represents the current state of a dependency.
type DependencyStatus string

const (
	StatusNotInstalled DependencyStatus = "not_installed"
	StatusInstalled    DependencyStatus = "installed"
	StatusOutdated     DependencyStatus = "outdated"
	StatusDownloading  DependencyStatus = "downloading"
)

// Dependency represents an external dependency that can be checked and downloaded.
type Dependency struct {
	ID            string
	Name          string
	Description   string
	TargetDir     string // Base directory for installation
	LatestVersion string
	DownloadURL   string
	ExpectedSize  int64

	// Check function verifies if dependency exists and returns its version
	Check func(ctx context.Context) (exists bool, version string, err error)

	// Download function downloads and installs the dependency
	Download func(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error
}

// DependencyRegistry stores all registered dependencies.
type DependencyRegistry map[string]*Dependency

var (
	registry DependencyRegistry
	mu       sync.RWMutex
)

func init() {
	registry = make(DependencyRegistry)
}

// Register adds a dependency to the global registry.
func Register(dep *Dependency) {
	mu.Lock()
	defer mu.Unlock()
	registry[dep.ID] = dep
}

// GetAll returns all registered dependencies.
func GetAll() []*Dependency {
	mu.RLock()
	defer mu.RUnlock()

	deps := make([]*Dependency, 0, len(registry))
	for _, d := range registry {
		deps = append(deps, d)
	}
	return deps
}

// Get retrieves a dependency by its ID.
func Get(id string) (*Dependency, bool) {
	mu.RLock()
	defer mu.RUnlock()

	dep, ok := registry[id]
	return dep, ok
}

// EnsureAvailable checks if a dependency is available.
// If not available, it returns an error indicating the dependency is missing.
// Users must manually download dependencies via the /dependencies UI.
func EnsureAvailable(ctx context.Context, q *jobqueue.Queue, depID string) error {
	dep, ok := Get(depID)
	if !ok {
		return fmt.Errorf("unknown dependency: %s", depID)
	}

	// Check if dependency exists
	exists, _, err := dep.Check(ctx)
	if err != nil {
		return fmt.Errorf("failed to check dependency %s: %w", depID, err)
	}

	if !exists {
		return fmt.Errorf("dependency %s is not installed. Please download it from the Dependencies page", dep.Name)
	}

	return nil
}

// GetFilePath retrieves the full path to a specific file within a dependency.
// It first checks the metadata store for tracked file paths, then falls back to
// constructing the path from the dependency's target directory.
//
// Parameters:
//   - depID: the dependency ID (e.g., "faster-whisper", "onnx-bundle")
//   - fileName: the name of the file to retrieve (e.g., "faster-whisper-xxl.exe", "model.onnx")
//
// Returns the full path to the file, or an error if the dependency is not found.
func GetFilePath(depID, fileName string) (string, error) {
	// Get dependency metadata from store
	metadata := GetMetadataStore()
	meta, ok := metadata.Get(depID)

	// If metadata exists and has the file tracked, use that path
	if ok && meta.Files != nil {
		if fileInfo, exists := meta.Files[fileName]; exists && fileInfo.Path != "" {
			return fileInfo.Path, nil
		}
	}

	// Fall back to constructing path from dependency definition
	dep, ok := Get(depID)
	if !ok {
		return "", fmt.Errorf("unknown dependency: %s", depID)
	}

	// Default: file is directly in the target directory
	return filepath.Join(dep.TargetDir, fileName), nil
}

// GetInstallPath retrieves the base installation directory for a dependency.
//
// Parameters:
//   - depID: the dependency ID (e.g., "faster-whisper", "onnx-bundle")
//
// Returns the installation directory path, or an error if the dependency is not found.
func GetInstallPath(depID string) (string, error) {
	// Try metadata store first
	metadata := GetMetadataStore()
	meta, ok := metadata.Get(depID)
	if ok && meta.InstallPath != "" {
		return meta.InstallPath, nil
	}

	// Fall back to dependency definition
	dep, ok := Get(depID)
	if !ok {
		return "", fmt.Errorf("unknown dependency: %s", depID)
	}

	return dep.TargetDir, nil
}

// CheckAnyMissing checks if any registered dependencies are missing.
// Returns true if at least one dependency is not installed, false otherwise.
func CheckAnyMissing(ctx context.Context) bool {
	mu.RLock()
	defer mu.RUnlock()

	for _, dep := range registry {
		exists, _, err := dep.Check(ctx)
		if err != nil || !exists {
			return true
		}
	}

	return false
}
