package deps

import (
	"context"
	"fmt"
	"path/filepath"
	"sync"

	"github.com/stevecastle/shrike/downloads"
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

	// Optional indicates if this dependency is optional (doesn't block setup)
	Optional bool
	// ManualOnly indicates if this dependency must be installed manually (shows link instead of auto-download)
	ManualOnly bool
	// InstallURL is the manual installation link for ManualOnly dependencies
	InstallURL string

	// Check function verifies if dependency exists and returns its version
	Check func(ctx context.Context) (exists bool, version string, err error)

	// Download function downloads and installs the dependency (legacy jobqueue-based)
	Download func(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error

	// DownloadFn is the new download function signature without jobqueue dependency
	DownloadFn func(ctx context.Context, progress downloads.ProgressCallback) error
}

// DependencyRegistry stores all registered dependencies.
type DependencyRegistry map[string]*Dependency

var (
	registry DependencyRegistry = make(DependencyRegistry)
	mu       sync.RWMutex
)

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

// CheckAnyMissing checks if any registered dependencies are missing and not ignored.
// Returns true if at least one dependency is not installed and not ignored, false otherwise.
func CheckAnyMissing(ctx context.Context) bool {
	mu.RLock()
	defer mu.RUnlock()

	metadata := GetMetadataStore()

	for _, dep := range registry {
		// Skip ignored dependencies
		if metadata.IsIgnored(dep.ID) {
			continue
		}

		// Skip optional dependencies
		if dep.Optional {
			continue
		}

		exists, _, err := dep.Check(ctx)
		if err != nil || !exists {
			return true
		}
	}

	return false
}

// GetRequired returns all non-optional dependencies.
func GetRequired() []*Dependency {
	mu.RLock()
	defer mu.RUnlock()

	deps := make([]*Dependency, 0)
	for _, d := range registry {
		if !d.Optional {
			deps = append(deps, d)
		}
	}
	return deps
}

// GetOptional returns all optional dependencies.
func GetOptional() []*Dependency {
	mu.RLock()
	defer mu.RUnlock()

	deps := make([]*Dependency, 0)
	for _, d := range registry {
		if d.Optional {
			deps = append(deps, d)
		}
	}
	return deps
}

// GetAutoDownloadable returns all dependencies that can be auto-downloaded (not ManualOnly).
func GetAutoDownloadable() []*Dependency {
	mu.RLock()
	defer mu.RUnlock()

	deps := make([]*Dependency, 0)
	for _, d := range registry {
		if !d.ManualOnly {
			deps = append(deps, d)
		}
	}
	return deps
}

// GetMissingRequired returns all required dependencies that are not installed or ignored.
func GetMissingRequired(ctx context.Context) []*Dependency {
	mu.RLock()
	defer mu.RUnlock()

	metadata := GetMetadataStore()
	deps := make([]*Dependency, 0)

	for _, d := range registry {
		// Skip optional dependencies
		if d.Optional {
			continue
		}

		// Skip ignored dependencies
		if metadata.IsIgnored(d.ID) {
			continue
		}

		// Check if installed
		exists, _, err := d.Check(ctx)
		if err != nil || !exists {
			deps = append(deps, d)
		}
	}
	return deps
}
