package deps

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// FileInfo represents information about an installed file.
type FileInfo struct {
	Path string `json:"path"`
	Hash string `json:"hash"`
	Size int64  `json:"size"`
}

// DependencyMetadata stores information about an installed dependency.
type DependencyMetadata struct {
	InstalledVersion string              `json:"installedVersion"`
	Status           DependencyStatus    `json:"status"`
	InstallPath      string              `json:"installPath"`
	LastChecked      time.Time           `json:"lastChecked"`
	LastUpdated      time.Time           `json:"lastUpdated"`
	Files            map[string]FileInfo `json:"files"` // Track installed files
	JobID            string              `json:"jobId"`  // Active download job ID
}

// MetadataStore manages dependency metadata persistence.
type MetadataStore struct {
	Dependencies map[string]DependencyMetadata `json:"dependencies"`
	mu           sync.RWMutex
	filePath     string
}

var (
	metadataStore *MetadataStore
	metadataOnce  sync.Once
)

// GetMetadataStore returns the singleton metadata store instance.
func GetMetadataStore() *MetadataStore {
	metadataOnce.Do(func() {
		store, err := LoadMetadata()
		if err != nil {
			// If loading fails, create a new empty store
			store = &MetadataStore{
				Dependencies: make(map[string]DependencyMetadata),
				filePath:     getMetadataPath(),
			}
		}
		metadataStore = store
	})
	return metadataStore
}

// getMetadataPath returns the path to the metadata file.
func getMetadataPath() string {
	appData := os.Getenv("APPDATA")
	if appData == "" {
		appData = "."
	}
	return filepath.Join(appData, "Lowkey Media Viewer", "dependencies.json")
}

// LoadMetadata loads the metadata store from disk.
func LoadMetadata() (*MetadataStore, error) {
	metadataPath := getMetadataPath()

	data, err := os.ReadFile(metadataPath)
	if err != nil {
		if os.IsNotExist(err) {
			// File doesn't exist yet, return empty store
			return &MetadataStore{
				Dependencies: make(map[string]DependencyMetadata),
				filePath:     metadataPath,
			}, nil
		}
		return nil, err
	}

	var store MetadataStore
	if err := json.Unmarshal(data, &store); err != nil {
		return nil, err
	}

	store.filePath = metadataPath
	if store.Dependencies == nil {
		store.Dependencies = make(map[string]DependencyMetadata)
	}

	return &store, nil
}

// Save persists the metadata store to disk.
func (m *MetadataStore) Save() error {
	m.mu.RLock()
	defer m.mu.RUnlock()

	// Ensure directory exists
	dir := filepath.Dir(m.filePath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}

	data, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(m.filePath, data, 0644)
}

// GetStatus returns the status of a dependency.
func (m *MetadataStore) GetStatus(depID string) DependencyStatus {
	m.mu.RLock()
	defer m.mu.RUnlock()

	meta, ok := m.Dependencies[depID]
	if !ok {
		return StatusNotInstalled
	}

	return meta.Status
}

// Get returns the metadata for a dependency.
func (m *MetadataStore) Get(depID string) (DependencyMetadata, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	meta, ok := m.Dependencies[depID]
	return meta, ok
}

// Update updates the metadata for a dependency.
func (m *MetadataStore) Update(depID string, meta DependencyMetadata) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.Dependencies[depID] = meta
	return nil
}

// UpdateStatus updates just the status of a dependency.
func (m *MetadataStore) UpdateStatus(depID string, status DependencyStatus) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	meta, ok := m.Dependencies[depID]
	if !ok {
		meta = DependencyMetadata{
			Files: make(map[string]FileInfo),
		}
	}

	meta.Status = status
	meta.LastChecked = time.Now()
	m.Dependencies[depID] = meta

	return nil
}

// SetJobID sets the active job ID for a dependency download.
func (m *MetadataStore) SetJobID(depID string, jobID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	meta, ok := m.Dependencies[depID]
	if !ok {
		meta = DependencyMetadata{
			Files: make(map[string]FileInfo),
		}
	}

	meta.JobID = jobID
	meta.LastChecked = time.Now()
	m.Dependencies[depID] = meta

	return nil
}

// ClearJobID clears the active job ID for a dependency.
func (m *MetadataStore) ClearJobID(depID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	meta, ok := m.Dependencies[depID]
	if !ok {
		return nil
	}

	meta.JobID = ""
	m.Dependencies[depID] = meta

	return nil
}

// GetJobID returns the active job ID for a dependency.
func (m *MetadataStore) GetJobID(depID string) string {
	m.mu.RLock()
	defer m.mu.RUnlock()

	meta, ok := m.Dependencies[depID]
	if !ok {
		return ""
	}

	return meta.JobID
}
