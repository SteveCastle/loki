package deps

import (
	"context"
	"errors"
	"sync"
	"testing"

	"github.com/stevecastle/shrike/jobqueue"
)

// mockDependency creates a test dependency with the given check result
func mockDependency(id string, exists bool, version string, checkErr error) *Dependency {
	return &Dependency{
		ID:            id,
		Name:          id + " Name",
		Description:   id + " Description",
		TargetDir:     "/test/" + id,
		LatestVersion: "1.0.0",
		ExpectedSize:  1024,
		Check: func(ctx context.Context) (bool, string, error) {
			return exists, version, checkErr
		},
		Download: func(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
			return nil
		},
	}
}

// TestRegisterAndGet tests dependency registration and retrieval
func TestRegisterAndGet(t *testing.T) {
	// Save and restore original registry
	origRegistry := registry
	mu.Lock()
	registry = make(DependencyRegistry)
	mu.Unlock()
	defer func() {
		mu.Lock()
		registry = origRegistry
		mu.Unlock()
	}()

	// Register a test dependency
	testDep := mockDependency("test-dep", true, "1.0.0", nil)
	Register(testDep)

	// Get the dependency
	retrieved, ok := Get("test-dep")
	if !ok {
		t.Fatal("Get() should find registered dependency")
	}
	if retrieved.ID != "test-dep" {
		t.Errorf("Retrieved dependency ID = %q; want %q", retrieved.ID, "test-dep")
	}
	if retrieved.Name != "test-dep Name" {
		t.Errorf("Retrieved dependency Name = %q; want %q", retrieved.Name, "test-dep Name")
	}
}

// TestGetNotFound tests getting a non-existent dependency
func TestGetNotFound(t *testing.T) {
	_, ok := Get("nonexistent-dependency-xyz")
	if ok {
		t.Error("Get() should return false for nonexistent dependency")
	}
}

// TestGetAll tests retrieving all dependencies
func TestGetAll(t *testing.T) {
	// Save and restore original registry
	origRegistry := registry
	mu.Lock()
	registry = make(DependencyRegistry)
	mu.Unlock()
	defer func() {
		mu.Lock()
		registry = origRegistry
		mu.Unlock()
	}()

	// Register multiple dependencies
	dep1 := mockDependency("dep-1", true, "1.0.0", nil)
	dep2 := mockDependency("dep-2", true, "2.0.0", nil)
	dep3 := mockDependency("dep-3", false, "", nil)

	Register(dep1)
	Register(dep2)
	Register(dep3)

	// Get all dependencies
	allDeps := GetAll()

	if len(allDeps) != 3 {
		t.Errorf("GetAll() returned %d dependencies; want 3", len(allDeps))
	}

	// Verify all dependencies are present
	ids := make(map[string]bool)
	for _, dep := range allDeps {
		ids[dep.ID] = true
	}

	for _, expectedID := range []string{"dep-1", "dep-2", "dep-3"} {
		if !ids[expectedID] {
			t.Errorf("GetAll() missing dependency %q", expectedID)
		}
	}
}

// TestCheckAnyMissing tests the CheckAnyMissing function
func TestCheckAnyMissing(t *testing.T) {
	// Save and restore original registry
	origRegistry := registry
	mu.Lock()
	registry = make(DependencyRegistry)
	mu.Unlock()
	defer func() {
		mu.Lock()
		registry = origRegistry
		mu.Unlock()
	}()

	// Register all existing dependencies
	Register(mockDependency("exists-1", true, "1.0.0", nil))
	Register(mockDependency("exists-2", true, "2.0.0", nil))

	// All dependencies exist - should return false
	ctx := context.Background()
	hasMissing := CheckAnyMissing(ctx)
	if hasMissing {
		t.Error("CheckAnyMissing() should return false when all dependencies exist")
	}

	// Add a missing dependency
	Register(mockDependency("missing-1", false, "", nil))

	hasMissing = CheckAnyMissing(ctx)
	if !hasMissing {
		t.Error("CheckAnyMissing() should return true when a dependency is missing")
	}
}

// TestCheckAnyMissingWithError tests CheckAnyMissing when check returns error
func TestCheckAnyMissingWithError(t *testing.T) {
	// Save and restore original registry
	origRegistry := registry
	mu.Lock()
	registry = make(DependencyRegistry)
	mu.Unlock()
	defer func() {
		mu.Lock()
		registry = origRegistry
		mu.Unlock()
	}()

	// Register dependency that returns error on check
	Register(mockDependency("error-dep", false, "", errors.New("check failed")))

	ctx := context.Background()
	hasMissing := CheckAnyMissing(ctx)

	// Error during check should be treated as missing
	if !hasMissing {
		t.Error("CheckAnyMissing() should return true when check returns error")
	}
}

// TestEnsureAvailable tests the EnsureAvailable function
func TestEnsureAvailable(t *testing.T) {
	// Save and restore original registry
	origRegistry := registry
	mu.Lock()
	registry = make(DependencyRegistry)
	mu.Unlock()
	defer func() {
		mu.Lock()
		registry = origRegistry
		mu.Unlock()
	}()

	// Register an existing dependency
	Register(mockDependency("available-dep", true, "1.0.0", nil))

	ctx := context.Background()
	err := EnsureAvailable(ctx, nil, "available-dep")
	if err != nil {
		t.Errorf("EnsureAvailable() should succeed for available dependency; got %v", err)
	}
}

// TestEnsureAvailableMissing tests EnsureAvailable with missing dependency
func TestEnsureAvailableMissing(t *testing.T) {
	// Save and restore original registry
	origRegistry := registry
	mu.Lock()
	registry = make(DependencyRegistry)
	mu.Unlock()
	defer func() {
		mu.Lock()
		registry = origRegistry
		mu.Unlock()
	}()

	// Register a missing dependency
	Register(mockDependency("missing-dep", false, "", nil))

	ctx := context.Background()
	err := EnsureAvailable(ctx, nil, "missing-dep")
	if err == nil {
		t.Error("EnsureAvailable() should return error for missing dependency")
	}
}

// TestEnsureAvailableUnknown tests EnsureAvailable with unknown dependency
func TestEnsureAvailableUnknown(t *testing.T) {
	ctx := context.Background()
	err := EnsureAvailable(ctx, nil, "completely-unknown-dep")
	if err == nil {
		t.Error("EnsureAvailable() should return error for unknown dependency")
	}
}

// TestGetFilePath tests the GetFilePath function
func TestGetFilePath(t *testing.T) {
	// Save and restore original registry
	origRegistry := registry
	mu.Lock()
	registry = make(DependencyRegistry)
	mu.Unlock()
	defer func() {
		mu.Lock()
		registry = origRegistry
		mu.Unlock()
	}()

	// Register a dependency
	dep := &Dependency{
		ID:        "file-dep",
		Name:      "File Dep",
		TargetDir: "/install/path",
		Check: func(ctx context.Context) (bool, string, error) {
			return true, "1.0.0", nil
		},
	}
	Register(dep)

	// Get file path (without metadata, should use target dir)
	path, err := GetFilePath("file-dep", "executable.exe")
	if err != nil {
		t.Errorf("GetFilePath() error = %v", err)
	}
	if path != "/install/path/executable.exe" && path != "\\install\\path\\executable.exe" {
		t.Errorf("GetFilePath() = %q; want /install/path/executable.exe", path)
	}
}

// TestGetFilePathUnknown tests GetFilePath with unknown dependency
func TestGetFilePathUnknown(t *testing.T) {
	_, err := GetFilePath("unknown-dep-xyz", "file.exe")
	if err == nil {
		t.Error("GetFilePath() should return error for unknown dependency")
	}
}

// TestGetInstallPath tests the GetInstallPath function
func TestGetInstallPath(t *testing.T) {
	// Save and restore original registry
	origRegistry := registry
	mu.Lock()
	registry = make(DependencyRegistry)
	mu.Unlock()
	defer func() {
		mu.Lock()
		registry = origRegistry
		mu.Unlock()
	}()

	// Register a dependency
	dep := &Dependency{
		ID:        "install-dep",
		Name:      "Install Dep",
		TargetDir: "/install/dir",
		Check: func(ctx context.Context) (bool, string, error) {
			return true, "1.0.0", nil
		},
	}
	Register(dep)

	// Get install path
	path, err := GetInstallPath("install-dep")
	if err != nil {
		t.Errorf("GetInstallPath() error = %v", err)
	}
	if path != "/install/dir" {
		t.Errorf("GetInstallPath() = %q; want /install/dir", path)
	}
}

// TestGetInstallPathUnknown tests GetInstallPath with unknown dependency
func TestGetInstallPathUnknown(t *testing.T) {
	_, err := GetInstallPath("unknown-install-dep")
	if err == nil {
		t.Error("GetInstallPath() should return error for unknown dependency")
	}
}

// TestDependencyStruct tests Dependency struct fields
func TestDependencyStruct(t *testing.T) {
	checkCalled := false
	downloadCalled := false

	dep := &Dependency{
		ID:            "struct-test",
		Name:          "Struct Test",
		Description:   "Test description",
		TargetDir:     "/target",
		LatestVersion: "2.5.0",
		DownloadURL:   "https://example.com/download",
		ExpectedSize:  1024 * 1024,
		Check: func(ctx context.Context) (bool, string, error) {
			checkCalled = true
			return true, "2.5.0", nil
		},
		Download: func(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
			downloadCalled = true
			return nil
		},
	}

	// Verify fields
	if dep.ID != "struct-test" {
		t.Errorf("Dependency.ID = %q; want %q", dep.ID, "struct-test")
	}
	if dep.ExpectedSize != 1024*1024 {
		t.Errorf("Dependency.ExpectedSize = %d; want %d", dep.ExpectedSize, 1024*1024)
	}

	// Test Check function
	exists, version, err := dep.Check(context.Background())
	if !checkCalled {
		t.Error("Check function was not called")
	}
	if !exists || version != "2.5.0" || err != nil {
		t.Errorf("Check() = (%v, %q, %v); want (true, 2.5.0, nil)", exists, version, err)
	}

	// Test Download function
	err = dep.Download(nil, nil, nil)
	if !downloadCalled {
		t.Error("Download function was not called")
	}
	if err != nil {
		t.Errorf("Download() error = %v", err)
	}
}

// TestDependencyStatusConstants tests DependencyStatus constants
func TestDependencyStatusConstants(t *testing.T) {
	tests := []struct {
		status   DependencyStatus
		expected string
	}{
		{StatusNotInstalled, "not_installed"},
		{StatusInstalled, "installed"},
		{StatusOutdated, "outdated"},
		{StatusDownloading, "downloading"},
	}

	for _, tt := range tests {
		if string(tt.status) != tt.expected {
			t.Errorf("DependencyStatus = %q; want %q", tt.status, tt.expected)
		}
	}
}

// TestConcurrentRegistration tests thread-safety of Register
func TestConcurrentRegistration(t *testing.T) {
	// Save and restore original registry
	origRegistry := registry
	mu.Lock()
	registry = make(DependencyRegistry)
	mu.Unlock()
	defer func() {
		mu.Lock()
		registry = origRegistry
		mu.Unlock()
	}()

	done := make(chan bool)

	// Multiple goroutines registering
	for i := 0; i < 10; i++ {
		go func(id int) {
			dep := mockDependency("concurrent-"+string(rune('0'+id)), true, "1.0", nil)
			Register(dep)
			done <- true
		}(i)
	}

	// Wait for all to complete
	for i := 0; i < 10; i++ {
		<-done
	}

	// Verify all registered
	allDeps := GetAll()
	if len(allDeps) != 10 {
		t.Errorf("Expected 10 registered dependencies; got %d", len(allDeps))
	}
}

// TestConcurrentGet tests thread-safety of Get
func TestConcurrentGet(t *testing.T) {
	// Save and restore original registry
	origRegistry := registry
	mu.Lock()
	registry = make(DependencyRegistry)
	mu.Unlock()
	defer func() {
		mu.Lock()
		registry = origRegistry
		mu.Unlock()
	}()

	// Register a dependency
	Register(mockDependency("concurrent-get", true, "1.0", nil))

	done := make(chan bool)

	// Multiple goroutines reading
	for i := 0; i < 100; i++ {
		go func() {
			_, _ = Get("concurrent-get")
			done <- true
		}()
	}

	// Wait for all to complete
	for i := 0; i < 100; i++ {
		<-done
	}
}
