package downloads

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/stevecastle/shrike/stream"
)

// DownloadManager orchestrates dependency downloads with progress tracking.
type DownloadManager struct {
	mu          sync.RWMutex
	progress    map[string]*Progress
	cancelFuncs map[string]context.CancelFunc
	installing  bool
}

// NewDownloadManager creates a new DownloadManager instance.
func NewDownloadManager() *DownloadManager {
	return &DownloadManager{
		progress:    make(map[string]*Progress),
		cancelFuncs: make(map[string]context.CancelFunc),
	}
}

// Global manager instance
var globalManager = NewDownloadManager()

// GetManager returns the global DownloadManager instance.
func GetManager() *DownloadManager {
	return globalManager
}

// Install starts a download for a single dependency.
// The downloadFn is the dependency's download function.
func (m *DownloadManager) Install(ctx context.Context, depID string, depName string, downloadFn func(context.Context, ProgressCallback) error) error {
	// Create cancellable context
	ctx, cancel := context.WithCancel(ctx)

	m.mu.Lock()
	m.cancelFuncs[depID] = cancel
	m.progress[depID] = &Progress{
		DependencyID:   depID,
		DependencyName: depName,
		Status:         StatusPending,
	}
	m.mu.Unlock()

	// Progress callback that updates internal state and broadcasts via SSE
	progressCb := func(p Progress) {
		p.DependencyID = depID
		p.DependencyName = depName
		m.updateProgress(depID, &p)
		m.broadcastProgress()
	}

	// Mark as downloading
	progressCb(Progress{Status: StatusDownloading, Message: "Starting download..."})

	// Run the download
	err := downloadFn(ctx, progressCb)

	m.mu.Lock()
	delete(m.cancelFuncs, depID)
	m.mu.Unlock()

	if err != nil {
		if ctx.Err() == context.Canceled {
			progressCb(Progress{Status: StatusCancelled, Message: "Download cancelled"})
		} else {
			progressCb(Progress{Status: StatusError, Error: err.Error(), Message: "Download failed"})
		}
		return err
	}

	progressCb(Progress{Status: StatusComplete, Message: "Installation complete", Percent: 100})
	return nil
}

// InstallAll starts downloads for all provided dependencies concurrently.
func (m *DownloadManager) InstallAll(ctx context.Context, deps []DependencyDownload, callback ProgressCallback) error {
	m.mu.Lock()
	m.installing = true
	m.mu.Unlock()

	defer func() {
		m.mu.Lock()
		m.installing = false
		m.mu.Unlock()
	}()

	// Initialize progress for all dependencies
	for _, dep := range deps {
		m.mu.Lock()
		m.progress[dep.ID] = &Progress{
			DependencyID:   dep.ID,
			DependencyName: dep.Name,
			Status:         StatusPending,
		}
		m.mu.Unlock()
	}

	m.broadcastProgress()

	// Use WaitGroup to track all downloads
	var wg sync.WaitGroup
	errChan := make(chan error, len(deps))

	for _, dep := range deps {
		wg.Add(1)
		go func(d DependencyDownload) {
			defer wg.Done()
			err := m.Install(ctx, d.ID, d.Name, d.DownloadFn)
			if err != nil {
				errChan <- fmt.Errorf("%s: %w", d.ID, err)
			}
		}(dep)
	}

	// Wait for all downloads to complete
	wg.Wait()
	close(errChan)

	// Collect errors
	var errors []error
	for err := range errChan {
		errors = append(errors, err)
	}

	if len(errors) > 0 {
		return fmt.Errorf("%d downloads failed", len(errors))
	}

	return nil
}

// Cancel cancels a specific download.
func (m *DownloadManager) Cancel(depID string) {
	m.mu.Lock()
	if cancel, ok := m.cancelFuncs[depID]; ok {
		cancel()
	}
	m.mu.Unlock()
}

// CancelAll cancels all active downloads.
func (m *DownloadManager) CancelAll() {
	m.mu.Lock()
	for _, cancel := range m.cancelFuncs {
		cancel()
	}
	m.mu.Unlock()
}

// GetProgress returns the current progress of all downloads.
func (m *DownloadManager) GetProgress() OverallProgress {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var overall OverallProgress
	overall.Dependencies = make([]Progress, 0, len(m.progress))
	overall.Installing = m.installing

	var totalPercent float64
	for _, p := range m.progress {
		overall.Dependencies = append(overall.Dependencies, *p)
		overall.TotalDeps++
		if p.Status == StatusComplete {
			overall.CompletedCount++
		}
		totalPercent += p.Percent
	}

	if overall.TotalDeps > 0 {
		overall.OverallPercent = totalPercent / float64(overall.TotalDeps)
	}

	return overall
}

// GetDependencyProgress returns the progress for a specific dependency.
func (m *DownloadManager) GetDependencyProgress(depID string) (Progress, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if p, ok := m.progress[depID]; ok {
		return *p, true
	}
	return Progress{}, false
}

// ClearProgress clears all progress data.
func (m *DownloadManager) ClearProgress() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.progress = make(map[string]*Progress)
}

// IsInstalling returns whether any installation is in progress.
func (m *DownloadManager) IsInstalling() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.installing
}

// updateProgress updates the progress for a specific dependency.
func (m *DownloadManager) updateProgress(depID string, p *Progress) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.progress[depID] = p
}

// broadcastProgress sends the current progress to all SSE clients.
func (m *DownloadManager) broadcastProgress() {
	progress := m.GetProgress()
	data, err := json.Marshal(progress)
	if err != nil {
		return
	}
	stream.Broadcast(stream.Message{
		Type: "download-progress",
		Msg:  string(data),
	})
}

// DependencyDownload represents a dependency to be downloaded.
type DependencyDownload struct {
	ID         string
	Name       string
	DownloadFn func(context.Context, ProgressCallback) error
}

// SpeedTracker tracks download speed over time.
type SpeedTracker struct {
	mu          sync.Mutex
	lastBytes   int64
	lastTime    time.Time
	speedWindow []int64
}

// NewSpeedTracker creates a new SpeedTracker.
func NewSpeedTracker() *SpeedTracker {
	return &SpeedTracker{
		lastTime:    time.Now(),
		speedWindow: make([]int64, 0, 10),
	}
}

// Update updates the speed tracker with new byte count.
func (s *SpeedTracker) Update(totalBytes int64) int64 {
	s.mu.Lock()
	defer s.mu.Unlock()

	now := time.Now()
	elapsed := now.Sub(s.lastTime).Seconds()

	if elapsed < 0.1 {
		// Not enough time passed for accurate measurement
		if len(s.speedWindow) > 0 {
			return s.averageSpeed()
		}
		return 0
	}

	bytesDownloaded := totalBytes - s.lastBytes
	speed := int64(float64(bytesDownloaded) / elapsed)

	s.lastBytes = totalBytes
	s.lastTime = now

	// Keep a sliding window of speed measurements
	s.speedWindow = append(s.speedWindow, speed)
	if len(s.speedWindow) > 10 {
		s.speedWindow = s.speedWindow[1:]
	}

	return s.averageSpeed()
}

func (s *SpeedTracker) averageSpeed() int64 {
	if len(s.speedWindow) == 0 {
		return 0
	}
	var sum int64
	for _, v := range s.speedWindow {
		sum += v
	}
	return sum / int64(len(s.speedWindow))
}
