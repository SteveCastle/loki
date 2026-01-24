package downloads

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"
)

const (
	// DefaultRetryAttempts is the number of times to retry a failed download.
	DefaultRetryAttempts = 3
	// DefaultRetryDelay is the delay between retry attempts.
	DefaultRetryDelay = 5 * time.Second
	// DefaultBufferSize is the buffer size for file downloads.
	DefaultBufferSize = 32 * 1024 // 32KB
)

// DownloadFile downloads a file from a URL to a local path with progress tracking.
// It supports resuming interrupted downloads using HTTP Range headers.
func DownloadFile(ctx context.Context, destPath string, url string, progressCb ByteProgressCallback) error {
	// Check if partial file exists for resume
	var existingSize int64
	if stat, err := os.Stat(destPath); err == nil {
		existingSize = stat.Size()
	}

	// Create HTTP request with context
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	// Add Range header for resume if we have partial data
	if existingSize > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", existingSize))
	}

	// Execute request
	client := &http.Client{
		Timeout: 0, // No timeout for large downloads
	}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to execute request: %w", err)
	}
	defer resp.Body.Close()

	// Check response status
	switch resp.StatusCode {
	case http.StatusOK:
		// Fresh download, reset existing size
		existingSize = 0
	case http.StatusPartialContent:
		// Resume supported, keep existing size
	default:
		return fmt.Errorf("bad status: %s", resp.Status)
	}

	// Determine total size
	totalSize := resp.ContentLength
	if totalSize > 0 && existingSize > 0 {
		totalSize += existingSize
	}

	// Open output file (append if resuming, create new otherwise)
	var out *os.File
	if existingSize > 0 && resp.StatusCode == http.StatusPartialContent {
		out, err = os.OpenFile(destPath, os.O_APPEND|os.O_WRONLY, 0644)
	} else {
		out, err = os.Create(destPath)
		existingSize = 0
	}
	if err != nil {
		return fmt.Errorf("failed to open output file: %w", err)
	}
	defer out.Close()

	// Copy with progress tracking
	downloaded := existingSize
	buffer := make([]byte, DefaultBufferSize)
	lastReport := time.Now()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		n, err := resp.Body.Read(buffer)
		if n > 0 {
			if _, writeErr := out.Write(buffer[:n]); writeErr != nil {
				return fmt.Errorf("failed to write to file: %w", writeErr)
			}
			downloaded += int64(n)

			// Report progress periodically
			if progressCb != nil && time.Since(lastReport) >= 100*time.Millisecond {
				progressCb(downloaded, totalSize)
				lastReport = time.Now()
			}
		}

		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("failed to read response: %w", err)
		}
	}

	// Final progress report
	if progressCb != nil {
		progressCb(downloaded, totalSize)
	}

	return nil
}

// DownloadWithRetry downloads a file with automatic retry on failure.
func DownloadWithRetry(ctx context.Context, destPath string, url string, progressCb ByteProgressCallback) error {
	var lastErr error

	for attempt := 1; attempt <= DefaultRetryAttempts; attempt++ {
		err := DownloadFile(ctx, destPath, url, progressCb)
		if err == nil {
			return nil
		}

		lastErr = err

		// Don't retry on context cancellation
		if ctx.Err() != nil {
			return err
		}

		// Wait before retry (unless it's the last attempt)
		if attempt < DefaultRetryAttempts {
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(DefaultRetryDelay):
			}
		}
	}

	return fmt.Errorf("download failed after %d attempts: %w", DefaultRetryAttempts, lastErr)
}

// FormatBytes formats bytes as human-readable size.
func FormatBytes(bytes int64) string {
	const unit = 1024
	if bytes < unit {
		return fmt.Sprintf("%d B", bytes)
	}
	div, exp := int64(unit), 0
	for n := bytes / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(bytes)/float64(div), "KMGTPE"[exp])
}

// FormatSpeed formats bytes per second as human-readable speed.
func FormatSpeed(bytesPerSec int64) string {
	return FormatBytes(bytesPerSec) + "/s"
}
