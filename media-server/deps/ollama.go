package deps

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"

	"github.com/stevecastle/shrike/jobqueue"
)

func init() {
	Register(&Dependency{
		ID:            "ollama",
		Name:          "Ollama",
		Description:   "Local AI model server for vision-based image descriptions and text generation",
		TargetDir:     "", // Ollama manages its own installation
		Check:         checkOllama,
		Download:      downloadOllama,
		LatestVersion: "0.13.3",                      // Version comparison not applicable for API service
		DownloadURL:   "https://ollama.com/download", // Official download page
		ExpectedSize:  0,                             // N/A for API service
	})
}

// checkOllama verifies if the Ollama API server is running locally.
func checkOllama(ctx context.Context) (bool, string, error) {
	// Default Ollama API endpoint
	apiURL := "http://localhost:11434/api/version"

	// Create HTTP request with timeout
	client := &http.Client{
		Timeout: 5 * time.Second,
	}

	req, err := http.NewRequestWithContext(ctx, "GET", apiURL, nil)
	if err != nil {
		return false, "", fmt.Errorf("failed to create request: %w", err)
	}

	// Try to connect to Ollama API
	resp, err := client.Do(req)
	if err != nil {
		// Server not running or not reachable
		return false, "", nil
	}
	defer resp.Body.Close()

	// If we get a successful response, Ollama is running
	if resp.StatusCode == http.StatusOK {
		// Parse JSON response to get version
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			return true, "unknown", fmt.Errorf("failed to read response: %w", err)
		}

		var versionResp struct {
			Version string `json:"version"`
		}

		if err := json.Unmarshal(body, &versionResp); err != nil {
			return true, "unknown", fmt.Errorf("failed to parse version response: %w", err)
		}

		return true, versionResp.Version, nil
	}

	return false, "", nil
}

// downloadOllama provides instructions for manual installation.
// Unlike other dependencies, Ollama cannot be automatically downloaded.
func downloadOllama(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	q.PushJobStdout(j.ID, "")
	q.PushJobStdout(j.ID, "Ollama Installation Required")
	q.PushJobStdout(j.ID, "================================")
	q.PushJobStdout(j.ID, "")
	q.PushJobStdout(j.ID, "Ollama is a local AI model server that must be installed manually.")
	q.PushJobStdout(j.ID, "")
	q.PushJobStdout(j.ID, "Download Ollama:")
	q.PushJobStdout(j.ID, "  https://ollama.com/download")
	q.PushJobStdout(j.ID, "")
	q.PushJobStdout(j.ID, "Installation Instructions:")
	q.PushJobStdout(j.ID, "  https://github.com/ollama/ollama/blob/main/README.md#quickstart")
	q.PushJobStdout(j.ID, "")
	q.PushJobStdout(j.ID, "After installation:")
	q.PushJobStdout(j.ID, "  1. Run 'ollama serve' to start the API server")
	q.PushJobStdout(j.ID, "  2. Install a vision model (recommended):")
	q.PushJobStdout(j.ID, "     ollama pull llama3.2-vision")
	q.PushJobStdout(j.ID, "")
	q.PushJobStdout(j.ID, "The Ollama API server will run at: http://localhost:11434")
	q.PushJobStdout(j.ID, "")

	return fmt.Errorf("manual installation required - please follow the instructions above")
}
