package workerapi

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// RunPod is a client of a RunPod queue-based serverless endpoint running the
// loki-worker image (the worker polls RunPod for jobs; nothing addresses it).
type RunPod struct {
	// Endpoint is the endpoint id (or its https://api.runpod.ai/v2/<id> URL).
	Endpoint string
	APIKey   string
	HTTP     *http.Client
	// Base overrides https://api.runpod.ai/v2 (tests).
	Base string
}

func (r *RunPod) url(p string) string {
	e := strings.TrimRight(strings.TrimSpace(r.Endpoint), "/")
	if strings.HasPrefix(e, "http://") || strings.HasPrefix(e, "https://") {
		// a pasted URL: keep it up to the endpoint id
		for _, s := range []string{"/run", "/runsync", "/health"} {
			e = strings.TrimSuffix(e, s)
		}
		return e + p
	}
	base := r.Base
	if base == "" {
		base = "https://api.runpod.ai/v2"
	}
	return strings.TrimRight(base, "/") + "/" + e + p
}

func (r *RunPod) do(ctx context.Context, method, p string, in, out any, timeout time.Duration) error {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	var body io.Reader
	if in != nil {
		b, err := json.Marshal(in)
		if err != nil {
			return err
		}
		body = bytes.NewReader(b)
	}
	req, err := http.NewRequestWithContext(ctx, method, r.url(p), body)
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+r.APIKey)
	if in != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	c := r.HTTP
	if c == nil {
		c = http.DefaultClient
	}
	resp, err := c.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden {
		return fmt.Errorf("RunPod rejected the API key (%d)", resp.StatusCode)
	}
	if resp.StatusCode/100 != 2 {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return &HTTPError{Status: resp.StatusCode, Body: string(b)}
	}
	if out == nil {
		return nil
	}
	return json.NewDecoder(resp.Body).Decode(out)
}

// Run queues a job; returns RunPod's job id.
func (r *RunPod) Run(ctx context.Context, in RunPodInput) (string, error) {
	var resp struct {
		ID     string `json:"id"`
		Status string `json:"status"`
	}
	// a 10 MB inline payload over a slow uplink takes a while
	if err := r.do(ctx, http.MethodPost, "/run", map[string]any{"input": in}, &resp, 5*time.Minute); err != nil {
		return "", err
	}
	if resp.ID == "" {
		return "", fmt.Errorf("RunPod returned no job id")
	}
	return resp.ID, nil
}

// RunPodStreamPoll is one /stream poll: the job's status and the chunks it
// streamed since the previous poll.
type RunPodStreamPoll struct {
	Status string
	Error  string
	Chunks []RunPodStream
}

// Stream polls a job's new stream chunks.
func (r *RunPod) Stream(ctx context.Context, id string) (*RunPodStreamPoll, error) {
	var resp struct {
		Status string `json:"status"`
		Error  string `json:"error"`
		Stream []struct {
			Output json.RawMessage `json:"output"`
		} `json:"stream"`
	}
	if err := r.do(ctx, http.MethodGet, "/stream/"+id, nil, &resp, 30*time.Second); err != nil {
		return nil, err
	}
	p := &RunPodStreamPoll{Status: resp.Status, Error: resp.Error}
	for _, s := range resp.Stream {
		var c RunPodStream
		if json.Unmarshal(s.Output, &c) == nil {
			p.Chunks = append(p.Chunks, c)
		}
	}
	return p, nil
}

// Result fetches a completed job's output (the worker's JobStatus).
func (r *RunPod) Result(ctx context.Context, id string) (*JobStatus, error) {
	var resp struct {
		Status string          `json:"status"`
		Error  string          `json:"error"`
		Output json.RawMessage `json:"output"`
	}
	if err := r.do(ctx, http.MethodGet, "/status/"+id, nil, &resp, time.Minute); err != nil {
		return nil, err
	}
	if resp.Status != "COMPLETED" {
		return nil, fmt.Errorf("RunPod job %s: %s %s", id, resp.Status, resp.Error)
	}
	var st JobStatus
	if err := json.Unmarshal(resp.Output, &st); err != nil {
		return nil, fmt.Errorf("RunPod job %s: unreadable output: %v", id, err)
	}
	return &st, nil
}

// Cancel cancels a queued or running job.
func (r *RunPod) Cancel(ctx context.Context, id string) error {
	return r.do(ctx, http.MethodPost, "/cancel/"+id, nil, nil, 15*time.Second)
}

// RunPodHealth is GET /health: the endpoint's queue and workers.
type RunPodHealth struct {
	Jobs struct {
		Completed  int `json:"completed"`
		Failed     int `json:"failed"`
		InProgress int `json:"inProgress"`
		InQueue    int `json:"inQueue"`
		Retried    int `json:"retried"`
	} `json:"jobs"`
	Workers struct {
		Idle         int `json:"idle"`
		Initializing int `json:"initializing"`
		Ready        int `json:"ready"`
		Running      int `json:"running"`
		Throttled    int `json:"throttled"`
		Unhealthy    int `json:"unhealthy"`
	} `json:"workers"`
}

// Health reports the endpoint's queue and worker counts.
func (r *RunPod) Health(ctx context.Context) (*RunPodHealth, error) {
	var h RunPodHealth
	if err := r.do(ctx, http.MethodGet, "/health", nil, &h, 20*time.Second); err != nil {
		return nil, err
	}
	return &h, nil
}

func randHex(n int) string {
	b := make([]byte, n)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

// NewToken returns a random worker token.
func NewToken() string { return "lw_" + randHex(24) }
