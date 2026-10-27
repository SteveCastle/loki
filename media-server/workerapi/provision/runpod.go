// Package provision creates and removes cloud GPU workers running the
// loki-worker image. RunPod is supported directly (persistent pods, and
// queue-based serverless endpoints); elsewhere (AWS, GCP, any Docker host)
// the image runs as-is and is added to a pool by URL (tools/worker/README.md).
package provision

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// DefaultImage is the published worker image.
const DefaultImage = "ghcr.io/stevecastle/loki-worker:latest"

// DefaultGPUs are RunPod GPU types every engine runs on (FP8 tensor cores,
// 24 GB or more), cheapest first; RunPod rents whichever is available.
var DefaultGPUs = []string{"NVIDIA GeForce RTX 4090", "NVIDIA L40S", "NVIDIA RTX 6000 Ada Generation", "NVIDIA GeForce RTX 5090", "NVIDIA H100 80GB HBM3"}

// WorkerPort is the port the image serves on.
const WorkerPort = 8080

// RunPod is a client of RunPod's REST API (https://rest.runpod.io/v1).
type RunPod struct {
	APIKey string
	Base   string // tests
	HTTP   *http.Client
}

func (r *RunPod) do(ctx context.Context, method, p string, in, out any) error {
	ctx, cancel := context.WithTimeout(ctx, 60*time.Second)
	defer cancel()
	var body io.Reader
	if in != nil {
		b, err := json.Marshal(in)
		if err != nil {
			return err
		}
		body = bytes.NewReader(b)
	}
	base := r.Base
	if base == "" {
		base = "https://rest.runpod.io/v1"
	}
	req, err := http.NewRequestWithContext(ctx, method, strings.TrimRight(base, "/")+p, body)
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
	b, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if resp.StatusCode/100 != 2 {
		return fmt.Errorf("RunPod %s %s: %d %s", method, p, resp.StatusCode, strings.TrimSpace(string(b)))
	}
	if out != nil && len(bytes.TrimSpace(b)) > 0 {
		return json.Unmarshal(b, out)
	}
	return nil
}

// PodSpec describes persistent worker pods.
type PodSpec struct {
	Name     string
	Image    string
	GPUs     []string
	GPUCount int
	// Cloud is SECURE (default) or COMMUNITY.
	Cloud string
	// VolumeGB is the pod volume mounted at /models: the models stay across
	// pod restarts (ignored with NetworkVolumeID).
	VolumeGB        int
	ContainerDiskGB int
	NetworkVolumeID string
	DataCenters     []string
	Token           string
	// Preload lists engines whose models load at start (LOKI_WORKER_PRELOAD).
	Preload []string
	Env     map[string]string
	// RegistryAuthID is a RunPod registry credential for a private image.
	RegistryAuthID string
}

// Pod is a created pod.
type Pod struct {
	ID   string `json:"id"`
	Name string `json:"name"`
	// URL is the worker's address through RunPod's HTTPS proxy.
	URL           string `json:"url"`
	DesiredStatus string `json:"desiredStatus,omitempty"`
}

// PodURL is the HTTPS proxy address of a pod's worker port.
func PodURL(id string) string { return fmt.Sprintf("https://%s-%d.proxy.runpod.net", id, WorkerPort) }

// CreatePod starts one worker pod.
func (r *RunPod) CreatePod(ctx context.Context, s PodSpec) (*Pod, error) {
	if s.Image == "" {
		s.Image = DefaultImage
	}
	if len(s.GPUs) == 0 {
		s.GPUs = DefaultGPUs
	}
	env := map[string]string{"LOKI_WORKER_TOKEN": s.Token, "LOKI_MODELS": "/models"}
	if len(s.Preload) > 0 {
		env["LOKI_WORKER_PRELOAD"] = strings.Join(s.Preload, ",")
	}
	for k, v := range s.Env {
		env[k] = v
	}
	body := map[string]any{
		"name":              s.Name,
		"imageName":         s.Image,
		"gpuTypeIds":        s.GPUs,
		"gpuCount":          max(1, s.GPUCount),
		"cloudType":         firstNonEmpty(s.Cloud, "SECURE"),
		"containerDiskInGb": max(s.ContainerDiskGB, 30),
		"volumeMountPath":   "/models",
		"ports":             []string{fmt.Sprintf("%d/http", WorkerPort)},
		"env":               env,
	}
	if s.NetworkVolumeID != "" {
		body["networkVolumeId"] = s.NetworkVolumeID
	} else {
		body["volumeInGb"] = max(s.VolumeGB, 100)
	}
	if len(s.DataCenters) > 0 {
		body["dataCenterIds"] = s.DataCenters
	}
	if s.RegistryAuthID != "" {
		body["containerRegistryAuthId"] = s.RegistryAuthID
	}
	var p Pod
	if err := r.do(ctx, http.MethodPost, "/pods", body, &p); err != nil {
		return nil, err
	}
	if p.ID == "" {
		return nil, fmt.Errorf("RunPod created no pod")
	}
	p.URL = PodURL(p.ID)
	return &p, nil
}

// DeletePod terminates a pod (its pod volume goes with it).
func (r *RunPod) DeletePod(ctx context.Context, id string) error {
	return r.do(ctx, http.MethodDelete, "/pods/"+id, nil, nil)
}

// ServerlessSpec describes a queue-based serverless endpoint.
type ServerlessSpec struct {
	Name       string
	Image      string
	GPUs       []string
	WorkersMin int
	WorkersMax int
	// IdleSeconds keeps a worker (and its loaded models) around after a job.
	IdleSeconds     int
	ExecutionSecs   int
	NetworkVolumeID string
	ContainerDiskGB int
	Flashboot       bool
	Env             map[string]string
	RegistryAuthID  string
}

// Endpoint is a created serverless endpoint.
type Endpoint struct {
	ID         string `json:"id"`
	TemplateID string `json:"templateId"`
	Name       string `json:"name"`
}

// CreateServerless creates a template for the image and an endpoint on it.
func (r *RunPod) CreateServerless(ctx context.Context, s ServerlessSpec) (*Endpoint, error) {
	if s.Image == "" {
		s.Image = DefaultImage
	}
	if len(s.GPUs) == 0 {
		s.GPUs = DefaultGPUs
	}
	env := map[string]string{}
	for k, v := range s.Env {
		env[k] = v
	}
	tmpl := map[string]any{
		"name":              s.Name + "-" + fmt.Sprint(time.Now().Unix()),
		"imageName":         s.Image,
		"isServerless":      true,
		"containerDiskInGb": max(s.ContainerDiskGB, 30),
		"env":               env,
		"readme":            "Lowkey loki-worker (queue-based serverless). Models live on the network volume (/runpod-volume/models).",
	}
	if s.RegistryAuthID != "" {
		tmpl["containerRegistryAuthId"] = s.RegistryAuthID
	}
	var t struct {
		ID string `json:"id"`
	}
	if err := r.do(ctx, http.MethodPost, "/templates", tmpl, &t); err != nil {
		return nil, err
	}
	ep := map[string]any{
		"name":               s.Name,
		"templateId":         t.ID,
		"gpuTypeIds":         s.GPUs,
		"workersMin":         max(0, s.WorkersMin),
		"workersMax":         max(1, s.WorkersMax),
		"idleTimeout":        min(max(s.IdleSeconds, 5), 3600),
		"executionTimeoutMs": max(s.ExecutionSecs, 600) * 1000,
		"flashboot":          s.Flashboot,
		"scalerType":         "QUEUE_DELAY",
		"scalerValue":        4,
	}
	if s.NetworkVolumeID != "" {
		ep["networkVolumeId"] = s.NetworkVolumeID
	}
	var e Endpoint
	if err := r.do(ctx, http.MethodPost, "/endpoints", ep, &e); err != nil {
		_ = r.DeleteTemplate(ctx, t.ID)
		return nil, err
	}
	e.TemplateID = t.ID
	return &e, nil
}

// DeleteEndpoint removes a serverless endpoint.
func (r *RunPod) DeleteEndpoint(ctx context.Context, id string) error {
	return r.do(ctx, http.MethodDelete, "/endpoints/"+id, nil, nil)
}

// DeleteTemplate removes a template.
func (r *RunPod) DeleteTemplate(ctx context.Context, id string) error {
	return r.do(ctx, http.MethodDelete, "/templates/"+id, nil, nil)
}

func firstNonEmpty(a ...string) string {
	for _, s := range a {
		if s != "" {
			return s
		}
	}
	return ""
}
