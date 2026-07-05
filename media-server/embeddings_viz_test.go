package main

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stevecastle/shrike/embedvec"
)

func seedVizEmbedding(t *testing.T, deps *Dependencies, path, model string, vec []float32) {
	t.Helper()
	if _, err := deps.DB.Exec(
		`INSERT INTO media_embedding (media_path, model, dim, vector, created_at) VALUES (?, ?, ?, ?, 0)`,
		path, model, len(vec), embedvec.Encode(vec)); err != nil {
		t.Fatalf("insert embedding: %v", err)
	}
}

type projectionResp struct {
	Model    string       `json:"model"`
	Dim      int          `json:"dim"`
	Total    int          `json:"total"`
	Count    int          `json:"count"`
	Variance []float64    `json:"variance"`
	Paths    []string     `json:"paths"`
	Points   [][3]float32 `json:"points"`
}

func getProjection(t *testing.T, deps *Dependencies, query string) projectionResp {
	t.Helper()
	req := httptest.NewRequest(http.MethodGet, "/api/embeddings/projection"+query, nil)
	rr := httptest.NewRecorder()
	embeddingsProjectionHandler(deps).ServeHTTP(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d; body = %s", rr.Code, rr.Body.String())
	}
	var resp projectionResp
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("bad json: %v", err)
	}
	return resp
}

func TestEmbeddingsProjectionBasic(t *testing.T) {
	deps := newIndexTestDeps(t)
	rng := rand.New(rand.NewSource(5))
	for i := 0; i < 30; i++ {
		vec := []float32{rng.Float32(), rng.Float32(), rng.Float32(), rng.Float32()}
		seedVizEmbedding(t, deps, fmt.Sprintf("m-%02d.jpg", i), "viz-model", vec)
	}

	resp := getProjection(t, deps, "?model=viz-model")
	if resp.Model != "viz-model" || resp.Dim != 4 || resp.Total != 30 || resp.Count != 30 {
		t.Fatalf("unexpected header fields: %+v", resp)
	}
	if len(resp.Paths) != 30 || len(resp.Points) != 30 {
		t.Fatalf("expected 30 paths+points, got %d/%d", len(resp.Paths), len(resp.Points))
	}
	for _, p := range resp.Points {
		for _, x := range p {
			if x < -1.001 || x > 1.001 {
				t.Fatalf("point out of [-1,1] bounds: %v", p)
			}
		}
	}
}

func TestEmbeddingsProjectionSamplesDeterministically(t *testing.T) {
	deps := newIndexTestDeps(t)
	rng := rand.New(rand.NewSource(9))
	for i := 0; i < 50; i++ {
		vec := []float32{rng.Float32(), rng.Float32(), rng.Float32()}
		seedVizEmbedding(t, deps, fmt.Sprintf("s-%02d.jpg", i), "viz-model", vec)
	}

	a := getProjection(t, deps, "?model=viz-model&limit=10")
	b := getProjection(t, deps, "?model=viz-model&limit=10")
	if a.Count != 10 || a.Total != 50 {
		t.Fatalf("expected count 10 of total 50, got %d of %d", a.Count, a.Total)
	}
	if strings.Join(a.Paths, "|") != strings.Join(b.Paths, "|") {
		t.Errorf("sample not deterministic across calls:\n%v\n%v", a.Paths, b.Paths)
	}
	for i := range a.Points {
		if a.Points[i] != b.Points[i] {
			t.Fatalf("projection not deterministic at %d: %v vs %v", i, a.Points[i], b.Points[i])
		}
	}
}

func TestEmbeddingsProjectionSkipsMismatchedDims(t *testing.T) {
	deps := newIndexTestDeps(t)
	seedVizEmbedding(t, deps, "a.jpg", "viz-model", []float32{1, 0, 0})
	seedVizEmbedding(t, deps, "b.jpg", "viz-model", []float32{0, 1, 0})
	seedVizEmbedding(t, deps, "stray.jpg", "viz-model", []float32{1, 0}) // wrong dim

	resp := getProjection(t, deps, "?model=viz-model")
	if resp.Total != 2 || resp.Count != 2 {
		t.Fatalf("expected the dim-2 stray to be dropped, got total %d count %d", resp.Total, resp.Count)
	}
	for _, p := range resp.Paths {
		if p == "stray.jpg" {
			t.Fatalf("stray row leaked into projection: %v", resp.Paths)
		}
	}
}

func TestEmbeddingsProjectionEmptyModel(t *testing.T) {
	deps := newIndexTestDeps(t)
	resp := getProjection(t, deps, "?model=nothing-here")
	if resp.Total != 0 || resp.Count != 0 {
		t.Fatalf("expected empty projection, got %+v", resp)
	}
}

func TestEmbeddingsProjectionRejectsBadLimit(t *testing.T) {
	deps := newIndexTestDeps(t)
	req := httptest.NewRequest(http.MethodGet, "/api/embeddings/projection?limit=bogus", nil)
	rr := httptest.NewRecorder()
	embeddingsProjectionHandler(deps).ServeHTTP(rr, req)
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for bad limit, got %d", rr.Code)
	}
}

func TestEmbeddingsVizPageServed(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/viz/embeddings", nil)
	rr := httptest.NewRecorder()
	embeddingsVizPageHandler().ServeHTTP(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d", rr.Code)
	}
	if ct := rr.Header().Get("Content-Type"); !strings.HasPrefix(ct, "text/html") {
		t.Errorf("content-type = %q", ct)
	}
	body := rr.Body.String()
	for _, want := range []string{"<canvas", "/api/embeddings/projection", "/api/index/models"} {
		if !strings.Contains(body, want) {
			t.Errorf("page missing %q", want)
		}
	}
}
