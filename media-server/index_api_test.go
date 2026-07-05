package main

import (
	"database/sql"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stevecastle/shrike/embedindex"
	"github.com/stevecastle/shrike/embedvec"
	"github.com/stevecastle/shrike/tasks"
)

func newIndexTestDeps(t *testing.T) *Dependencies {
	t.Helper()
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatalf("open db: %v", err)
	}
	t.Cleanup(func() { db.Close() })
	for _, stmt := range []string{
		`CREATE TABLE media (path TEXT PRIMARY KEY)`,
		`CREATE TABLE media_embedding (
			media_path TEXT NOT NULL, model TEXT NOT NULL, dim INTEGER NOT NULL,
			vector BLOB NOT NULL, created_at INTEGER,
			PRIMARY KEY (media_path, model))`,
	} {
		if _, err := db.Exec(stmt); err != nil {
			t.Fatalf("create table: %v", err)
		}
	}
	return &Dependencies{DB: db}
}

// otherEmbedModel returns a registered model different from active, so tests
// hold regardless of which model the host's config marks active.
func otherEmbedModel(active string) string {
	if active == "dinov2-base" {
		return "siglip2-base-patch16-224"
	}
	return "dinov2-base"
}

func seedEmbedding(t *testing.T, deps *Dependencies, path, model string, withMedia bool) {
	t.Helper()
	if withMedia {
		if _, err := deps.DB.Exec(`INSERT OR IGNORE INTO media (path) VALUES (?)`, path); err != nil {
			t.Fatalf("insert media: %v", err)
		}
	}
	blob := embedvec.Encode([]float32{1, 0, 0})
	if _, err := deps.DB.Exec(
		`INSERT INTO media_embedding (media_path, model, dim, vector, created_at) VALUES (?, ?, 3, ?, 42)`,
		path, model, blob); err != nil {
		t.Fatalf("insert embedding: %v", err)
	}
}

func TestIndexStatus(t *testing.T) {
	deps := newIndexTestDeps(t)
	activeModel := tasks.ActiveEmbedModel().ID
	seedEmbedding(t, deps, "a.jpg", activeModel, true)
	seedEmbedding(t, deps, "b.jpg", otherEmbedModel(activeModel), true)
	seedEmbedding(t, deps, "gone.jpg", activeModel, false) // orphan
	deps.DB.Exec(`INSERT INTO media (path) VALUES ('never-embedded.jpg')`)

	req := httptest.NewRequest(http.MethodGet, "/api/index/status", nil)
	rr := httptest.NewRecorder()
	indexStatusHandler(deps).ServeHTTP(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d; body = %s", rr.Code, rr.Body.String())
	}
	var resp struct {
		ActiveModel        string `json:"active_model"`
		MediaTotal         int    `json:"media_total"`
		MissingActiveModel int    `json:"missing_active_model"`
		Orphaned           int    `json:"orphaned"`
		Embeddings         []struct {
			Model string `json:"model"`
			Count int    `json:"count"`
			Dim   int    `json:"dim"`
		} `json:"embeddings"`
	}
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("bad JSON: %v", err)
	}
	if resp.ActiveModel != activeModel {
		t.Errorf("active_model = %q, want %q", resp.ActiveModel, activeModel)
	}
	// media: a.jpg, b.jpg, never-embedded.jpg
	if resp.MediaTotal != 3 {
		t.Errorf("media_total = %d, want 3", resp.MediaTotal)
	}
	// b.jpg (other model) + never-embedded.jpg lack an active-model embedding
	if resp.MissingActiveModel != 2 {
		t.Errorf("missing_active_model = %d, want 2", resp.MissingActiveModel)
	}
	if resp.Orphaned != 1 {
		t.Errorf("orphaned = %d, want 1", resp.Orphaned)
	}
	if len(resp.Embeddings) != 2 {
		t.Errorf("embeddings groups = %d, want 2 (%+v)", len(resp.Embeddings), resp.Embeddings)
	}
}

func TestIndexModels(t *testing.T) {
	deps := newIndexTestDeps(t)
	req := httptest.NewRequest(http.MethodGet, "/api/index/models", nil)
	rr := httptest.NewRecorder()
	indexModelsHandler(deps).ServeHTTP(rr, req)
	var resp struct {
		Models []struct {
			ID     string `json:"id"`
			Dim    int    `json:"dim"`
			Active bool   `json:"active"`
		} `json:"models"`
	}
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil || len(resp.Models) < 2 {
		t.Fatalf("models = %+v, err = %v", resp.Models, err)
	}
	activeCount := 0
	for _, m := range resp.Models {
		if m.Active {
			activeCount++
		}
		if m.Dim == 0 || m.ID == "" {
			t.Errorf("incomplete model entry: %+v", m)
		}
	}
	if activeCount != 1 {
		t.Errorf("active models = %d, want exactly 1", activeCount)
	}
}

func TestIndexRebuild(t *testing.T) {
	deps := newIndexTestDeps(t)
	t.Cleanup(func() { tasks.SetVectorIndex(nil) })
	activeModel := tasks.ActiveEmbedModel().ID
	seedEmbedding(t, deps, "a.jpg", activeModel, true)
	seedEmbedding(t, deps, "b.jpg", activeModel, true)

	req := httptest.NewRequest(http.MethodPost, "/api/index/rebuild", nil)
	rr := httptest.NewRecorder()
	indexRebuildHandler(deps).ServeHTTP(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d; body = %s", rr.Code, rr.Body.String())
	}
	var resp struct {
		Model   string `json:"model"`
		Vectors int    `json:"vectors"`
	}
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("bad JSON: %v", err)
	}
	if resp.Model != activeModel || resp.Vectors != 2 {
		t.Errorf("rebuild = %+v, want model %s with 2 vectors", resp, activeModel)
	}
	if got := tasks.IndexSize(); got != 2 {
		t.Errorf("IndexSize() = %d, want 2", got)
	}
}

func TestIndexMissing(t *testing.T) {
	deps := newIndexTestDeps(t)
	activeModel := tasks.ActiveEmbedModel().ID
	seedEmbedding(t, deps, "a.jpg", activeModel, true)
	deps.DB.Exec(`INSERT INTO media (path) VALUES ('x.jpg'), ('y.jpg')`)

	req := httptest.NewRequest(http.MethodGet, "/api/index/missing?limit=1", nil)
	rr := httptest.NewRecorder()
	indexMissingHandler(deps).ServeHTTP(rr, req)
	var resp struct {
		Model        string   `json:"model"`
		TotalMissing int      `json:"total_missing"`
		Paths        []string `json:"paths"`
	}
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("bad JSON: %v; body = %s", err, rr.Body.String())
	}
	if resp.Model != activeModel || resp.TotalMissing != 2 || len(resp.Paths) != 1 {
		t.Errorf("missing = %+v, want total 2 with 1 path", resp)
	}

	// Unknown model is a 400.
	req = httptest.NewRequest(http.MethodGet, "/api/index/missing?model=nope", nil)
	rr = httptest.NewRecorder()
	indexMissingHandler(deps).ServeHTTP(rr, req)
	if rr.Code != http.StatusBadRequest {
		t.Errorf("unknown model: status = %d, want 400", rr.Code)
	}
}

func TestEmbeddingsGetAndDelete(t *testing.T) {
	deps := newIndexTestDeps(t)
	activeModel := tasks.ActiveEmbedModel().ID
	secondModel := otherEmbedModel(activeModel)
	seedEmbedding(t, deps, "a.jpg", activeModel, true)
	seedEmbedding(t, deps, "a.jpg", secondModel, true)

	// GET with vector
	req := httptest.NewRequest(http.MethodGet, "/api/embeddings?path=a.jpg&vector=true", nil)
	rr := httptest.NewRecorder()
	embeddingsHandler(deps).ServeHTTP(rr, req)
	var resp struct {
		Embeddings []struct {
			Model     string    `json:"model"`
			Dim       int       `json:"dim"`
			CreatedAt int64     `json:"created_at"`
			Vector    []float32 `json:"vector"`
		} `json:"embeddings"`
	}
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil || len(resp.Embeddings) != 2 {
		t.Fatalf("embeddings = %+v, err = %v", resp.Embeddings, err)
	}
	if len(resp.Embeddings[0].Vector) != 3 {
		t.Errorf("vector len = %d, want 3", len(resp.Embeddings[0].Vector))
	}

	// DELETE one model only
	req = httptest.NewRequest(http.MethodDelete, "/api/embeddings?path=a.jpg&model="+secondModel, nil)
	rr = httptest.NewRecorder()
	embeddingsHandler(deps).ServeHTTP(rr, req)
	var del struct {
		Deleted int64 `json:"deleted"`
	}
	if err := json.Unmarshal(rr.Body.Bytes(), &del); err != nil || del.Deleted != 1 {
		t.Fatalf("deleted = %+v, err = %v; body = %s", del, err, rr.Body.String())
	}

	// Missing path param is a 400.
	req = httptest.NewRequest(http.MethodGet, "/api/embeddings", nil)
	rr = httptest.NewRecorder()
	embeddingsHandler(deps).ServeHTTP(rr, req)
	if rr.Code != http.StatusBadRequest {
		t.Errorf("no path: status = %d, want 400", rr.Code)
	}
}

func TestEmbeddingsPrune(t *testing.T) {
	deps := newIndexTestDeps(t)
	activeModel := tasks.ActiveEmbedModel().ID
	seedEmbedding(t, deps, "kept.jpg", activeModel, true)
	seedEmbedding(t, deps, "gone.jpg", activeModel, false)
	seedEmbedding(t, deps, "gone.jpg", otherEmbedModel(activeModel), false)

	// Install an index holding the orphan so prune's IndexDelete is exercised.
	idx := embedindex.New()
	idx.Add("kept.jpg", []float32{1, 0, 0})
	idx.Add("gone.jpg", []float32{0, 1, 0})
	tasks.SetVectorIndexForModel(idx, activeModel)
	t.Cleanup(func() { tasks.SetVectorIndex(nil) })

	req := httptest.NewRequest(http.MethodPost, "/api/embeddings/prune", nil)
	rr := httptest.NewRecorder()
	embeddingsPruneHandler(deps).ServeHTTP(rr, req)
	var resp struct {
		PrunedRows  int64 `json:"pruned_rows"`
		PrunedPaths int   `json:"pruned_paths"`
	}
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("bad JSON: %v; body = %s", err, rr.Body.String())
	}
	if resp.PrunedRows != 2 || resp.PrunedPaths != 1 {
		t.Errorf("prune = %+v, want 2 rows / 1 path", resp)
	}
	var left int
	deps.DB.QueryRow(`SELECT COUNT(*) FROM media_embedding`).Scan(&left)
	if left != 1 {
		t.Errorf("remaining embeddings = %d, want 1", left)
	}
	if got := tasks.IndexSize(); got != 1 {
		t.Errorf("IndexSize() after prune = %d, want 1", got)
	}
}
