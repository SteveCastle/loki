package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// POST /api/media/merge-metadata — the synchronous "Merge" behind the
// viewer's discrete context-palette selection: consolidate tags, embeddings,
// and the transcript (column + .vtt sidecar) onto the FIRST path, then delete
// the source files and erase their database references. Mirrors the Electron
// main-process mergeItemMetadata.

func mergeReq(t *testing.T, deps *Dependencies, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/api/media/merge-metadata",
		strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	lokiMediaMergeMetadataHandler(deps)(rec, req)
	return rec
}

func TestMediaMergeMetadataMergesAndDeletesSources(t *testing.T) {
	db := newFacesTestDB(t)
	deps := &Dependencies{DB: db}

	dir := t.TempDir()
	target := filepath.Join(dir, "keep.mp4")
	src := filepath.Join(dir, "dup.mp4")
	for _, p := range []string{target, src} {
		if err := os.WriteFile(p, []byte("media"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(dir, "dup.vtt"), []byte("WEBVTT dup"), 0o644); err != nil {
		t.Fatal(err)
	}

	mustExec := func(q string, args ...any) {
		t.Helper()
		if _, err := db.Exec(q, args...); err != nil {
			t.Fatal(err)
		}
	}
	mustExec(`INSERT INTO media (path) VALUES (?)`, target)
	mustExec(`INSERT INTO media (path, transcript) VALUES (?, 'from dup')`, src)
	mustExec(`INSERT INTO media_tag_by_category (media_path, tag_label, category_label, weight, time_stamp)
	          VALUES (?, 'sunset', 'Scene', 1, 0)`, target)
	mustExec(`INSERT INTO media_tag_by_category (media_path, tag_label, category_label, weight, time_stamp)
	          VALUES (?, 'sunset', 'Scene', 1, 0), (?, 'beach', 'Scene', 1, 0)`, src, src)
	mustExec(`INSERT INTO media_embedding (media_path, model, dim, vector) VALUES (?, 'm1', 1, x'aa')`, target)
	mustExec(`INSERT INTO media_embedding (media_path, model, dim, vector) VALUES (?, 'm1', 1, x'bb'), (?, 'm2', 1, x'cc')`, src, src)
	mustExec(`INSERT INTO battle (winner_path, loser_path, outcome) VALUES (?, ?, 1)`, src, target)

	rec := mergeReq(t, deps,
		`{"paths":["`+jsonEscape(target)+`","`+jsonEscape(src)+`"]}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("merge: %d %s", rec.Code, rec.Body.String())
	}
	var out struct {
		Tags       int64    `json:"tags"`
		Embeddings int64    `json:"embeddings"`
		Transcript bool     `json:"transcript"`
		Deleted    []string `json:"deleted"`
		Failed     []string `json:"failed"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	if out.Tags != 1 || out.Embeddings != 1 || !out.Transcript {
		t.Errorf("payload = %+v, want tags=1 embeddings=1 transcript=true", out)
	}
	if len(out.Deleted) != 1 || out.Deleted[0] != src || len(out.Failed) != 0 {
		t.Errorf("deleted/failed = %v / %v", out.Deleted, out.Failed)
	}

	// Disk: source and its sidecar gone; sidecar moved next to the target.
	if _, err := os.Stat(src); !os.IsNotExist(err) {
		t.Errorf("source file survived: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "dup.vtt")); !os.IsNotExist(err) {
		t.Errorf("source sidecar survived: %v", err)
	}
	if b, err := os.ReadFile(filepath.Join(dir, "keep.vtt")); err != nil || string(b) != "WEBVTT dup" {
		t.Errorf("moved sidecar = %q, %v", b, err)
	}

	countRows := func(q string, args ...any) int {
		t.Helper()
		var n int
		if err := db.QueryRow(q, args...).Scan(&n); err != nil {
			t.Fatal(err)
		}
		return n
	}
	// Target: tag union, own m1 vector kept, m2 copied, transcript filled.
	if n := countRows(`SELECT COUNT(*) FROM media_tag_by_category WHERE media_path = ?`, target); n != 2 {
		t.Errorf("target tags = %d, want 2", n)
	}
	var vec []byte
	if err := db.QueryRow(
		`SELECT vector FROM media_embedding WHERE media_path = ? AND model = 'm1'`, target,
	).Scan(&vec); err != nil || len(vec) != 1 || vec[0] != 0xaa {
		t.Errorf("target m1 vector = %x, %v — the target's own row must win", vec, err)
	}
	if n := countRows(`SELECT COUNT(*) FROM media_embedding WHERE media_path = ?`, target); n != 2 {
		t.Errorf("target embeddings = %d, want 2", n)
	}
	var transcript string
	if err := db.QueryRow(`SELECT transcript FROM media WHERE path = ?`, target).Scan(&transcript); err != nil || transcript != "from dup" {
		t.Errorf("target transcript = %q, %v", transcript, err)
	}
	// Source: every reference erased.
	for _, q := range []string{
		`SELECT COUNT(*) FROM media WHERE path = ?`,
		`SELECT COUNT(*) FROM media_tag_by_category WHERE media_path = ?`,
		`SELECT COUNT(*) FROM media_embedding WHERE media_path = ?`,
	} {
		if n := countRows(q, src); n != 0 {
			t.Errorf("source rows survived (%s): %d", q, n)
		}
	}
	if n := countRows(`SELECT COUNT(*) FROM battle WHERE winner_path = ? OR loser_path = ?`, src, src); n != 0 {
		t.Errorf("source battle rows survived: %d", n)
	}
}

func TestMediaMergeMetadataRejectsBadInput(t *testing.T) {
	deps := &Dependencies{DB: newFacesTestDB(t)}
	for _, body := range []string{`{}`, `{"paths":[]}`, `{"paths":["only-one"]}`, `not json`} {
		if rec := mergeReq(t, deps, body); rec.Code != http.StatusBadRequest {
			t.Errorf("body %q → %d, want 400", body, rec.Code)
		}
	}
}

// jsonEscape backslash-escapes a path for embedding in a JSON string literal
// (Windows temp paths contain backslashes).
func jsonEscape(s string) string {
	return strings.ReplaceAll(s, `\`, `\\`)
}
