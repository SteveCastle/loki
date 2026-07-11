package main

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/stevecastle/shrike/media"
	"github.com/stevecastle/shrike/storage"
	_ "modernc.org/sqlite"
)

// memBackend is an in-memory storage.Backend standing in for an S3 root at
// s3://demo/. Upload/Download/Exists operate on a map; Contains gates by the
// root prefix.
type memBackend struct {
	root  string // e.g. "s3://demo/"
	name  string
	files map[string][]byte
}

func newMemBackend(root, name string) *memBackend {
	return &memBackend{root: root, name: name, files: map[string][]byte{}}
}
func (b *memBackend) List(context.Context, string) ([]storage.Entry, error) { return nil, nil }
func (b *memBackend) Scan(context.Context, string, bool) ([]storage.FileInfo, error) {
	return nil, nil
}
func (b *memBackend) Download(_ context.Context, p string) (io.ReadCloser, error) {
	if v, ok := b.files[p]; ok {
		return io.NopCloser(bytes.NewReader(v)), nil
	}
	return nil, os.ErrNotExist
}
func (b *memBackend) Upload(_ context.Context, p string, r io.Reader, _ string) error {
	data, err := io.ReadAll(r)
	if err != nil {
		return err
	}
	b.files[p] = data
	return nil
}
func (b *memBackend) MediaURL(p string) (string, error) { return p, nil }
func (b *memBackend) Exists(_ context.Context, p string) (bool, error) {
	_, ok := b.files[p]
	return ok, nil
}
func (b *memBackend) Contains(p string) bool {
	return len(p) >= len(b.root) && p[:len(b.root)] == b.root
}
func (b *memBackend) Root() storage.Entry {
	return storage.Entry{Name: b.name, Path: b.root, IsDir: true, Type: "s3"}
}

func xferSchema(t *testing.T) *sql.DB {
	t.Helper()
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	db.SetMaxOpenConns(1)
	t.Cleanup(func() { db.Close() })
	if err := media.InitializeSchema(db); err != nil {
		t.Fatal(err)
	}
	return db
}

// TestExportImportRoundTrip_LocalToS3: a local library with a tagged item +
// embedding + face exports, then imports into a fresh server whose only root
// is an S3 bucket. The item's paths must be rebased to s3://demo/…, the file
// bytes uploaded there, and the tag/embedding/face preserved. An untagged
// item must NOT be exported.
func TestExportImportRoundTrip_LocalToS3(t *testing.T) {
	// ---- source: local root /srcroot with two files, one tagged ----
	srcDir := t.TempDir()
	os.WriteFile(filepath.Join(srcDir, "a.jpg"), []byte("AAA-bytes"), 0o644)
	os.WriteFile(filepath.Join(srcDir, "untagged.jpg"), []byte("U"), 0o644)

	srcDB := xferSchema(t)
	aPath := filepath.Join(srcDir, "a.jpg")
	uPath := filepath.Join(srcDir, "untagged.jpg")
	srcDB.Exec(`INSERT INTO media(path, width, height) VALUES(?, 100, 200)`, aPath)
	srcDB.Exec(`INSERT INTO media(path) VALUES(?)`, uPath)
	srcDB.Exec(`INSERT INTO category(label, weight) VALUES('Places', 0)`)
	srcDB.Exec(`INSERT INTO tag(label, category_label, weight) VALUES('beach','Places',0)`)
	srcDB.Exec(`INSERT INTO media_tag_by_category(media_path, tag_label, category_label, weight, time_stamp) VALUES(?, 'beach','Places',0,0)`, aPath)
	vec := []byte{1, 2, 3, 4}
	srcDB.Exec(`INSERT INTO media_embedding(media_path, model, dim, vector, created_at) VALUES(?, 'siglip2-base-patch16-224', 1, ?, 0)`, aPath, vec)
	srcDB.Exec(`INSERT INTO person(name) VALUES('Alice')`)
	var personID int64
	srcDB.QueryRow(`SELECT id FROM person WHERE name='Alice'`).Scan(&personID)
	srcDB.Exec(`INSERT INTO face(media_path, model, frame_ts, bbox_x, bbox_y, bbox_w, bbox_h, det_score, vector, person_id, assigned_by, created_at)
		VALUES(?, 'sface', 0, 0.1, 0.1, 0.2, 0.2, 0.9, ?, ?, 'user', 0)`, aPath, vec, personID)

	srcReg := storage.NewRegistry([]storage.Backend{storage.NewLocalBackend(srcDir, "Local")})
	srcDeps := &Dependencies{DB: srcDB, Storage: srcReg}

	// ---- export ----
	rr := httptest.NewRecorder()
	exportHandler(srcDeps)(rr, httptest.NewRequest(http.MethodGet, "/api/export", nil))
	if rr.Code != http.StatusOK {
		t.Fatalf("export: %d %s", rr.Code, rr.Body.String())
	}
	archive := rr.Body.Bytes()
	if len(archive) == 0 {
		t.Fatal("empty archive")
	}

	// ---- destination: fresh DB, only an S3 root ----
	dstDB := xferSchema(t)
	mem := newMemBackend("s3://demo/", "Demo")
	dstReg := storage.NewRegistry([]storage.Backend{mem})
	dstReg.ReplaceWithDefault([]storage.Backend{mem}, 0)
	dstDeps := &Dependencies{DB: dstDB, Storage: dstReg}

	// ---- import (multipart file=archive) ----
	var body bytes.Buffer
	mw := multipart.NewWriter(&body)
	fw, _ := mw.CreateFormFile("file", "lib.lokiexport")
	fw.Write(archive)
	mw.Close()
	req := httptest.NewRequest(http.MethodPost, "/api/import", &body)
	req.Header.Set("Content-Type", mw.FormDataContentType())
	rr2 := httptest.NewRecorder()
	importHandler(dstDeps)(rr2, req)
	if rr2.Code != http.StatusOK {
		t.Fatalf("import: %d %s", rr2.Code, rr2.Body.String())
	}

	// ---- assertions on the destination ----
	wantPath := "s3://demo/a.jpg" // rebased from /srcroot/a.jpg (relkey a.jpg)

	var n int
	dstDB.QueryRow(`SELECT COUNT(*) FROM media`).Scan(&n)
	if n != 1 {
		t.Fatalf("dest media count = %d, want 1 (only the tagged item)", n)
	}
	var gotPath string
	if err := dstDB.QueryRow(`SELECT path FROM media`).Scan(&gotPath); err != nil || gotPath != wantPath {
		t.Fatalf("dest media path = %q (err %v), want %q", gotPath, err, wantPath)
	}
	// file bytes uploaded to the S3 backend at the rebased path
	if got, ok := mem.files[wantPath]; !ok || string(got) != "AAA-bytes" {
		t.Fatalf("file bytes at %s = %q (present=%v), want AAA-bytes", wantPath, got, ok)
	}
	// tag preserved, keyed by the new path
	var tagN int
	dstDB.QueryRow(`SELECT COUNT(*) FROM media_tag_by_category WHERE media_path = ? AND tag_label='beach'`, wantPath).Scan(&tagN)
	if tagN != 1 {
		t.Fatalf("tag not preserved for %s", wantPath)
	}
	// embedding preserved
	var embN int
	dstDB.QueryRow(`SELECT COUNT(*) FROM media_embedding WHERE media_path = ?`, wantPath).Scan(&embN)
	if embN != 1 {
		t.Fatalf("embedding not preserved for %s", wantPath)
	}
	// face preserved and linked to a remapped person named Alice
	var faceN int
	dstDB.QueryRow(`SELECT COUNT(*) FROM face f JOIN person p ON p.id=f.person_id WHERE f.media_path=? AND p.name='Alice'`, wantPath).Scan(&faceN)
	if faceN != 1 {
		t.Fatalf("face/person not preserved+remapped for %s", wantPath)
	}
}

// TestExportImportMergeSkipsExisting: importing the same archive twice must
// not duplicate — the second run skips the already-present item.
func TestExportImportMergeSkipsExisting(t *testing.T) {
	srcDir := t.TempDir()
	os.WriteFile(filepath.Join(srcDir, "a.jpg"), []byte("A"), 0o644)
	srcDB := xferSchema(t)
	aPath := filepath.Join(srcDir, "a.jpg")
	srcDB.Exec(`INSERT INTO media(path) VALUES(?)`, aPath)
	srcDB.Exec(`INSERT INTO media_tag_by_category(media_path, tag_label, category_label, weight, time_stamp) VALUES(?, 't','c',0,0)`, aPath)
	srcDeps := &Dependencies{DB: srcDB, Storage: storage.NewRegistry([]storage.Backend{storage.NewLocalBackend(srcDir, "L")})}

	rr := httptest.NewRecorder()
	exportHandler(srcDeps)(rr, httptest.NewRequest(http.MethodGet, "/api/export", nil))
	archive := rr.Body.Bytes()

	dstDB := xferSchema(t)
	mem := newMemBackend("s3://demo/", "Demo")
	reg := storage.NewRegistry([]storage.Backend{mem})
	reg.ReplaceWithDefault([]storage.Backend{mem}, 0)
	dstDeps := &Dependencies{DB: dstDB, Storage: reg}

	doImport := func() *importResult {
		var body bytes.Buffer
		mw := multipart.NewWriter(&body)
		fw, _ := mw.CreateFormFile("file", "lib.lokiexport")
		fw.Write(archive)
		mw.Close()
		req := httptest.NewRequest(http.MethodPost, "/api/import", &body)
		req.Header.Set("Content-Type", mw.FormDataContentType())
		rr := httptest.NewRecorder()
		importHandler(dstDeps)(rr, req)
		if rr.Code != http.StatusOK {
			t.Fatalf("import: %d %s", rr.Code, rr.Body.String())
		}
		var res importResult
		json.Unmarshal(rr.Body.Bytes(), &res)
		return &res
	}

	first := doImport()
	if first.Imported != 1 {
		t.Fatalf("first import: imported=%d, want 1", first.Imported)
	}
	second := doImport()
	if second.Imported != 0 || second.Skipped != 1 {
		t.Fatalf("second import: imported=%d skipped=%d, want 0/1", second.Imported, second.Skipped)
	}
	var n int
	dstDB.QueryRow(`SELECT COUNT(*) FROM media`).Scan(&n)
	if n != 1 {
		t.Fatalf("dest media count after 2 imports = %d, want 1 (no dup)", n)
	}
}
