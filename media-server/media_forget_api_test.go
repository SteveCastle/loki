package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stevecastle/shrike/media"
	_ "modernc.org/sqlite"
)

// /api/media/forget and /api/media/delete share one cleanup routine
// (eraseMediaReferences): erase every database reference to a path without
// touching the file — media row, tags, embeddings, faces plus the assertions
// keyed by their ids, scan markers, and the battle log. Delete used to leave
// faces, assertions, and battles behind; this scenario now pins both to the
// full contract.
func runEraseEveryReferenceTest(
	t *testing.T, makeHandler func(*Dependencies) http.HandlerFunc, url string,
) {
	t.Helper()
	db := newFacesTestDB(t)
	deps := &Dependencies{DB: db}
	handler := makeHandler(deps)

	const gone = "/photos/gone.jpg"
	const kept = "/photos/kept.jpg"
	for _, p := range []string{gone, kept} {
		if _, err := db.Exec(`INSERT INTO media (path) VALUES (?)`, p); err != nil {
			t.Fatal(err)
		}
		if _, err := db.Exec(
			`INSERT INTO media_tag_by_category (media_path, tag_label, category_label, weight, time_stamp)
			 VALUES (?, 'sunset', 'Subject', 1, 0)`, p,
		); err != nil {
			t.Fatal(err)
		}
		if _, err := db.Exec(
			`INSERT INTO media_embedding (media_path, model, dim, vector) VALUES (?, 'siglip2', 2, x'0000')`, p,
		); err != nil {
			t.Fatal(err)
		}
		if _, err := db.Exec(
			`INSERT INTO face_scan (media_path, model, face_count) VALUES (?, 'sface', 1)`, p,
		); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := db.Exec(
		`INSERT INTO battle (winner_path, loser_path, outcome) VALUES (?, ?, 1), (?, ?, 1)`,
		gone, kept, kept, gone,
	); err != nil {
		t.Fatal(err)
	}

	// A face on each item, both in the same person, plus the curation
	// assertions and cover pointer keyed by the doomed face id.
	alice, err := media.CreatePerson(db, "Alice")
	if err != nil {
		t.Fatal(err)
	}
	goneFaces, err := media.ReplaceFaces(db, gone, "sface", []media.NewFace{
		{X: 0.1, Y: 0.1, W: 0.2, H: 0.2, Score: 0.9, Vec: []float32{1, 0}},
	}, 1)
	if err != nil {
		t.Fatal(err)
	}
	keptFaces, err := media.ReplaceFaces(db, kept, "sface", []media.NewFace{
		{X: 0.1, Y: 0.1, W: 0.2, H: 0.2, Score: 0.9, Vec: []float32{0, 1}},
	}, 1)
	if err != nil {
		t.Fatal(err)
	}
	goneFace, keptFace := goneFaces[0], keptFaces[0]
	if err := media.AssignFace(db, goneFace, alice, "auto"); err != nil {
		t.Fatal(err)
	}
	if err := media.AssignFace(db, keptFace, alice, "user"); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(`UPDATE person SET cover_face_id = ? WHERE id = ?`, goneFace, alice); err != nil {
		t.Fatal(err)
	}
	if err := media.AddFaceVeto(db, goneFace, 999); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(
		`INSERT INTO face_cannot_link (face_a, face_b) VALUES (?, ?)`, goneFace, keptFace,
	); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(
		`INSERT INTO face_group_ban_member (ban_id, face_id) VALUES (1, ?)`, goneFace,
	); err != nil {
		t.Fatal(err)
	}

	req := httptest.NewRequest(http.MethodPost, url,
		strings.NewReader(`{"path":"`+gone+`"}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	handler(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("%s: %d %s", url, rec.Code, rec.Body.String())
	}
	var out map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	// tags = 2: the seeded 'sunset' plus the People-category bridge row that
	// assigning the face to Alice wrote — a person tag is a tag, and it has to
	// go with the item.
	for key, want := range map[string]float64{
		"media": 1, "tags": 2, "embeddings": 1, "faces": 1, "battles": 2,
	} {
		if got, _ := out[key].(float64); got != want {
			t.Errorf("payload %q = %v, want %v (full: %v)", key, out[key], want, out)
		}
	}

	countRows := func(query string, args ...any) int {
		t.Helper()
		var n int
		if err := db.QueryRow(query, args...).Scan(&n); err != nil {
			t.Fatal(err)
		}
		return n
	}
	for _, c := range []struct {
		name     string
		query    string
		wantKept int
	}{
		{"media", `SELECT COUNT(*) FROM media WHERE path = ?`, 1},
		{"tags", `SELECT COUNT(*) FROM media_tag_by_category WHERE media_path = ?`, 2},
		{"embeddings", `SELECT COUNT(*) FROM media_embedding WHERE media_path = ?`, 1},
		{"faces", `SELECT COUNT(*) FROM face WHERE media_path = ?`, 1},
		{"face_scan", `SELECT COUNT(*) FROM face_scan WHERE media_path = ?`, 1},
	} {
		if n := countRows(c.query, gone); n != 0 {
			t.Errorf("%s rows survived: %d", c.name, n)
		}
		// Every one of them is still there for the OTHER item.
		if n := countRows(c.query, kept); n != c.wantKept {
			t.Errorf("%s rows for the kept item = %d, want %d", c.name, n, c.wantKept)
		}
	}
	if n := countRows(`SELECT COUNT(*) FROM battle WHERE winner_path = ? OR loser_path = ?`, gone, gone); n != 0 {
		t.Errorf("battle rows survived: %d", n)
	}
	// Assertions keyed by the deleted face id can never be resolved again.
	if n := countRows(`SELECT COUNT(*) FROM face_veto WHERE face_id = ?`, goneFace); n != 0 {
		t.Errorf("face_veto rows survived: %d", n)
	}
	if n := countRows(
		`SELECT COUNT(*) FROM face_cannot_link WHERE face_a = ? OR face_b = ?`, goneFace, goneFace,
	); n != 0 {
		t.Errorf("face_cannot_link rows survived: %d", n)
	}
	if n := countRows(`SELECT COUNT(*) FROM face_group_ban_member WHERE face_id = ?`, goneFace); n != 0 {
		t.Errorf("face_group_ban_member rows survived: %d", n)
	}
	// The person survives with the dangling cover cleared.
	var cover any
	if err := db.QueryRow(`SELECT cover_face_id FROM person WHERE id = ?`, alice).Scan(&cover); err != nil {
		t.Fatalf("person row gone: %v", err)
	}
	if cover != nil {
		t.Errorf("cover_face_id still points at the deleted face: %v", cover)
	}
}

func TestMediaForgetErasesEveryReference(t *testing.T) {
	runEraseEveryReferenceTest(t, lokiMediaForgetHandler, "/api/media/forget")
}

func TestMediaDeleteErasesEveryReference(t *testing.T) {
	runEraseEveryReferenceTest(t, lokiMediaDeleteHandler, "/api/media/delete")
}

func TestMediaForgetAndDeleteRejectEmptyPath(t *testing.T) {
	db := newFacesTestDB(t)
	for name, handler := range map[string]http.HandlerFunc{
		"forget": lokiMediaForgetHandler(&Dependencies{DB: db}),
		"delete": lokiMediaDeleteHandler(&Dependencies{DB: db}),
	} {
		for _, body := range []string{`{}`, `{"path":"  "}`, `not json`} {
			req := httptest.NewRequest(http.MethodPost, "/api/media/"+name, strings.NewReader(body))
			req.Header.Set("Content-Type", "application/json")
			rec := httptest.NewRecorder()
			handler(rec, req)
			if rec.Code != http.StatusBadRequest {
				t.Errorf("%s: body %q → %d, want 400", name, body, rec.Code)
			}
		}
	}
}
