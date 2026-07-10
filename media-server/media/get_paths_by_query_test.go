package media

import (
	"database/sql"
	"testing"
)

// seedTaggedMedia inserts a small fixture: two files tagged "Exchange Student"
// and one untagged file, so we can prove a tag query selects a subset rather
// than the whole table.
func seedTaggedMedia(t *testing.T, db *sql.DB) {
	t.Helper()
	rows := []struct {
		path, tag string
	}{
		{"/lib/a.jpg", "Exchange Student"},
		{"/lib/b.jpg", "Exchange Student"},
		{"/lib/c.jpg", ""}, // untagged — must NOT be selected by the tag query
	}
	for _, r := range rows {
		if _, err := db.Exec("INSERT INTO media (path) VALUES (?)", r.path); err != nil {
			t.Fatalf("insert media %s: %v", r.path, err)
		}
		if r.tag != "" {
			if _, err := db.Exec(
				"INSERT INTO media_tag_by_category (media_path, tag_label, category_label) VALUES (?, ?, ?)",
				r.path, r.tag, "people"); err != nil {
				t.Fatalf("insert tag for %s: %v", r.path, err)
			}
		}
	}
}

// TestGetPathsByQuery_QuotedMultiWordTag verifies that a properly quoted
// multi-word tag value selects exactly the tagged subset. This is the query the
// renderer now emits for tags containing spaces.
func TestGetPathsByQuery_QuotedMultiWordTag(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()
	seedTaggedMedia(t, db)

	paths, err := GetPathsByQuery(db, `tag:"Exchange Student"`)
	if err != nil {
		t.Fatalf("GetPathsByQuery() error = %v", err)
	}
	if len(paths) != 2 {
		t.Fatalf("quoted tag query selected %d paths, want 2 (%v)", len(paths), paths)
	}
}

// TestGetPathsByQuery_UnparseableQueryErrors is the regression guard for the
// reported bug: an unquoted multi-word tag (`tag:Exchange Student`) fails to
// parse. Previously this silently fell back to selecting the ENTIRE library,
// so a "describe these files" job ran against every item. It must now return an
// error so the calling job fails loudly instead.
func TestGetPathsByQuery_UnparseableQueryErrors(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()
	seedTaggedMedia(t, db)

	paths, err := GetPathsByQuery(db, "tag:Exchange Student")
	if err == nil {
		t.Fatalf("expected an error for an unparseable query, got %d paths (%v)", len(paths), paths)
	}
	if paths != nil {
		t.Fatalf("expected no paths on parse failure, got %v", paths)
	}
}

// seedNestedMedia inserts a directory with a file directly inside it plus a
// file in a subdirectory, so we can distinguish a one-level match from a
// recursive one.
func seedNestedMedia(t *testing.T, db *sql.DB) {
	t.Helper()
	paths := []string{
		"/lib/cats/top.jpg",        // directly in /lib/cats
		"/lib/cats/sub/nested.jpg", // one level deeper
		"/lib/dogs/other.jpg",      // sibling dir — never matched
	}
	for _, p := range paths {
		if _, err := db.Exec("INSERT INTO media (path) VALUES (?)", p); err != nil {
			t.Fatalf("insert media %s: %v", p, err)
		}
	}
}

// TestGetPathsByQuery_PathDirIsOneLevel pins the non-recursive contract the
// context palette relies on: pathdir matches only files directly in the
// directory, not files in subdirectories.
func TestGetPathsByQuery_PathDirIsOneLevel(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()
	seedNestedMedia(t, db)

	paths, err := GetPathsByQuery(db, `pathdir:"/lib/cats"`)
	if err != nil {
		t.Fatalf("GetPathsByQuery() error = %v", err)
	}
	if len(paths) != 1 || paths[0] != "/lib/cats/top.jpg" {
		t.Fatalf("pathdir matched %v, want only [/lib/cats/top.jpg]", paths)
	}
}

// TestGetPathsByQuery_WildcardPathIsRecursive pins the recursive contract: the
// trailing-wildcard path query the context palette emits when recursive
// browsing is on matches every file under the directory, at any depth.
func TestGetPathsByQuery_WildcardPathIsRecursive(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()
	seedNestedMedia(t, db)

	paths, err := GetPathsByQuery(db, `path:"/lib/cats/*"`)
	if err != nil {
		t.Fatalf("GetPathsByQuery() error = %v", err)
	}
	if len(paths) != 2 {
		t.Fatalf("wildcard path matched %d paths, want 2 (top + nested) (%v)", len(paths), paths)
	}
	for _, p := range paths {
		if p == "/lib/dogs/other.jpg" {
			t.Fatalf("wildcard path leaked a sibling directory: %v", paths)
		}
	}
}

// seedMixedMedia inserts videos and images in various transcript states so the
// filetype/transcript predicates the home page emits can be pinned.
func seedMixedMedia(t *testing.T, db *sql.DB) {
	t.Helper()
	rows := []struct {
		path       string
		transcript interface{}
	}{
		{"/lib/talk.MP4", nil},          // video, no transcript (uppercase ext)
		{"/lib/clip.webm", ""},          // video, empty transcript
		{"/lib/done.mkv", "WEBVTT ..."}, // video, transcribed
		{"/lib/photo.jpg", nil},         // image — never a transcript target
	}
	for _, r := range rows {
		if _, err := db.Exec(
			"INSERT INTO media (path, transcript) VALUES (?, ?)", r.path, r.transcript,
		); err != nil {
			t.Fatalf("insert media %s: %v", r.path, err)
		}
	}
}

// TestGetPathsByQuery_FiletypeVideo pins the filetype:video predicate: only
// video extensions match, case-insensitively.
func TestGetPathsByQuery_FiletypeVideo(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()
	seedMixedMedia(t, db)

	paths, err := GetPathsByQuery(db, "filetype:video")
	if err != nil {
		t.Fatalf("GetPathsByQuery() error = %v", err)
	}
	if len(paths) != 3 {
		t.Fatalf("filetype:video matched %d paths, want 3 (%v)", len(paths), paths)
	}
	for _, p := range paths {
		if p == "/lib/photo.jpg" {
			t.Fatalf("filetype:video matched an image: %v", paths)
		}
	}
}

// TestGetPathsByQuery_TranscriptTargeting pins the exact query the home page's
// Transcripts card emits: videos that still need a transcript — NULL or empty
// column — and nothing else. Before the transcript/filetype predicates existed,
// unknown keys compiled to 1=1 and this selected the entire library.
func TestGetPathsByQuery_TranscriptTargeting(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()
	seedMixedMedia(t, db)

	paths, err := GetPathsByQuery(db, `filetype:video AND (transcript:null OR transcript:"")`)
	if err != nil {
		t.Fatalf("GetPathsByQuery() error = %v", err)
	}
	want := map[string]bool{"/lib/talk.MP4": true, "/lib/clip.webm": true}
	if len(paths) != len(want) {
		t.Fatalf("transcript targeting matched %d paths, want %d (%v)", len(paths), len(want), paths)
	}
	for _, p := range paths {
		if !want[p] {
			t.Fatalf("transcript targeting selected unexpected path %s (%v)", p, paths)
		}
	}
}

// TestGetPathsByQuery_FacesUngrouped pins the faces:ungrouped predicate the
// People panel's Ungrouped card emits: only media whose detected faces are
// ALL unassigned match — one grouped face disqualifies the item (it already
// carries a person tag; secondary detections used to drag fully-tagged media
// into the Ungrouped view). Unknown values fail closed (unknown keys used to
// compile to 1=1 and select the whole library).
func TestGetPathsByQuery_FacesUngrouped(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()
	for _, p := range []string{"/lib/stray.jpg", "/lib/known.jpg", "/lib/mixed.jpg", "/lib/empty.jpg"} {
		if _, err := db.Exec("INSERT INTO media (path) VALUES (?)", p); err != nil {
			t.Fatal(err)
		}
	}
	// setupTestDB's manual schema has the face table but not the constraint
	// tables ReplaceFaces reconciles — create those minimally.
	for _, stmt := range []string{
		`CREATE TABLE IF NOT EXISTS face_veto (face_id INTEGER, person_id INTEGER)`,
		`CREATE TABLE IF NOT EXISTS face_cannot_link (face_a INTEGER, face_b INTEGER)`,
		`CREATE TABLE IF NOT EXISTS face_group_ban_member (ban_id INTEGER, face_id INTEGER)`,
	} {
		if _, err := db.Exec(stmt); err != nil {
			t.Fatal(err)
		}
	}
	// A stray face (never grouped) and a face assigned to a person.
	if _, err := ReplaceFaces(db, "/lib/stray.jpg", "m1", []NewFace{
		{X: 0.1, Y: 0.1, W: 0.2, H: 0.2, Score: 0.9, Vec: []float32{1, 0}},
	}, 1); err != nil {
		t.Fatal(err)
	}
	knownIDs, err := ReplaceFaces(db, "/lib/known.jpg", "m1", []NewFace{
		{X: 0.1, Y: 0.1, W: 0.2, H: 0.2, Score: 0.9, Vec: []float32{0, 1}},
	}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(
		`UPDATE face SET person_id = 7, assigned_by = 'user' WHERE id = ?`, knownIDs[0],
	); err != nil {
		t.Fatal(err)
	}
	// Mixed: the main face is grouped, a secondary detection isn't. The item
	// already carries its person tag, so it must NOT read as "ungrouped".
	mixedIDs, err := ReplaceFaces(db, "/lib/mixed.jpg", "m1", []NewFace{
		{X: 0.1, Y: 0.1, W: 0.4, H: 0.4, Score: 0.95, Vec: []float32{1, 1}},
		{X: 0.7, Y: 0.7, W: 0.1, H: 0.1, Score: 0.4, Vec: []float32{1, 2}},
	}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(
		`UPDATE face SET person_id = 7, assigned_by = 'auto' WHERE id = ?`, mixedIDs[0],
	); err != nil {
		t.Fatal(err)
	}

	paths, err := GetPathsByQuery(db, "faces:ungrouped")
	if err != nil {
		t.Fatalf("GetPathsByQuery() error = %v", err)
	}
	if len(paths) != 1 || paths[0] != "/lib/stray.jpg" {
		t.Fatalf("faces:ungrouped matched %v, want only [/lib/stray.jpg]", paths)
	}

	paths, err = GetPathsByQuery(db, "faces:bogus")
	if err != nil {
		t.Fatalf("GetPathsByQuery() error = %v", err)
	}
	if len(paths) != 0 {
		t.Fatalf("faces:bogus matched %d paths, want 0 (%v)", len(paths), paths)
	}
}

// TestGetPathsByQuery_FiletypeUnknownMatchesNothing guards the fail-closed
// contract: an unrecognized filetype value must select zero rows, not the
// whole library.
func TestGetPathsByQuery_FiletypeUnknownMatchesNothing(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()
	seedMixedMedia(t, db)

	paths, err := GetPathsByQuery(db, "filetype:audio")
	if err != nil {
		t.Fatalf("GetPathsByQuery() error = %v", err)
	}
	if len(paths) != 0 {
		t.Fatalf("filetype:audio matched %d paths, want 0 (%v)", len(paths), paths)
	}
}

// TestGetPathsByQuery_EmptyQuerySelectsAll confirms the legitimate "no filter"
// case still returns everything — only parse *failures* are treated as errors.
func TestGetPathsByQuery_EmptyQuerySelectsAll(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()
	seedTaggedMedia(t, db)

	paths, err := GetPathsByQuery(db, "")
	if err != nil {
		t.Fatalf("GetPathsByQuery() error = %v", err)
	}
	if len(paths) != 3 {
		t.Fatalf("empty query selected %d paths, want 3 (all) (%v)", len(paths), paths)
	}
}
