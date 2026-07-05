package media

import (
	"database/sql"
	"testing"

	_ "modernc.org/sqlite"
)

func newPeopleDB(t *testing.T) *sql.DB {
	t.Helper()
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	if err := InitializeSchema(db); err != nil {
		t.Fatal(err)
	}
	// Bridge rows are only written for library media (older DBs enforce a
	// media_path foreign key), so tests register their paths as media rows.
	for _, p := range []string{"a.jpg", "b.jpg"} {
		if _, err := db.Exec(`INSERT INTO media (path) VALUES (?)`, p); err != nil {
			t.Fatal(err)
		}
	}
	return db
}

// bridgeRows returns (path, tag, ts) rows in the People category.
func bridgeRows(t *testing.T, db *sql.DB) map[string]int {
	t.Helper()
	rows, err := db.Query(`SELECT media_path, tag_label, time_stamp FROM media_tag_by_category WHERE category_label = ?`, PeopleCategory)
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	out := map[string]int{}
	for rows.Next() {
		var p, tag string
		var ts float64
		if err := rows.Scan(&p, &tag, &ts); err != nil {
			t.Fatal(err)
		}
		out[p+"|"+tag]++
	}
	return out
}

func tagCategory(t *testing.T, db *sql.DB, label string) (string, bool) {
	t.Helper()
	var cat sql.NullString
	err := db.QueryRow(`SELECT category_label FROM tag WHERE label = ?`, label).Scan(&cat)
	if err == sql.ErrNoRows {
		return "", false
	}
	if err != nil {
		t.Fatal(err)
	}
	return cat.String, true
}

func TestCreatePersonAndTag(t *testing.T) {
	db := newPeopleDB(t)
	id, err := CreatePerson(db, "Alice")
	if err != nil {
		t.Fatal(err)
	}
	if id == 0 {
		t.Fatal("zero id")
	}
	if cat, ok := tagCategory(t, db, "Alice"); !ok || cat != PeopleCategory {
		t.Fatalf("tag = (%q, %v), want People tag", cat, ok)
	}
	// Duplicate name rejected.
	if _, err := CreatePerson(db, "Alice"); err == nil {
		t.Fatal("duplicate person allowed")
	}
	// A name owned by a curated tag in another category is stored with the
	// _cluster suffix instead of erroring — the curated tag and the face
	// cluster deliberately coexist.
	if _, err := db.Exec(`INSERT INTO tag (label, category_label) VALUES ('blonde', 'Appearance')`); err != nil {
		t.Fatal(err)
	}
	pid, err := CreatePerson(db, "blonde")
	if err != nil {
		t.Fatalf("collision create: %v", err)
	}
	p, _, _ := GetPersonByID(db, pid)
	if p.Name != "blonde"+PersonClusterSuffix {
		t.Fatalf("stored name = %q, want suffixed", p.Name)
	}
	if cat, ok := tagCategory(t, db, "blonde"+PersonClusterSuffix); !ok || cat != PeopleCategory {
		t.Fatalf("suffixed tag = (%q, %v)", cat, ok)
	}
	// The curated tag is untouched.
	if cat, _ := tagCategory(t, db, "blonde"); cat != "Appearance" {
		t.Fatalf("curated tag harmed: %q", cat)
	}
	// Display-name lookup finds the suffixed person.
	if got, ok, _ := GetPersonByDisplayName(db, "blonde"); !ok || got.ID != pid {
		t.Fatalf("display lookup = (%+v, %v)", got, ok)
	}
}

func TestNextUnknownName(t *testing.T) {
	db := newPeopleDB(t)
	n1, err := NextUnknownName(db)
	if err != nil || n1 != "Unknown #1" {
		t.Fatalf("first = %q, %v", n1, err)
	}
	if _, err := CreatePerson(db, "Unknown #1"); err != nil {
		t.Fatal(err)
	}
	if _, err := CreatePerson(db, "Unknown #7"); err != nil {
		t.Fatal(err)
	}
	n, err := NextUnknownName(db)
	if err != nil || n != "Unknown #8" {
		t.Fatalf("next = %q, %v", n, err)
	}
}

func TestAssignUnassignFaceBridge(t *testing.T) {
	db := newPeopleDB(t)
	pid, _ := CreatePerson(db, "Alice")
	ids, err := ReplaceFaces(db, "a.jpg", "m1", []NewFace{
		{X: 0.1, Y: 0.1, W: 0.2, H: 0.2, Score: 0.9, Vec: []float32{1, 0}},
		{X: 0.5, Y: 0.5, W: 0.2, H: 0.2, Score: 0.8, Vec: []float32{0, 1}},
	}, 1)
	if err != nil {
		t.Fatal(err)
	}

	if err := AssignFace(db, ids[0], pid, "user"); err != nil {
		t.Fatal(err)
	}
	if rows := bridgeRows(t, db); rows["a.jpg|Alice"] != 1 {
		t.Fatalf("bridge rows = %v", rows)
	}
	f, _, _ := GetFaceByID(db, ids[0])
	if f.PersonID != pid || f.AssignedBy != "user" {
		t.Fatalf("face = %+v", f)
	}
	p, _, _ := GetPersonByID(db, pid)
	if p.CoverFaceID != ids[0] {
		t.Fatalf("cover = %d, want %d", p.CoverFaceID, ids[0])
	}

	// Second face of the same person on the same media: bridge row is
	// deduped (same path+tag+ts) — unassigning ONE keeps the row.
	if err := AssignFace(db, ids[1], pid, "user"); err != nil {
		t.Fatal(err)
	}
	if rows := bridgeRows(t, db); rows["a.jpg|Alice"] != 1 {
		t.Fatalf("bridge rows after 2nd assign = %v", rows)
	}
	if err := UnassignFace(db, ids[1]); err != nil {
		t.Fatal(err)
	}
	if rows := bridgeRows(t, db); rows["a.jpg|Alice"] != 1 {
		t.Fatalf("bridge row dropped while sibling face remains: %v", rows)
	}
	// Unassign the last face → bridge row goes away.
	if err := UnassignFace(db, ids[0]); err != nil {
		t.Fatal(err)
	}
	if rows := bridgeRows(t, db); len(rows) != 0 {
		t.Fatalf("bridge rows remain after last unassign: %v", rows)
	}
	p, _, _ = GetPersonByID(db, pid)
	if p.CoverFaceID != 0 {
		t.Fatalf("cover face not cleared: %d", p.CoverFaceID)
	}
}

func TestAutoAssignNeverOverwritesUser(t *testing.T) {
	db := newPeopleDB(t)
	alice, _ := CreatePerson(db, "Alice")
	bob, _ := CreatePerson(db, "Bob")
	ids, _ := ReplaceFaces(db, "a.jpg", "m1", []NewFace{{Score: 0.9, Vec: []float32{1}}}, 1)

	if err := AssignFace(db, ids[0], alice, "user"); err != nil {
		t.Fatal(err)
	}
	if err := AssignFace(db, ids[0], bob, "auto"); err != nil {
		t.Fatal(err)
	}
	f, _, _ := GetFaceByID(db, ids[0])
	if f.PersonID != alice || f.AssignedBy != "user" {
		t.Fatalf("auto overwrote user assignment: %+v", f)
	}
	// User reassignment IS allowed and moves the bridge row.
	if err := AssignFace(db, ids[0], bob, "user"); err != nil {
		t.Fatal(err)
	}
	rows := bridgeRows(t, db)
	if rows["a.jpg|Alice"] != 0 || rows["a.jpg|Bob"] != 1 {
		t.Fatalf("reassign bridge rows = %v", rows)
	}
}

func TestRenamePersonCascades(t *testing.T) {
	db := newPeopleDB(t)
	pid, _ := CreatePerson(db, "Unknown #1")
	ids, _ := ReplaceFaces(db, "a.jpg", "m1", []NewFace{{Score: 0.9, Vec: []float32{1}}}, 1)
	_ = AssignFace(db, ids[0], pid, "auto")

	if err := RenamePerson(db, pid, "Alice"); err != nil {
		t.Fatal(err)
	}
	p, _, _ := GetPersonByID(db, pid)
	if p.Name != "Alice" {
		t.Fatalf("name = %q", p.Name)
	}
	rows := bridgeRows(t, db)
	if rows["a.jpg|Alice"] != 1 || rows["a.jpg|Unknown #1"] != 0 {
		t.Fatalf("bridge rows = %v", rows)
	}
	if _, ok := tagCategory(t, db, "Unknown #1"); ok {
		t.Fatal("old tag row remains")
	}
	if cat, ok := tagCategory(t, db, "Alice"); !ok || cat != PeopleCategory {
		t.Fatalf("new tag = (%q, %v)", cat, ok)
	}
	// Renaming to another existing person's name → error (must merge).
	if _, err := CreatePerson(db, "Bob"); err != nil {
		t.Fatal(err)
	}
	if err := RenamePerson(db, pid, "Bob"); err == nil {
		t.Fatal("rename onto existing person allowed")
	}
	// Renaming to a curated tag's name stores the suffixed form and
	// cascades the bridge rows to it.
	if _, err := db.Exec(`INSERT INTO tag (label, category_label) VALUES ('redhead', 'Appearance')`); err != nil {
		t.Fatal(err)
	}
	if err := RenamePerson(db, pid, "redhead"); err != nil {
		t.Fatalf("collision rename: %v", err)
	}
	p, _, _ = GetPersonByID(db, pid)
	if p.Name != "redhead"+PersonClusterSuffix {
		t.Fatalf("stored name = %q, want suffixed", p.Name)
	}
	rows = bridgeRows(t, db)
	if rows["a.jpg|redhead"+PersonClusterSuffix] != 1 {
		t.Fatalf("bridge rows after collision rename = %v", rows)
	}
	if cat, _ := tagCategory(t, db, "redhead"); cat != "Appearance" {
		t.Fatalf("curated tag harmed: %q", cat)
	}
}

func TestMergePersons(t *testing.T) {
	db := newPeopleDB(t)
	unknown, _ := CreatePerson(db, "Unknown #1")
	alice, _ := CreatePerson(db, "Alice")
	idsA, _ := ReplaceFaces(db, "a.jpg", "m1", []NewFace{{Score: 0.9, Vec: []float32{1}}}, 1)
	idsB, _ := ReplaceFaces(db, "b.jpg", "m1", []NewFace{{Score: 0.9, Vec: []float32{1}}}, 1)
	_ = AssignFace(db, idsA[0], unknown, "auto")
	_ = AssignFace(db, idsB[0], alice, "user")

	if err := MergePersons(db, unknown, alice); err != nil {
		t.Fatal(err)
	}
	// Faces moved, assigned_by preserved.
	fa, _, _ := GetFaceByID(db, idsA[0])
	if fa.PersonID != alice || fa.AssignedBy != "auto" {
		t.Fatalf("merged face = %+v", fa)
	}
	// Source person + tag gone; bridge rows now under Alice.
	if _, ok, _ := GetPersonByID(db, unknown); ok {
		t.Fatal("source person remains")
	}
	if _, ok := tagCategory(t, db, "Unknown #1"); ok {
		t.Fatal("source tag remains")
	}
	rows := bridgeRows(t, db)
	if rows["a.jpg|Alice"] != 1 || rows["b.jpg|Alice"] != 1 {
		t.Fatalf("bridge rows = %v", rows)
	}
	if err := MergePersons(db, alice, alice); err == nil {
		t.Fatal("self-merge allowed")
	}
}

func TestDeletePersonKeepsFaces(t *testing.T) {
	db := newPeopleDB(t)
	pid, _ := CreatePerson(db, "Alice")
	ids, _ := ReplaceFaces(db, "a.jpg", "m1", []NewFace{{Score: 0.9, Vec: []float32{1}}}, 1)
	_ = AssignFace(db, ids[0], pid, "user")

	if err := DeletePerson(db, pid); err != nil {
		t.Fatal(err)
	}
	f, ok, _ := GetFaceByID(db, ids[0])
	if !ok || f.PersonID != 0 || f.AssignedBy != "" {
		t.Fatalf("face after person delete = %+v (ok=%v)", f, ok)
	}
	if rows := bridgeRows(t, db); len(rows) != 0 {
		t.Fatalf("bridge rows remain: %v", rows)
	}
	if _, ok := tagCategory(t, db, "Alice"); ok {
		t.Fatal("tag remains")
	}
}

func TestPersonMediaPathsAndGetPeople(t *testing.T) {
	db := newPeopleDB(t)
	pid, _ := CreatePerson(db, "Alice")
	idsA, _ := ReplaceFaces(db, "a.jpg", "m1", []NewFace{
		{Score: 0.5, Vec: []float32{1}},
		{Score: 0.7, Vec: []float32{1}},
	}, 1)
	idsB, _ := ReplaceFaces(db, "b.jpg", "m1", []NewFace{{Score: 0.99, Vec: []float32{1}}}, 1)
	for _, id := range append(idsA, idsB...) {
		_ = AssignFace(db, id, pid, "auto")
	}

	paths, err := PersonMediaPaths(db, pid)
	if err != nil {
		t.Fatal(err)
	}
	if len(paths) != 2 || paths[0] != "b.jpg" {
		t.Fatalf("paths = %v, want [b.jpg a.jpg] (best det first)", paths)
	}

	people, err := GetPeople(db)
	if err != nil {
		t.Fatal(err)
	}
	if len(people) != 1 || people[0].FaceCount != 3 || people[0].MediaCount != 2 {
		t.Fatalf("people = %+v", people)
	}
}

func TestAssignFaceSkipsBridgeForNonLibraryMedia(t *testing.T) {
	db := newPeopleDB(t)
	pid, _ := CreatePerson(db, "Alice")
	// A path with NO media row (e.g. an ad-hoc scan of a non-library file).
	ids, _ := ReplaceFaces(db, `C:\outside\query.jpg`, "m1", []NewFace{{Score: 0.9, Vec: []float32{1}}}, 1)

	if err := AssignFace(db, ids[0], pid, "user"); err != nil {
		t.Fatalf("assign to non-library media failed: %v", err)
	}
	f, _, _ := GetFaceByID(db, ids[0])
	if f.PersonID != pid {
		t.Fatalf("assignment lost: %+v", f)
	}
	if rows := bridgeRows(t, db); len(rows) != 0 {
		t.Fatalf("bridge row written for non-library media: %v", rows)
	}
}

func TestGetPeopleCoverSelfHeals(t *testing.T) {
	db := newPeopleDB(t)
	pid, _ := CreatePerson(db, "Alice")
	// Two faces: a tiny low-confidence one and a big clear one. Assign the
	// tiny one FIRST so it becomes the stored cover.
	ids, _ := ReplaceFaces(db, "a.jpg", "m1", []NewFace{
		{X: 0.8, Y: 0.8, W: 0.05, H: 0.05, Score: 0.71, Vec: []float32{1}},
		{X: 0.2, Y: 0.2, W: 0.5, H: 0.5, Score: 0.98, Vec: []float32{1}},
	}, 1)
	_ = AssignFace(db, ids[0], pid, "user")
	_ = AssignFace(db, ids[1], pid, "user")

	people, err := GetPeople(db)
	if err != nil {
		t.Fatal(err)
	}
	// Stored cover (the tiny face) is still valid → honoured.
	if people[0].CoverFaceID != ids[0] {
		t.Fatalf("cover = %d, want stored %d", people[0].CoverFaceID, ids[0])
	}

	// Rescan the media: old rows (including the cover) are replaced. The
	// stored cover must be cleared, and GetPeople must fall back to the
	// person's best remaining face.
	otherIDs, _ := ReplaceFaces(db, "b.jpg", "m1", []NewFace{
		{X: 0.1, Y: 0.1, W: 0.3, H: 0.3, Score: 0.9, Vec: []float32{1}},
		{X: 0.5, Y: 0.5, W: 0.6, H: 0.6, Score: 0.95, Vec: []float32{1}},
	}, 2)
	_ = AssignFace(db, otherIDs[0], pid, "user")
	_ = AssignFace(db, otherIDs[1], pid, "user")
	if _, err := ReplaceFaces(db, "a.jpg", "m1", nil, 3); err != nil {
		t.Fatal(err)
	}

	var stored any
	if err := db.QueryRow(`SELECT cover_face_id FROM person WHERE id=?`, pid).Scan(&stored); err != nil {
		t.Fatal(err)
	}
	if stored != nil {
		t.Fatalf("stale cover not cleared by rescan: %v", stored)
	}
	people, err = GetPeople(db)
	if err != nil {
		t.Fatal(err)
	}
	// Fallback = best remaining face by det_score × bbox area: otherIDs[1]
	// (0.95 × 0.36) beats otherIDs[0] (0.9 × 0.09).
	if people[0].CoverFaceID != otherIDs[1] {
		t.Fatalf("effective cover = %d, want best face %d", people[0].CoverFaceID, otherIDs[1])
	}
}

func TestPersonFacesByQualityAndSetCover(t *testing.T) {
	db := newPeopleDB(t)
	pid, _ := CreatePerson(db, "Alice")
	ids, _ := ReplaceFaces(db, "a.jpg", "m1", []NewFace{
		{X: 0.1, Y: 0.1, W: 0.05, H: 0.05, Score: 0.99, Vec: []float32{1}}, // tiny, high conf
		{X: 0.2, Y: 0.2, W: 0.5, H: 0.5, Score: 0.80, Vec: []float32{1}},   // big, decent conf
	}, 1)
	for _, id := range ids {
		_ = AssignFace(db, id, pid, "user")
	}

	faces, err := PersonFacesByQuality(db, pid)
	if err != nil {
		t.Fatal(err)
	}
	// Quality = det_score × area: 0.80×0.25 ≫ 0.99×0.0025.
	if len(faces) != 2 || faces[0].ID != ids[1] {
		t.Fatalf("quality order wrong: %+v", faces)
	}

	if err := SetPersonCover(db, pid, ids[0]); err != nil {
		t.Fatal(err)
	}
	p, _, _ := GetPersonByID(db, pid)
	if p.CoverFaceID != ids[0] {
		t.Fatalf("cover = %d, want %d", p.CoverFaceID, ids[0])
	}
	// A face belonging to someone else (or nobody) is rejected.
	otherIDs, _ := ReplaceFaces(db, "b.jpg", "m1", []NewFace{{Score: 0.9, Vec: []float32{1}}}, 1)
	if err := SetPersonCover(db, pid, otherIDs[0]); err == nil {
		t.Fatal("cover set to a face outside the person")
	}
}

func TestDeleteAllFaceData(t *testing.T) {
	db := newPeopleDB(t)
	pid, _ := CreatePerson(db, "Alice")
	ids, _ := ReplaceFaces(db, "a.jpg", "m1", []NewFace{{Score: 0.9, Vec: []float32{1}}}, 1)
	_ = AssignFace(db, ids[0], pid, "user")
	// A non-People tag must survive the wipe.
	if err := AddTag(db, "a.jpg", "sunset", "Scene"); err != nil {
		t.Fatal(err)
	}

	if err := DeleteAllFaceData(db); err != nil {
		t.Fatal(err)
	}
	for _, q := range []string{
		`SELECT COUNT(*) FROM face`,
		`SELECT COUNT(*) FROM face_scan`,
		`SELECT COUNT(*) FROM person`,
		`SELECT COUNT(*) FROM media_tag_by_category WHERE category_label = 'People'`,
		`SELECT COUNT(*) FROM tag WHERE category_label = 'People'`,
	} {
		var n int
		if err := db.QueryRow(q).Scan(&n); err != nil {
			t.Fatal(err)
		}
		if n != 0 {
			t.Fatalf("%s = %d, want 0", q, n)
		}
	}
	if cat, ok := tagCategory(t, db, "sunset"); !ok || cat != "Scene" {
		t.Fatalf("foreign tag harmed: (%q, %v)", cat, ok)
	}
}
