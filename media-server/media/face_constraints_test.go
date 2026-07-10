package media

import (
	"database/sql"
	"testing"

	_ "modernc.org/sqlite"
)

// addFace inserts one face row for path (registered as media) and returns its
// id.
func addFace(t *testing.T, db *sql.DB, path, model string, vec []float32) int64 {
	t.Helper()
	if _, err := db.Exec(`INSERT OR IGNORE INTO media (path) VALUES (?)`, path); err != nil {
		t.Fatal(err)
	}
	ids, err := ReplaceFaces(db, path, model, []NewFace{
		{X: 0.1, Y: 0.1, W: 0.2, H: 0.2, Score: 0.9, Vec: vec},
	}, 1)
	if err != nil {
		t.Fatal(err)
	}
	return ids[0]
}

func vetoCount(t *testing.T, db *sql.DB) int {
	t.Helper()
	var n int
	if err := db.QueryRow(`SELECT COUNT(*) FROM face_veto`).Scan(&n); err != nil {
		t.Fatal(err)
	}
	return n
}

func cannotLinkCount(t *testing.T, db *sql.DB) int {
	t.Helper()
	var n int
	if err := db.QueryRow(`SELECT COUNT(*) FROM face_cannot_link`).Scan(&n); err != nil {
		t.Fatal(err)
	}
	return n
}

func TestRejectFaceRecordsVetoLinksAndUnassigns(t *testing.T) {
	db := newPeopleDB(t)
	alice, _ := CreatePerson(db, "Alice")
	f1 := addFace(t, db, "r1.jpg", "m1", []float32{1, 0})
	f2 := addFace(t, db, "r2.jpg", "m1", []float32{1, 0.05})
	f3 := addFace(t, db, "r3.jpg", "m1", []float32{1, -0.05})
	if err := AssignFace(db, f1, alice, "user"); err != nil {
		t.Fatal(err)
	}
	if err := AssignFace(db, f2, alice, "auto"); err != nil {
		t.Fatal(err)
	}
	if err := AssignFace(db, f3, alice, "auto"); err != nil {
		t.Fatal(err)
	}

	links, err := RejectFaceFromPerson(db, f3, alice)
	if err != nil {
		t.Fatal(err)
	}
	if links != 2 {
		t.Fatalf("cannot-links = %d, want 2 (f1, f2 exemplars)", links)
	}
	if vetoed, _ := FaceVetoExists(db, f3, alice); !vetoed {
		t.Fatal("veto not recorded")
	}
	f, _, _ := GetFaceByID(db, f3)
	if f.PersonID != 0 {
		t.Fatalf("rejected face still assigned: %+v", f)
	}
	// The veto is enforced: auto assignment silently refuses.
	if err := AssignFace(db, f3, alice, "auto"); err != nil {
		t.Fatal(err)
	}
	f, _, _ = GetFaceByID(db, f3)
	if f.PersonID != 0 {
		t.Fatalf("auto assignment overrode a veto: %+v", f)
	}
	// …but the user changing their mind wins and clears the assertions.
	if err := AssignFace(db, f3, alice, "user"); err != nil {
		t.Fatal(err)
	}
	f, _, _ = GetFaceByID(db, f3)
	if f.PersonID != alice || f.AssignedBy != "user" {
		t.Fatalf("user re-assign blocked: %+v", f)
	}
	if vetoed, _ := FaceVetoExists(db, f3, alice); vetoed {
		t.Fatal("contradicted veto survived a user assignment")
	}
	if n := cannotLinkCount(t, db); n != 0 {
		t.Fatalf("contradicted cannot-links survived: %d", n)
	}
}

func TestUserMoveRecordsVetoAgainstOldPerson(t *testing.T) {
	db := newPeopleDB(t)
	alice, _ := CreatePerson(db, "Alice")
	bob, _ := CreatePerson(db, "Bob")
	f := addFace(t, db, "mv.jpg", "m1", []float32{1, 0})
	if err := AssignFace(db, f, alice, "auto"); err != nil {
		t.Fatal(err)
	}
	// The user corrects the assignment → an implicit "not Alice".
	if err := AssignFace(db, f, bob, "user"); err != nil {
		t.Fatal(err)
	}
	if vetoed, _ := FaceVetoExists(db, f, alice); !vetoed {
		t.Fatal("moving a face off a person did not record a veto")
	}
	// An auto pass can no longer put it back.
	if err := AssignFace(db, f, alice, "auto"); err != nil {
		t.Fatal(err)
	}
	got, _, _ := GetFaceByID(db, f)
	if got.PersonID != bob {
		t.Fatalf("face moved back to the vetoed person: %+v", got)
	}
}

func TestMergeMigratesAndReconcilesConstraints(t *testing.T) {
	db := newPeopleDB(t)
	alice, _ := CreatePerson(db, "Alice")
	bob, _ := CreatePerson(db, "Bob")
	carol, _ := CreatePerson(db, "Carol")
	fa := addFace(t, db, "ma.jpg", "m1", []float32{1, 0})
	fb := addFace(t, db, "mb.jpg", "m1", []float32{0, 1})
	fc := addFace(t, db, "mc.jpg", "m1", []float32{1, 1})
	_ = AssignFace(db, fa, alice, "user")
	_ = AssignFace(db, fb, bob, "user")
	_ = AssignFace(db, fc, carol, "user")

	// fc was rejected from Bob; fa↔fb are cannot-linked (fa was rejected from
	// Bob's group at some point).
	if _, err := RejectFaceFromPerson(db, fc, bob); err != nil {
		t.Fatal(err)
	}
	if err := AddFaceVeto(db, fa, bob); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(
		`INSERT INTO face_cannot_link (face_a, face_b, created_at) VALUES (?, ?, 0)`,
		min64(fa, fb), max64(fa, fb),
	); err != nil {
		t.Fatal(err)
	}

	// Merging Bob into Alice: fc's veto follows to Alice; fa's veto against
	// Bob is dropped (fa is IN Alice — the merge contradicts it), and the
	// fa↔fb cannot-link dissolves because both now sit in Alice.
	if err := MergePersons(db, bob, alice); err != nil {
		t.Fatal(err)
	}
	if vetoed, _ := FaceVetoExists(db, fc, alice); !vetoed {
		t.Fatal("veto did not follow the merge target")
	}
	if vetoed, _ := FaceVetoExists(db, fa, alice); vetoed {
		t.Fatal("merge left a veto against a face's own person")
	}
	// fc's rejection recorded fc↔fb (Bob's only face); the manual fa↔fb made
	// two links total. The merge dissolves fa↔fb (both ends now in Alice) and
	// keeps fc↔fb (fc is Carol's) — exactly one link remains.
	if n := cannotLinkCount(t, db); n != 1 {
		rows, _ := db.Query(`SELECT face_a, face_b FROM face_cannot_link`)
		defer rows.Close()
		var pairs [][2]int64
		for rows.Next() {
			var a, b int64
			_ = rows.Scan(&a, &b)
			pairs = append(pairs, [2]int64{a, b})
		}
		t.Fatalf("cannot-links after merge = %v (fa=%d fb=%d fc=%d)", pairs, fa, fb, fc)
	}
}

func min64(a, b int64) int64 {
	if a < b {
		return a
	}
	return b
}

func max64(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}

func TestRescanDropsStaleConstraints(t *testing.T) {
	db := newPeopleDB(t)
	alice, _ := CreatePerson(db, "Alice")
	f1 := addFace(t, db, "rs.jpg", "m1", []float32{1, 0})
	other := addFace(t, db, "rs2.jpg", "m1", []float32{0, 1})
	_ = AssignFace(db, other, alice, "user")
	if _, err := RejectFaceFromPerson(db, f1, alice); err != nil {
		t.Fatal(err)
	}
	if vetoCount(t, db) != 1 || cannotLinkCount(t, db) != 1 {
		t.Fatalf("setup: vetoes=%d links=%d", vetoCount(t, db), cannotLinkCount(t, db))
	}
	// Rescanning rs.jpg replaces its face rows → the old row's assertions go.
	if _, err := ReplaceFaces(db, "rs.jpg", "m1", []NewFace{
		{X: 0.1, Y: 0.1, W: 0.2, H: 0.2, Score: 0.9, Vec: []float32{1, 0}},
	}, 2); err != nil {
		t.Fatal(err)
	}
	if vetoCount(t, db) != 0 || cannotLinkCount(t, db) != 0 {
		t.Fatalf("stale constraints survived rescan: vetoes=%d links=%d", vetoCount(t, db), cannotLinkCount(t, db))
	}
}

func TestDeletePersonDropsItsVetoesKeepsCannotLinks(t *testing.T) {
	db := newPeopleDB(t)
	alice, _ := CreatePerson(db, "Alice")
	f1 := addFace(t, db, "dp.jpg", "m1", []float32{1, 0})
	member := addFace(t, db, "dp2.jpg", "m1", []float32{0, 1})
	_ = AssignFace(db, member, alice, "auto")
	if _, err := RejectFaceFromPerson(db, f1, alice); err != nil {
		t.Fatal(err)
	}
	if err := DeletePerson(db, alice); err != nil {
		t.Fatal(err)
	}
	if vetoCount(t, db) != 0 {
		t.Fatal("veto against a deleted person survived")
	}
	if cannotLinkCount(t, db) != 1 {
		t.Fatal("cannot-link must outlive the person (that's its whole job)")
	}
}

func TestLockPersonFacesPromotesAutoToUser(t *testing.T) {
	db := newPeopleDB(t)
	alice, _ := CreatePerson(db, "Alice")
	f1 := addFace(t, db, "lk1.jpg", "m1", []float32{1, 0})
	f2 := addFace(t, db, "lk2.jpg", "m1", []float32{1, 0.1})
	_ = AssignFace(db, f1, alice, "user")
	_ = AssignFace(db, f2, alice, "auto")

	n, err := LockPersonFaces(db, alice)
	if err != nil {
		t.Fatal(err)
	}
	if n != 1 {
		t.Fatalf("locked %d, want 1 (f2 only; f1 already user)", n)
	}
	for _, id := range []int64{f1, f2} {
		f, _, _ := GetFaceByID(db, id)
		if f.AssignedBy != "user" {
			t.Fatalf("face %d not locked: %+v", id, f)
		}
	}
	// Locked faces now count in GetPeople's LockedCount.
	people, err := GetPeople(db)
	if err != nil {
		t.Fatal(err)
	}
	if len(people) != 1 || people[0].LockedCount != 2 {
		t.Fatalf("people = %+v, want lockedCount 2", people)
	}
}
