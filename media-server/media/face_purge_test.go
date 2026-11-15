package media

import (
	"testing"
)

// The retired anime pipeline's rows are purged: auto-only groups vanish with
// their People tags, curated (named / user-assigned) people survive with their
// tags, photo faces are untouched, and the scan markers become skip markers
// so drawn items are not rescanned.
func TestPurgeRetiredFaceModel(t *testing.T) {
	db := newPeopleDB(t)
	const anime, photo = RetiredAnimeFaceModelID, "arcface"

	animeA := addFace(t, db, "toon1.png", anime, []float32{1, 0})
	animeB := addFace(t, db, "toon2.png", anime, []float32{0, 1})
	animeC := addFace(t, db, "toon3.png", anime, []float32{1, 1})
	photoF := addFace(t, db, "photo.jpg", photo, []float32{1, 0})

	unknown, _ := CreatePerson(db, "Unknown #1") // auto blob → deleted
	zelda, _ := CreatePerson(db, "Zelda")        // curated by name → kept
	alice, _ := CreatePerson(db, "Alice")        // photo person → untouched
	for _, a := range []struct {
		face, person int64
	}{{animeA, unknown}, {animeB, unknown}, {animeC, zelda}, {photoF, alice}} {
		if err := AssignFace(db, a.face, a.person, "auto"); err != nil {
			t.Fatal(err)
		}
	}

	faces, people, err := PurgeRetiredFaceModel(db, anime, DrawnSkipModelID)
	if err != nil {
		t.Fatal(err)
	}
	if faces != 3 || people != 1 {
		t.Fatalf("purged faces=%d people=%d, want 3 and 1", faces, people)
	}
	if _, ok, _ := GetPersonByID(db, unknown); ok {
		t.Fatal("auto-only anime group survived")
	}
	if _, ok, _ := GetPersonByID(db, zelda); !ok {
		t.Fatal("named person was deleted")
	}
	rows := bridgeRows(t, db)
	if rows["toon3.png|Zelda"] != 1 || rows["photo.jpg|Alice"] != 1 {
		t.Fatalf("curated/photo tags lost: %v", rows)
	}
	if rows["toon1.png|Unknown #1"] != 0 || rows["toon2.png|Unknown #1"] != 0 {
		t.Fatalf("auto group tags left behind: %v", rows)
	}
	if _, ok, _ := GetFaceByID(db, photoF); !ok {
		t.Fatal("photo face removed")
	}
	var markers, skip int
	db.QueryRow(`SELECT COUNT(*) FROM face_scan WHERE model=?`, anime).Scan(&markers)
	db.QueryRow(`SELECT COUNT(*) FROM face_scan WHERE model=? AND face_count=0`, DrawnSkipModelID).Scan(&skip)
	if markers != 0 || skip != 3 {
		t.Fatalf("scan markers: retired=%d skip=%d, want 0 and 3", markers, skip)
	}

	// Idempotent.
	if f, p, err := PurgeRetiredFaceModel(db, anime, DrawnSkipModelID); err != nil || f != 0 || p != 0 {
		t.Fatalf("second purge = %d,%d,%v", f, p, err)
	}
}
