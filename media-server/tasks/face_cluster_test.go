package tasks

import (
	"testing"

	"github.com/stevecastle/shrike/media"
	_ "modernc.org/sqlite"
)

// vecNear returns a unit-ish vector close to base (cosine ≈ 0.99).
func vecNear(base []float32, eps float32) []float32 {
	out := make([]float32, len(base))
	copy(out, base)
	out[0] += eps
	return out
}

func TestClusterFacesJoinsSeedsAndMintsUnknowns(t *testing.T) {
	db := newFaceIndexDB(t)
	resetFaceIndex(t)

	// Person Alice with one user-labeled seed face pointing at [1,0,0].
	alice, err := media.CreatePerson(db, "Alice")
	if err != nil {
		t.Fatal(err)
	}
	seedIDs := seedFaces(t, db, "seed.jpg", "m1", []float32{1, 0, 0})
	if err := media.AssignFace(db, seedIDs[0], alice, "user"); err != nil {
		t.Fatal(err)
	}

	// Two unassigned faces near Alice's seed → should join Alice.
	seedFaces(t, db, "a1.jpg", "m1", vecNear([]float32{1, 0, 0}, 0.05))
	seedFaces(t, db, "a2.jpg", "m1", vecNear([]float32{1, 0, 0}, -0.05))

	// Three mutually-similar faces around [0,1,0] → new "Unknown #1".
	seedFaces(t, db, "b1.jpg", "m1", vecNear([]float32{0, 1, 0}, 0.03))
	seedFaces(t, db, "b2.jpg", "m1", vecNear([]float32{0, 1, 0}, -0.03))
	seedFaces(t, db, "b3.jpg", "m1", []float32{0, 1, 0})

	// One lone face → stays unassigned (below min cluster size).
	seedFaces(t, db, "lone.jpg", "m1", []float32{0, 0, 1})

	model := FaceModel{ID: "m1", MatchThreshold: 0.9}
	stats, err := clusterFaces(db, model, model.MatchThreshold, 3)
	if err != nil {
		t.Fatal(err)
	}
	if stats.JoinedExisting != 2 {
		t.Fatalf("joined = %d, want 2", stats.JoinedExisting)
	}
	if stats.NewPeople != 1 || stats.NewlyClustered != 3 {
		t.Fatalf("new people = %d (%d faces), want 1 (3)", stats.NewPeople, stats.NewlyClustered)
	}
	if stats.Unassigned != 1 {
		t.Fatalf("unassigned = %d, want 1", stats.Unassigned)
	}

	// The Alice joins are auto, bridge rows exist, and the unknown person is named.
	people, err := media.GetPeople(db)
	if err != nil {
		t.Fatal(err)
	}
	byName := map[string]media.Person{}
	for _, p := range people {
		byName[p.Name] = p
	}
	if byName["Alice"].FaceCount != 3 {
		t.Fatalf("Alice faces = %d, want 3 (seed + 2 joined)", byName["Alice"].FaceCount)
	}
	if byName["Unknown #1"].FaceCount != 3 {
		t.Fatalf("Unknown #1 faces = %d, want 3: %+v", byName["Unknown #1"].FaceCount, people)
	}

	// Idempotency: a second pass with nothing new changes nothing.
	stats2, err := clusterFaces(db, model, model.MatchThreshold, 3)
	if err != nil {
		t.Fatal(err)
	}
	if stats2.JoinedExisting != 0 || stats2.NewPeople != 0 {
		t.Fatalf("second pass not idempotent: %+v", stats2)
	}
}

func TestResetAutoAssignmentsKeepsUserLabels(t *testing.T) {
	db := newFaceIndexDB(t)
	resetFaceIndex(t)

	alice, _ := media.CreatePerson(db, "Alice")
	unknown, _ := media.CreatePerson(db, "Unknown #1")
	userIDs := seedFaces(t, db, "u.jpg", "m1", []float32{1, 0})
	autoIDs := seedFaces(t, db, "a.jpg", "m1", []float32{0, 1})
	_ = media.AssignFace(db, userIDs[0], alice, "user")
	_ = media.AssignFace(db, autoIDs[0], unknown, "auto")

	n, err := resetAutoAssignments(db, "m1")
	if err != nil {
		t.Fatal(err)
	}
	if n != 1 {
		t.Fatalf("reset %d, want 1", n)
	}
	fUser, _, _ := media.GetFaceByID(db, userIDs[0])
	if fUser.PersonID != alice || fUser.AssignedBy != "user" {
		t.Fatalf("user label harmed: %+v", fUser)
	}
	fAuto, _, _ := media.GetFaceByID(db, autoIDs[0])
	if fAuto.PersonID != 0 {
		t.Fatalf("auto assignment survived: %+v", fAuto)
	}
	// Emptied anonymous person dissolved; Alice kept.
	if _, ok, _ := media.GetPersonByID(db, unknown); ok {
		t.Fatal("empty Unknown person survived reset")
	}
	if _, ok, _ := media.GetPersonByID(db, alice); !ok {
		t.Fatal("Alice dissolved")
	}
}

func TestAutoAssignNewFacesViaIndex(t *testing.T) {
	db := newFaceIndexDB(t)
	resetFaceIndex(t)

	model := FaceModel{ID: "m1", MatchThreshold: 0.9}
	alice, _ := media.CreatePerson(db, "Alice")
	seedIDs := seedFaces(t, db, "seed.jpg", "m1", []float32{1, 0, 0})
	_ = media.AssignFace(db, seedIDs[0], alice, "user")

	// Install the index for m1 (prerequisite for incremental assignment).
	idx, pathKeys, err := BuildFaceIndexFromDB(db, "m1", nil)
	if err != nil {
		t.Fatal(err)
	}
	SetFaceIndexForModel(idx, "m1", pathKeys)

	// Simulate the collector: store a new face near Alice, index it, auto-assign.
	newFaces := []media.NewFace{{Score: 0.9, Vec: vecNear([]float32{1, 0, 0}, 0.05)}}
	ids, err := media.ReplaceFaces(db, "new.jpg", "m1", newFaces, 1)
	if err != nil {
		t.Fatal(err)
	}
	faceIndexReplacePath("m1", "new.jpg", ids, newFaces)

	if n := autoAssignNewFaces(db, model, ids, newFaces); n != 1 {
		t.Fatalf("auto-assigned %d, want 1", n)
	}
	f, _, _ := media.GetFaceByID(db, ids[0])
	if f.PersonID != alice || f.AssignedBy != "auto" {
		t.Fatalf("face = %+v", f)
	}

	// A face UNLIKE anything assigned stays unassigned.
	far := []media.NewFace{{Score: 0.9, Vec: []float32{0, 0, 1}}}
	farIDs, _ := media.ReplaceFaces(db, "far.jpg", "m1", far, 1)
	faceIndexReplacePath("m1", "far.jpg", farIDs, far)
	if n := autoAssignNewFaces(db, model, farIDs, far); n != 0 {
		t.Fatalf("dissimilar face auto-assigned (%d)", n)
	}

	// Wrong indexed model → no-op.
	SetFaceIndexForModel(embedindexNewForTest(), "other", nil)
	if n := autoAssignNewFaces(db, model, farIDs, far); n != 0 {
		t.Fatalf("assigned despite model mismatch (%d)", n)
	}
}

func TestClusterFacesNeverTouchesUserAssignments(t *testing.T) {
	db := newFaceIndexDB(t)
	resetFaceIndex(t)

	alice, _ := media.CreatePerson(db, "Alice")
	bob, _ := media.CreatePerson(db, "Bob")
	// A user says this face is Bob even though its vector matches Alice's seed.
	seedIDs := seedFaces(t, db, "seed.jpg", "m1", []float32{1, 0})
	_ = media.AssignFace(db, seedIDs[0], alice, "user")
	contraIDs := seedFaces(t, db, "contra.jpg", "m1", vecNear([]float32{1, 0}, 0.01))
	_ = media.AssignFace(db, contraIDs[0], bob, "user")

	model := FaceModel{ID: "m1", MatchThreshold: 0.9}
	if _, err := clusterFaces(db, model, model.MatchThreshold, 3); err != nil {
		t.Fatal(err)
	}
	f, _, _ := media.GetFaceByID(db, contraIDs[0])
	if f.PersonID != bob || f.AssignedBy != "user" {
		t.Fatalf("clustering moved a user label: %+v", f)
	}
}
