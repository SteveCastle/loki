package tasks

import (
	"fmt"
	"math"
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
	params := clusterParams{joinThreshold: 0.9, formThreshold: 0.95, minQuality: 0.75, minCluster: 3, passes: 2}
	stats, err := clusterFaces(db, model, params)
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
	stats2, err := clusterFaces(db, model, params)
	if err != nil {
		t.Fatal(err)
	}
	if stats2.JoinedExisting != 0 || stats2.NewPeople != 0 {
		t.Fatalf("second pass not idempotent: %+v", stats2)
	}
}

func TestResetAutoAssignmentsKeepsUserLabelsAndNamedPeople(t *testing.T) {
	db := newFaceIndexDB(t)
	resetFaceIndex(t)

	alice, _ := media.CreatePerson(db, "Alice")
	unknown, _ := media.CreatePerson(db, "Unknown #1")
	userIDs := seedFaces(t, db, "u.jpg", "m1", []float32{1, 0})
	// An auto face inside a NAMED person (e.g. an Unknown cluster the user
	// renamed/merged into Alice) — naming endorses the contents.
	namedAutoIDs := seedFaces(t, db, "n.jpg", "m1", []float32{1, 0.1})
	autoIDs := seedFaces(t, db, "a.jpg", "m1", []float32{0, 1})
	_ = media.AssignFace(db, userIDs[0], alice, "user")
	_ = media.AssignFace(db, namedAutoIDs[0], alice, "auto")
	_ = media.AssignFace(db, autoIDs[0], unknown, "auto")

	n, err := resetAutoAssignments(db, "m1")
	if err != nil {
		t.Fatal(err)
	}
	if n != 1 {
		t.Fatalf("reset %d, want 1 (only the Unknown cluster's face)", n)
	}
	fUser, _, _ := media.GetFaceByID(db, userIDs[0])
	if fUser.PersonID != alice || fUser.AssignedBy != "user" {
		t.Fatalf("user label harmed: %+v", fUser)
	}
	// The auto face inside the NAMED person survives a reset.
	fNamedAuto, _, _ := media.GetFaceByID(db, namedAutoIDs[0])
	if fNamedAuto.PersonID != alice || fNamedAuto.AssignedBy != "auto" {
		t.Fatalf("named person's auto face scattered: %+v", fNamedAuto)
	}
	fAuto, _, _ := media.GetFaceByID(db, autoIDs[0])
	if fAuto.PersonID != 0 {
		t.Fatalf("unnamed cluster's auto assignment survived: %+v", fAuto)
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
	if _, err := clusterFaces(db, model, defaultClusterParams(model)); err != nil {
		t.Fatal(err)
	}
	f, _, _ := media.GetFaceByID(db, contraIDs[0])
	if f.PersonID != bob || f.AssignedBy != "user" {
		t.Fatalf("clustering moved a user label: %+v", f)
	}
}

// vecAt returns the 2D unit vector at angle rad — precise cosine control for
// threshold tests (cos between vecAt(a) and vecAt(b) = cos(a-b)).
func vecAt(rad float64) []float32 {
	return []float32{float32(math.Cos(rad)), float32(math.Sin(rad))}
}

func TestCorroboratedJoinWidensTheGate(t *testing.T) {
	db := newFaceIndexDB(t)
	resetFaceIndex(t)

	// Alice has THREE user faces ~29.5° from the query (cosine ≈ 0.87 — below
	// the 0.9 threshold but within corroboration slack). One alone must not
	// clear the bar; three together must (0.87 + 2×0.02 bonus ≥ 0.9).
	alice, _ := media.CreatePerson(db, "Alice")
	for i, rad := range []float64{0.510, 0.515, 0.520} {
		ids := seedFaces(t, db, fmt.Sprintf("seed%d.jpg", i), "m1", vecAt(rad))
		_ = media.AssignFace(db, ids[0], alice, "user")
	}
	// Bob has ONE face at the same distance — insufficient evidence.
	bob, _ := media.CreatePerson(db, "Bob")
	bobIDs := seedFaces(t, db, "bobseed.jpg", "m1", vecAt(-0.515))
	_ = media.AssignFace(db, bobIDs[0], bob, "user")

	queryIDs := seedFaces(t, db, "query.jpg", "m1", vecAt(0))

	model := FaceModel{ID: "m1", MatchThreshold: 0.9}
	stats, err := clusterFaces(db, model, clusterParams{
		joinThreshold: 0.9, formThreshold: 0.95, minQuality: 0.75, minCluster: 3, passes: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	if stats.JoinedExisting != 1 {
		t.Fatalf("joined = %d, want 1 (corroborated Alice join)", stats.JoinedExisting)
	}
	f, _, _ := media.GetFaceByID(db, queryIDs[0])
	if f.PersonID != alice {
		t.Fatalf("query joined person %d, want Alice %d (corroboration should beat Bob's single equal match)", f.PersonID, alice)
	}
}

func TestSecondPassPullsInNearDuplicates(t *testing.T) {
	db := newFaceIndexDB(t)
	resetFaceIndex(t)

	// Seed at 0°. Face A at ~23° (cos 0.92 ≥ 0.9 → joins pass 1). Face B at
	// ~31° — cos 0.86 to the seed (no join, no corroborators) but cos 0.99 to
	// A, so it should join once A has become a seed on pass 2.
	alice, _ := media.CreatePerson(db, "Alice")
	seedIDs := seedFaces(t, db, "seed.jpg", "m1", vecAt(0))
	_ = media.AssignFace(db, seedIDs[0], alice, "user")
	seedFaces(t, db, "a.jpg", "m1", vecAt(0.4027)) // cos ≈ 0.920
	bIDs := seedFaces(t, db, "b.jpg", "m1", vecAt(0.5359)) // cos ≈ 0.860 to seed

	model := FaceModel{ID: "m1", MatchThreshold: 0.9}
	onePass := clusterParams{joinThreshold: 0.9, formThreshold: 0.95, minQuality: 0.75, minCluster: 3, passes: 1}
	stats, err := clusterFaces(db, model, onePass)
	if err != nil {
		t.Fatal(err)
	}
	if stats.JoinedExisting != 1 {
		t.Fatalf("one pass joined %d, want 1 (A only)", stats.JoinedExisting)
	}
	fB, _, _ := media.GetFaceByID(db, bIDs[0])
	if fB.PersonID != 0 {
		t.Fatalf("B joined on a single pass: %+v", fB)
	}

	// Second run (equivalent to passes=2 from scratch): A is now a seed.
	stats, err = clusterFaces(db, model, onePass)
	if err != nil {
		t.Fatal(err)
	}
	if stats.JoinedExisting != 1 {
		t.Fatalf("second run joined %d, want 1 (B via A)", stats.JoinedExisting)
	}
	fB, _, _ = media.GetFaceByID(db, bIDs[0])
	if fB.PersonID != alice {
		t.Fatalf("B not pulled in transitively: %+v", fB)
	}
}

func TestQualityFloorBlocksClusterFounding(t *testing.T) {
	db := newFaceIndexDB(t)
	resetFaceIndex(t)

	// Three identical low-confidence detections (blurry background faces)
	// must NOT mint a person.
	for i := 0; i < 3; i++ {
		faces := []media.NewFace{{X: 0.1, Y: 0.1, W: 0.1, H: 0.1, Score: 0.5, Vec: vecAt(0)}}
		if _, err := media.ReplaceFaces(db, fmt.Sprintf("junk%d.jpg", i), "m1", faces, 1); err != nil {
			t.Fatal(err)
		}
	}
	model := FaceModel{ID: "m1", MatchThreshold: 0.9}
	stats, err := clusterFaces(db, model, clusterParams{
		joinThreshold: 0.9, formThreshold: 0.95, minQuality: 0.75, minCluster: 3, passes: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	if stats.NewPeople != 0 || stats.QualitySkipped != 3 {
		t.Fatalf("stats = %+v, want 0 new people and 3 quality-skipped", stats)
	}
}

func TestFormationThresholdStricterThanJoin(t *testing.T) {
	db := newFaceIndexDB(t)
	resetFaceIndex(t)

	// Two clear faces at mutual cosine ≈ 0.92: enough to JOIN an existing
	// person at 0.9, not enough to FORM a new identity at 0.95.
	seedFaces(t, db, "x.jpg", "m1", vecAt(0))
	seedFaces(t, db, "y.jpg", "m1", vecAt(0.4027))

	model := FaceModel{ID: "m1", MatchThreshold: 0.9}
	strict := clusterParams{joinThreshold: 0.9, formThreshold: 0.95, minQuality: 0.75, minCluster: 2, passes: 1}
	stats, err := clusterFaces(db, model, strict)
	if err != nil {
		t.Fatal(err)
	}
	if stats.NewPeople != 0 || stats.Unassigned != 2 {
		t.Fatalf("strict formation: %+v, want no new people", stats)
	}

	relaxed := strict
	relaxed.formThreshold = 0.9
	stats, err = clusterFaces(db, model, relaxed)
	if err != nil {
		t.Fatal(err)
	}
	if stats.NewPeople != 1 || stats.NewlyClustered != 2 {
		t.Fatalf("relaxed formation: %+v, want one new 2-face person", stats)
	}
}
