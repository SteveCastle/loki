package tasks

import (
	"fmt"
	"math"
	"testing"

	"github.com/stevecastle/shrike/appconfig"
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

// TestClusterFacesOnlyFaceIDsRestrictsCandidates pins the incremental-pass
// contract: with onlyFaceIDs set, faces outside the batch are not candidates
// — the unassigned backlog is neither joined nor formed into clusters (its
// cost is skipped entirely) — while batch faces still join people and mint
// Unknowns, matched against ALL assigned seeds.
func TestClusterFacesOnlyFaceIDsRestrictsCandidates(t *testing.T) {
	db := newFaceIndexDB(t)
	resetFaceIndex(t)

	alice, err := media.CreatePerson(db, "Alice")
	if err != nil {
		t.Fatal(err)
	}
	seedIDs := seedFaces(t, db, "seed.jpg", "m1", []float32{1, 0, 0})
	if err := media.AssignFace(db, seedIDs[0], alice, "user"); err != nil {
		t.Fatal(err)
	}

	// Backlog (NOT in the batch): a face near Alice and a trio that would
	// mint an Unknown in a full pass. None of them may move.
	backlogJoin := seedFaces(t, db, "old.jpg", "m1", vecNear([]float32{1, 0, 0}, 0.05))
	backlogTrio := [][]int64{
		seedFaces(t, db, "ob1.jpg", "m1", vecNear([]float32{0, 1, 0}, 0.03)),
		seedFaces(t, db, "ob2.jpg", "m1", vecNear([]float32{0, 1, 0}, -0.03)),
		seedFaces(t, db, "ob3.jpg", "m1", []float32{0, 1, 0}),
	}

	// The batch: a face near Alice (joins) and a trio elsewhere (mints).
	batchJoin := seedFaces(t, db, "new.jpg", "m1", vecNear([]float32{1, 0, 0}, -0.05))
	batchTrio := [][]int64{
		seedFaces(t, db, "nb1.jpg", "m1", vecNear([]float32{0, 0, 1}, 0.03)),
		seedFaces(t, db, "nb2.jpg", "m1", vecNear([]float32{0, 0, 1}, -0.03)),
		seedFaces(t, db, "nb3.jpg", "m1", []float32{0, 0, 1}),
	}
	only := map[int64]bool{batchJoin[0]: true}
	for _, ids := range batchTrio {
		only[ids[0]] = true
	}

	model := FaceModel{ID: "m1", MatchThreshold: 0.9}
	params := clusterParams{joinThreshold: 0.9, formThreshold: 0.95, minQuality: 0.75, minCluster: 3, passes: 2, onlyFaceIDs: only}
	stats, err := clusterFaces(db, model, params)
	if err != nil {
		t.Fatal(err)
	}
	if stats.JoinedExisting != 1 {
		t.Fatalf("joined = %d, want 1 (batch face only)", stats.JoinedExisting)
	}
	if stats.NewPeople != 1 || stats.NewlyClustered != 3 {
		t.Fatalf("new people = %d (%d faces), want 1 (3, the batch trio)", stats.NewPeople, stats.NewlyClustered)
	}

	f, _, _ := media.GetFaceByID(db, batchJoin[0])
	if f.PersonID != alice {
		t.Fatalf("batch face should have joined Alice: %+v", f)
	}
	f, _, _ = media.GetFaceByID(db, backlogJoin[0])
	if f.PersonID != 0 {
		t.Fatalf("backlog face must not be touched by a restricted pass: %+v", f)
	}
	for _, ids := range backlogTrio {
		f, _, _ := media.GetFaceByID(db, ids[0])
		if f.PersonID != 0 {
			t.Fatalf("backlog trio face must not cluster in a restricted pass: %+v", f)
		}
	}
	for _, ids := range batchTrio {
		f, _, _ := media.GetFaceByID(db, ids[0])
		if f.PersonID == 0 {
			t.Fatalf("batch trio face should have minted an Unknown: %+v", f)
		}
	}

	// A follow-up FULL pass (nil restriction) settles the backlog exactly as
	// the end-of-scan pass does.
	params.onlyFaceIDs = nil
	stats, err = clusterFaces(db, model, params)
	if err != nil {
		t.Fatal(err)
	}
	if stats.JoinedExisting != 1 || stats.NewPeople != 1 {
		t.Fatalf("full pass should settle the backlog: %+v", stats)
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

// Deleting a group records its membership as a dissolved-group ban — a
// negative attractor: no automatic pass may reunite the majority of it, so
// the same nonsense blob can't re-form. Genuine subsets (below the majority
// line) plus fresh faces must still group freely.
func TestDissolvedGroupDoesNotReform(t *testing.T) {
	db := newFaceIndexDB(t)
	resetFaceIndex(t)

	x := []float32{1, 0, 0}
	y := []float32{0, 1, 0}
	// A nonsense blob spanning two identities: 2 faces at Y, 4 at X.
	var members []int64
	members = append(members, seedFaces(t, db, "y.jpg", "m1", y, y)...)
	members = append(members, seedFaces(t, db, "x.jpg", "m1", x, x, x, x)...)
	blob, err := media.CreatePerson(db, "Unknown #1")
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range members {
		if err := media.AssignFace(db, id, blob, "auto"); err != nil {
			t.Fatal(err)
		}
	}

	// The user deletes the blob (handler order: ban first, then delete).
	if n, err := media.BanFaceGroup(db, blob, "Unknown #1"); err != nil || n != 6 {
		t.Fatalf("ban: n=%d err=%v", n, err)
	}
	if err := media.DeletePerson(db, blob); err != nil {
		t.Fatal(err)
	}

	// Four fresh faces land at Y — plenty of new evidence for a Y group.
	newIDs := seedFaces(t, db, "new.jpg", "m1", y, y, y, y)

	model := FaceModel{ID: "m1", MatchThreshold: 0.9}
	params := clusterParams{joinThreshold: 0.9, formThreshold: 0.95, minQuality: 0.75, minCluster: 3, passes: 2}
	stats, err := clusterFaces(db, model, params)
	if err != nil {
		t.Fatal(err)
	}
	// The X cluster would reunite 4/6 of the banned group → blocked. The Y
	// cluster holds only 2 banned members (under the overlap floor) → minted.
	if stats.NewPeople != 1 {
		t.Fatalf("new people = %d, want 1 (the Y group only): %+v", stats.NewPeople, stats)
	}
	if stats.BanBlocked != 4 {
		t.Fatalf("ban-blocked = %d, want 4 (the X cluster): %+v", stats.BanBlocked, stats)
	}
	fX, _, _ := media.GetFaceByID(db, members[2])
	if fX.PersonID != 0 {
		t.Fatalf("banned X face regrouped: %+v", fX)
	}
	fY, _, _ := media.GetFaceByID(db, members[0])
	fNew, _, _ := media.GetFaceByID(db, newIDs[0])
	if fNew.PersonID == 0 || fY.PersonID != fNew.PersonID {
		t.Fatalf("Y faces should form one new group: banned=%+v fresh=%+v", fY, fNew)
	}
}

// The saved grouping tuner (server config) must reach every clustering pass
// via defaultClusterParams, with zero values reading as "built-in default"
// and the incremental pass keeping the stricter of its floor and the tuned one.
func TestClusterParamsUseSavedTuning(t *testing.T) {
	prev := appconfig.Get()
	t.Cleanup(func() { appconfig.Set(prev) })

	cfg := prev
	cfg.FaceClusterThresholdOffset = 0.05
	cfg.FaceClusterMinCluster = 5
	cfg.FaceClusterMinQuality = 0.6
	appconfig.Set(cfg)

	model := FaceModel{ID: "m1", MatchThreshold: 0.5}
	p := defaultClusterParams(model)
	if math.Abs(float64(p.joinThreshold)-0.55) > 1e-6 {
		t.Fatalf("joinThreshold = %v, want 0.55", p.joinThreshold)
	}
	if math.Abs(float64(p.formThreshold)-0.60) > 1e-6 {
		t.Fatalf("formThreshold = %v, want 0.60", p.formThreshold)
	}
	if p.minCluster != 5 || p.minQuality != 0.6 {
		t.Fatalf("minCluster=%d minQuality=%v, want 5 / 0.6", p.minCluster, p.minQuality)
	}
	inc := incrementalClusterParams(model)
	if inc.minQuality != 0.8 {
		t.Fatalf("incremental minQuality = %v, want 0.8 (stricter of tuned 0.6 and 0.8)", inc.minQuality)
	}
	if inc.minCluster != 7 {
		t.Fatalf("incremental minCluster = %d, want 7 (tuned 5 + 2)", inc.minCluster)
	}

	// A tuned floor ABOVE the incremental one stays in force.
	cfg.FaceClusterMinQuality = 0.9
	appconfig.Set(cfg)
	if inc := incrementalClusterParams(model); inc.minQuality != 0.9 {
		t.Fatalf("incremental minQuality = %v, want 0.9", inc.minQuality)
	}

	// Zero values = built-in defaults.
	cfg.FaceClusterThresholdOffset = 0
	cfg.FaceClusterMinCluster = 0
	cfg.FaceClusterMinQuality = 0
	appconfig.Set(cfg)
	p = defaultClusterParams(model)
	if math.Abs(float64(p.joinThreshold)-0.5) > 1e-6 || p.minCluster != minAutoClusterSize || p.minQuality != 0.75 {
		t.Fatalf("defaults not restored: %+v", p)
	}
}

// A confirmed core must anchor its cluster. Before anchoring, the mean guard
// used a weighted all-member mean, so once enough of a lookalike's faces had
// snuck in they outvoted the confirmed faces (user weight 3 loses to volume),
// the center drifted onto the lookalike, and the cluster hoovered up the rest
// of their images. Now the guard measures candidates against the
// user-confirmed faces alone.
func TestClusterAnchorsToConfirmedCore(t *testing.T) {
	db := newFaceIndexDB(t)
	resetFaceIndex(t)

	alice, err := media.CreatePerson(db, "Alice")
	if err != nil {
		t.Fatal(err)
	}
	// Two hand-confirmed faces define Alice at [1,0,0].
	for _, id := range seedFaces(t, db, "core.jpg", "m1", []float32{1, 0, 0}, []float32{1, 0, 0}) {
		if err := media.AssignFace(db, id, alice, "user"); err != nil {
			t.Fatal(err)
		}
	}
	// Three drifted AUTO members — a lookalike's faces that snuck in
	// (cosine 0.7 to the confirmed core).
	drift := []float32{0.7, 0.7141, 0}
	for _, id := range seedFaces(t, db, "drift.jpg", "m1", drift, drift, drift) {
		if err := media.AssignFace(db, id, alice, "auto"); err != nil {
			t.Fatal(err)
		}
	}

	// More of the lookalike: identical to the drifted members, so best match
	// is 1.0 with heavy corroboration, and the old weighted all-member mean
	// ((2·3·0.7 + 3·1.0)/9 = 0.8) cleared the 0.78 floor — it would have
	// joined. Against the confirmed core alone it scores 0.7 and must not.
	bobIDs := seedFaces(t, db, "bob.jpg", "m1", drift)
	// A genuine face near the confirmed core still joins.
	genuineIDs := seedFaces(t, db, "alice2.jpg", "m1", []float32{0.95, 0.3122, 0})

	model := FaceModel{ID: "m1", MatchThreshold: 0.9}
	params := clusterParams{joinThreshold: 0.9, formThreshold: 0.95, minQuality: 0.75, minCluster: 3, passes: 2}
	stats, err := clusterFaces(db, model, params)
	if err != nil {
		t.Fatal(err)
	}
	if stats.JoinedExisting != 1 {
		t.Fatalf("joined = %d, want 1 (the genuine face only)", stats.JoinedExisting)
	}
	fBob, _, _ := media.GetFaceByID(db, bobIDs[0])
	if fBob.PersonID != 0 {
		t.Fatalf("lookalike joined the confirmed cluster: %+v", fBob)
	}
	fGenuine, _, _ := media.GetFaceByID(db, genuineIDs[0])
	if fGenuine.PersonID != alice {
		t.Fatalf("genuine face refused: %+v", fGenuine)
	}
}

func TestCountUngroupedFaces(t *testing.T) {
	db := newFaceIndexDB(t)
	resetFaceIndex(t)

	// Routing is on by default, so faces under every KNOWN model count:
	// two unassigned sface + one unassigned anime-ccip = 3.
	sfaceIDs := seedFaces(t, db, "s.jpg", "sface", []float32{1, 0}, []float32{0, 1})
	seedFaces(t, db, "a.jpg", "anime-ccip", []float32{1, 0})
	// Assigned faces don't count.
	alice, err := media.CreatePerson(db, "Alice")
	if err != nil {
		t.Fatal(err)
	}
	assignedIDs := seedFaces(t, db, "b.jpg", "sface", []float32{0.5, 0.5})
	if err := media.AssignFace(db, assignedIDs[0], alice, "user"); err != nil {
		t.Fatal(err)
	}
	// Faces stored under a model no longer known (retired recognizer) are
	// invisible to clustering and must not inflate the count.
	seedFaces(t, db, "old.jpg", "retired-model", []float32{1, 0})

	n, err := CountUngroupedFaces(db)
	if err != nil {
		t.Fatal(err)
	}
	if n != 3 {
		t.Fatalf("ungrouped = %d, want 3", n)
	}

	// Assigning one drops the count.
	if err := media.AssignFace(db, sfaceIDs[0], alice, "auto"); err != nil {
		t.Fatal(err)
	}
	n, err = CountUngroupedFaces(db)
	if err != nil {
		t.Fatal(err)
	}
	if n != 2 {
		t.Fatalf("ungrouped after assign = %d, want 2", n)
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

// The join rule is single-linkage at heart (best match against ANY member),
// so without a guard a person grows by transitive chaining: seed→A→B where B
// barely resembles the seed. The mean-similarity guard must stop the chain at
// the hop where the face no longer resembles the person AS A WHOLE. This is
// the mechanism that produced the "341 of 376 faces in one Unknown" collapse
// with weak (SFace) embeddings.
func TestMeanGuardBlocksChainDrift(t *testing.T) {
	db := newFaceIndexDB(t)
	resetFaceIndex(t)

	alice, _ := media.CreatePerson(db, "Alice")
	seedIDs := seedFaces(t, db, "seed.jpg", "m1", vecAt(0))
	_ = media.AssignFace(db, seedIDs[0], alice, "user")
	// A at 0.449 rad: cos ≈ 0.9008 to the seed → joins pass 1 (mean 0.90).
	seedFaces(t, db, "a.jpg", "m1", vecAt(0.449))
	// B at 0.898 rad: cos ≈ 0.9008 to A (single-linkage would chain it in on
	// pass 2) but only ≈ 0.623 to the seed — mean (0.623+0.901)/2 ≈ 0.762,
	// below the 0.9 − meanJoinSlack = 0.78 floor. Must stay out.
	bIDs := seedFaces(t, db, "b.jpg", "m1", vecAt(0.898))

	model := FaceModel{ID: "m1", MatchThreshold: 0.9}
	stats, err := clusterFaces(db, model, clusterParams{
		joinThreshold: 0.9, formThreshold: 0.95, minQuality: 0.75, minCluster: 3, passes: 3,
	})
	if err != nil {
		t.Fatal(err)
	}
	if stats.JoinedExisting != 1 {
		t.Fatalf("joined = %d, want 1 (A only; B is a chain hop)", stats.JoinedExisting)
	}
	fB, _, _ := media.GetFaceByID(db, bIDs[0])
	if fB.PersonID != 0 {
		t.Fatalf("chain drift: B joined via A despite low mean similarity: %+v", fB)
	}
}

// meanPairwise must report the true mean pairwise cosine, NOT the inflated
// cosine-to-normalized-centroid. For n unit vectors a·e0 + b·eᵢ (orthogonal
// noise axes), pairwise cosine is a² everywhere, while mean-to-centroid is
// ||Σx||/n = sqrt-scale inflated (the old coherence gate's bug: a blob at a
// near-random 0.2 internal similarity scored 0.6 against its own centroid).
func TestMeanPairwiseNotCentroidInflated(t *testing.T) {
	const n, dim = 5, 6
	a := float32(math.Sqrt(0.2))
	b := float32(math.Sqrt(0.8))
	sum := make([]float32, dim)
	for i := 0; i < n; i++ {
		v := make([]float32, dim)
		v[0] = a
		v[1+i] = b
		for k := range sum {
			sum[k] += v[k]
		}
	}
	got := meanPairwise(sum, n)
	if math.Abs(float64(got)-0.2) > 1e-4 {
		t.Fatalf("meanPairwise = %.4f, want 0.2 (centroid metric would say 0.6)", got)
	}
}

func TestResetAllAutoAssignmentsKeepsOnlyUserLabels(t *testing.T) {
	db := newFaceIndexDB(t)
	resetFaceIndex(t)

	alice, _ := media.CreatePerson(db, "Alice")
	unknown, _ := media.CreatePerson(db, "Unknown #1")
	userIDs := seedFaces(t, db, "u.jpg", "m1", []float32{1, 0})
	namedAutoIDs := seedFaces(t, db, "n.jpg", "m1", []float32{1, 0.1})
	autoIDs := seedFaces(t, db, "a.jpg", "m1", []float32{0, 1})
	_ = media.AssignFace(db, userIDs[0], alice, "user")
	_ = media.AssignFace(db, namedAutoIDs[0], alice, "auto")
	_ = media.AssignFace(db, autoIDs[0], unknown, "auto")

	n, err := resetAllAutoAssignments(db, "m1")
	if err != nil {
		t.Fatal(err)
	}
	if n != 2 {
		t.Fatalf("reset %d, want 2 (every auto face, even inside named Alice)", n)
	}
	fUser, _, _ := media.GetFaceByID(db, userIDs[0])
	if fUser.PersonID != alice || fUser.AssignedBy != "user" {
		t.Fatalf("user label harmed: %+v", fUser)
	}
	fNamedAuto, _, _ := media.GetFaceByID(db, namedAutoIDs[0])
	if fNamedAuto.PersonID != 0 {
		t.Fatalf("named person's auto face survived a full reset: %+v", fNamedAuto)
	}
	// Alice survives (user-made, holds a user face); the Unknown dissolves.
	if _, ok, _ := media.GetPersonByID(db, alice); !ok {
		t.Fatal("Alice dissolved by full reset")
	}
	if _, ok, _ := media.GetPersonByID(db, unknown); ok {
		t.Fatal("emptied Unknown person survived full reset")
	}
}

// A rejected face must never rejoin the person it was removed from, no matter
// how well its vector matches — the veto is a standing user assertion.
func TestRejectedFaceNeverRejoinsPerson(t *testing.T) {
	db := newFaceIndexDB(t)
	resetFaceIndex(t)

	alice, _ := media.CreatePerson(db, "Alice")
	seedIDs := seedFaces(t, db, "seed.jpg", "m1", []float32{1, 0, 0})
	_ = media.AssignFace(db, seedIDs[0], alice, "user")
	// Identical twin lookalike the user has rejected from Alice.
	twinIDs := seedFaces(t, db, "twin.jpg", "m1", vecNear([]float32{1, 0, 0}, 0.01))
	if _, err := media.RejectFaceFromPerson(db, twinIDs[0], alice); err != nil {
		t.Fatal(err)
	}

	model := FaceModel{ID: "m1", MatchThreshold: 0.9}
	stats, err := clusterFaces(db, model, defaultClusterParams(model))
	if err != nil {
		t.Fatal(err)
	}
	if stats.JoinedExisting != 0 {
		t.Fatalf("joined = %d, want 0 (only candidate is vetoed)", stats.JoinedExisting)
	}
	f, _, _ := media.GetFaceByID(db, twinIDs[0])
	if f.PersonID != 0 {
		t.Fatalf("vetoed face rejoined: %+v", f)
	}
}

// The cannot-link half of a rejection must hold even when the group it was
// recorded against is DISSOLVED (reset deletes the anonymous person, killing
// the veto) and re-forms from the same faces under a new person id — the
// user's "never again" survives reclustering with new settings.
func TestRejectionSurvivesClusterDissolveAndReform(t *testing.T) {
	db := newFaceIndexDB(t)
	resetFaceIndex(t)

	model := FaceModel{ID: "m1", MatchThreshold: 0.9}
	params := clusterParams{joinThreshold: 0.9, formThreshold: 0.9, minQuality: 0.75, minCluster: 3, passes: 2}

	// Four near-identical faces form "Unknown #1".
	var ids []int64
	for i, eps := range []float32{0, 0.01, -0.01, 0.02} {
		got := seedFaces(t, db, fmt.Sprintf("c%d.jpg", i), "m1", vecNear([]float32{0, 1, 0}, eps))
		ids = append(ids, got[0])
	}
	if _, err := clusterFaces(db, model, params); err != nil {
		t.Fatal(err)
	}
	f, _, _ := media.GetFaceByID(db, ids[3])
	unknown := f.PersonID
	if unknown == 0 {
		t.Fatal("setup: face did not cluster")
	}

	// The user throws one face out of the group…
	if _, err := media.RejectFaceFromPerson(db, ids[3], unknown); err != nil {
		t.Fatal(err)
	}
	// …then rebuilds everything from scratch (the veto's person dies here).
	if _, err := resetAllAutoAssignments(db, "m1"); err != nil {
		t.Fatal(err)
	}
	if _, ok, _ := media.GetPersonByID(db, unknown); ok {
		t.Fatal("setup: emptied Unknown person should have dissolved")
	}
	stats, err := clusterFaces(db, model, params)
	if err != nil {
		t.Fatal(err)
	}
	if stats.NewPeople != 1 {
		t.Fatalf("reform: %+v, want the trio to re-cluster", stats)
	}
	f, _, _ = media.GetFaceByID(db, ids[3])
	if f.PersonID != 0 {
		t.Fatalf("rejected face rejoined the re-formed cluster (person %d)", f.PersonID)
	}
	// And phase 1 of yet another pass can't pull it in either.
	if _, err := clusterFaces(db, model, params); err != nil {
		t.Fatal(err)
	}
	f, _, _ = media.GetFaceByID(db, ids[3])
	if f.PersonID != 0 {
		t.Fatalf("rejected face joined via phase 1 after reform: %+v", f)
	}
}

// A single USER seed at borderline distance must beat a single AUTO seed at
// the same distance: human validation carries userSeedWeight in corroboration,
// so the user-anchored person clears the gate where the auto one can't.
func TestUserSeedsOutweighAutoSeeds(t *testing.T) {
	db := newFaceIndexDB(t)
	resetFaceIndex(t)

	// Both seeds sit ~29.5° from the query (cosine ≈ 0.87: inside the 0.06
	// corroboration slack of the 0.9 threshold, but below it).
	alice, _ := media.CreatePerson(db, "Alice")
	aliceIDs := seedFaces(t, db, "aliceseed.jpg", "m1", vecAt(0.515))
	_ = media.AssignFace(db, aliceIDs[0], alice, "user")
	bob, _ := media.CreatePerson(db, "Bob")
	bobIDs := seedFaces(t, db, "bobseed.jpg", "m1", vecAt(-0.515))
	_ = media.AssignFace(db, bobIDs[0], bob, "auto")

	queryIDs := seedFaces(t, db, "query.jpg", "m1", vecAt(0))

	model := FaceModel{ID: "m1", MatchThreshold: 0.9}
	stats, err := clusterFaces(db, model, clusterParams{
		joinThreshold: 0.9, formThreshold: 0.95, minQuality: 0.75, minCluster: 3, passes: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	if stats.JoinedExisting != 1 {
		t.Fatalf("joined = %d, want 1 (the user-seeded Alice)", stats.JoinedExisting)
	}
	f, _, _ := media.GetFaceByID(db, queryIDs[0])
	if f.PersonID != alice {
		t.Fatalf("query joined person %d, want Alice %d — user seed must outweigh an equal auto seed", f.PersonID, alice)
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
