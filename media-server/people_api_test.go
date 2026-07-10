package main

import (
	"encoding/json"
	"image"
	"image/png"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stevecastle/shrike/appconfig"
	"github.com/stevecastle/shrike/media"
	"github.com/stevecastle/shrike/tasks"
	_ "modernc.org/sqlite"
)

// muxWithPeopleRoutes builds a ServeMux with the face+people routes but no
// auth middleware interference (renderer.AuthMiddleware is nil in tests, and
// ApplyMiddlewares passes through when it is).
func muxWithPeopleRoutes(t *testing.T) (*http.ServeMux, *Dependencies) {
	t.Helper()
	db := newFacesTestDB(t)
	deps := &Dependencies{DB: db}
	mux := http.NewServeMux()
	RegisterPeopleRoutes(mux, deps)
	return mux, deps
}

func doJSON(t *testing.T, mux *http.ServeMux, method, url, body string) (*httptest.ResponseRecorder, map[string]any) {
	t.Helper()
	var reader *strings.Reader
	if body == "" {
		reader = strings.NewReader("{}")
	} else {
		reader = strings.NewReader(body)
	}
	req := httptest.NewRequest(method, url, reader)
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	var out map[string]any
	_ = json.Unmarshal(rec.Body.Bytes(), &out)
	return rec, out
}

func TestPeopleCreateListRenameDelete(t *testing.T) {
	mux, deps := muxWithPeopleRoutes(t)

	rec, created := doJSON(t, mux, http.MethodPost, "/api/people", `{"name":"Alice"}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("create: %d %s", rec.Code, rec.Body.String())
	}
	id := int64(created["id"].(float64))

	// Duplicate name → 400.
	rec, _ = doJSON(t, mux, http.MethodPost, "/api/people", `{"name":"Alice"}`)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("duplicate create: %d", rec.Code)
	}

	// List includes Alice with zero counts.
	req := httptest.NewRequest(http.MethodGet, "/api/people", nil)
	lrec := httptest.NewRecorder()
	mux.ServeHTTP(lrec, req)
	var people []media.Person
	if err := json.Unmarshal(lrec.Body.Bytes(), &people); err != nil {
		t.Fatal(err)
	}
	if len(people) != 1 || people[0].Name != "Alice" {
		t.Fatalf("people = %+v", people)
	}

	// Rename cascades (verified deeper in media tests; here: status + list).
	rec, _ = doJSON(t, mux, http.MethodPost, "/api/people/"+jsonNum(id)+"/rename", `{"name":"Alicia"}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("rename: %d %s", rec.Code, rec.Body.String())
	}
	// Rename of a missing person → 404.
	rec, _ = doJSON(t, mux, http.MethodPost, "/api/people/99999/rename", `{"name":"X"}`)
	if rec.Code != http.StatusNotFound {
		t.Fatalf("rename missing: %d", rec.Code)
	}

	// Delete.
	rec, _ = doJSON(t, mux, http.MethodDelete, "/api/people/"+jsonNum(id), "")
	if rec.Code != http.StatusOK {
		t.Fatalf("delete: %d %s", rec.Code, rec.Body.String())
	}
	if _, found, _ := media.GetPersonByID(deps.DB, id); found {
		t.Fatal("person survived delete")
	}
}

func TestFaceAssignByNameAndUnassign(t *testing.T) {
	mux, deps := muxWithPeopleRoutes(t)
	ids, err := media.ReplaceFaces(deps.DB, "a.jpg", "m1", []media.NewFace{{Score: 0.9, Vec: []float32{1, 0}}}, 1)
	if err != nil {
		t.Fatal(err)
	}
	faceURL := "/api/faces/" + jsonNum(ids[0])

	// Assign by NEW name creates the person.
	rec, out := doJSON(t, mux, http.MethodPost, faceURL+"/assign", `{"name":"Bob"}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("assign: %d %s", rec.Code, rec.Body.String())
	}
	personID := int64(out["personId"].(float64))
	f, _, _ := media.GetFaceByID(deps.DB, ids[0])
	if f.PersonID != personID || f.AssignedBy != "user" {
		t.Fatalf("face = %+v", f)
	}

	// Assign by the SAME name reuses the person (no duplicate).
	ids2, _ := media.ReplaceFaces(deps.DB, "b.jpg", "m1", []media.NewFace{{Score: 0.9, Vec: []float32{0, 1}}}, 1)
	rec, out = doJSON(t, mux, http.MethodPost, "/api/faces/"+jsonNum(ids2[0])+"/assign", `{"name":"Bob"}`)
	if rec.Code != http.StatusOK || int64(out["personId"].(float64)) != personID {
		t.Fatalf("re-assign by name: %d %+v", rec.Code, out)
	}

	// Unassign.
	rec, _ = doJSON(t, mux, http.MethodPost, faceURL+"/unassign", "")
	if rec.Code != http.StatusOK {
		t.Fatalf("unassign: %d", rec.Code)
	}
	f, _, _ = media.GetFaceByID(deps.DB, ids[0])
	if f.PersonID != 0 {
		t.Fatalf("face still assigned: %+v", f)
	}

	// Assign to a nonexistent person id → 404.
	rec, _ = doJSON(t, mux, http.MethodPost, faceURL+"/assign", `{"personId":424242}`)
	if rec.Code != http.StatusNotFound {
		t.Fatalf("assign missing person: %d", rec.Code)
	}
	// Neither personId nor name → 400.
	rec, _ = doJSON(t, mux, http.MethodPost, faceURL+"/assign", `{}`)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("assign empty body: %d", rec.Code)
	}
}

func TestPersonMergeEndpoint(t *testing.T) {
	mux, deps := muxWithPeopleRoutes(t)
	fromID, _ := media.CreatePerson(deps.DB, "Unknown #1")
	intoID, _ := media.CreatePerson(deps.DB, "Alice")
	ids, _ := media.ReplaceFaces(deps.DB, "a.jpg", "m1", []media.NewFace{{Score: 0.9, Vec: []float32{1}}}, 1)
	_ = media.AssignFace(deps.DB, ids[0], fromID, "auto")

	rec, _ := doJSON(t, mux, http.MethodPost, "/api/people/"+jsonNum(fromID)+"/merge", `{"intoId":`+jsonNum(intoID)+`}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("merge: %d %s", rec.Code, rec.Body.String())
	}
	f, _, _ := media.GetFaceByID(deps.DB, ids[0])
	if f.PersonID != intoID {
		t.Fatalf("face not moved: %+v", f)
	}
	// Self-merge → 400.
	rec, _ = doJSON(t, mux, http.MethodPost, "/api/people/"+jsonNum(intoID)+"/merge", `{"intoId":`+jsonNum(intoID)+`}`)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("self merge: %d", rec.Code)
	}
}

func TestPersonMediaEndpoint(t *testing.T) {
	mux, deps := muxWithPeopleRoutes(t)
	pid, _ := media.CreatePerson(deps.DB, "Alice")
	if _, err := deps.DB.Exec(`INSERT INTO media (path, width, height) VALUES ('a.jpg', 10, 10)`); err != nil {
		t.Fatal(err)
	}
	ids, _ := media.ReplaceFaces(deps.DB, "a.jpg", "m1", []media.NewFace{{Score: 0.9, Vec: []float32{1}}}, 1)
	_ = media.AssignFace(deps.DB, ids[0], pid, "user")

	req := httptest.NewRequest(http.MethodGet, "/api/people/"+jsonNum(pid)+"/media", nil)
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("media: %d %s", rec.Code, rec.Body.String())
	}
	var items []map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &items); err != nil {
		t.Fatal(err)
	}
	if len(items) != 1 || items[0]["path"] != "a.jpg" {
		t.Fatalf("items = %+v", items)
	}
	// Missing person → 404.
	req = httptest.NewRequest(http.MethodGet, "/api/people/99999/media", nil)
	rec = httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	if rec.Code != http.StatusNotFound {
		t.Fatalf("missing person media: %d", rec.Code)
	}
}

func TestPersonCoverRegenerateEndpoint(t *testing.T) {
	mux, deps := muxWithPeopleRoutes(t)
	pid, _ := media.CreatePerson(deps.DB, "Alice")

	// One face on an undecodable path (simulates the broken-preview state)
	// and one on a real image file — regenerate must pick the real one even
	// though the broken face scores higher on quality.
	imgPath := filepath.Join(t.TempDir(), "cover.png")
	img := image.NewRGBA(image.Rect(0, 0, 100, 100))
	fh, err := os.Create(imgPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := png.Encode(fh, img); err != nil {
		t.Fatal(err)
	}
	fh.Close()

	brokenIDs, _ := media.ReplaceFaces(deps.DB, `C:\gone\missing.jpg`, "m1", []media.NewFace{
		{X: 0.1, Y: 0.1, W: 0.9, H: 0.9, Score: 0.99, Vec: []float32{1}},
	}, 1)
	goodIDs, _ := media.ReplaceFaces(deps.DB, imgPath, "m1", []media.NewFace{
		{X: 0.2, Y: 0.2, W: 0.4, H: 0.4, Score: 0.9, Vec: []float32{1}},
	}, 1)
	_ = media.AssignFace(deps.DB, brokenIDs[0], pid, "user")
	_ = media.AssignFace(deps.DB, goodIDs[0], pid, "user")
	// Force the stored cover onto the broken face.
	if err := media.SetPersonCover(deps.DB, pid, brokenIDs[0]); err != nil {
		t.Fatal(err)
	}

	rec, out := doJSON(t, mux, http.MethodPost, "/api/people/"+jsonNum(pid)+"/cover", "")
	if rec.Code != http.StatusOK {
		t.Fatalf("regenerate: %d %s", rec.Code, rec.Body.String())
	}
	if int64(out["coverFaceId"].(float64)) != goodIDs[0] {
		t.Fatalf("cover = %v, want renderable face %d", out["coverFaceId"], goodIDs[0])
	}
	p, _, _ := media.GetPersonByID(deps.DB, pid)
	if p.CoverFaceID != goodIDs[0] {
		t.Fatalf("stored cover = %d, want %d", p.CoverFaceID, goodIDs[0])
	}

	// A person with no faces → 422.
	emptyID, _ := media.CreatePerson(deps.DB, "Empty")
	rec, _ = doJSON(t, mux, http.MethodPost, "/api/people/"+jsonNum(emptyID)+"/cover", "")
	if rec.Code != http.StatusUnprocessableEntity {
		t.Fatalf("empty person: %d", rec.Code)
	}
	// Unknown person → 404.
	rec, _ = doJSON(t, mux, http.MethodPost, "/api/people/99999/cover", "")
	if rec.Code != http.StatusNotFound {
		t.Fatalf("missing person: %d", rec.Code)
	}
}

func TestMediaAssignPersonEndpoint(t *testing.T) {
	mux, deps := muxWithPeopleRoutes(t)
	model := tasks.ActiveFaceModel().ID

	// Alice exists with one face pointing at [1,0].
	alice, _ := media.CreatePerson(deps.DB, "Alice")
	seedIDs, _ := media.ReplaceFaces(deps.DB, "seed.jpg", model, []media.NewFace{
		{X: 0.1, Y: 0.1, W: 0.3, H: 0.3, Score: 0.9, Vec: []float32{1, 0}},
	}, 1)
	_ = media.AssignFace(deps.DB, seedIDs[0], alice, "user")

	// Target media (pre-scanned, so no ONNX subprocess is needed) with two
	// faces: a LARGE dissimilar one and a small one similar to Alice. The
	// similar one must win the assignment despite its size.
	targetIDs, _ := media.ReplaceFaces(deps.DB, "target.jpg", model, []media.NewFace{
		{X: 0.0, Y: 0.0, W: 0.8, H: 0.8, Score: 0.95, Vec: []float32{0, 1}},
		{X: 0.7, Y: 0.7, W: 0.1, H: 0.1, Score: 0.85, Vec: []float32{0.98, 0.2}},
	}, 1)

	rec, out := doJSON(t, mux, http.MethodPost, "/api/media/assign-person",
		`{"path":"target.jpg","personId":`+jsonNum(alice)+`,"setCover":true}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("assign: %d %s", rec.Code, rec.Body.String())
	}
	if int64(out["faceId"].(float64)) != targetIDs[1] {
		t.Fatalf("assigned face %v, want the Alice-similar one %d", out["faceId"], targetIDs[1])
	}
	f, _, _ := media.GetFaceByID(deps.DB, targetIDs[1])
	if f.PersonID != alice || f.AssignedBy != "user" {
		t.Fatalf("face = %+v", f)
	}
	// Shift-drop semantics: cover replaced with this face.
	p, _, _ := media.GetPersonByID(deps.DB, alice)
	if p.CoverFaceID != targetIDs[1] {
		t.Fatalf("cover = %d, want %d", p.CoverFaceID, targetIDs[1])
	}
	// The large face stays unassigned (it's someone else in the shot).
	other, _, _ := media.GetFaceByID(deps.DB, targetIDs[0])
	if other.PersonID != 0 {
		t.Fatalf("dissimilar face was assigned too: %+v", other)
	}

	// A pre-scanned no-face item → 422.
	if _, err := media.ReplaceFaces(deps.DB, "empty.jpg", model, nil, 1); err != nil {
		t.Fatal(err)
	}
	rec, _ = doJSON(t, mux, http.MethodPost, "/api/media/assign-person",
		`{"path":"empty.jpg","personId":`+jsonNum(alice)+`}`)
	if rec.Code != http.StatusUnprocessableEntity {
		t.Fatalf("no-face media: %d %s", rec.Code, rec.Body.String())
	}
	// Missing person → 404; missing fields → 400.
	rec, _ = doJSON(t, mux, http.MethodPost, "/api/media/assign-person",
		`{"path":"target.jpg","personId":99999}`)
	if rec.Code != http.StatusNotFound {
		t.Fatalf("missing person: %d", rec.Code)
	}
	rec, _ = doJSON(t, mux, http.MethodPost, "/api/media/assign-person", `{}`)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("missing fields: %d", rec.Code)
	}
}

func TestPersonDeleteWithFaces(t *testing.T) {
	mux, deps := muxWithPeopleRoutes(t)
	pid, _ := media.CreatePerson(deps.DB, "Messy Cluster")
	ids, _ := media.ReplaceFaces(deps.DB, "a.jpg", "m1", []media.NewFace{
		{Score: 0.9, Vec: []float32{1, 0}},
		{Score: 0.8, Vec: []float32{0, 1}},
	}, 1)
	otherIDs, _ := media.ReplaceFaces(deps.DB, "b.jpg", "m1", []media.NewFace{{Score: 0.9, Vec: []float32{1, 1}}}, 1)
	_ = media.AssignFace(deps.DB, ids[0], pid, "auto")
	_ = media.AssignFace(deps.DB, ids[1], pid, "user")

	rec, out := doJSON(t, mux, http.MethodDelete, "/api/people/"+jsonNum(pid)+"?deleteFaces=true", "")
	if rec.Code != http.StatusOK {
		t.Fatalf("purge: %d %s", rec.Code, rec.Body.String())
	}
	if int(out["facesDeleted"].(float64)) != 2 {
		t.Fatalf("facesDeleted = %v, want 2", out["facesDeleted"])
	}
	// Person + its faces gone; unrelated faces and the scan marker survive.
	if _, ok, _ := media.GetPersonByID(deps.DB, pid); ok {
		t.Fatal("person survived purge")
	}
	for _, id := range ids {
		if _, ok, _ := media.GetFaceByID(deps.DB, id); ok {
			t.Fatalf("face %d survived purge", id)
		}
	}
	if _, ok, _ := media.GetFaceByID(deps.DB, otherIDs[0]); !ok {
		t.Fatal("unrelated face was deleted")
	}
	if scanned, _ := media.HasFaceScan(deps.DB, "a.jpg", "m1"); !scanned {
		t.Fatal("scan marker deleted — media would rescan and the junk would return")
	}
}

func TestFacesWipeEndpoint(t *testing.T) {
	mux, deps := muxWithPeopleRoutes(t)
	pid, _ := media.CreatePerson(deps.DB, "Alice")
	ids, _ := media.ReplaceFaces(deps.DB, "a.jpg", "m1", []media.NewFace{{Score: 0.9, Vec: []float32{1}}}, 1)
	_ = media.AssignFace(deps.DB, ids[0], pid, "user")

	// Without confirm → 400 and nothing deleted.
	rec, _ := doJSON(t, mux, http.MethodDelete, "/api/faces/all", "")
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("wipe without confirm: %d", rec.Code)
	}
	if faces, _ := media.GetFaces(deps.DB, "a.jpg", "m1"); len(faces) != 1 {
		t.Fatal("unconfirmed wipe deleted data")
	}

	rec, _ = doJSON(t, mux, http.MethodDelete, "/api/faces/all?confirm=true", "")
	if rec.Code != http.StatusOK {
		t.Fatalf("wipe: %d %s", rec.Code, rec.Body.String())
	}
	if faces, _ := media.GetFaces(deps.DB, "a.jpg", "m1"); len(faces) != 0 {
		t.Fatal("faces survived wipe")
	}
	if people, _ := media.GetPeople(deps.DB); len(people) != 0 {
		t.Fatal("people survived wipe")
	}
}

// The review loop: list a person's faces (least typical first), lock the
// group, reject a face — and the rejection is permanent (veto recorded).
func TestFaceReviewLockAndRejectEndpoints(t *testing.T) {
	mux, deps := muxWithPeopleRoutes(t)

	rec, created := doJSON(t, mux, http.MethodPost, "/api/people", `{"name":"Alice"}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("create: %d", rec.Code)
	}
	personID := int64(created["id"].(float64))

	// Two on-axis faces + one off-axis outlier, all assigned to Alice.
	var faceIDs []int64
	for path, vec := range map[string][]float32{
		"t1.jpg": {1, 0, 0},
		"t2.jpg": {1, 0.05, 0},
		"t3.jpg": {0.2, 1, 0}, // the outlier
	} {
		ids, err := media.ReplaceFaces(deps.DB, path, "m1", []media.NewFace{
			{X: 0.1, Y: 0.1, W: 0.2, H: 0.2, Score: 0.9, Vec: vec},
		}, 1)
		if err != nil {
			t.Fatal(err)
		}
		faceIDs = append(faceIDs, ids[0])
		by := "auto"
		if path == "t1.jpg" {
			by = "user"
		}
		if err := media.AssignFace(deps.DB, ids[0], personID, by); err != nil {
			t.Fatal(err)
		}
	}

	// Review listing: 3 faces, outlier first.
	req := httptest.NewRequest(http.MethodGet, "/api/people/"+jsonNum(personID)+"/faces", nil)
	lrec := httptest.NewRecorder()
	mux.ServeHTTP(lrec, req)
	if lrec.Code != http.StatusOK {
		t.Fatalf("faces: %d %s", lrec.Code, lrec.Body.String())
	}
	var review struct {
		Faces []struct {
			ID         int64   `json:"id"`
			MediaPath  string  `json:"path"`
			AssignedBy string  `json:"assignedBy"`
			Typicality float64 `json:"typicality"`
		} `json:"faces"`
	}
	if err := json.Unmarshal(lrec.Body.Bytes(), &review); err != nil {
		t.Fatal(err)
	}
	if len(review.Faces) != 3 {
		t.Fatalf("faces = %+v", review.Faces)
	}
	if review.Faces[0].MediaPath != "t3.jpg" {
		t.Fatalf("least typical first, got %+v", review.Faces)
	}

	// Lock the group: the two auto faces become user assignments.
	rec, locked := doJSON(t, mux, http.MethodPost, "/api/people/"+jsonNum(personID)+"/lock", "")
	if rec.Code != http.StatusOK || locked["locked"].(float64) != 2 {
		t.Fatalf("lock: %d %v", rec.Code, locked)
	}

	// Reject the outlier: unassigned, vetoed, cannot-linked to the exemplars.
	outlier := review.Faces[0].ID
	rec, rejected := doJSON(t, mux, http.MethodPost, "/api/faces/"+jsonNum(outlier)+"/reject", "")
	if rec.Code != http.StatusOK {
		t.Fatalf("reject: %d %s", rec.Code, rec.Body.String())
	}
	if rejected["cannotLinks"].(float64) != 2 {
		t.Fatalf("reject payload = %v", rejected)
	}
	f, _, _ := media.GetFaceByID(deps.DB, outlier)
	if f.PersonID != 0 {
		t.Fatalf("rejected face still assigned: %+v", f)
	}
	if vetoed, _ := media.FaceVetoExists(deps.DB, outlier, personID); !vetoed {
		t.Fatal("reject did not record a veto")
	}

	// Rejecting an unassigned face without a personId is a 400.
	rec, _ = doJSON(t, mux, http.MethodPost, "/api/faces/"+jsonNum(outlier)+"/reject", "")
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("re-reject without person: %d", rec.Code)
	}
}

// One-shot cleanup: keep the checked faces, permanently discard the rest.
func TestPersonCurateKeepsCheckedDiscardsRest(t *testing.T) {
	mux, deps := muxWithPeopleRoutes(t)

	rec, created := doJSON(t, mux, http.MethodPost, "/api/people", `{"name":"Alice"}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("create: %d", rec.Code)
	}
	personID := int64(created["id"].(float64))

	// One user face, three auto faces; the user checks the user face + one auto.
	assign := []struct {
		path string
		by   string
	}{
		{"k1.jpg", "user"},
		{"k2.jpg", "auto"},
		{"d1.jpg", "auto"},
		{"d2.jpg", "auto"},
	}
	ids := map[string]int64{}
	for i, a := range assign {
		got, err := media.ReplaceFaces(deps.DB, a.path, "m1", []media.NewFace{
			{X: 0.1, Y: 0.1, W: 0.2, H: 0.2, Score: 0.9 - float64(i)*0.01, Vec: []float32{1, 0}},
		}, 1)
		if err != nil {
			t.Fatal(err)
		}
		ids[a.path] = got[0]
		if err := media.AssignFace(deps.DB, got[0], personID, a.by); err != nil {
			t.Fatal(err)
		}
	}

	body := `{"keepFaceIds":[` + jsonNum(ids["k1.jpg"]) + `,` + jsonNum(ids["k2.jpg"]) + `]}`
	rec, out := doJSON(t, mux, http.MethodPost, "/api/people/"+jsonNum(personID)+"/curate", body)
	if rec.Code != http.StatusOK {
		t.Fatalf("curate: %d %s", rec.Code, rec.Body.String())
	}
	if out["kept"].(float64) != 2 || out["rejected"].(float64) != 2 {
		t.Fatalf("curate payload = %v, want kept 2 rejected 2", out)
	}

	// Keeps are locked; discards are unassigned AND vetoed.
	for _, p := range []string{"k1.jpg", "k2.jpg"} {
		f, _, _ := media.GetFaceByID(deps.DB, ids[p])
		if f.PersonID != personID || f.AssignedBy != "user" {
			t.Fatalf("keeper %s not locked: %+v", p, f)
		}
	}
	for _, p := range []string{"d1.jpg", "d2.jpg"} {
		f, _, _ := media.GetFaceByID(deps.DB, ids[p])
		if f.PersonID != 0 {
			t.Fatalf("discard %s still assigned: %+v", p, f)
		}
		if vetoed, _ := media.FaceVetoExists(deps.DB, ids[p], personID); !vetoed {
			t.Fatalf("discard %s has no veto", p)
		}
	}
}

// Dragging the "New group" chip onto media mints a person from that image's
// face: the face is pulled out of its current cluster (user move + veto) and
// becomes the new group's locked seed and cover.
func TestMediaAssignNewPersonCreatesGroup(t *testing.T) {
	mux, deps := muxWithPeopleRoutes(t)
	model := tasks.ActiveFaceModel().ID

	// The image's face currently sits in an auto cluster.
	oldGroup, _ := media.CreatePerson(deps.DB, "Old Cluster")
	ids, _ := media.ReplaceFaces(deps.DB, "hero.jpg", model, []media.NewFace{
		{X: 0.1, Y: 0.1, W: 0.5, H: 0.5, Score: 0.95, Vec: []float32{1, 0}},
	}, 1)
	_ = media.AssignFace(deps.DB, ids[0], oldGroup, "auto")

	rec, out := doJSON(t, mux, http.MethodPost, "/api/media/assign-person",
		`{"path":"hero.jpg","newPerson":true}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("new person: %d %s", rec.Code, rec.Body.String())
	}
	if out["created"] != true {
		t.Fatalf("payload = %v", out)
	}
	newID := int64(out["personId"].(float64))
	if newID == oldGroup {
		t.Fatal("did not create a new person")
	}
	if out["name"].(string) != "Unknown #1" {
		t.Fatalf("name = %v, want auto-generated Unknown #1", out["name"])
	}

	// The face moved, is locked, and is the new group's cover.
	f, _, _ := media.GetFaceByID(deps.DB, ids[0])
	if f.PersonID != newID || f.AssignedBy != "user" {
		t.Fatalf("face = %+v, want locked into new person %d", f, newID)
	}
	p, _, _ := media.GetPersonByID(deps.DB, newID)
	if p.CoverFaceID != ids[0] {
		t.Fatalf("cover = %d, want %d", p.CoverFaceID, ids[0])
	}
	// Pulling it out recorded the negative assertion against the old group.
	if vetoed, _ := media.FaceVetoExists(deps.DB, ids[0], oldGroup); !vetoed {
		t.Fatal("user move off the old cluster did not record a veto")
	}

	// A custom name is honored.
	ids2, _ := media.ReplaceFaces(deps.DB, "hero2.jpg", model, []media.NewFace{
		{X: 0.1, Y: 0.1, W: 0.5, H: 0.5, Score: 0.95, Vec: []float32{0, 1}},
	}, 1)
	rec, out = doJSON(t, mux, http.MethodPost, "/api/media/assign-person",
		`{"path":"hero2.jpg","newPerson":true,"name":"Zed"}`)
	if rec.Code != http.StatusOK || out["name"].(string) != "Zed" {
		t.Fatalf("named new person: %d %v", rec.Code, out)
	}
	f2, _, _ := media.GetFaceByID(deps.DB, ids2[0])
	if f2.AssignedBy != "user" {
		t.Fatalf("named group's seed not locked: %+v", f2)
	}

	// A face-less item must NOT leave an empty group behind.
	if _, err := media.ReplaceFaces(deps.DB, "blank.jpg", model, nil, 1); err != nil {
		t.Fatal(err)
	}
	before, _ := media.GetPeople(deps.DB)
	rec, _ = doJSON(t, mux, http.MethodPost, "/api/media/assign-person",
		`{"path":"blank.jpg","newPerson":true}`)
	if rec.Code != http.StatusUnprocessableEntity {
		t.Fatalf("no-face new person: %d", rec.Code)
	}
	after, _ := media.GetPeople(deps.DB)
	if len(after) != len(before) {
		t.Fatalf("empty group created: %d -> %d people", len(before), len(after))
	}
}

func TestFacesUngroupedListAndNewGroup(t *testing.T) {
	mux, deps := muxWithPeopleRoutes(t)
	model := tasks.ActiveFaceModel().ID
	ids, err := media.ReplaceFaces(deps.DB, "a.jpg", model, []media.NewFace{
		{Score: 0.7, Vec: []float32{1, 0}},
		{Score: 0.95, Vec: []float32{0, 1}},
	}, 1)
	if err != nil {
		t.Fatal(err)
	}

	// ?faces=1 lists the ungrouped faces, best detections first.
	req := httptest.NewRequest(http.MethodGet, "/api/faces/ungrouped?faces=1&limit=10", nil)
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	var out struct {
		Count int `json:"count"`
		Faces []struct {
			ID       int64   `json:"id"`
			Path     string  `json:"path"`
			DetScore float64 `json:"detScore"`
		} `json:"faces"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	if out.Count != 2 || len(out.Faces) != 2 {
		t.Fatalf("list = %+v", out)
	}
	if out.Faces[0].DetScore < out.Faces[1].DetScore {
		t.Fatalf("not best-first: %+v", out.Faces)
	}

	// {newPerson:true} mints an auto-named group seeded (and covered) by the
	// face, as a locked user assignment.
	arec, aout := doJSON(t, mux, http.MethodPost,
		"/api/faces/"+jsonNum(ids[0])+"/assign", `{"newPerson":true}`)
	if arec.Code != http.StatusOK {
		t.Fatalf("newPerson assign: %d %s", arec.Code, arec.Body.String())
	}
	if aout["created"] != true || !strings.HasPrefix(aout["name"].(string), "Unknown #") {
		t.Fatalf("assign out = %v", aout)
	}
	f, _, _ := media.GetFaceByID(deps.DB, ids[0])
	if f.PersonID == 0 || f.AssignedBy != "user" {
		t.Fatalf("face = %+v, want locked into the new person", f)
	}
	p, found, _ := media.GetPersonByID(deps.DB, f.PersonID)
	if !found || p.CoverFaceID != ids[0] {
		t.Fatalf("person = %+v, want cover face %d", p, ids[0])
	}

	// The assigned face left the ungrouped pool.
	req = httptest.NewRequest(http.MethodGet, "/api/faces/ungrouped", nil)
	rec = httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	var after map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &after); err != nil {
		t.Fatal(err)
	}
	if int(after["count"].(float64)) != 1 {
		t.Fatalf("count after assign = %v, want 1", after["count"])
	}
}

func TestPersonDeleteRecordsGroupBan(t *testing.T) {
	mux, deps := muxWithPeopleRoutes(t)
	model := tasks.ActiveFaceModel().ID

	makeBlob := func(path, name string) int64 {
		ids, err := media.ReplaceFaces(deps.DB, path, model, []media.NewFace{
			{Score: 0.9, Vec: []float32{1, 0}},
			{Score: 0.9, Vec: []float32{0, 1}},
			{Score: 0.9, Vec: []float32{0.7, 0.7}},
		}, 1)
		if err != nil {
			t.Fatal(err)
		}
		pid, err := media.CreatePerson(deps.DB, name)
		if err != nil {
			t.Fatal(err)
		}
		for _, id := range ids {
			if err := media.AssignFace(deps.DB, id, pid, "auto"); err != nil {
				t.Fatal(err)
			}
		}
		return pid
	}

	// Default delete records the dissolved membership as a ban.
	pid := makeBlob("a.jpg", "Blob A")
	rec, out := doJSON(t, mux, http.MethodDelete, "/api/people/"+jsonNum(pid), "")
	if rec.Code != http.StatusOK {
		t.Fatalf("delete: %d %s", rec.Code, rec.Body.String())
	}
	if out["banned"].(float64) != 3 {
		t.Fatalf("banned = %v, want 3", out["banned"])
	}
	bans, err := media.FaceGroupBans(deps.DB, model)
	if err != nil {
		t.Fatal(err)
	}
	if len(bans) != 1 || len(bans[0].Members) != 3 {
		t.Fatalf("bans = %+v", bans)
	}

	// ?ban=false dissolves without recording.
	pid2 := makeBlob("b.jpg", "Blob B")
	rec, out = doJSON(t, mux, http.MethodDelete, "/api/people/"+jsonNum(pid2)+"?ban=false", "")
	if rec.Code != http.StatusOK || out["banned"].(float64) != 0 {
		t.Fatalf("ban=false delete: %d %v", rec.Code, out)
	}
	if bans, _ = media.FaceGroupBans(deps.DB, model); len(bans) != 1 {
		t.Fatalf("bans after ban=false = %+v", bans)
	}
}

func TestFacesTuningRoundTrip(t *testing.T) {
	prev := appconfig.Get()
	t.Cleanup(func() { appconfig.Set(prev) })
	cfg := prev
	cfg.FaceClusterThresholdOffset = 0
	cfg.FaceClusterMinCluster = 0
	cfg.FaceClusterMinQuality = 0
	appconfig.Set(cfg)
	mux, _ := muxWithPeopleRoutes(t)

	// Nothing saved → effective built-in defaults.
	req := httptest.NewRequest(http.MethodGet, "/api/faces/tuning", nil)
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	var out map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	if out["thresholdOffset"].(float64) != 0 || out["minCluster"].(float64) != 3 || out["minQuality"].(float64) != 0.75 {
		t.Fatalf("defaults = %v", out)
	}

	// Save and read back — the in-memory config (what clustering reads) is
	// updated too.
	prec, pout := doJSON(t, mux, http.MethodPost, "/api/faces/tuning",
		`{"thresholdOffset":0.05,"minCluster":5,"minQuality":0.6}`)
	if prec.Code != http.StatusOK {
		t.Fatalf("save: %d %s", prec.Code, prec.Body.String())
	}
	if pout["minCluster"].(float64) != 5 {
		t.Fatalf("saved = %v", pout)
	}
	got := appconfig.Get()
	if got.FaceClusterThresholdOffset != 0.05 || got.FaceClusterMinCluster != 5 || got.FaceClusterMinQuality != 0.6 {
		t.Fatalf("config = %+v", got)
	}

	// Out-of-range values are rejected.
	for _, body := range []string{
		`{"thresholdOffset":0.9,"minCluster":5,"minQuality":0.6}`,
		`{"thresholdOffset":0,"minCluster":0,"minQuality":0.6}`,
		`{"thresholdOffset":0,"minCluster":5,"minQuality":1.5}`,
	} {
		if brec, _ := doJSON(t, mux, http.MethodPost, "/api/faces/tuning", body); brec.Code != http.StatusBadRequest {
			t.Fatalf("body %s: %d, want 400", body, brec.Code)
		}
	}
}

func TestFacesUngroupedCount(t *testing.T) {
	mux, deps := muxWithPeopleRoutes(t)

	// Two unassigned faces under the shipped default model, one assigned.
	// (Only known models count — the tasks-package test covers that split.)
	model := tasks.ActiveFaceModel().ID
	ids, err := media.ReplaceFaces(deps.DB, "a.jpg", model, []media.NewFace{
		{Score: 0.9, Vec: []float32{1, 0}},
		{Score: 0.9, Vec: []float32{0, 1}},
	}, 1)
	if err != nil {
		t.Fatal(err)
	}
	alice, _ := media.CreatePerson(deps.DB, "Alice")
	if err := media.AssignFace(deps.DB, ids[0], alice, "user"); err != nil {
		t.Fatal(err)
	}

	req := httptest.NewRequest(http.MethodGet, "/api/faces/ungrouped", nil)
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("ungrouped: %d %s", rec.Code, rec.Body.String())
	}
	var out map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	if int(out["count"].(float64)) != 1 {
		t.Fatalf("count = %v, want 1", out["count"])
	}

	// Wrong method → 405.
	rec, _ = doJSON(t, mux, http.MethodPost, "/api/faces/ungrouped", "")
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("POST: %d", rec.Code)
	}
}

// Removing a person's tag from a media item must discard their faces from the
// group — veto + unassign, so clustering can't put them back — via both entry
// points: the dedicated reject-person endpoint (Electron viewer) and the
// assignments DELETE (web UI / lokictl).
func TestRejectPersonFacesOnTagRemoval(t *testing.T) {
	mux, deps := muxWithPeopleRoutes(t)

	// Two of Carol's faces on a.jpg, one on b.jpg.
	idsA, err := media.ReplaceFaces(deps.DB, "a.jpg", "m1", []media.NewFace{
		{Score: 0.9, Vec: []float32{1, 0}},
		{Score: 0.8, Vec: []float32{0.9, 0.1}},
	}, 1)
	if err != nil {
		t.Fatal(err)
	}
	idsB, err := media.ReplaceFaces(deps.DB, "b.jpg", "m1", []media.NewFace{
		{Score: 0.9, Vec: []float32{0.8, 0.2}},
	}, 1)
	if err != nil {
		t.Fatal(err)
	}
	carol, err := media.CreatePerson(deps.DB, "Carol")
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range []int64{idsA[0], idsA[1], idsB[0]} {
		if err := media.AssignFace(deps.DB, id, carol, "user"); err != nil {
			t.Fatal(err)
		}
	}

	// Electron path: reject by name, one item.
	rec, out := doJSON(t, mux, http.MethodPost, "/api/media/reject-person",
		`{"path":"a.jpg","name":"Carol"}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("reject-person: %d %s", rec.Code, rec.Body.String())
	}
	if got := len(out["rejectedFaceIds"].([]any)); got != 2 {
		t.Fatalf("rejected %d faces, want 2", got)
	}
	for _, id := range idsA {
		f, _, _ := media.GetFaceByID(deps.DB, id)
		if f.PersonID != 0 {
			t.Fatalf("face %d still assigned after reject", id)
		}
		if vetoed, _ := media.FaceVetoExists(deps.DB, id, carol); !vetoed {
			t.Fatalf("face %d has no veto", id)
		}
	}
	// The other item's face is untouched.
	if f, _, _ := media.GetFaceByID(deps.DB, idsB[0]); f.PersonID != carol {
		t.Fatalf("b.jpg face lost its assignment: %+v", f)
	}

	// Unknown person → 404 (the caller treats it as nothing-to-discard).
	rec, _ = doJSON(t, mux, http.MethodPost, "/api/media/reject-person",
		`{"path":"a.jpg","name":"Nobody"}`)
	if rec.Code != http.StatusNotFound {
		t.Fatalf("unknown person: %d", rec.Code)
	}

	// Web path: deleting the person-tag assignment rejects the faces too.
	amux := http.NewServeMux()
	amux.HandleFunc("/api/assignments", lokiDeleteAssignmentHandler(deps))
	rec, _ = doJSON(t, amux, http.MethodDelete, "/api/assignments",
		`{"mediaPath":"b.jpg","tag":{"tag_label":"Carol"}}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("delete assignment: %d %s", rec.Code, rec.Body.String())
	}
	f, _, _ := media.GetFaceByID(deps.DB, idsB[0])
	if f.PersonID != 0 {
		t.Fatal("face still assigned after tag removal")
	}
	if vetoed, _ := media.FaceVetoExists(deps.DB, idsB[0], carol); !vetoed {
		t.Fatal("tag removal recorded no veto")
	}
}
