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
