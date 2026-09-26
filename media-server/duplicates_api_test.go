package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/stevecastle/shrike/media"
)

func dupReq(t *testing.T, h http.HandlerFunc, method, target, id, body string) *httptest.ResponseRecorder {
	t.Helper()
	var req *http.Request
	if body != "" {
		req = httptest.NewRequest(method, target, strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
	} else {
		req = httptest.NewRequest(method, target, nil)
	}
	if id != "" {
		req.SetPathValue("id", id)
	}
	rec := httptest.NewRecorder()
	h(rec, req)
	return rec
}

func decodeJSON(t *testing.T, rec *httptest.ResponseRecorder, into any) {
	t.Helper()
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	if err := json.Unmarshal(rec.Body.Bytes(), into); err != nil {
		t.Fatalf("decode %q: %v", rec.Body.String(), err)
	}
}

func TestDuplicatesAPIReviewFlow(t *testing.T) {
	db := newFacesTestDB(t)
	deps := &Dependencies{DB: db}
	dir := t.TempDir()
	a, b, c := filepath.Join(dir, "a.jpg"), filepath.Join(dir, "b.jpg"), filepath.Join(dir, "c.jpg")
	for _, p := range []string{a, b, c} {
		if err := os.WriteFile(p, []byte("bytes"), 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := db.Exec(`INSERT INTO media (path, width, height) VALUES (?, 640, 480)`, p); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := db.Exec(`INSERT INTO media_tag_by_category (media_path, tag_label, category_label, weight, time_stamp)
		VALUES (?, 'beach', 'Scene', 1, 0)`, b); err != nil {
		t.Fatal(err)
	}
	gid, err := media.CreateDuplicateGroup(db, "m", 0.9995, a, []media.DuplicateMember{
		{Path: a, Score: 1}, {Path: b, Score: 0.9999}, {Path: c, Score: 0.9996},
	})
	if err != nil || gid == 0 {
		t.Fatal(err)
	}
	id := strconv.FormatInt(gid, 10)

	// List: one pending group, members in the renderer item shape with
	// score/anchor/excluded attached and media columns joined in.
	var list struct {
		Groups []struct {
			ID      int64 `json:"id"`
			Status  string
			Members []map[string]any
		}
		Total int
	}
	decodeJSON(t, dupReq(t, duplicatesListHandler(deps), http.MethodGet, "/api/duplicates", "", ""), &list)
	if list.Total != 1 || len(list.Groups) != 1 || len(list.Groups[0].Members) != 3 {
		t.Fatalf("list = %+v", list)
	}
	first := list.Groups[0].Members[0]
	if first["path"] != a || first["anchor"] != true || first["width"] != float64(640) || first["score"] != float64(1) {
		t.Errorf("first member = %v; want the anchor with media columns", first)
	}
	// Equal dimensions and sizes everywhere: the keeper rule falls back to
	// the anchor, and the view says so.
	if kp, ok := list.Groups[0].Members[0]["keeper"].(bool); !ok || !kp {
		t.Errorf("anchor should be the keeper on an all-equal group: %v", list.Groups[0].Members[0])
	}

	// Stats and per-path lookup.
	var stats media.DuplicateStats
	decodeJSON(t, dupReq(t, duplicatesStatsHandler(deps), http.MethodGet, "/api/duplicates/stats", "", ""), &stats)
	if stats.Pending != 1 || stats.PendingItems != 3 {
		t.Errorf("stats = %+v", stats)
	}
	var forPath struct {
		GroupID int64 `json:"groupId"`
		Status  string
	}
	decodeJSON(t, dupReq(t, duplicatesForPathHandler(deps), http.MethodGet, "/api/duplicates/for-path?path="+urlQuery(c), "", ""), &forPath)
	if forPath.GroupID != gid || forPath.Status != "pending" {
		t.Errorf("for-path = %+v; want group %d pending", forPath, gid)
	}

	// Exclude c: it is not a duplicate; the group stays pending with a+b.
	var excl struct {
		Group struct{ ActiveCount int }
	}
	decodeJSON(t, dupReq(t, duplicateExcludeHandler(deps), http.MethodPost, "/api/duplicates/"+id+"/exclude", id,
		`{"path":`+jsonString(c)+`}`), &excl)
	if excl.Group.ActiveCount != 2 {
		t.Errorf("active after exclude = %d; want 2", excl.Group.ActiveCount)
	}
	if rec := dupReq(t, duplicateExcludeHandler(deps), http.MethodPost, "/api/duplicates/"+id+"/exclude", id,
		`{"path":"nope"}`); rec.Code != http.StatusNotFound {
		t.Errorf("excluding a non-member: status %d; want 404", rec.Code)
	}

	// Dismiss / restore.
	if rec := dupReq(t, duplicateStatusHandler(deps, media.DuplicateStatusDismissed), http.MethodPost,
		"/api/duplicates/"+id+"/dismiss", id, ""); rec.Code != http.StatusOK {
		t.Fatalf("dismiss: %d %s", rec.Code, rec.Body.String())
	}
	decodeJSON(t, dupReq(t, duplicatesStatsHandler(deps), http.MethodGet, "/api/duplicates/stats", "", ""), &stats)
	if stats.Pending != 0 || stats.Dismissed != 1 {
		t.Errorf("stats after dismiss = %+v", stats)
	}
	if rec := dupReq(t, duplicateStatusHandler(deps, media.DuplicateStatusPending), http.MethodPost,
		"/api/duplicates/"+id+"/restore", id, ""); rec.Code != http.StatusOK {
		t.Fatalf("restore: %d", rec.Code)
	}

	// Merge: keep b (the tagged one); default sources = the other ACTIVE
	// members, so a is deleted and the excluded c is untouched. With only b
	// and c (excluded) left the group is pruned? No — two member rows remain
	// (b active, c excluded), so the group survives as a dismissed-by-count
	// record; what matters is that a is gone and c is still on disk.
	var merged struct {
		Merge struct {
			Target  string
			Deleted []string
			Tags    int64
		}
		Group *struct {
			MemberCount int
			ActiveCount int
		}
	}
	decodeJSON(t, dupReq(t, duplicateMergeHandler(deps), http.MethodPost, "/api/duplicates/"+id+"/merge", id,
		`{"keep":`+jsonString(b)+`}`), &merged)
	if merged.Merge.Target != b || len(merged.Merge.Deleted) != 1 || merged.Merge.Deleted[0] != a {
		t.Errorf("merge = %+v; want b kept, a deleted", merged.Merge)
	}
	if _, err := os.Stat(a); !os.IsNotExist(err) {
		t.Errorf("a.jpg should be deleted from disk")
	}
	if _, err := os.Stat(c); err != nil {
		t.Errorf("excluded c.jpg must not be merged: %v", err)
	}
	if merged.Group == nil || merged.Group.MemberCount != 2 || merged.Group.ActiveCount != 1 {
		t.Errorf("group after merge = %+v; want 2 rows (b + excluded c), 1 active", merged.Group)
	}
	var mediaRows int
	_ = db.QueryRow(`SELECT COUNT(*) FROM media`).Scan(&mediaRows)
	if mediaRows != 2 {
		t.Errorf("media rows = %d; want 2", mediaRows)
	}

	// Merging with nothing active left is a 400, not a crash.
	if rec := dupReq(t, duplicateMergeHandler(deps), http.MethodPost, "/api/duplicates/"+id+"/merge", id,
		`{"keep":`+jsonString(b)+`}`); rec.Code != http.StatusBadRequest {
		t.Errorf("merge with no sources: %d; want 400", rec.Code)
	}
	// A keep outside the group is rejected.
	if rec := dupReq(t, duplicateMergeHandler(deps), http.MethodPost, "/api/duplicates/"+id+"/merge", id,
		`{"keep":"/elsewhere.jpg"}`); rec.Code != http.StatusBadRequest {
		t.Errorf("merge with foreign keep: %d; want 400", rec.Code)
	}

	// Delete the group record; then wipe requires confirm.
	if rec := dupReq(t, duplicateGroupHandler(deps), http.MethodDelete, "/api/duplicates/"+id, id, ""); rec.Code != http.StatusOK {
		t.Fatalf("delete: %d %s", rec.Code, rec.Body.String())
	}
	if rec := dupReq(t, duplicateGroupHandler(deps), http.MethodGet, "/api/duplicates/"+id, id, ""); rec.Code != http.StatusNotFound {
		t.Errorf("get after delete: %d; want 404", rec.Code)
	}
	if rec := dupReq(t, duplicatesWipeHandler(deps), http.MethodDelete, "/api/duplicates/all", "", ""); rec.Code != http.StatusBadRequest {
		t.Errorf("wipe without confirm: %d; want 400", rec.Code)
	}
	if rec := dupReq(t, duplicatesWipeHandler(deps), http.MethodDelete, "/api/duplicates/all?confirm=true", "", ""); rec.Code != http.StatusOK {
		t.Errorf("wipe: %d", rec.Code)
	}
}

func TestDuplicatesAPIRejectsBadInput(t *testing.T) {
	deps := &Dependencies{DB: newFacesTestDB(t)}
	if rec := dupReq(t, duplicatesListHandler(deps), http.MethodGet, "/api/duplicates?status=bogus", "", ""); rec.Code != http.StatusBadRequest {
		t.Errorf("bad status: %d", rec.Code)
	}
	if rec := dupReq(t, duplicateGroupHandler(deps), http.MethodGet, "/api/duplicates/x", "x", ""); rec.Code != http.StatusBadRequest {
		t.Errorf("bad id: %d", rec.Code)
	}
	if rec := dupReq(t, duplicateStatusHandler(deps, "pending"), http.MethodPost, "/api/duplicates/99/restore", "99", ""); rec.Code != http.StatusNotFound {
		t.Errorf("unknown group: %d; want 404", rec.Code)
	}
	if rec := dupReq(t, duplicatesForPathHandler(deps), http.MethodGet, "/api/duplicates/for-path", "", ""); rec.Code != http.StatusBadRequest {
		t.Errorf("for-path without path: %d", rec.Code)
	}
}

func jsonString(s string) string {
	b, _ := json.Marshal(s)
	return string(b)
}

func urlQuery(s string) string {
	return strings.NewReplacer("\\", "%5C", " ", "%20", ":", "%3A").Replace(s)
}

// TestDuplicatesAPISortByMembers pins the server-side ordering the panel's
// "Most items" toggle relies on (sorting must happen in SQL so paging holds).
func TestDuplicatesAPISortByMembers(t *testing.T) {
	db := newFacesTestDB(t)
	deps := &Dependencies{DB: db}
	mk := func(anchor string, others ...string) int64 {
		t.Helper()
		members := []media.DuplicateMember{{Path: anchor, Score: 1}}
		for _, o := range others {
			members = append(members, media.DuplicateMember{Path: o, Score: 0.999})
		}
		id, err := media.CreateDuplicateGroup(db, "m", 0.9995, anchor, members)
		if err != nil || id == 0 {
			t.Fatalf("create %s: id=%d err=%v", anchor, id, err)
		}
		return id
	}
	small := mk("/s1.jpg", "/s2.jpg")
	big := mk("/b1.jpg", "/b2.jpg", "/b3.jpg", "/b4.jpg")
	newest := mk("/n1.jpg", "/n2.jpg", "/n3.jpg")

	ids := func(query string) []int64 {
		t.Helper()
		var list struct {
			Groups []struct {
				ID int64 `json:"id"`
			}
		}
		decodeJSON(t, dupReq(t, duplicatesListHandler(deps), http.MethodGet, "/api/duplicates"+query, "", ""), &list)
		out := make([]int64, 0, len(list.Groups))
		for _, g := range list.Groups {
			out = append(out, g.ID)
		}
		return out
	}
	if got := ids(""); len(got) != 3 || got[0] != newest || got[1] != big || got[2] != small {
		t.Errorf("default order = %v; want newest first [%d %d %d]", got, newest, big, small)
	}
	if got := ids("?sort=members"); len(got) != 3 || got[0] != big || got[1] != newest || got[2] != small {
		t.Errorf("members order = %v; want biggest first [%d %d %d]", got, big, newest, small)
	}
	// Paging under the members sort: page 2 of size 1 is the second-biggest.
	if got := ids("?sort=members&limit=1&offset=1"); len(got) != 1 || got[0] != newest {
		t.Errorf("members page 2 = %v; want [%d]", got, newest)
	}
	if rec := dupReq(t, duplicatesListHandler(deps), http.MethodGet, "/api/duplicates?sort=bogus", "", ""); rec.Code != http.StatusBadRequest {
		t.Errorf("bogus sort: %d; want 400", rec.Code)
	}
}
