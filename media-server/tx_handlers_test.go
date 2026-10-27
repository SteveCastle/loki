package main

import (
	"net/http"
	"testing"
)

// Renaming a tag onto an existing label must fail as a whole: before, the
// label UPDATE failed (UNIQUE) while the assignments still moved, and the
// handler answered 200.
func TestRenameTagOntoExistingLabelIsAtomic(t *testing.T) {
	deps := testDeps(t)
	rr := postJSON(lokiRenameTagHandler(deps), renameRequest{Label: "sunset", NewLabel: "portrait"})
	if rr.Code != http.StatusConflict {
		t.Fatalf("status = %d, want 409: %s", rr.Code, rr.Body.String())
	}
	var n int
	deps.DB.QueryRow(`SELECT COUNT(*) FROM media_tag_by_category WHERE tag_label = 'sunset'`).Scan(&n)
	if n != 1 {
		t.Fatalf("assignments moved despite the failed rename: %d left on sunset", n)
	}
}

func TestRenameTagSucceeds(t *testing.T) {
	deps := testDeps(t)
	rr := postJSON(lokiRenameTagHandler(deps), renameRequest{Label: "sunset", NewLabel: "dusk"})
	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d: %s", rr.Code, rr.Body.String())
	}
	var n int
	deps.DB.QueryRow(`SELECT COUNT(*) FROM media_tag_by_category WHERE tag_label = 'dusk'`).Scan(&n)
	if n != 1 {
		t.Fatalf("dusk assignments = %d", n)
	}
}
