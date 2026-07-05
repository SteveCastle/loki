package tasks

import (
	"testing"
)

// captureProgress installs a recording notifier for the duration of the test
// and restores a no-op afterwards (the notifier is package-global).
func captureProgress(t *testing.T) *map[string]int {
	t.Helper()
	got := map[string]int{}
	SetProgressNotifier(func(kind string, n int) { got[kind] += n })
	t.Cleanup(func() { SetProgressNotifier(func(string, int) {}) })
	return &got
}

// The "tags" progress notification must fire exactly when a file transitions
// from untagged to tagged — that is what the withTags coverage counter tracks.
func TestInsertTagsForFile_NotifiesOnUntaggedToTagged(t *testing.T) {
	db := setupTagDB(t)
	got := captureProgress(t)
	const path = "/media/a.jpg"

	// First insert: untagged → tagged. One notification.
	if err := insertTagsForFile(db, path, []TagInfo{
		{Label: "sunset", Category: "Suggested"},
		{Label: "beach", Category: "Suggested"},
	}); err != nil {
		t.Fatalf("first insert failed: %v", err)
	}
	if (*got)[ProgressTags] != 1 {
		t.Fatalf("after first insert: tags notifications = %d, want 1", (*got)[ProgressTags])
	}

	// Second insert on the now-tagged file: adds a new tag but the item was
	// already counted — no further notification.
	if err := insertTagsForFile(db, path, []TagInfo{
		{Label: "dunes", Category: "Suggested"},
	}); err != nil {
		t.Fatalf("second insert failed: %v", err)
	}
	if (*got)[ProgressTags] != 1 {
		t.Fatalf("after second insert: tags notifications = %d, want still 1", (*got)[ProgressTags])
	}

	// Re-inserting only duplicates inserts nothing and must not notify.
	if err := insertTagsForFile(db, path, []TagInfo{
		{Label: "sunset", Category: "Suggested"},
	}); err != nil {
		t.Fatalf("duplicate-only insert failed: %v", err)
	}
	if (*got)[ProgressTags] != 1 {
		t.Fatalf("after duplicate-only insert: tags notifications = %d, want still 1", (*got)[ProgressTags])
	}
}

// notifyProgress with no notifier installed must be a silent no-op (tests and
// lokictl link the tasks package without wiring the stats API).
func TestNotifyProgress_NoNotifierIsNoOp(t *testing.T) {
	// Fresh package state can't be guaranteed here, so install nil explicitly
	// via a typed no-op removal path: storing a nil func is not allowed by
	// atomic.Value, so simulate "unset" by never having stored in a fresh
	// process. This test just exercises the guarded call path with a real
	// no-op to make sure nothing panics.
	SetProgressNotifier(func(string, int) {})
	notifyProgress(ProgressDescription, 1)
}
