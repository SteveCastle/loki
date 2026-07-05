package media

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// findOfflineDriveRoot returns a drive root (e.g. `Q:\`) that does not exist
// on this machine, or "" if every letter is taken.
func findOfflineDriveRoot() string {
	for c := 'Z'; c >= 'D'; c-- {
		root := string(c) + `:\`
		if _, err := os.Stat(root); err != nil {
			return root
		}
	}
	return ""
}

// TestVolumeRoot pins the guard's path classification.
func TestVolumeRoot(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("volume roots are Windows drive/UNC concepts")
	}
	cases := map[string]string{
		`C:\media\a.jpg`:        `C:\`,
		`\\nas\share\b\c.jpg`:   `\\nas\share\`,
		`relative\path\img.jpg`: ``,
		`/unix/style/img.jpg`:   ``,
	}
	for path, want := range cases {
		if got := volumeRoot(path); got != want {
			t.Errorf("volumeRoot(%q) = %q, want %q", path, got, want)
		}
	}
}

// TestStreamingCleanup_SkipsOfflineVolumes is the data-safety guard: a path on
// an unmounted drive stats exactly like a deleted file, so cleanup must NOT
// remove it — otherwise unplugging a drive and running cleanup would purge
// that whole volume's library metadata unrecoverably.
func TestStreamingCleanup_SkipsOfflineVolumes(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("volume guard is drive-letter/UNC based")
	}
	offlineRoot := findOfflineDriveRoot()
	if offlineRoot == "" {
		t.Skip("no free drive letter on this machine to simulate an offline volume")
	}

	db := setupTestDB(t)
	defer db.Close()

	// Two rows on the offline volume, plus one genuinely-deleted file on a
	// real volume that cleanup should still remove.
	tmpDir := t.TempDir()
	genuinelyMissing := filepath.Join(tmpDir, "deleted.jpg")
	offlinePaths := []string{
		offlineRoot + `media\one.jpg`,
		offlineRoot + `media\two.jpg`,
	}
	for _, p := range append(offlinePaths, genuinelyMissing) {
		if _, err := db.Exec("INSERT INTO media (path) VALUES (?)", p); err != nil {
			t.Fatalf("insert %s: %v", p, err)
		}
	}

	result, err := StreamingCleanupNonExistentItems(context.Background(), db, nil)
	if err != nil {
		t.Fatalf("StreamingCleanupNonExistentItems() error = %v", err)
	}

	if result.MediaItemsRemoved != 1 {
		t.Errorf("removed %d items, want 1 (only the genuinely-deleted file)", result.MediaItemsRemoved)
	}
	if result.SkippedUnavailable != 2 {
		t.Errorf("SkippedUnavailable = %d, want 2", result.SkippedUnavailable)
	}
	if len(result.UnavailableRoots) != 1 || result.UnavailableRoots[0] != offlineRoot {
		t.Errorf("UnavailableRoots = %v, want [%s]", result.UnavailableRoots, offlineRoot)
	}

	// The offline volume's rows must still be in the database.
	var remaining int
	if err := db.QueryRow("SELECT COUNT(*) FROM media").Scan(&remaining); err != nil {
		t.Fatal(err)
	}
	if remaining != 2 {
		t.Errorf("%d rows remain, want 2 (the offline volume's items)", remaining)
	}
}

// TestRemoveItemsFromDB_CallsRemovalHook verifies the hook fires with each
// committed batch so derived state (the live vector index) can evict paths.
func TestRemoveItemsFromDB_CallsRemovalHook(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	if _, err := db.Exec("INSERT INTO media (path) VALUES ('/lib/a.jpg'), ('/lib/b.jpg')"); err != nil {
		t.Fatal(err)
	}

	var hooked []string
	SetMediaRemovalHook(func(paths []string) { hooked = append(hooked, paths...) })
	defer SetMediaRemovalHook(nil)

	if _, err := RemoveItemsFromDB(context.Background(), db, []string{"/lib/a.jpg", "/lib/b.jpg"}); err != nil {
		t.Fatalf("RemoveItemsFromDB() error = %v", err)
	}

	if len(hooked) != 2 {
		t.Fatalf("removal hook saw %d paths, want 2 (%v)", len(hooked), hooked)
	}
}
