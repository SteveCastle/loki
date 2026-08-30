package tasks

import (
	"context"
	"database/sql"
	"os"
	"path/filepath"
	"testing"
	"time"

	_ "modernc.org/sqlite"
)

func newThumbCleanupDB(t *testing.T) *sql.DB {
	t.Helper()
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatalf("open db: %v", err)
	}
	t.Cleanup(func() { db.Close() })
	for _, stmt := range []string{
		`CREATE TABLE media ("path" TEXT PRIMARY KEY, thumbnail_path_600 TEXT, thumbnail_path_1200 TEXT)`,
		`CREATE TABLE tag (label TEXT PRIMARY KEY, thumbnail_path_600 TEXT)`,
		`CREATE TABLE media_tag_by_category (media_path TEXT, tag_label TEXT, category_label TEXT, weight REAL, time_stamp REAL)`,
	} {
		if _, err := db.Exec(stmt); err != nil {
			t.Fatalf("schema: %v", err)
		}
	}
	return db
}

func TestThumbTimeStampKeyMatchesJSNumberToString(t *testing.T) {
	// The hash input must match what the Electron worker builds with
	// `timeStamp.toString()` — no trailing zeros, no forced decimal point.
	cases := map[float64]string{
		5:      "5",
		5.5:    "5.5",
		0.04:   "0.04",
		120.25: "120.25",
	}
	for in, want := range cases {
		if got := thumbTimeStampKey(in); got != want {
			t.Errorf("thumbTimeStampKey(%v) = %q, want %q", in, got, want)
		}
	}
}

func TestThumbnailCleanupClassification(t *testing.T) {
	db := newThumbCleanupDB(t)
	baseDir := t.TempDir()
	cacheDir := filepath.Join(baseDir, "thumbnail_path_600")
	if err := os.MkdirAll(cacheDir, 0755); err != nil {
		t.Fatal(err)
	}

	imgPath := `C:\pics\a.jpg`
	vidPath := `C:\vids\b.mp4`
	if _, err := db.Exec(`INSERT INTO media ("path") VALUES (?), (?)`, imgPath, vidPath); err != nil {
		t.Fatal(err)
	}
	// A tag pinned at 5.5s on the live video: its timestamped thumbnail must
	// survive. The same timestamp on a path with no media row must not.
	if _, err := db.Exec(
		`INSERT INTO media_tag_by_category (media_path, tag_label, category_label, weight, time_stamp)
		 VALUES (?, 'scene', 'Marks', 0, 5.5), ('C:\gone\dead.mp4', 'scene', 'Marks', 0, 5.5)`,
		vidPath); err != nil {
		t.Fatal(err)
	}

	// A tag preview referenced by literal path — not derivable from any media
	// row, protected purely by the reference.
	refName := thumbHashHex("some historic input") + ".png"
	refPath := filepath.Join(cacheDir, refName)
	if _, err := db.Exec(`INSERT INTO tag (label, thumbnail_path_600) VALUES ('sunset', ?)`, refPath); err != nil {
		t.Fatal(err)
	}

	mk := func(name string) string {
		p := filepath.Join(cacheDir, name)
		if err := os.WriteFile(p, []byte("x"), 0644); err != nil {
			t.Fatal(err)
		}
		old := time.Now().Add(-2 * time.Hour)
		if err := os.Chtimes(p, old, old); err != nil {
			t.Fatal(err)
		}
		return p
	}

	kept := map[string]bool{
		mk(thumbHashHex(imgPath)):                     true, // image thumb: bare digest
		mk(thumbHashHex(`C:/pics/a.jpg`)):             true, // separator-variant spelling
		mk(thumbHashHex(vidPath) + ".mp4"):            true, // video thumb
		mk(thumbHashHex(vidPath+"5.5") + ".mp4"):      true, // tagged-timestamp thumb
		mk(refName):                                   true, // literal DB reference
		mk(thumbHashHex(`C:\gone\z.jpg`)):             false,
		mk(thumbHashHex(`C:\gone\dead.mp4`) + ".mp4"): false,
		mk(thumbHashHex(`C:\gone\dead.mp4`+"5.5") + ".mp4"): false, // orphaned tag timestamp
	}
	mk("notes.txt") // not our naming scheme — must be ignored entirely

	keep, err := loadThumbKeepSet(context.Background(), db)
	if err != nil {
		t.Fatalf("loadThumbKeepSet: %v", err)
	}
	files, unrecognized, err := listThumbCacheFiles(baseDir)
	if err != nil {
		t.Fatalf("listThumbCacheFiles: %v", err)
	}
	if unrecognized != 1 {
		t.Errorf("unrecognized = %d, want 1 (notes.txt)", unrecognized)
	}
	if len(files) != len(kept) {
		t.Fatalf("listed %d candidate files, want %d", len(files), len(kept))
	}
	for _, f := range files {
		want, known := kept[f.Path]
		if !known {
			t.Errorf("unexpected candidate %s", f.Path)
			continue
		}
		if got := keep.keeps(f); got != want {
			t.Errorf("keeps(%s) = %v, want %v", filepath.Base(f.Path), got, want)
		}
	}
}
