package tasks

import (
	"context"
	"database/sql"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// openFileLibrary creates an on-disk library database with (optionally) a
// media table, so discovery can find it as a sibling of the main database.
func openFileLibrary(t *testing.T, path string, withMedia bool) *sql.DB {
	t.Helper()
	db, err := sql.Open("sqlite", "file:"+path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	stmts := []string{`CREATE TABLE tag (label TEXT PRIMARY KEY, thumbnail_path_600 TEXT)`}
	if withMedia {
		stmts = append(stmts,
			`CREATE TABLE media ("path" TEXT PRIMARY KEY, thumbnail_path_600 TEXT)`,
			`CREATE TABLE media_tag_by_category (media_path TEXT, tag_label TEXT, category_label TEXT, weight REAL, time_stamp REAL)`)
	} else {
		stmts = append(stmts, `CREATE TABLE settings (k TEXT PRIMARY KEY, v TEXT)`)
	}
	for _, s := range stmts {
		if _, err := db.Exec(s); err != nil {
			t.Fatal(err)
		}
	}
	return db
}

func TestThumbnailCleanupDiscoversSiblingLibraries(t *testing.T) {
	dir := t.TempDir()
	mainPath := filepath.Join(dir, "dream.sqlite")
	mainDB := openFileLibrary(t, mainPath, true)
	otherPath := filepath.Join(dir, "dream-x.sqlite")
	otherDB := openFileLibrary(t, otherPath, true)
	openFileLibrary(t, filepath.Join(dir, "settings.db"), false) // SQLite, but not a library
	elsewhere := filepath.Join(t.TempDir(), "archive.sqlite")
	archiveDB := openFileLibrary(t, elsewhere, true)
	// Noise that must never be opened as a library.
	for _, name := range []string{"dream.sqlite-wal", "dream.sqlite-shm", "notes.txt", "fake.sqlite"} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte("not a database"), 0644); err != nil {
			t.Fatal(err)
		}
	}

	onlyMain := `C:\lib\main-only.jpg`
	onlyOther := `C:\lib\other-only.jpg`
	both := `C:\lib\both.jpg`
	onlyArchive := `C:\lib\archive-only.jpg`
	if _, err := mainDB.Exec(`INSERT INTO media ("path") VALUES (?), (?)`, onlyMain, both); err != nil {
		t.Fatal(err)
	}
	if _, err := otherDB.Exec(`INSERT INTO media ("path") VALUES (?), (?)`, onlyOther, both); err != nil {
		t.Fatal(err)
	}
	if _, err := archiveDB.Exec(`INSERT INTO media ("path") VALUES (?)`, onlyArchive); err != nil {
		t.Fatal(err)
	}

	expect := map[string]string{ // cache file -> expected owner labels
		writeThumb(t, dir, "thumbnail_path_600", thumbHashHex(onlyMain)):          "dream.sqlite",
		writeThumb(t, dir, "thumbnail_path_600", thumbHashHex(onlyOther)):         "dream-x.sqlite",
		writeThumb(t, dir, "thumbnail_path_600", thumbHashHex(both)):              "dream.sqlite;dream-x.sqlite",
		writeThumb(t, dir, "thumbnail_path_600", thumbHashHex(onlyArchive)):       "archive.sqlite",
		writeThumb(t, dir, "thumbnail_path_600", thumbHashHex(`C:\lib\gone.jpg`)): "-",
	}

	var logs []string
	logf := func(s string) { logs = append(logs, s) }

	// Discovery alone: siblings found, the settings database skipped, the
	// noise ignored, the explicit extra appended after the siblings.
	libs, err := discoverThumbLibraries(context.Background(), mainPath, mainDB, []string{elsewhere, mainPath}, true, logf)
	if err != nil {
		t.Fatalf("discoverThumbLibraries: %v (logs %v)", err, logs)
	}
	defer closeThumbLibraries(libs)
	var labels []string
	for _, l := range libs {
		labels = append(labels, l.Label)
	}
	if got := strings.Join(labels, ","); got != "dream.sqlite,dream-x.sqlite,archive.sqlite" {
		t.Fatalf("libraries = %s, want dream.sqlite,dream-x.sqlite,archive.sqlite (logs %v)", got, logs)
	}
	if len(logs) != 1 || !strings.Contains(logs[0], "settings.db") {
		t.Errorf("expected one skip note for settings.db, got %v", logs)
	}
	for i, l := range libs {
		if l.bit != 1<<uint(i) {
			t.Errorf("library %d bit = %d", i, l.bit)
		}
	}

	ix, err := buildThumbCacheIndex(context.Background(), dir)
	if err != nil {
		t.Fatal(err)
	}
	for _, lib := range libs {
		ix.bit = lib.bit
		if _, err := markLiveThumbnails(context.Background(), lib.DB, ix, true); err != nil {
			t.Fatalf("mark %s: %v", lib.Label, err)
		}
	}

	manifest := filepath.Join(t.TempDir(), "manifest.tsv")
	if err := writeThumbManifest(manifest, ix, libs); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(manifest)
	if err != nil {
		t.Fatal(err)
	}
	got := map[string]string{}
	for i, line := range strings.Split(strings.TrimSpace(string(raw)), "\n") {
		if i == 0 {
			if line != "file\tlibraries" {
				t.Errorf("manifest header = %q", line)
			}
			continue
		}
		file, owners, ok := strings.Cut(line, "\t")
		if !ok {
			t.Errorf("bad manifest line %q", line)
			continue
		}
		got[file] = owners
	}
	for file, want := range expect {
		if got[file] != want {
			t.Errorf("%s owners = %q, want %q", filepath.Base(file), got[file], want)
		}
	}
	if len(got) != len(expect) {
		t.Errorf("manifest has %d rows, want %d", len(got), len(expect))
	}

	report := strings.Join(thumbOwnershipReport(ix, libs), "\n")
	for _, want := range []string{
		"dream.sqlite needs 2 file(s)",
		"dream-x.sqlite needs 2 file(s)",
		"archive.sqlite needs 1 file(s)",
		"shared by two or more: 1",
		"named by none (orphans): 1",
	} {
		if !strings.Contains(report, want) {
			t.Errorf("report missing %q:\n%s", want, report)
		}
	}

	// Only the orphan is unkept.
	for i := range ix.entries {
		want := expect[ix.path(i)] != "-"
		if ix.isKept(i) != want {
			t.Errorf("kept(%s) = %v, want %v", filepath.Base(ix.path(i)), ix.isKept(i), want)
		}
	}
}

func TestThumbnailCleanupDiscoveryFailsClosed(t *testing.T) {
	dir := t.TempDir()
	mainPath := filepath.Join(dir, "dream.sqlite")
	mainDB := openFileLibrary(t, mainPath, true)
	// An explicit library that does not exist must abort, not be skipped:
	// the caller believes it shares the cache.
	_, err := discoverThumbLibraries(context.Background(), mainPath, mainDB,
		[]string{filepath.Join(dir, "missing.sqlite")}, false, func(string) {})
	if err == nil {
		t.Fatal("expected an error for an unreadable explicit library")
	}
}

func TestThumbnailCleanupScopedAcrossLibraries(t *testing.T) {
	dir := t.TempDir()
	mainPath := filepath.Join(dir, "dream.sqlite")
	mainDB := openFileLibrary(t, mainPath, true)
	otherDB := openFileLibrary(t, filepath.Join(dir, "dream-x.sqlite"), true)
	scope := filepath.Join(t.TempDir(), "scope")
	if err := os.MkdirAll(scope, 0755); err != nil {
		t.Fatal(err)
	}
	inOther := filepath.Join(scope, "other.jpg") // on disk, only the second library knows it
	orphan := filepath.Join(scope, "gone.jpg")   // on disk, nobody knows it
	for _, p := range []string{inOther, orphan} {
		if err := os.WriteFile(p, []byte("m"), 0644); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := otherDB.Exec(`INSERT INTO media ("path") VALUES (?)`, inOther); err != nil {
		t.Fatal(err)
	}
	keepFile := writeThumb(t, dir, "thumbnail_path_600", thumbHashHex(inOther))
	dropFile := writeThumb(t, dir, "thumbnail_path_600", thumbHashHex(orphan))

	libs, err := discoverThumbLibraries(context.Background(), mainPath, mainDB, nil, true, func(string) {})
	if err != nil {
		t.Fatal(err)
	}
	defer closeThumbLibraries(libs)
	if len(libs) != 2 {
		t.Fatalf("libraries = %d, want 2", len(libs))
	}
	ix, err := buildThumbCacheIndex(context.Background(), dir)
	if err != nil {
		t.Fatal(err)
	}
	cands, st, err := thumbScopeCandidates(context.Background(), libs, scope, func(string) {})
	if err != nil {
		t.Fatal(err)
	}
	victims, err := thumbScopeVictims(context.Background(), libs, ix, cands, &st)
	if err != nil {
		t.Fatal(err)
	}
	if st.Live != 1 || st.OrphanPaths != 1 {
		t.Errorf("stats = %+v, want 1 live (via the second library), 1 orphan", st)
	}
	if len(victims) != 1 || ix.path(victims[0].Index) != dropFile {
		t.Errorf("victims = %v, want only %s", victims, filepath.Base(dropFile))
	}
	_ = keepFile
}
