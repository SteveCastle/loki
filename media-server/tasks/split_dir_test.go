package tasks

import (
	"database/sql"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/stevecastle/shrike/embedindex"
	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/media"
)

// The point of the split task is that the filesystem and the library move
// together. These tests assert both halves: where the bytes ended up, and that
// EVERY path-bearing table followed them.

func newSplitDB(t *testing.T) *sql.DB {
	t.Helper()
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	t.Cleanup(func() { db.Close() })
	if err := media.InitializeSchema(db); err != nil {
		t.Fatalf("schema: %v", err)
	}
	return db
}

// seedSplitItem writes a file into dir and gives it a row in every table that
// stores a media path, so a move that misses one is visible.
func seedSplitItem(t *testing.T, db *sql.DB, dir, name string) string {
	t.Helper()
	p := filepath.Join(dir, name)
	if err := os.WriteFile(p, []byte("data-"+name), 0o644); err != nil {
		t.Fatal(err)
	}
	seedSplitRows(t, db, p)
	return p
}

// seedSplitRows inserts the library rows for a path that already exists on
// disk, spelled exactly as given.
func seedSplitRows(t *testing.T, db *sql.DB, storedPath string) {
	t.Helper()
	stmts := []struct {
		sql  string
		args []any
	}{
		{`INSERT INTO media (path) VALUES (?)`, []any{storedPath}},
		{`INSERT INTO media_tag_by_category (media_path, tag_label, category_label, weight, time_stamp)
		  VALUES (?, 'sunset', 'Subject', 1, 0)`, []any{storedPath}},
		{`INSERT INTO media_embedding (media_path, model, dim, vector) VALUES (?, 'siglip2', 2, x'0000')`, []any{storedPath}},
		{`INSERT INTO face_scan (media_path, model, face_count) VALUES (?, 'sface', 1)`, []any{storedPath}},
		{`INSERT INTO face (media_path, model, bbox_x, bbox_y, bbox_w, bbox_h, det_score, vector)
		  VALUES (?, 'sface', 0.1, 0.1, 0.2, 0.2, 0.9, x'0000')`, []any{storedPath}},
		{`INSERT INTO battle (winner_path, loser_path, outcome) VALUES (?, 'elsewhere.jpg', 1)`, []any{storedPath}},
		{`INSERT INTO battle (winner_path, loser_path, outcome) VALUES ('elsewhere.jpg', ?, 1)`, []any{storedPath}},
	}
	for _, s := range stmts {
		if _, err := db.Exec(s.sql, s.args...); err != nil {
			t.Fatalf("seed %q: %v", s.sql, err)
		}
	}
}

// rowsNaming counts, per table, how many rows still name a path.
func rowsNaming(t *testing.T, db *sql.DB, path string) map[string]int {
	t.Helper()
	queries := map[string]string{
		"media":                 `SELECT COUNT(*) FROM media WHERE "path" = ?`,
		"media_tag_by_category": `SELECT COUNT(*) FROM media_tag_by_category WHERE media_path = ?`,
		"media_embedding":       `SELECT COUNT(*) FROM media_embedding WHERE media_path = ?`,
		"face":                  `SELECT COUNT(*) FROM face WHERE media_path = ?`,
		"face_scan":             `SELECT COUNT(*) FROM face_scan WHERE media_path = ?`,
		"battle":                `SELECT COUNT(*) FROM battle WHERE winner_path = ? OR loser_path = ?`,
	}
	out := map[string]int{}
	for name, q := range queries {
		args := []any{path}
		if name == "battle" {
			args = append(args, path)
		}
		var n int
		if err := db.QueryRow(q, args...).Scan(&n); err != nil {
			t.Fatalf("count %s: %v", name, err)
		}
		out[name] = n
	}
	return out
}

// assertMovedInLibrary checks that nothing names from and everything names to.
func assertMovedInLibrary(t *testing.T, db *sql.DB, from, to string) {
	t.Helper()
	for table, n := range rowsNaming(t, db, from) {
		if n != 0 {
			t.Errorf("%s still has %d row(s) naming the old path %q", table, n, from)
		}
	}
	want := map[string]int{
		"media": 1, "media_tag_by_category": 1, "media_embedding": 1,
		"face": 1, "face_scan": 1, "battle": 2,
	}
	for table, n := range rowsNaming(t, db, to) {
		if n != want[table] {
			t.Errorf("%s has %d row(s) naming the new path %q, want %d", table, n, to, want[table])
		}
	}
}

// runSplitDir runs the task to completion against a real queue, returning the
// job so its state and stdout can be inspected.
func runSplitDir(t *testing.T, db *sql.DB, args []string) *jobqueue.Job {
	t.Helper()
	q := jobqueue.NewQueueWithDB(db)
	id, err := q.AddJob("", "split-dir", args, "", nil)
	if err != nil {
		t.Fatalf("add job: %v", err)
	}
	j, err := q.ClaimJob()
	if err != nil || j == nil {
		t.Fatalf("claim job: %v (job=%v)", err, j)
	}
	if err := splitDirTask(j, q, nil); err != nil {
		t.Fatalf("split-dir: %v", err)
	}
	if got := q.Jobs[id].State; got != jobqueue.StateCompleted {
		t.Fatalf("job status = %v, want completed. stdout:\n%s", got, strings.Join(q.Jobs[id].Stdout, "\n"))
	}
	return q.Jobs[id]
}

func TestSplitDirAlphaMovesFilesAndEveryTable(t *testing.T) {
	db := newSplitDB(t)
	dir := t.TempDir()

	apple := seedSplitItem(t, db, dir, "apple.jpg")
	avocado := seedSplitItem(t, db, dir, "Avocado.jpg")
	banana := seedSplitItem(t, db, dir, "banana.png")
	numbered := seedSplitItem(t, db, dir, "9lives.jpg")
	odd := seedSplitItem(t, db, dir, "_weird.jpg")

	runSplitDir(t, db, []string{"--target", dir, "--mode", "alpha"})

	// Case only decides the letter, not the folder: both A files land together.
	for _, tc := range []struct {
		from, folder string
	}{
		{apple, "A"}, {avocado, "A"}, {banana, "B"}, {numbered, "0-9"}, {odd, "#"},
	} {
		to := filepath.Join(dir, tc.folder, filepath.Base(tc.from))
		if _, err := os.Stat(to); err != nil {
			t.Errorf("%s not at %s: %v", filepath.Base(tc.from), to, err)
		}
		if _, err := os.Stat(tc.from); !os.IsNotExist(err) {
			t.Errorf("%s is still in the source directory", tc.from)
		}
		assertMovedInLibrary(t, db, tc.from, to)
	}
}

func TestSplitDirDateBucketsByModTimeAtEachGranularity(t *testing.T) {
	stamps := map[string]time.Time{
		"jan.jpg":  time.Date(2025, 1, 4, 10, 0, 0, 0, time.Local),
		"feb.jpg":  time.Date(2025, 2, 9, 10, 0, 0, 0, time.Local),
		"feb2.jpg": time.Date(2025, 2, 20, 10, 0, 0, 0, time.Local),
		"next.jpg": time.Date(2026, 2, 9, 10, 0, 0, 0, time.Local),
	}
	cases := []struct {
		granularity string
		want        map[string]string
	}{
		{"year", map[string]string{"jan.jpg": "2025", "feb.jpg": "2025", "feb2.jpg": "2025", "next.jpg": "2026"}},
		{"month", map[string]string{"jan.jpg": "2025-01", "feb.jpg": "2025-02", "feb2.jpg": "2025-02", "next.jpg": "2026-02"}},
		{"day", map[string]string{"jan.jpg": "2025-01-04", "feb.jpg": "2025-02-09", "feb2.jpg": "2025-02-20", "next.jpg": "2026-02-09"}},
	}
	for _, tc := range cases {
		t.Run(tc.granularity, func(t *testing.T) {
			db := newSplitDB(t)
			dir := t.TempDir()
			for name, when := range stamps {
				p := seedSplitItem(t, db, dir, name)
				if err := os.Chtimes(p, when, when); err != nil {
					t.Fatal(err)
				}
			}

			runSplitDir(t, db, []string{"--target", dir, "--mode", "date", "--granularity", tc.granularity})

			for name, folder := range tc.want {
				to := filepath.Join(dir, folder, name)
				if _, err := os.Stat(to); err != nil {
					t.Errorf("%s not at %s: %v", name, to, err)
				}
				assertMovedInLibrary(t, db, filepath.Join(dir, name), to)
			}
		})
	}
}

func TestSplitDirCapsSubfolderSize(t *testing.T) {
	db := newSplitDB(t)
	dir := t.TempDir()
	for _, n := range []string{"a1.jpg", "a2.jpg", "a3.jpg", "a4.jpg", "a5.jpg"} {
		seedSplitItem(t, db, dir, n)
	}

	runSplitDir(t, db, []string{"--target", dir, "--mode", "alpha", "--max-per-dir", "2"})

	// Five files, cap of two: A_1 and A_2 full, A_3 holding the remainder.
	for folder, want := range map[string]int{"A_1": 2, "A_2": 2, "A_3": 1} {
		entries, err := os.ReadDir(filepath.Join(dir, folder))
		if err != nil {
			t.Fatalf("read %s: %v", folder, err)
		}
		if len(entries) != want {
			t.Errorf("%s holds %d files, want %d", folder, len(entries), want)
		}
	}
}

// A second run must respect what the first one already put in each folder,
// which is what makes pause/resume and repeated sweeps safe.
func TestSplitDirCapCountsFilesAlreadyInSubfolders(t *testing.T) {
	db := newSplitDB(t)
	dir := t.TempDir()
	for _, n := range []string{"a1.jpg", "a2.jpg", "a3.jpg"} {
		seedSplitItem(t, db, dir, n)
	}
	runSplitDir(t, db, []string{"--target", dir, "--mode", "alpha", "--max-per-dir", "2"})

	// New arrivals in the now-empty top level.
	for _, n := range []string{"a4.jpg", "a5.jpg"} {
		seedSplitItem(t, db, dir, n)
	}
	runSplitDir(t, db, []string{"--target", dir, "--mode", "alpha", "--max-per-dir", "2"})

	for folder, want := range map[string]int{"A_1": 2, "A_2": 2, "A_3": 1} {
		entries, err := os.ReadDir(filepath.Join(dir, folder))
		if err != nil {
			t.Fatalf("read %s: %v", folder, err)
		}
		if len(entries) != want {
			t.Errorf("%s holds %d files, want %d", folder, len(entries), want)
		}
	}
}

// The library may spell a path with different separators than the directory
// listing reports. A miss there would move the file and orphan every row.
func TestSplitDirMatchesLibraryPathSpelling(t *testing.T) {
	db := newSplitDB(t)
	dir := t.TempDir()

	name := "apple.jpg"
	onDisk := filepath.Join(dir, name)
	if err := os.WriteFile(onDisk, []byte("data"), 0o644); err != nil {
		t.Fatal(err)
	}
	// Stored with forward slashes regardless of what filepath.Join produced.
	storedFrom := slashKey(onDisk)
	seedSplitRows(t, db, storedFrom)

	runSplitDir(t, db, []string{"--target", dir, "--mode", "alpha"})

	if _, err := os.Stat(filepath.Join(dir, "A", name)); err != nil {
		t.Fatalf("file not moved: %v", err)
	}
	// And the rewrite keeps the library's separator style.
	assertMovedInLibrary(t, db, storedFrom, slashKey(dir)+"/A/"+name)
}

func TestSplitDirKeepsSidecarsWithTheirMedia(t *testing.T) {
	db := newSplitDB(t)
	dir := t.TempDir()

	// A downloader's pairing: the sidecar's own name would bucket it under "Z".
	seedSplitItem(t, db, dir, "zebra.jpg")
	sidecar := filepath.Join(dir, "zebra.jpg.json")
	if err := os.WriteFile(sidecar, []byte("{}"), 0o644); err != nil {
		t.Fatal(err)
	}
	// A date split is where the two would otherwise diverge, since the sidecar
	// is written after the image.
	older := time.Date(2024, 3, 1, 12, 0, 0, 0, time.Local)
	if err := os.Chtimes(filepath.Join(dir, "zebra.jpg"), older, older); err != nil {
		t.Fatal(err)
	}

	runSplitDir(t, db, []string{"--target", dir, "--mode", "date", "--granularity", "month"})

	image := filepath.Join(dir, "2024-03", "zebra.jpg")
	if _, err := os.Stat(image); err != nil {
		t.Fatalf("image not in its date folder: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "2024-03", "zebra.jpg.json")); err != nil {
		t.Errorf("sidecar was separated from its image: %v", err)
	}
}

// The "video.srt" spelling: same stem, different extension, no library row of
// its own. It still has to travel with its video.
func TestSplitDirKeepsSameStemSidecarsWithTheirMedia(t *testing.T) {
	db := newSplitDB(t)
	dir := t.TempDir()

	seedSplitItem(t, db, dir, "lecture.mp4")
	for _, sidecar := range []string{"lecture.srt", "lecture.vtt", "lecture.json"} {
		if err := os.WriteFile(filepath.Join(dir, sidecar), []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	when := time.Date(2024, 5, 2, 12, 0, 0, 0, time.Local)
	if err := os.Chtimes(filepath.Join(dir, "lecture.mp4"), when, when); err != nil {
		t.Fatal(err)
	}

	runSplitDir(t, db, []string{"--target", dir, "--mode", "date", "--granularity", "month"})

	for _, sidecar := range []string{"lecture.srt", "lecture.vtt", "lecture.json"} {
		if _, err := os.Stat(filepath.Join(dir, "2024-05", sidecar)); err != nil {
			t.Errorf("%s was separated from lecture.mp4: %v", sidecar, err)
		}
	}
}

// Two media files can share a stem without either describing the other, so
// neither may drag the other out of its own bucket.
func TestSplitDirDoesNotPairTwoMediaFilesSharingAStem(t *testing.T) {
	db := newSplitDB(t)
	dir := t.TempDir()

	stamps := map[string]time.Time{
		"holiday.jpg": time.Date(2024, 5, 2, 12, 0, 0, 0, time.Local),
		"holiday.png": time.Date(2026, 1, 3, 12, 0, 0, 0, time.Local),
	}
	for name, when := range stamps {
		p := seedSplitItem(t, db, dir, name)
		if err := os.Chtimes(p, when, when); err != nil {
			t.Fatal(err)
		}
	}

	runSplitDir(t, db, []string{"--target", dir, "--mode", "date", "--granularity", "month"})

	for name, folder := range map[string]string{"holiday.jpg": "2024-05", "holiday.png": "2026-01"} {
		if _, err := os.Stat(filepath.Join(dir, folder, name)); err != nil {
			t.Errorf("%s is not in %s — it was paired with the other: %v", name, folder, err)
		}
	}
}

// An ambiguous stem is left alone rather than guessed at.
func TestSplitDirLeavesAmbiguousSidecarsOnTheirOwn(t *testing.T) {
	db := newSplitDB(t)
	dir := t.TempDir()

	old := time.Date(2024, 5, 2, 12, 0, 0, 0, time.Local)
	for _, n := range []string{"clip.mp4", "clip.mov"} {
		seedSplitItem(t, db, dir, n)
		if err := os.Chtimes(filepath.Join(dir, n), old, old); err != nil {
			t.Fatal(err)
		}
	}
	// Which one does clip.srt belong to? Unknowable.
	newer := time.Date(2026, 1, 3, 12, 0, 0, 0, time.Local)
	srt := filepath.Join(dir, "clip.srt")
	if err := os.WriteFile(srt, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(srt, newer, newer); err != nil {
		t.Fatal(err)
	}

	runSplitDir(t, db, []string{"--target", dir, "--mode", "date", "--granularity", "month"})

	if _, err := os.Stat(filepath.Join(dir, "2026-01", "clip.srt")); err != nil {
		t.Errorf("clip.srt should have been bucketed on its own date: %v", err)
	}
}

// scope=library is the answer to a directory that holds a media library AND
// unrelated clutter nobody asked to reorganize.
func TestSplitDirScopeLibraryLeavesUnrelatedFilesInPlace(t *testing.T) {
	db := newSplitDB(t)
	dir := t.TempDir()

	tracked := seedSplitItem(t, db, dir, "apple.jpg")
	// A sidecar of a library item: no row of its own, but it travels.
	sidecar := filepath.Join(dir, "apple.jpg.json")
	if err := os.WriteFile(sidecar, []byte("{}"), 0o644); err != nil {
		t.Fatal(err)
	}
	// Clutter: no library row, describes nothing.
	clutter := []string{"installer.msi", "notes.md", "backup.sqlite", "archive.torrent"}
	for _, n := range clutter {
		if err := os.WriteFile(filepath.Join(dir, n), []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	j := runSplitDir(t, db, []string{"--target", dir, "--mode", "alpha", "--scope", "library"})

	to := filepath.Join(dir, "A", "apple.jpg")
	if _, err := os.Stat(to); err != nil {
		t.Errorf("library item was not moved: %v", err)
	}
	assertMovedInLibrary(t, db, tracked, to)
	if _, err := os.Stat(filepath.Join(dir, "A", "apple.jpg.json")); err != nil {
		t.Errorf("sidecar of a library item did not travel with it: %v", err)
	}
	for _, n := range clutter {
		if _, err := os.Stat(filepath.Join(dir, n)); err != nil {
			t.Errorf("%s should have been left in place: %v", n, err)
		}
	}
	if out := strings.Join(j.Stdout, "\n"); !strings.Contains(out, "4 left in place") {
		t.Errorf("scope summary did not report the skipped files:\n%s", out)
	}
}

func TestSplitDirMovesFilesWithNoLibraryRow(t *testing.T) {
	db := newSplitDB(t)
	dir := t.TempDir()
	loose := filepath.Join(dir, "apple.jpg")
	if err := os.WriteFile(loose, []byte("data"), 0o644); err != nil {
		t.Fatal(err)
	}

	j := runSplitDir(t, db, []string{"--target", dir, "--mode", "alpha"})

	if _, err := os.Stat(filepath.Join(dir, "A", "apple.jpg")); err != nil {
		t.Errorf("untracked file was not moved: %v", err)
	}
	if out := strings.Join(j.Stdout, "\n"); !strings.Contains(out, "1 not in the library") {
		t.Errorf("summary did not report the untracked file:\n%s", out)
	}
}

// A destination the library already owns is a conflict, not a merge — and the
// file must go back so disk and database never disagree.
func TestSplitDirRevertsTheFileWhenTheDatabaseRefuses(t *testing.T) {
	db := newSplitDB(t)
	dir := t.TempDir()

	src := seedSplitItem(t, db, dir, "apple.jpg")
	// Another item already claims where apple.jpg is headed.
	occupied := filepath.Join(dir, "A", "apple.jpg")
	if _, err := db.Exec(`INSERT INTO media (path) VALUES (?)`, occupied); err != nil {
		t.Fatal(err)
	}

	j := runSplitDir(t, db, []string{"--target", dir, "--mode", "alpha"})

	if _, err := os.Stat(src); err != nil {
		t.Errorf("file was not put back after the database refused: %v", err)
	}
	if _, err := os.Stat(occupied); err == nil {
		t.Errorf("file was left at the conflicting destination %s", occupied)
	}
	for table, n := range rowsNaming(t, db, src) {
		want := map[string]int{
			"media": 1, "media_tag_by_category": 1, "media_embedding": 1,
			"face": 1, "face_scan": 1, "battle": 2,
		}[table]
		if n != want {
			t.Errorf("%s has %d row(s) naming the reverted path, want %d", table, n, want)
		}
	}
	if out := strings.Join(j.Stdout, "\n"); !strings.Contains(out, "reverting move") {
		t.Errorf("the revert was not reported:\n%s", out)
	}
}

// Rewriting the rows is only half of it: the in-memory indexes are keyed by
// path too, and a stale key makes similarity search return a path the media
// handler can no longer serve.
func TestSplitDirReKeysTheLiveVectorIndex(t *testing.T) {
	db := newSplitDB(t)
	dir := t.TempDir()
	t.Cleanup(func() { SetVectorIndexForModel(nil, "") })

	src := filepath.Join(dir, "apple.jpg")
	if err := os.WriteFile(src, []byte("data"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(`INSERT INTO media (path) VALUES (?)`, src); err != nil {
		t.Fatal(err)
	}
	const model = "m1"
	if err := media.UpsertEmbedding(db, src, model, []float32{1, 0, 0}, 0); err != nil {
		t.Fatal(err)
	}
	idx, err := BuildIndexFromDB(db, model, nil)
	if err != nil {
		t.Fatal(err)
	}
	SetVectorIndexForModel(idx, model)

	runSplitDir(t, db, []string{"--target", dir, "--mode", "alpha"})

	hits, ok := indexSearch(model, []float32{1, 0, 0}, 5, nil, embedindex.ScoreDefault)
	if !ok {
		t.Fatal("no index installed")
	}
	if len(hits) != 1 {
		t.Fatalf("hits = %+v, want exactly the moved vector", hits)
	}
	if want := filepath.Join(dir, "A", "apple.jpg"); hits[0].Path != want {
		t.Errorf("index hit = %q, want %q", hits[0].Path, want)
	}
}

func TestSplitDirDryRunChangesNothing(t *testing.T) {
	db := newSplitDB(t)
	dir := t.TempDir()
	src := seedSplitItem(t, db, dir, "apple.jpg")

	j := runSplitDir(t, db, []string{"--target", dir, "--mode", "alpha", "--dry-run"})

	if _, err := os.Stat(src); err != nil {
		t.Errorf("dry run moved the file: %v", err)
	}
	if entries, err := os.ReadDir(dir); err == nil {
		for _, e := range entries {
			if e.IsDir() {
				t.Errorf("dry run created subfolder %s", e.Name())
			}
		}
	}
	if n := rowsNaming(t, db, src)["media"]; n != 1 {
		t.Errorf("dry run changed the media row")
	}
	if out := strings.Join(j.Stdout, "\n"); !strings.Contains(out, "Would move") {
		t.Errorf("dry run did not preview the plan:\n%s", out)
	}
}

// Subfolders are the destination, not the input: descending into them would
// re-shuffle a previous run's output on every re-run.
func TestSplitDirIgnoresExistingSubfolders(t *testing.T) {
	db := newSplitDB(t)
	dir := t.TempDir()
	sub := filepath.Join(dir, "A")
	if err := os.MkdirAll(sub, 0o755); err != nil {
		t.Fatal(err)
	}
	settled := seedSplitItem(t, db, sub, "already.jpg")
	seedSplitItem(t, db, dir, "banana.jpg")

	runSplitDir(t, db, []string{"--target", dir, "--mode", "alpha"})

	if _, err := os.Stat(settled); err != nil {
		t.Errorf("a file already in a subfolder was disturbed: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "B", "banana.jpg")); err != nil {
		t.Errorf("top-level file was not moved: %v", err)
	}
}

// /create moves the last token of a typed command into the job's Input, so the
// options have to survive being split across Arguments and Input wherever the
// break lands.
func TestSplitDirReadsOptionsSplitBetweenArgumentsAndInput(t *testing.T) {
	cases := []struct {
		name  string
		args  []string
		input string
	}{
		{"flag value stranded in input", []string{"--target", "@dir@", "--mode"}, "date"},
		{"whole flag stranded in input", []string{"--target=@dir@"}, "--mode=date"},
		{"directory as the positional input", []string{"--mode=date"}, "@dir@"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			db := newSplitDB(t)
			dir := t.TempDir()
			p := seedSplitItem(t, db, dir, "apple.jpg")
			when := time.Date(2024, 3, 1, 12, 0, 0, 0, time.Local)
			if err := os.Chtimes(p, when, when); err != nil {
				t.Fatal(err)
			}

			args := make([]string, len(tc.args))
			for i, a := range tc.args {
				args[i] = strings.ReplaceAll(a, "@dir@", dir)
			}
			input := strings.ReplaceAll(tc.input, "@dir@", dir)

			q := jobqueue.NewQueueWithDB(db)
			id, err := q.AddJob("", "split-dir", args, input, nil)
			if err != nil {
				t.Fatal(err)
			}
			j, err := q.ClaimJob()
			if err != nil || j == nil {
				t.Fatalf("claim: %v", err)
			}
			if err := splitDirTask(j, q, nil); err != nil {
				t.Fatalf("split-dir: %v\n%s", err, strings.Join(q.Jobs[id].Stdout, "\n"))
			}

			// Landing in the date folder proves BOTH options were read.
			to := filepath.Join(dir, "2024-03", "apple.jpg")
			if _, err := os.Stat(to); err != nil {
				t.Fatalf("not at %s: %v\n%s", to, err, strings.Join(q.Jobs[id].Stdout, "\n"))
			}
			assertMovedInLibrary(t, db, p, to)
		})
	}
}

// A row whose file is gone is dead weight: it shows up in queries, in
// similarity results, and in people groups, and nothing can ever serve it.
func TestSplitDirPruneMissingRemovesEveryTrace(t *testing.T) {
	db := newSplitDB(t)
	dir := t.TempDir()

	live := seedSplitItem(t, db, dir, "apple.jpg")
	// Rows with no file behind them.
	gone := filepath.Join(dir, "deleted.jpg")
	seedSplitRows(t, db, gone)

	j := runSplitDir(t, db, []string{"--target", dir, "--mode", "alpha", "--prune-missing"})

	for table, n := range rowsNaming(t, db, gone) {
		if n != 0 {
			t.Errorf("%s still has %d row(s) for the missing file", table, n)
		}
	}
	// And the live item was moved as usual.
	assertMovedInLibrary(t, db, live, filepath.Join(dir, "A", "apple.jpg"))
	if out := strings.Join(j.Stdout, "\n"); !strings.Contains(out, "Pruned 1 path") {
		t.Errorf("prune was not reported:\n%s", out)
	}
}

// An embedding (or face, or battle row) can outlive the media row it belonged
// to. Nothing reaches those rows again — every query joins through media — so
// only a sweep like this will ever find them.
func TestSplitDirPruneMissingRemovesRowsWithNoMediaRow(t *testing.T) {
	db := newSplitDB(t)
	dir := t.TempDir()
	seedSplitItem(t, db, dir, "apple.jpg")

	// No media row, no file: reachable from nowhere.
	stray := filepath.Join(dir, "vanished.jpg")
	if err := media.UpsertEmbedding(db, stray, "siglip2", []float32{1, 0, 0}, 0); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(
		`INSERT INTO battle (winner_path, loser_path, outcome) VALUES (?, 'other.jpg', 1)`, stray,
	); err != nil {
		t.Fatal(err)
	}

	runSplitDir(t, db, []string{"--target", dir, "--mode", "alpha", "--prune-missing"})

	for table, n := range rowsNaming(t, db, stray) {
		if n != 0 {
			t.Errorf("%s still has %d stray row(s) for a path with no media row and no file", table, n)
		}
	}
}

// A stray row whose FILE still exists is left alone: it is not a missing-file
// reference, and this task is not a general-purpose garbage collector.
func TestSplitDirPruneMissingKeepsStrayRowsWhoseFileExists(t *testing.T) {
	db := newSplitDB(t)
	dir := t.TempDir()

	// On disk, embedded, but never given a media row.
	loose := filepath.Join(dir, "loose.jpg")
	if err := os.WriteFile(loose, []byte("data"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := media.UpsertEmbedding(db, loose, "siglip2", []float32{1, 0, 0}, 0); err != nil {
		t.Fatal(err)
	}

	runSplitDir(t, db, []string{"--target", dir, "--mode", "alpha", "--prune-missing"})

	var n int
	if err := db.QueryRow(
		`SELECT COUNT(*) FROM media_embedding WHERE media_path LIKE ?`, filepath.Join(dir, "%"),
	).Scan(&n); err != nil {
		t.Fatal(err)
	}
	if n != 1 {
		t.Errorf("embedding rows = %d, want 1 (a row whose file exists was pruned)", n)
	}
}

// Pruning must never reach past the one directory the task listed. Rows in
// subdirectories are outside its remit, and it has no idea whether those files
// exist.
func TestSplitDirPruneMissingIgnoresSubdirectoryRows(t *testing.T) {
	db := newSplitDB(t)
	dir := t.TempDir()

	seedSplitItem(t, db, dir, "apple.jpg")
	// A row for a file in a subdirectory that does not exist on disk at all.
	childRow := filepath.Join(dir, "sub", "childless.jpg")
	seedSplitRows(t, db, childRow)

	runSplitDir(t, db, []string{"--target", dir, "--mode", "alpha", "--prune-missing"})

	if n := rowsNaming(t, db, childRow)["media"]; n != 1 {
		t.Errorf("a subdirectory row was pruned; media rows naming it = %d, want 1", n)
	}
}

func TestSplitDirPruneMissingIsOffByDefault(t *testing.T) {
	db := newSplitDB(t)
	dir := t.TempDir()
	seedSplitItem(t, db, dir, "apple.jpg")
	gone := filepath.Join(dir, "deleted.jpg")
	seedSplitRows(t, db, gone)

	runSplitDir(t, db, []string{"--target", dir, "--mode", "alpha"})

	if n := rowsNaming(t, db, gone)["media"]; n != 1 {
		t.Errorf("rows were pruned without --prune-missing; media rows = %d, want 1", n)
	}
}

func TestSplitDirPruneMissingDryRunOnlyReports(t *testing.T) {
	db := newSplitDB(t)
	dir := t.TempDir()
	gone := filepath.Join(dir, "deleted.jpg")
	seedSplitRows(t, db, gone)
	seedSplitItem(t, db, dir, "apple.jpg")

	j := runSplitDir(t, db, []string{"--target", dir, "--mode", "alpha", "--prune-missing", "--dry-run"})

	if n := rowsNaming(t, db, gone)["media"]; n != 1 {
		t.Errorf("dry run pruned rows; media rows = %d, want 1", n)
	}
	if out := strings.Join(j.Stdout, "\n"); !strings.Contains(out, "Would prune 1 library row") {
		t.Errorf("dry run did not preview the prune:\n%s", out)
	}
}

// A file present under a different letter case is still present. Refusing to
// prune costs nothing; pruning a file that exists costs everything.
func TestSplitDirPruneMissingKeepsCaseVariantMatches(t *testing.T) {
	db := newSplitDB(t)
	dir := t.TempDir()

	if err := os.WriteFile(filepath.Join(dir, "Apple.JPG"), []byte("data"), 0o644); err != nil {
		t.Fatal(err)
	}
	// The library recorded a different casing of the same file.
	seedSplitRows(t, db, filepath.Join(dir, "apple.jpg"))

	runSplitDir(t, db, []string{"--target", dir, "--mode", "alpha", "--prune-missing"})

	// It must not have been pruned — either it moved with the file, or it
	// stayed put, but its rows must still exist somewhere.
	var n int
	if err := db.QueryRow(`SELECT COUNT(*) FROM media`).Scan(&n); err != nil {
		t.Fatal(err)
	}
	if n != 1 {
		t.Errorf("media rows = %d, want 1 (a case-variant match was pruned)", n)
	}
}

// keep-recent turns the directory into an archive whose root is the working
// folder: freshly saved files stay put, settled ones get filed away.
func TestSplitDirKeepRecentLeavesTheCurrentPeriodInPlace(t *testing.T) {
	db := newSplitDB(t)
	dir := t.TempDir()
	now := time.Now()

	// Two files in the working window, one long settled.
	fresh := seedSplitItem(t, db, dir, "fresh.jpg")
	alsoFresh := seedSplitItem(t, db, dir, "also-fresh.jpg")
	old := seedSplitItem(t, db, dir, "old.jpg")
	settled := now.AddDate(0, -6, 0)
	if err := os.Chtimes(old, settled, settled); err != nil {
		t.Fatal(err)
	}

	runSplitDir(t, db, []string{"--target", dir, "--mode", "date", "--granularity", "month", "--keep-recent", "1"})

	for _, p := range []string{fresh, alsoFresh} {
		if _, err := os.Stat(p); err != nil {
			t.Errorf("%s should have stayed in the working directory: %v", filepath.Base(p), err)
		}
	}
	to := filepath.Join(dir, settled.Format("2006-01"), "old.jpg")
	if _, err := os.Stat(to); err != nil {
		t.Errorf("settled file was not filed away to %s: %v", to, err)
	}
	assertMovedInLibrary(t, db, old, to)
}

func TestSplitDirKeepRecentCoversMultiplePeriods(t *testing.T) {
	db := newSplitDB(t)
	dir := t.TempDir()
	now := time.Now()

	// Dated mid-month so a run near a month boundary can't drift a file into
	// the wrong bucket.
	lastMonth := time.Date(now.Year(), now.Month(), 15, 12, 0, 0, 0, time.Local).AddDate(0, -1, 0)
	older := lastMonth.AddDate(0, -3, 0)

	recent := seedSplitItem(t, db, dir, "recent.jpg")
	previous := seedSplitItem(t, db, dir, "previous.jpg")
	if err := os.Chtimes(previous, lastMonth, lastMonth); err != nil {
		t.Fatal(err)
	}
	ancient := seedSplitItem(t, db, dir, "ancient.jpg")
	if err := os.Chtimes(ancient, older, older); err != nil {
		t.Fatal(err)
	}

	runSplitDir(t, db, []string{"--target", dir, "--mode", "date", "--granularity", "month", "--keep-recent", "2"})

	for _, p := range []string{recent, previous} {
		if _, err := os.Stat(p); err != nil {
			t.Errorf("%s should have stayed (within the 2-month window): %v", filepath.Base(p), err)
		}
	}
	if _, err := os.Stat(filepath.Join(dir, older.Format("2006-01"), "ancient.jpg")); err != nil {
		t.Errorf("file outside the window was not filed: %v", err)
	}
}

// A sidecar inherits its primary's bucket, so holding the primary back must
// hold the sidecar back too — otherwise the transcript is archived while its
// video stays in the working folder.
func TestSplitDirKeepRecentHoldsSidecarsWithTheirPrimary(t *testing.T) {
	db := newSplitDB(t)
	dir := t.TempDir()

	seedSplitItem(t, db, dir, "today.mp4")
	for _, s := range []string{"today.mp4.vtt", "today.srt"} {
		p := filepath.Join(dir, s)
		if err := os.WriteFile(p, []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
		// Sidecars written long before the video they describe: only the
		// inherited bucket can keep them together.
		old := time.Now().AddDate(0, -8, 0)
		if err := os.Chtimes(p, old, old); err != nil {
			t.Fatal(err)
		}
	}

	runSplitDir(t, db, []string{"--target", dir, "--mode", "date", "--granularity", "month", "--keep-recent", "1"})

	for _, s := range []string{"today.mp4", "today.mp4.vtt", "today.srt"} {
		if _, err := os.Stat(filepath.Join(dir, s)); err != nil {
			t.Errorf("%s was separated from its primary and filed away: %v", s, err)
		}
	}
}

// A file dated in the future is newer than the working window, not older.
func TestSplitDirKeepRecentHoldsFutureDatedFiles(t *testing.T) {
	db := newSplitDB(t)
	dir := t.TempDir()

	future := seedSplitItem(t, db, dir, "postdated.jpg")
	when := time.Now().AddDate(0, 3, 0)
	if err := os.Chtimes(future, when, when); err != nil {
		t.Fatal(err)
	}

	runSplitDir(t, db, []string{"--target", dir, "--mode", "date", "--granularity", "month", "--keep-recent", "1"})

	if _, err := os.Stat(future); err != nil {
		t.Errorf("a future-dated file should stay in the working directory: %v", err)
	}
}

func TestRecentBucketCutoff(t *testing.T) {
	cases := []struct {
		now         time.Time
		granularity string
		keep        int
		want        string
	}{
		{time.Date(2026, 8, 8, 10, 0, 0, 0, time.Local), "month", 1, "2026-08"},
		{time.Date(2026, 8, 8, 10, 0, 0, 0, time.Local), "month", 2, "2026-07"},
		{time.Date(2026, 8, 8, 10, 0, 0, 0, time.Local), "month", 3, "2026-06"},
		// Month stepping from a 31st: AddDate(0,-1,0) would overflow March 31
		// back to March 3 and skip February entirely.
		{time.Date(2026, 3, 31, 10, 0, 0, 0, time.Local), "month", 2, "2026-02"},
		{time.Date(2026, 3, 31, 10, 0, 0, 0, time.Local), "month", 3, "2026-01"},
		{time.Date(2026, 5, 31, 10, 0, 0, 0, time.Local), "month", 4, "2026-02"},
		// Crossing a year boundary.
		{time.Date(2026, 1, 15, 10, 0, 0, 0, time.Local), "month", 2, "2025-12"},
		{time.Date(2026, 1, 15, 10, 0, 0, 0, time.Local), "year", 1, "2026"},
		{time.Date(2026, 1, 15, 10, 0, 0, 0, time.Local), "year", 3, "2024"},
		{time.Date(2026, 8, 8, 10, 0, 0, 0, time.Local), "day", 1, "2026-08-08"},
		{time.Date(2026, 8, 8, 10, 0, 0, 0, time.Local), "day", 3, "2026-08-06"},
		{time.Date(2026, 3, 1, 10, 0, 0, 0, time.Local), "day", 2, "2026-02-28"},
		// keep < 1 is treated as the current period rather than the future.
		{time.Date(2026, 8, 8, 10, 0, 0, 0, time.Local), "month", 0, "2026-08"},
	}
	for _, tc := range cases {
		if got := recentBucketCutoff(tc.now, tc.granularity, tc.keep); got != tc.want {
			t.Errorf("recentBucketCutoff(%s, %s, %d) = %q, want %q",
				tc.now.Format("2006-01-02"), tc.granularity, tc.keep, got, tc.want)
		}
	}
}

func TestPartitionRecent(t *testing.T) {
	files := []splitFile{
		{Name: "a", Bucket: "2026-06"},
		{Name: "b", Bucket: "2026-07"},
		{Name: "c", Bucket: "2026-08"},
		{Name: "d", Bucket: "2026-09"}, // future
	}
	toFile, held := partitionRecent(files, "2026-07")
	if held != 3 {
		t.Errorf("held = %d, want 3 (July, August and the future-dated one)", held)
	}
	if len(toFile) != 1 || toFile[0].Name != "a" {
		t.Errorf("toFile = %+v, want only the June file", toFile)
	}
}

// Alphabetical buckets have no recency; ignoring the flag would archive the
// working files it was asked to protect.
func TestSplitDirKeepRecentRequiresDateMode(t *testing.T) {
	db := newSplitDB(t)
	dir := t.TempDir()
	src := seedSplitItem(t, db, dir, "apple.jpg")

	q := jobqueue.NewQueueWithDB(db)
	if _, err := q.AddJob("", "split-dir",
		[]string{"--target", dir, "--mode", "alpha", "--keep-recent", "1"}, "", nil); err != nil {
		t.Fatal(err)
	}
	j, err := q.ClaimJob()
	if err != nil || j == nil {
		t.Fatalf("claim: %v", err)
	}
	if err := splitDirTask(j, q, nil); err == nil {
		t.Fatal("expected an error for --keep-recent with alphabetical mode")
	}
	if _, err := os.Stat(src); err != nil {
		t.Errorf("the rejected run moved files anyway: %v", err)
	}
}

func TestSplitDirRejectsBadInput(t *testing.T) {
	cases := []struct {
		name string
		args []string
	}{
		{"no target", []string{"--mode", "alpha"}},
		{"unknown mode", []string{"--target", t.TempDir(), "--mode", "sideways"}},
		{"unknown granularity", []string{"--target", t.TempDir(), "--mode", "date", "--granularity", "fortnight"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			db := newSplitDB(t)
			q := jobqueue.NewQueueWithDB(db)
			if _, err := q.AddJob("", "split-dir", tc.args, "", nil); err != nil {
				t.Fatal(err)
			}
			j, err := q.ClaimJob()
			if err != nil || j == nil {
				t.Fatalf("claim: %v", err)
			}
			if err := splitDirTask(j, q, nil); err == nil {
				t.Fatal("expected an error")
			}
		})
	}
}

func TestSplitDirIsRegistered(t *testing.T) {
	task, ok := GetTasks()["split-dir"]
	if !ok {
		t.Fatal("split-dir is not registered")
	}
	names := make([]string, 0, len(task.Options))
	for _, o := range task.Options {
		names = append(names, o.Name)
	}
	sort.Strings(names)
	want := []string{"dry-run", "granularity", "keep-recent", "max-per-dir", "mode", "prune-missing", "scope", "target"}
	if strings.Join(names, ",") != strings.Join(want, ",") {
		t.Errorf("options = %v, want %v", names, want)
	}
}

func TestAlphaBucket(t *testing.T) {
	cases := map[string]string{
		"apple.jpg": "A", "Apple.jpg": "A", "9lives.jpg": "0-9",
		"_weird.jpg": "#", "": "#", "élan.jpg": "É", " space.jpg": "#",
	}
	for name, want := range cases {
		if got := alphaBucket(name); got != want {
			t.Errorf("alphaBucket(%q) = %q, want %q", name, got, want)
		}
	}
}

func TestSiblingUnderPreservesSeparatorStyle(t *testing.T) {
	cases := []struct{ stored, folder, want string }{
		{`C:\pics\a.jpg`, "A", `C:\pics\A\a.jpg`},
		{"/pics/a.jpg", "A", "/pics/A/a.jpg"},
		{`C:/pics/a.jpg`, "2026-08", "C:/pics/2026-08/a.jpg"},
	}
	for _, tc := range cases {
		if got := siblingUnder(tc.stored, tc.folder); got != tc.want {
			t.Errorf("siblingUnder(%q, %q) = %q, want %q", tc.stored, tc.folder, got, tc.want)
		}
	}
}
