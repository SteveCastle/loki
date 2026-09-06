package tasks

import (
	"context"
	"database/sql"
	"os"
	"path/filepath"
	"sort"
	"testing"

	_ "modernc.org/sqlite"
)

func newThumbCleanupDB(t *testing.T) *sql.DB {
	t.Helper()
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatalf("open db: %v", err)
	}
	// One connection: an in-memory database is per-connection, and the
	// task holds prepared statements open while running other queries.
	db.SetMaxOpenConns(1)
	t.Cleanup(func() { db.Close() })
	for _, stmt := range []string{
		`CREATE TABLE media ("path" TEXT PRIMARY KEY, thumbnail_path_600 TEXT, thumbnail_path_1200 TEXT)`,
		`CREATE TABLE tag (label TEXT PRIMARY KEY, thumbnail_path_600 TEXT)`,
		`CREATE TABLE media_tag_by_category (media_path TEXT, tag_label TEXT, category_label TEXT, weight REAL, time_stamp REAL)`,
		`CREATE TABLE media_embedding (media_path TEXT, model TEXT, vector BLOB)`,
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

func TestThumbHashMatchesGenerators(t *testing.T) {
	// Reference values computed with node's crypto exactly as the Electron
	// worker does: createHash('sha256').update(path + ts.toString()).digest('hex').
	cases := map[string]string{
		`C:\pics\a.jpg`:         "f866d21d582ee1c2ae46519f22d2f8d3e0586a07d53cf2237a63eb39722e8ca4",
		`C:\vids\b.mp4` + "5.5": "ee0068faa41f800008aacb8de8525b0ab15dc43744e450cef7ce77eeb6707dfe",
	}
	for in, want := range cases {
		got := thumbHashHex(in)
		if got != want {
			t.Errorf("thumbHashHex(%q) = %s, want %s (node reference)", in, got, want)
		}
		d, ok := decodeHex64(got)
		if !ok || d != thumbDigestOf(in) {
			t.Errorf("decodeHex64 round-trip failed for %q", in)
		}
	}
	if isHex64("ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789") {
		t.Errorf("uppercase stems must be rejected: the index rebuilds filenames in lowercase")
	}
}

// writeThumb creates a cache file under baseDir/<cacheDir>/<name>.
func writeThumb(t *testing.T, baseDir, cacheDir, name string) string {
	t.Helper()
	dir := filepath.Join(baseDir, cacheDir)
	if err := os.MkdirAll(dir, 0755); err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(dir, name)
	if err := os.WriteFile(p, []byte("x"), 0644); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestThumbnailCleanupFullSweepClassification(t *testing.T) {
	db := newThumbCleanupDB(t)
	baseDir := t.TempDir()

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
	refPath := writeThumb(t, baseDir, "thumbnail_path_600", refName)
	if _, err := db.Exec(`INSERT INTO tag (label, thumbnail_path_600) VALUES ('sunset', ?)`, refPath); err != nil {
		t.Fatal(err)
	}
	// A DB-recorded thumbnail on the media row, spelled with forward slashes.
	recName := thumbHashHex("recorded by an older build")
	recPath := writeThumb(t, baseDir, "thumbnail_path_1200", recName)
	if _, err := db.Exec(`UPDATE media SET thumbnail_path_1200 = ? WHERE "path" = ?`,
		filepath.ToSlash(recPath), imgPath); err != nil {
		t.Fatal(err)
	}

	kept := map[string]bool{
		writeThumb(t, baseDir, "thumbnail_path_600", thumbHashHex(imgPath)):              true, // image thumb: bare digest
		writeThumb(t, baseDir, "thumbnail_path_100", thumbHashHex(imgPath)):              true, // same digest, another size
		writeThumb(t, baseDir, "thumbnail_path_600", thumbHashHex(`C:/pics/a.jpg`)):      true, // separator-variant spelling
		writeThumb(t, baseDir, "thumbnail_path_600", thumbHashHex(vidPath)+".mp4"):       true, // video thumb
		writeThumb(t, baseDir, "thumbnail_path_600", thumbHashHex(vidPath+"5.5")+".mp4"): true, // tagged-timestamp thumb
		refPath: true, // literal tag reference
		recPath: true, // literal media reference
		writeThumb(t, baseDir, "thumbnail_path_600", thumbHashHex(`C:\gone\z.jpg`)):                 false,
		writeThumb(t, baseDir, "thumbnail_path_1200", thumbHashHex(`C:\gone\dead.mp4`)+".mp4"):      false,
		writeThumb(t, baseDir, "thumbnail_path_600", thumbHashHex(`C:\gone\dead.mp4`+"5.5")+".mp4"): false, // orphaned tag timestamp
	}
	writeThumb(t, baseDir, "thumbnail_path_600", "notes.txt") // not our naming scheme — must be ignored entirely
	writeThumb(t, baseDir, "thumbnail_path_600", "ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789")

	ix, err := buildThumbCacheIndex(context.Background(), baseDir)
	if err != nil {
		t.Fatalf("buildThumbCacheIndex: %v", err)
	}
	if ix.unrecognized != 2 {
		t.Errorf("unrecognized = %d, want 2 (notes.txt, uppercase stem)", ix.unrecognized)
	}
	if len(ix.entries) != len(kept) {
		t.Fatalf("indexed %d candidate files, want %d", len(ix.entries), len(kept))
	}
	if !sort.SliceIsSorted(ix.entries, func(i, j int) bool {
		return string(ix.entries[i].stem[:]) < string(ix.entries[j].stem[:])
	}) {
		t.Errorf("index is not sorted by digest")
	}

	st, err := markLiveThumbnails(context.Background(), db, ix, true)
	if err != nil {
		t.Fatalf("markLiveThumbnails: %v", err)
	}
	if st.MediaRows != 2 || st.Timestamps != 1 || st.Literals != 2 {
		t.Errorf("stats = %+v, want 2 rows, 1 timestamp, 2 literals", st)
	}
	check := func(label string) {
		for i := range ix.entries {
			p := ix.path(i)
			want, known := kept[p]
			if !known {
				t.Errorf("%s: unexpected candidate %s", label, p)
				continue
			}
			if got := ix.isKept(i); got != want {
				t.Errorf("%s: kept(%s) = %v, want %v", label, filepath.Base(p), got, want)
			}
		}
	}
	check("single chunk")

	// The same result must fall out of the keyset-paginated path when every
	// chunk holds one row.
	defer func(n int) { thumbScanChunk = n }(thumbScanChunk)
	thumbScanChunk = 1
	ix.owners = make([]uint32, len(ix.entries))
	st2, err := markLiveThumbnails(context.Background(), db, ix, true)
	if err != nil {
		t.Fatalf("markLiveThumbnails (chunk=1): %v", err)
	}
	if st2 != st {
		t.Errorf("chunked stats = %+v, want %+v", st2, st)
	}
	check("chunk=1")
}

func TestThumbnailCleanupScoped(t *testing.T) {
	db := newThumbCleanupDB(t)
	baseDir := t.TempDir()
	scope := filepath.Join(t.TempDir(), "scope")
	sub := filepath.Join(scope, "sub")
	if err := os.MkdirAll(sub, 0755); err != nil {
		t.Fatal(err)
	}
	mkMedia := func(p string) string {
		if err := os.WriteFile(p, []byte("m"), 0644); err != nil {
			t.Fatal(err)
		}
		return p
	}
	livePath := mkMedia(filepath.Join(scope, "live.jpg"))     // on disk, in DB
	lostPath := mkMedia(filepath.Join(sub, "lost.mp4"))       // on disk, not in DB
	noThumb := mkMedia(filepath.Join(sub, "nothumb.png"))     // on disk, not in DB, no thumbnail
	mkMedia(filepath.Join(sub, "readme.txt"))                 // not media — never a candidate
	deleted := filepath.Join(scope, "deleted.webm")           // gone from disk, tag rows linger
	embedded := filepath.ToSlash(filepath.Join(sub, "e.gif")) // gone from disk, embedding row lingers, slash-spelled
	outside := `C:\elsewhere\x.jpg`                           // orphan outside the scope — untouchable

	if _, err := db.Exec(`INSERT INTO media ("path") VALUES (?)`, livePath); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(
		`INSERT INTO media_tag_by_category (media_path, tag_label, category_label, weight, time_stamp)
		 VALUES (?, 'a', 'c', 0, 0), (?, 'b', 'c', 0, 12.25), (?, 'b', 'c', 0, 12.25), (?, 'a', 'c', 0, 3)`,
		deleted, deleted, deleted, livePath); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(`INSERT INTO media_embedding (media_path, model) VALUES (?, 'm')`, embedded); err != nil {
		t.Fatal(err)
	}
	// A tag preview literally pointing at an orphan's thumbnail keeps it in
	// the full sweep, so the scoped sweep must keep it too.
	protected := writeThumb(t, baseDir, "thumbnail_path_100", thumbHashHex(lostPath)+".mp4")
	if _, err := db.Exec(`INSERT INTO tag (label, thumbnail_path_600) VALUES ('t', ?)`, protected); err != nil {
		t.Fatal(err)
	}

	expect := map[string]string{ // path -> expected note ("" = must survive)
		writeThumb(t, baseDir, "thumbnail_path_600", thumbHashHex(livePath)):               "",
		writeThumb(t, baseDir, "thumbnail_path_600", thumbHashHex(livePath+"3")):           "",
		writeThumb(t, baseDir, "thumbnail_path_600", thumbHashHex(lostPath)+".mp4"):        lostPath,
		writeThumb(t, baseDir, "thumbnail_path_1200", thumbHashHex(lostPath)+".mp4"):       lostPath,
		writeThumb(t, baseDir, "thumbnail_path_600", thumbHashHex(deleted)+".mp4"):         deleted,
		writeThumb(t, baseDir, "thumbnail_path_600", thumbHashHex(deleted+"12.25")+".mp4"): deleted + " @12.25s",
		writeThumb(t, baseDir, "thumbnail_path_600", thumbHashHex(embedded)+".mp4"):        embedded,
		// The backslash spelling of the embedding's path is the same file to
		// the scope, so its thumbnail goes too — attributed to the spelling
		// the candidate was found under.
		writeThumb(t, baseDir, "thumbnail_path_600", thumbHashHex(filepath.FromSlash(embedded))+".mp4"): embedded,
		writeThumb(t, baseDir, "thumbnail_path_600", thumbHashHex(outside)):                             "",
		protected: "",
	}
	_ = noThumb

	ix, err := buildThumbCacheIndex(context.Background(), baseDir)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := markLiveThumbnails(context.Background(), db, ix, false); err != nil {
		t.Fatal(err)
	}
	var logs []string
	libs := []thumbLibrary{{Label: "test", DB: db, bit: 1}}
	cands, st, err := thumbScopeCandidates(context.Background(), libs, scope, func(s string) { logs = append(logs, s) })
	if err != nil {
		t.Fatalf("thumbScopeCandidates: %v (logs %v)", err, logs)
	}
	if st.Walked != 3 || st.Referenced != 2 || st.Candidates != 5 {
		t.Errorf("scope stats = %+v, want 3 walked, 2 referenced, 5 candidates (cands %v)", st, cands)
	}
	victims, err := thumbScopeVictims(context.Background(), libs, ix, cands, &st)
	if err != nil {
		t.Fatalf("thumbScopeVictims: %v", err)
	}
	if st.Live != 1 || st.OrphanPaths != 3 || st.OrphanNoThumb != 1 {
		t.Errorf("judged stats = %+v, want 1 live, 3 orphan paths with thumbs, 1 without", st)
	}

	got := map[string]string{}
	for _, v := range victims {
		p := ix.path(v.Index)
		if _, dup := got[p]; dup {
			t.Errorf("victim %s listed twice", p)
		}
		got[p] = v.Note
	}
	for p, note := range expect {
		gotNote, isVictim := got[p]
		if note == "" && isVictim {
			t.Errorf("%s must survive a scoped run, was marked for deletion (%s)", filepath.Base(p), gotNote)
		}
		if note != "" && !isVictim {
			t.Errorf("%s should be deleted (thumbnail of %s), was not", filepath.Base(p), note)
		}
		if note != "" && isVictim && gotNote != note {
			t.Errorf("%s attributed to %q, want %q", filepath.Base(p), gotNote, note)
		}
	}
	for p := range got {
		if _, known := expect[p]; !known {
			t.Errorf("unexpected victim %s", p)
		}
	}
}

func TestThumbScopePrefixesBothSeparators(t *testing.T) {
	got := thumbScopePrefixes(`C:\pics\2024\`)
	want := map[string]bool{}
	for _, p := range got {
		want[p] = true
	}
	if !want[`C:\pics\2024\`] || !want[`C:/pics/2024/`] {
		t.Errorf("prefixes = %q, want both separator spellings of C:\\pics\\2024\\", got)
	}
}
