package tasks

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/stevecastle/shrike/appconfig"
	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/media"
	"github.com/stevecastle/shrike/mediaext"
)

// Cleaning up orphaned thumbnails.
//
// Thumbnails live next to the database in thumbnail_path_100/600/1200
// directories, named by sha256 of the media path (thumbnail.go's
// getThumbnailPath, which the Electron app's getMediaCachePath in
// src/main/media.ts and image-processing-worker.js mirror exactly): the
// lowercase hex digest alone for images, digest + ".mp4" for videos and
// animated formats, and — when a video frame other than the default is
// wanted — sha256(path + timestamp) where the timestamp is formatted like
// JavaScript's Number.toString(). Nothing deletes those files when a media
// row goes away, so over time the cache accumulates thumbnails of items the
// library no longer knows.
//
// The hash is one-way, so a cache file cannot be traced back to its media
// path. The task therefore works from the other direction: it indexes every
// cache file by digest, then streams the database and marks every digest the
// library could still ask for — sha256(path) for every media row,
// sha256(path + ts) for every tagged timestamp on a live row, plus any file
// the media/tag tables reference by literal path. Unmarked files are
// orphans. A deleted thumbnail that turns out to be wanted again is
// regenerated on demand, so the worst case of an overly-aggressive pass is a
// one-time re-render, never data loss. Files whose names don't look like
// ours (not a 64-char lowercase hex stem) are left alone.
//
// Scale: libraries run to millions of media rows and the cache to a multiple
// of that. The index holds 34 bytes per cache file (raw digest + two small
// indices) in one sorted slice, digests are looked up by binary search, and
// the database is streamed once per table rather than materialised — nothing
// here is proportional to the length of a path or a hex string.
//
// Scoped mode (--dir) exists so a run can be tested on a bounded corner of
// the library. Since orphans can't be enumerated from the cache side, the
// scope enumerates candidate media paths instead: every file on disk under
// the directory plus every path under it that some table other than media
// still references (tags, embeddings, faces, battles — the same list the
// media move re-points). Each candidate that is not a media row and has a
// thumbnail in the cache is an orphan, and the output names the media path
// each deleted file belonged to. A scoped run deletes a subset of what the
// full run would.
//
// Several libraries can share one cache: both apps put the cache beside the
// database, so every database file in that folder writes to the same
// directories, and a file named by any of them must survive. Every SQLite
// file beside the configured database is therefore opened as a library
// (plus any passed with --db), each marks the index under its own owner
// bit, the run reports who needs what, and only files no library names are
// deleted. A library that cannot be read aborts the run before anything is
// removed.

var thumbnailCleanupOptions = []TaskOption{
	{Name: "dry-run", Label: "Dry Run", Type: "bool",
		Description: "Report which thumbnails would be deleted without removing anything"},
	{Name: "dir", Label: "Scope Directory", Type: "string",
		Description: "Only consider thumbnails of files under this directory (recursive): files on disk plus paths other tables still reference. Leave empty to sweep the whole cache."},
	{Name: "db", Label: "Additional Libraries", Type: "string",
		Description: "Other library databases that share this thumbnail cache, ';'-separated. Their thumbnails are kept too."},
	{Name: "discover", Label: "Discover Sibling Libraries", Type: "bool", Default: true,
		Description: "Treat every SQLite database in the configured database's folder as a library sharing the cache (they do: the cache lives next to the database)"},
	{Name: "manifest", Label: "Manifest File", Type: "string",
		Description: "Write a TSV of every cache file and the libraries that need it to this path"},
}

// thumbCacheDirs are the cache directories getThumbnailPath writes into,
// one per size the apps request. Their position is the dir index stored in
// each cache entry, so never reorder them.
var thumbCacheDirs = []string{"thumbnail_path_100", "thumbnail_path_600", "thumbnail_path_1200"}

// thumbCleanupMinAge protects freshly-written files: a thumbnail generated
// for a row inserted after this task snapshotted the media table would look
// orphaned, so anything newer than this is skipped and caught on a later run.
const thumbCleanupMinAge = 15 * time.Minute

// thumbCleanupPreviewMax caps how many individual deletions a run prints.
const thumbCleanupPreviewMax = 100

// thumbReadDirBatch is how many directory entries are read per syscall while
// indexing; the cache directories are flat and can hold millions of files,
// so they are streamed rather than read into one slice.
const thumbReadDirBatch = 4096

// thumbProgressEvery is how many cache entries the sweep visits between
// progress updates (SetJobProgress takes the queue lock).
const thumbProgressEvery = 2000

// thumbScanChunk is the row count per database read while marking; each
// chunk is its own short read transaction. A variable so tests can force
// the pagination path.
var thumbScanChunk = 20000

// thumbWarnMax caps the individual "failed to delete" lines. Every stdout
// line is persisted with the job record, so an unwritable cache directory
// must not turn into millions of warnings.
const thumbWarnMax = 20

// thumbDigest is a raw sha256 — the filename stem decoded from hex.
type thumbDigest [32]byte

func thumbDigestOf(input string) thumbDigest { return sha256.Sum256([]byte(input)) }

// thumbHashHex must produce byte-identical output to createHash in
// thumbnail.go and createHash in the Electron app: lowercase hex of sha256.
func thumbHashHex(input string) string {
	d := thumbDigestOf(input)
	return hex.EncodeToString(d[:])
}

// thumbTimeStampKey formats a timestamp the way both generators feed it into
// the hash: JavaScript Number.toString() semantics (no trailing zeros, no
// exponent for normal values), matching formatTimeStamp in thumbnail.go.
func thumbTimeStampKey(ts float64) string {
	return strconv.FormatFloat(ts, 'f', -1, 64)
}

// thumbSpellings returns the string plus its separator-swapped forms. The
// hash is exact-string sensitive but the two apps may spell the same file
// with different separators; treating every spelling as live costs a couple
// of extra lookups and prevents deleting a thumbnail that is still valid
// under the other spelling. The input itself is always element 0.
func thumbSpellings(p string) []string {
	out := []string{p}
	if alt := strings.ReplaceAll(p, `\`, "/"); alt != p {
		out = append(out, alt)
	}
	if alt := strings.ReplaceAll(p, "/", `\`); alt != p {
		out = append(out, alt)
	}
	return out
}

// isHex64 reports whether s is exactly 64 lowercase hex characters — the
// shape of every filename stem the thumbnail generators produce. Lowercase
// is required (not just accepted) because the index reconstructs filenames
// from the digest and must reproduce the on-disk name exactly.
func isHex64(s string) bool {
	if len(s) != 64 {
		return false
	}
	for _, c := range s {
		if (c < '0' || c > '9') && (c < 'a' || c > 'f') {
			return false
		}
	}
	return true
}

func decodeHex64(s string) (thumbDigest, bool) {
	var d thumbDigest
	if !isHex64(s) {
		return d, false
	}
	if _, err := hex.Decode(d[:], []byte(s)); err != nil {
		return d, false
	}
	return d, true
}

// thumbCacheEntry is one cache file: 34 bytes, no pointers, so an index of
// millions stays compact and invisible to the garbage collector.
type thumbCacheEntry struct {
	stem thumbDigest
	dir  uint8 // index into thumbCacheDirs
	ext  uint8 // index into thumbCacheIndex.exts
}

// thumbCacheIndex is every recognised cache file, sorted by digest, with a
// parallel owner mask set by the marking passes: one bit per library that
// can still name the file. Zero means orphan.
type thumbCacheIndex struct {
	baseDir      string
	exts         []string
	entries      []thumbCacheEntry
	owners       []uint32
	bit          uint32 // the library currently being marked
	unrecognized int
}

// isKept reports whether any library names entry i.
func (ix *thumbCacheIndex) isKept(i int) bool { return ix.owners[i] != 0 }

// buildThumbCacheIndex streams the cache directories under baseDir. Files
// that don't match the generators' naming scheme (README drops, foreign
// caches, whatever) are counted but never candidates for deletion. Missing
// directories are fine — a library that never rendered a size simply doesn't
// have that folder. Nothing is stat'ed here: size and mtime are only needed
// for the files that turn out to be orphans.
func buildThumbCacheIndex(ctx context.Context, baseDir string) (*thumbCacheIndex, error) {
	ix := &thumbCacheIndex{baseDir: baseDir}
	extIdx := map[string]uint8{}
	for di, dir := range thumbCacheDirs {
		f, err := os.Open(filepath.Join(baseDir, dir))
		if err != nil {
			if os.IsNotExist(err) {
				continue
			}
			return nil, fmt.Errorf("opening %s: %w", dir, err)
		}
		for {
			if err := ctx.Err(); err != nil {
				f.Close()
				return nil, err
			}
			batch, rerr := f.ReadDir(thumbReadDirBatch)
			for _, e := range batch {
				if !e.Type().IsRegular() {
					if !e.IsDir() {
						ix.unrecognized++
					}
					continue
				}
				name := e.Name()
				ext := filepath.Ext(name)
				switch strings.ToLower(ext) {
				case "", ".mp4", ".png", ".jpg", ".jpeg", ".webp":
				default:
					ix.unrecognized++
					continue
				}
				d, ok := decodeHex64(strings.TrimSuffix(name, ext))
				if !ok {
					ix.unrecognized++
					continue
				}
				ei, ok := extIdx[ext]
				if !ok {
					if len(ix.exts) >= 255 {
						ix.unrecognized++
						continue
					}
					ei = uint8(len(ix.exts))
					ix.exts = append(ix.exts, ext)
					extIdx[ext] = ei
				}
				ix.entries = append(ix.entries, thumbCacheEntry{stem: d, dir: uint8(di), ext: ei})
			}
			if rerr == io.EOF {
				break
			}
			if rerr != nil {
				f.Close()
				return nil, fmt.Errorf("reading %s: %w", dir, rerr)
			}
		}
		f.Close()
	}
	slices.SortFunc(ix.entries, func(a, b thumbCacheEntry) int {
		return bytes.Compare(a.stem[:], b.stem[:])
	})
	ix.owners = make([]uint32, len(ix.entries))
	ix.bit = 1
	return ix, nil
}

// path reconstructs the on-disk path of entry i.
func (ix *thumbCacheIndex) path(i int) string {
	e := ix.entries[i]
	return filepath.Join(ix.baseDir, thumbCacheDirs[e.dir], hex.EncodeToString(e.stem[:])+ix.exts[e.ext])
}

// find returns the index range [lo, hi) of entries with the given digest —
// the same stem can exist in every size directory and with more than one
// extension.
func (ix *thumbCacheIndex) find(stem thumbDigest) (int, int) {
	lo, _ := slices.BinarySearchFunc(ix.entries, stem, func(e thumbCacheEntry, t thumbDigest) int {
		return bytes.Compare(e.stem[:], t[:])
	})
	hi := lo
	for hi < len(ix.entries) && ix.entries[hi].stem == stem {
		hi++
	}
	return lo, hi
}

// keep marks every entry with this digest as named by the current library.
func (ix *thumbCacheIndex) keep(stem thumbDigest) {
	lo, hi := ix.find(stem)
	for i := lo; i < hi; i++ {
		ix.owners[i] |= ix.bit
	}
}

// keepInput marks the thumbnails of a hash input (a media path, or path +
// timestamp key) under every separator spelling.
func (ix *thumbCacheIndex) keepInput(input string) {
	for _, s := range thumbSpellings(input) {
		ix.keep(thumbDigestOf(s))
	}
}

// keepLiteral marks the file a database column points at by literal path:
// the entry whose digest, cache directory and extension the reference
// spells. The reference's parent directory is compared by name only (the
// cache may have been spelled through a different drive mapping or
// separator); a reference outside any known cache directory keeps every
// size with that digest, since over-keeping is the safe direction.
func (ix *thumbCacheIndex) keepLiteral(p string) {
	p = strings.TrimSpace(p)
	if p == "" {
		return
	}
	name, parent := p, ""
	if i := strings.LastIndexAny(p, `/\`); i >= 0 {
		name, parent = p[i+1:], p[:i]
		if j := strings.LastIndexAny(parent, `/\`); j >= 0 {
			parent = parent[j+1:]
		}
	}
	ext := filepath.Ext(name)
	d, ok := decodeHex64(strings.TrimSuffix(name, ext))
	if !ok {
		return
	}
	dir := -1
	for i, cd := range thumbCacheDirs {
		if strings.EqualFold(parent, cd) {
			dir = i
		}
	}
	lo, hi := ix.find(d)
	for i := lo; i < hi; i++ {
		e := ix.entries[i]
		if dir >= 0 && (int(e.dir) != dir || !strings.EqualFold(ix.exts[e.ext], ext)) {
			continue
		}
		ix.owners[i] |= ix.bit
	}
}

// thumbLibrary is one database whose thumbnails live in the cache being
// swept. Several libraries share a cache whenever their database files are
// siblings, because both apps derive the cache directory from the database's
// own folder.
type thumbLibrary struct {
	Label string
	Path  string
	DB    *sql.DB
	bit   uint32
	owned bool // opened by this task, so closed by it
}

// thumbLibraryMaxBits is how many libraries get a distinct owner bit; any
// beyond that share the last bit (the report merges them, deletion is still
// correct — a file any of them names is kept).
const thumbLibraryMaxBits = 32

// isSQLiteFile reports whether the file starts with the SQLite 3 header.
func isSQLiteFile(path string) bool {
	f, err := os.Open(path)
	if err != nil {
		return false
	}
	defer f.Close()
	var hdr [16]byte
	if _, err := io.ReadFull(f, hdr[:]); err != nil {
		return false
	}
	return string(hdr[:]) == "SQLite format 3\x00"
}

// discoverThumbLibraries assembles the libraries sharing the cache: the
// configured database first, then (with discover) every SQLite file beside
// it, then the explicit extras. A candidate without a media table is
// skipped with a note; a candidate that fails to open aborts the run,
// because a library we could not read might need the files about to go.
func discoverThumbLibraries(ctx context.Context, mainPath string, mainDB *sql.DB, extras []string, discover bool, log func(string)) ([]thumbLibrary, error) {
	libs := []thumbLibrary{{Label: filepath.Base(mainPath), Path: mainPath, DB: mainDB}}
	seen := map[string]bool{thumbPathKey(mainPath): true}

	var candidates []string
	if discover {
		entries, err := os.ReadDir(filepath.Dir(mainPath))
		if err != nil {
			return nil, fmt.Errorf("listing the database folder: %w", err)
		}
		for _, e := range entries {
			if !e.Type().IsRegular() {
				continue
			}
			name := strings.ToLower(e.Name())
			if strings.HasSuffix(name, "-wal") || strings.HasSuffix(name, "-shm") || strings.HasSuffix(name, "-journal") {
				continue
			}
			p := filepath.Join(filepath.Dir(mainPath), e.Name())
			if seen[thumbPathKey(p)] || !isSQLiteFile(p) {
				continue
			}
			candidates = append(candidates, p)
		}
	}
	for _, p := range extras {
		if p = strings.TrimSpace(p); p != "" && !seen[thumbPathKey(p)] {
			candidates = append(candidates, p)
		}
	}

	for _, p := range candidates {
		if seen[thumbPathKey(p)] {
			continue
		}
		seen[thumbPathKey(p)] = true
		// mode=rw (not the default rwc): a mistyped path must fail, not be
		// created as an empty database and skipped as "not a library".
		db, err := sql.Open("sqlite", "file:"+p+"?mode=rw&_pragma=busy_timeout=5000")
		if err != nil {
			return nil, fmt.Errorf("opening library %s: %w", p, err)
		}
		if err := db.PingContext(ctx); err != nil {
			db.Close()
			return nil, fmt.Errorf("opening library %s: %w", p, err)
		}
		cols, err := thumbTableColumns(ctx, db, "media")
		if err != nil {
			db.Close()
			return nil, fmt.Errorf("reading library %s: %w", p, err)
		}
		if cols == nil || !cols["path"] {
			db.Close()
			log(fmt.Sprintf("Skipping %s: not a media library (no media table)", filepath.Base(p)))
			continue
		}
		libs = append(libs, thumbLibrary{Label: filepath.Base(p), Path: p, DB: db, owned: true})
	}

	for i := range libs {
		b := i
		if b >= thumbLibraryMaxBits {
			b = thumbLibraryMaxBits - 1
		}
		libs[i].bit = 1 << uint(b)
	}
	return libs, nil
}

// thumbPathKey normalises a path for duplicate detection: cleaned,
// slash-separated, case-folded (the cache is a Windows-first feature and
// NTFS is case-insensitive; a spurious match on a case-sensitive filesystem
// only drops a duplicate candidate).
func thumbPathKey(p string) string {
	return strings.ToLower(filepath.ToSlash(filepath.Clean(p)))
}

func closeThumbLibraries(libs []thumbLibrary) {
	for _, l := range libs {
		if l.owned && l.DB != nil {
			l.DB.Close()
		}
	}
}

// thumbOwnershipReport summarises who needs what once every library has
// marked the index: per-library counts, how many files two or more
// libraries share, and how many nobody names.
func thumbOwnershipReport(ix *thumbCacheIndex, libs []thumbLibrary) []string {
	per := make([]int, len(libs))
	shared, none := 0, 0
	for _, o := range ix.owners {
		if o == 0 {
			none++
			continue
		}
		n := 0
		for i, l := range libs {
			if o&l.bit != 0 {
				per[i]++
				n++
			}
		}
		if n > 1 {
			shared++
		}
	}
	lines := []string{fmt.Sprintf("Ownership across %d librar%s:", len(libs), map[bool]string{true: "y", false: "ies"}[len(libs) == 1])}
	for i, l := range libs {
		lines = append(lines, fmt.Sprintf("  %s needs %d file(s)", l.Label, per[i]))
	}
	if len(libs) > 1 {
		lines = append(lines, fmt.Sprintf("  shared by two or more: %d", shared))
	}
	lines = append(lines, fmt.Sprintf("  named by none (orphans): %d", none))
	return lines
}

// writeThumbManifest writes one line per cache file with the libraries that
// name it, tab-separated, so an attribution can be inspected before a real
// run. "-" marks an orphan.
func writeThumbManifest(path string, ix *thumbCacheIndex, libs []thumbLibrary) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	w := bufio.NewWriterSize(f, 1<<20)
	fmt.Fprintln(w, "file\tlibraries")
	for i, o := range ix.owners {
		w.WriteString(ix.path(i))
		w.WriteByte('\t')
		if o == 0 {
			w.WriteByte('-')
		} else {
			first := true
			for _, l := range libs {
				if o&l.bit != 0 {
					if !first {
						w.WriteByte(';')
					}
					first = false
					w.WriteString(l.Label)
				}
			}
		}
		w.WriteByte('\n')
	}
	if err := w.Flush(); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// thumbTableExists reports whether a table is present — a viewer-only
// library has no face/embedding tables and a fresh server has no battle log.
func thumbTableExists(ctx context.Context, db *sql.DB, name string) (bool, error) {
	var one int
	err := db.QueryRowContext(ctx,
		`SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?`, name).Scan(&one)
	if errors.Is(err, sql.ErrNoRows) {
		return false, nil
	}
	return err == nil, err
}

// thumbTableColumns returns the column names of a table, or nil when the
// table doesn't exist.
func thumbTableColumns(ctx context.Context, db *sql.DB, table string) (map[string]bool, error) {
	rows, err := db.QueryContext(ctx, fmt.Sprintf(`PRAGMA table_info(%q)`, table))
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	cols := map[string]bool{}
	for rows.Next() {
		var (
			cid     int
			name    string
			typ     sql.NullString
			notnull int
			dflt    sql.NullString
			pk      int
		)
		if err := rows.Scan(&cid, &name, &typ, &notnull, &dflt, &pk); err != nil {
			return nil, err
		}
		cols[name] = true
	}
	if len(cols) == 0 {
		return nil, rows.Err()
	}
	return cols, rows.Err()
}

// thumbLiteralColumns are the database columns that store a thumbnail file
// path verbatim: DB-recorded thumbnails and tag previews, which may predate
// the current naming scheme. Column set differs between library generations
// (the viewer's schema and the server's have drifted before), so absent
// columns are simply skipped — a missing column can't be referencing
// anything.
var thumbLiteralColumns = []struct{ table, column string }{
	{"media", "thumbnail_path_100"},
	{"media", "thumbnail_path_600"},
	{"media", "thumbnail_path_1200"},
	{"tag", "thumbnail_path_600"},
}

// thumbMarkStats reports what the marking passes saw.
type thumbMarkStats struct {
	MediaRows  int
	Timestamps int
	Literals   int
}

// markLiveThumbnails streams the database and marks every cache entry the
// library can still name. With includePaths, the media table's own paths
// (and their tagged timestamps) are marked — the full-cache sweep. Without
// it only literal references are marked, which is what a scoped run needs
// before judging its candidates: a tag preview pointing at an orphan's
// thumbnail keeps that file in the full sweep, so it must in the scoped one.
//
// The media table is scanned exactly once whichever columns are wanted.
func markLiveThumbnails(ctx context.Context, db *sql.DB, ix *thumbCacheIndex, includePaths bool) (thumbMarkStats, error) {
	var st thumbMarkStats

	mediaCols, err := thumbTableColumns(ctx, db, "media")
	if err != nil {
		return st, fmt.Errorf("inspecting media table: %w", err)
	}
	if mediaCols == nil {
		return st, fmt.Errorf("no media table in this database")
	}
	var litCols []string
	for _, lc := range thumbLiteralColumns {
		if lc.table == "media" && mediaCols[lc.column] {
			litCols = append(litCols, lc.column)
		}
	}

	if includePaths || len(litCols) > 0 {
		// Keyset-paginated on the path index so no read transaction stays
		// open for more than one chunk: the library is shared with the
		// viewer, and a rollback-journal database blocks writers for as long
		// as a reader holds its cursor.
		sel := []string{`"path"`}
		sel = append(sel, litCols...)
		query := `SELECT ` + strings.Join(sel, ", ") + ` FROM media WHERE "path" > ? ORDER BY "path" LIMIT ?`
		lits := make([]sql.NullString, len(litCols))
		var p string
		dest := make([]any, 0, 1+len(litCols))
		dest = append(dest, &p)
		for i := range lits {
			dest = append(dest, &lits[i])
		}
		after := ""
		for {
			rows, err := db.QueryContext(ctx, query, after, thumbScanChunk)
			if err != nil {
				return st, fmt.Errorf("scanning media: %w", err)
			}
			n := 0
			for rows.Next() {
				if err := rows.Scan(dest...); err != nil {
					rows.Close()
					return st, err
				}
				n++
				st.MediaRows++
				after = p
				if includePaths {
					ix.keepInput(p)
				}
				for _, l := range lits {
					if l.Valid && l.String != "" {
						st.Literals++
						ix.keepLiteral(l.String)
					}
				}
			}
			rows.Close()
			if err := rows.Err(); err != nil {
				return st, err
			}
			if n < thumbScanChunk {
				break
			}
		}
	}

	if includePaths {
		// Timestamped variants: only for rows the media table still has — a
		// timestamp on an orphaned tag row names a thumbnail of a deleted
		// item, which is exactly what this task exists to remove. The join
		// is a primary-key probe per timestamped tag row.
		// Chunked on rowid for the same reason as the media scan; the
		// tag table can be an order of magnitude larger than media.
		var afterRow int64 = -1
		for {
			tsRows, err := db.QueryContext(ctx, `
				SELECT mtbc.rowid, mtbc.media_path, mtbc.time_stamp
				FROM media_tag_by_category mtbc
				JOIN media m ON m."path" = mtbc.media_path
				WHERE mtbc.rowid > ? AND mtbc.time_stamp > 0
				ORDER BY mtbc.rowid LIMIT ?`, afterRow, thumbScanChunk)
			if err != nil {
				return st, fmt.Errorf("loading tagged timestamps: %w", err)
			}
			n := 0
			for tsRows.Next() {
				var rowid int64
				var p string
				var ts float64
				if err := tsRows.Scan(&rowid, &p, &ts); err != nil {
					tsRows.Close()
					return st, err
				}
				n++
				afterRow = rowid
				st.Timestamps++
				ix.keepInput(p + thumbTimeStampKey(ts))
			}
			tsRows.Close()
			if err := tsRows.Err(); err != nil {
				return st, err
			}
			if n < thumbScanChunk {
				break
			}
		}
	}

	// Tag previews.
	tagCols, err := thumbTableColumns(ctx, db, "tag")
	if err != nil {
		return st, fmt.Errorf("inspecting tag table: %w", err)
	}
	if tagCols["thumbnail_path_600"] {
		rows, err := db.QueryContext(ctx,
			`SELECT thumbnail_path_600 FROM tag WHERE thumbnail_path_600 IS NOT NULL AND thumbnail_path_600 != ''`)
		if err != nil {
			return st, fmt.Errorf("loading tag previews: %w", err)
		}
		for rows.Next() {
			var p sql.NullString
			if err := rows.Scan(&p); err != nil {
				rows.Close()
				return st, err
			}
			if p.Valid {
				st.Literals++
				ix.keepLiteral(p.String)
			}
		}
		rows.Close()
		if err := rows.Err(); err != nil {
			return st, err
		}
	}
	return st, nil
}

// thumbVictim is one cache entry to delete, with the media path it belonged
// to when that is known (scoped mode; the full sweep can't know).
type thumbVictim struct {
	Index int
	Note  string
}

// thumbScopeStats reports what a scoped run found.
type thumbScopeStats struct {
	Walked        int // media files found on disk under the scope
	Referenced    int // paths under the scope some non-media table still references
	Candidates    int // distinct candidate paths after merging the two
	Live          int // candidates that are media rows (thumbnails kept)
	OrphanNoThumb int // candidates with no row and no thumbnail either
	OrphanPaths   int // candidates with no row and at least one thumbnail
}

// thumbScopePrefixes returns the prefixes a stored path must start with to
// sit under dir: the cleaned directory plus a separator, in both separator
// spellings.
func thumbScopePrefixes(dir string) []string {
	return thumbSpellings(filepath.Clean(dir) + string(filepath.Separator))
}

// thumbScopeCandidates enumerates the media paths a scoped run should judge:
// media files on disk under dir (recursive) and paths under dir that any
// non-media table still references. Both are deduplicated on the
// slash-normalised string, keeping the first spelling seen.
func thumbScopeCandidates(ctx context.Context, libs []thumbLibrary, dir string, log func(string)) ([]string, thumbScopeStats, error) {
	var st thumbScopeStats
	seen := map[string]struct{}{}
	var cands []string
	add := func(p string) bool {
		key := strings.ReplaceAll(p, `\`, "/")
		if _, dup := seen[key]; dup {
			return false
		}
		seen[key] = struct{}{}
		cands = append(cands, p)
		return true
	}

	root := filepath.Clean(dir)
	if info, err := os.Stat(root); err != nil {
		log(fmt.Sprintf("Scope directory is not on disk (%v); judging database references only", err))
	} else if !info.IsDir() {
		return nil, st, fmt.Errorf("scope %s is not a directory", root)
	} else {
		// The hash is case-exact and the walk spells every file with the
		// root as given, so a scope typed in the wrong case would match
		// nothing. On a case-insensitive filesystem EvalSymlinks returns
		// the directory's on-disk spelling, which is what the apps hashed
		// when they scanned it. Only a case difference is adopted: when the
		// scope is a junction or symlink the apps spelled paths through the
		// link, not its target, and resolving it would match nothing.
		if canon, err := filepath.EvalSymlinks(root); err == nil && canon != root && strings.EqualFold(canon, root) {
			log(fmt.Sprintf("Scope on disk is spelled %s", canon))
			root = canon
		}
		werr := filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
			if err := ctx.Err(); err != nil {
				return err
			}
			if err != nil {
				log(fmt.Sprintf("Warning: skipping %s: %v", p, err))
				if d != nil && d.IsDir() {
					return fs.SkipDir
				}
				return nil
			}
			if d.IsDir() {
				// The cache itself can sit under a broad scope; its files are
				// never media and hashing millions of them is pure waste.
				if strings.HasPrefix(d.Name(), "thumbnail_path_") && p != root {
					return fs.SkipDir
				}
				return nil
			}
			if !d.Type().IsRegular() {
				return nil
			}
			if !mediaext.IsImage(p) && !mediaext.IsVideo(p) && !mediaext.IsAudio(p) {
				return nil
			}
			st.Walked++
			add(p)
			return nil
		})
		if werr != nil {
			return nil, st, werr
		}
	}

	// Paths other tables still reference under the scope. Range comparisons
	// rather than LIKE so an index on the column is usable; the upper bound
	// is the prefix followed by the largest code point, which every path
	// under the prefix sorts below.
	prefixes := thumbScopePrefixes(dir)
	for _, p := range thumbScopePrefixes(root) {
		if !slices.Contains(prefixes, p) {
			prefixes = append(prefixes, p)
		}
	}
	for _, lib := range libs {
		for _, pc := range media.PathColumns() {
			if pc.Table == "media" {
				continue
			}
			ok, err := thumbTableExists(ctx, lib.DB, pc.Table)
			if err != nil {
				return nil, st, err
			}
			if !ok {
				continue
			}
			for _, prefix := range prefixes {
				rows, err := lib.DB.QueryContext(ctx, fmt.Sprintf(
					`SELECT DISTINCT %s FROM %s WHERE %s >= ? AND %s < ?`,
					pc.Quoted, pc.Table, pc.Quoted, pc.Quoted), prefix, prefix+"\U0010FFFF")
				if err != nil {
					return nil, st, fmt.Errorf("scanning %s.%s in %s: %w", pc.Table, pc.Column, lib.Label, err)
				}
				for rows.Next() {
					var p sql.NullString
					if err := rows.Scan(&p); err != nil {
						rows.Close()
						return nil, st, err
					}
					if p.Valid && p.String != "" && add(p.String) {
						st.Referenced++
					}
				}
				rows.Close()
				if err := rows.Err(); err != nil {
					return nil, st, err
				}
			}
		}
	}
	st.Candidates = len(cands)
	return cands, st, nil
}

// thumbScopeVictims judges each candidate against the media table and the
// cache. A candidate is live when any spelling of it is a media row; its
// thumbnails are then kept. Otherwise every cache entry answering to the
// candidate (default thumbnail, and tagged-timestamp thumbnails from its
// leftover tag rows) is a victim, unless a literal reference marked it kept.
//
// Judging is exact-string, like the hash: a row spelled with a different
// drive-letter case is a different key, and the thumbnail of the other
// spelling is an orphan in the full sweep too.
func thumbScopeVictims(ctx context.Context, libs []thumbLibrary, ix *thumbCacheIndex, cands []string, st *thumbScopeStats) ([]thumbVictim, error) {
	type libStmts struct{ live, ts *sql.Stmt }
	stmts := make([]libStmts, 0, len(libs))
	defer func() {
		for _, s := range stmts {
			if s.live != nil {
				s.live.Close()
			}
			if s.ts != nil {
				s.ts.Close()
			}
		}
	}()
	for _, lib := range libs {
		live, err := lib.DB.PrepareContext(ctx, `SELECT 1 FROM media WHERE "path" IN (?, ?, ?) LIMIT 1`)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", lib.Label, err)
		}
		var ts *sql.Stmt
		if ok, err := thumbTableExists(ctx, lib.DB, "media_tag_by_category"); err != nil {
			live.Close()
			return nil, err
		} else if ok {
			ts, err = lib.DB.PrepareContext(ctx,
				`SELECT DISTINCT time_stamp FROM media_tag_by_category WHERE media_path IN (?, ?, ?) AND time_stamp > 0`)
			if err != nil {
				live.Close()
				return nil, fmt.Errorf("%s: %w", lib.Label, err)
			}
		}
		stmts = append(stmts, libStmts{live: live, ts: ts})
	}

	var victims []thumbVictim
	claimed := map[int]struct{}{}
	collect := func(input, note string) int {
		n := 0
		for _, s := range thumbSpellings(input) {
			lo, hi := ix.find(thumbDigestOf(s))
			for i := lo; i < hi; i++ {
				if ix.isKept(i) {
					continue
				}
				if _, dup := claimed[i]; dup {
					continue
				}
				claimed[i] = struct{}{}
				victims = append(victims, thumbVictim{Index: i, Note: note})
				n++
			}
		}
		return n
	}

	for _, c := range cands {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		sp := thumbSpellings(c)
		args := []any{sp[0], sp[0], sp[0]}
		for i := 1; i < len(sp) && i < 3; i++ {
			args[i] = sp[i]
		}

		// Live in any library keeps it.
		live := false
		for _, s := range stmts {
			var one int
			err := s.live.QueryRowContext(ctx, args...).Scan(&one)
			if err == nil {
				live = true
				break
			}
			if !errors.Is(err, sql.ErrNoRows) {
				return nil, err
			}
		}
		if live {
			st.Live++
			continue
		}

		n := collect(c, c)
		seenTS := map[string]bool{}
		for _, s := range stmts {
			if s.ts == nil {
				continue
			}
			tsRows, err := s.ts.QueryContext(ctx, args...)
			if err != nil {
				return nil, err
			}
			for tsRows.Next() {
				var ts float64
				if err := tsRows.Scan(&ts); err != nil {
					tsRows.Close()
					return nil, err
				}
				key := thumbTimeStampKey(ts)
				if seenTS[key] {
					continue
				}
				seenTS[key] = true
				n += collect(c+key, c+" @"+key+"s")
			}
			tsRows.Close()
			if err := tsRows.Err(); err != nil {
				return nil, err
			}
		}
		if n == 0 {
			st.OrphanNoThumb++
		} else {
			st.OrphanPaths++
		}
	}
	return victims, nil
}

// thumbSweeper deletes (or, in a dry run, reports) victims one at a time,
// applying the minimum-age guard and keeping the tallies.
type thumbSweeper struct {
	j      *jobqueue.Job
	q      *jobqueue.Queue
	ix     *thumbCacheIndex
	dryRun bool
	cutoff time.Time

	removed, failed, tooNew, gone int
	freed                         int64
	previewShown                  int
	lastProgressLine              time.Time
}

func (s *thumbSweeper) visit(i int, note string) {
	p := s.ix.path(i)
	info, err := os.Stat(p)
	if err != nil {
		// Deleted underneath us (a concurrent run, the user) — nothing to do.
		s.gone++
		return
	}
	if info.ModTime().After(s.cutoff) {
		// Could be a thumbnail for a row added since our snapshot — leave it
		// for the next run rather than racing the generator.
		s.tooNew++
		return
	}
	line := ""
	if s.previewShown < thumbCleanupPreviewMax {
		verb := "Deleted: "
		if s.dryRun {
			verb = "Would delete: "
		}
		line = verb + p
		if note != "" {
			line += "  (thumbnail of " + note + ")"
		}
	}
	if !s.dryRun {
		if err := os.Remove(p); err != nil {
			s.failed++
			if s.failed <= thumbWarnMax {
				s.q.PushJobStdout(s.j.ID, fmt.Sprintf("Warning: failed to delete %s: %v", p, err))
			} else if s.failed == thumbWarnMax+1 {
				s.q.PushJobStdout(s.j.ID, "... (further deletion failures not listed; see the final count)")
			}
			return
		}
	}
	s.removed++
	s.freed += info.Size()
	if line != "" {
		s.q.PushJobStdout(s.j.ID, line)
		s.previewShown++
		if s.previewShown == thumbCleanupPreviewMax {
			s.q.PushJobStdout(s.j.ID, "... (further files not listed individually)")
		}
	} else if time.Since(s.lastProgressLine) >= 10*time.Second {
		// Every stdout line is persisted with the job, so progress is
		// reported on a clock, not a count — a first sweep of a big cache
		// can remove millions.
		s.lastProgressLine = time.Now()
		s.q.PushJobStdout(s.j.ID, fmt.Sprintf("Progress: %d orphaned thumbnail(s) so far", s.removed))
	}
}

// pauseOrCancel checks the job's control signals; the returned error (if
// any) is what the task should return after its own cleanup message.
func (s *thumbSweeper) pauseOrCancel(ctx context.Context, done, total int) error {
	select {
	case <-ctx.Done():
		s.q.PushJobStdout(s.j.ID, fmt.Sprintf("Task was canceled after %d thumbnail(s)", s.removed))
		_ = s.q.CancelJob(s.j.ID)
		return ctx.Err()
	default:
	}
	if s.q.PauseRequested(s.j.ID) {
		s.q.PushJobStdout(s.j.ID, fmt.Sprintf("Paused at %d/%d — resume to continue", done, total))
		return jobqueue.ErrPaused
	}
	return nil
}

func (s *thumbSweeper) summary(kept int) string {
	verb := "Removed"
	if s.dryRun {
		verb = "Would remove"
	}
	return fmt.Sprintf(
		"%s %d orphaned thumbnail(s), reclaiming %.1f MB (%d left in place, %d too new to judge, %d failed, %d already gone)",
		verb, s.removed, float64(s.freed)/(1024*1024), kept, s.tooNew, s.failed, s.gone)
}

func thumbnailCleanupTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	ctx := j.Ctx
	tokens := dirTaskTokens(j)
	opts := ParseOptions(&jobqueue.Job{Arguments: tokens}, thumbnailCleanupOptions)
	dryRun, _ := opts["dry-run"].(bool)
	discover, _ := opts["discover"].(bool)
	scopeDir, _ := opts["dir"].(string)
	scopeDir = strings.TrimSpace(scopeDir)
	manifestPath, _ := opts["manifest"].(string)
	manifestPath = strings.TrimSpace(manifestPath)

	// --db may repeat (ParseOptions keeps only the last value) and each
	// value may hold several ';'-separated paths.
	var extraDBs []string
	valueOpts := map[string]bool{}
	for _, o := range thumbnailCleanupOptions {
		if o.Type != "bool" {
			valueOpts[o.Name] = true
		}
	}
	for i := 0; i < len(tokens); i++ {
		name, val, hasEq := strings.Cut(strings.TrimPrefix(tokens[i], "--"), "=")
		if !strings.HasPrefix(tokens[i], "--") || name != "db" {
			continue
		}
		if !hasEq && i+1 < len(tokens) && !strings.HasPrefix(tokens[i+1], "--") {
			i++
			val = tokens[i]
		}
		for _, p := range strings.Split(val, ";") {
			if p = strings.TrimSpace(p); p != "" {
				extraDBs = append(extraDBs, p)
			}
		}
	}

	// Same shape as the move and split tasks: a bare positional token is the
	// directory. Tokens that are values of string options are skipped, so
	// the first remaining non-flag token is unambiguous.
	if scopeDir == "" {
		for i := 0; i < len(tokens); i++ {
			tok := tokens[i]
			if strings.HasPrefix(tok, "--") {
				if name, _, hasEq := strings.Cut(strings.TrimPrefix(tok, "--"), "="); !hasEq && valueOpts[name] {
					i++
				}
				continue
			}
			if !strings.HasPrefix(tok, "-") {
				scopeDir = tok
				break
			}
		}
	}
	if scopeDir != "" && !filepath.IsAbs(scopeDir) {
		q.PushJobStdout(j.ID, fmt.Sprintf("Error: scope directory must be an absolute path, got %q", scopeDir))
		q.ErrorJob(j.ID)
		return fmt.Errorf("scope directory must be absolute")
	}

	fail := func(msg string, err error) error {
		if ctx.Err() != nil {
			q.PushJobStdout(j.ID, "Task was canceled")
			_ = q.CancelJob(j.ID)
			return err
		}
		q.PushJobStdout(j.ID, fmt.Sprintf("Error: %s: %v", msg, err))
		q.ErrorJob(j.ID)
		return err
	}
	log := func(line string) { q.PushJobStdout(j.ID, line) }

	dbPath := appconfig.Get().DBPath
	if dbPath == "" {
		return fail("no database path configured", fmt.Errorf("cannot locate the thumbnail cache"))
	}
	baseDir := filepath.Dir(dbPath)
	if dryRun {
		log("Dry run: nothing will be deleted")
	}
	log(fmt.Sprintf("Indexing thumbnail cache under %s", baseDir))
	t0 := time.Now()
	ix, err := buildThumbCacheIndex(ctx, baseDir)
	if err != nil {
		return fail("scanning thumbnail directories", err)
	}
	log(fmt.Sprintf("Indexed %d thumbnail file(s) in %s (%d unrelated file(s) left alone)",
		len(ix.entries), time.Since(t0).Round(time.Millisecond), ix.unrecognized))
	if len(ix.entries) == 0 {
		log("Nothing to clean")
		q.CompleteJob(j.ID)
		return nil
	}

	sw := &thumbSweeper{j: j, q: q, ix: ix, dryRun: dryRun, cutoff: time.Now().Add(-thumbCleanupMinAge), lastProgressLine: time.Now()}

	libs, err := discoverThumbLibraries(ctx, dbPath, q.Db, extraDBs, discover, log)
	if err != nil {
		return fail("discovering libraries sharing the cache (nothing deleted)", err)
	}
	defer closeThumbLibraries(libs)
	labels := make([]string, 0, len(libs))
	for _, l := range libs {
		labels = append(labels, l.Label)
	}
	log(fmt.Sprintf("Libraries sharing this cache: %s", strings.Join(labels, ", ")))

	if scopeDir != "" {
		log(fmt.Sprintf("Scope: %s (and subdirectories)", filepath.Clean(scopeDir)))
		t0 = time.Now()
		literals := 0
		for _, lib := range libs {
			ix.bit = lib.bit
			st, err := markLiveThumbnails(ctx, lib.DB, ix, false)
			if err != nil {
				return fail("reading thumbnail references in "+lib.Label, err)
			}
			literals += st.Literals
		}
		cands, sst, err := thumbScopeCandidates(ctx, libs, scopeDir, log)
		if err != nil {
			return fail("enumerating the scope", err)
		}
		log(fmt.Sprintf("Scope holds %d media file(s) on disk and %d path(s) only a database remembers: %d candidate(s), %d literal reference(s) protected",
			sst.Walked, sst.Referenced, sst.Candidates, literals))
		victims, err := thumbScopeVictims(ctx, libs, ix, cands, &sst)
		if err != nil {
			return fail("judging candidates", err)
		}
		log(fmt.Sprintf("%d candidate(s) are still in the library, %d are orphaned with %d thumbnail(s) to remove, %d orphaned with no thumbnails (judged in %s)",
			sst.Live, sst.OrphanPaths, len(victims), sst.OrphanNoThumb, time.Since(t0).Round(time.Millisecond)))
		if sst.Candidates > 0 && sst.Live == 0 && len(victims) == 0 {
			log("Hint: nothing under the scope matched a media row or a cached thumbnail. Paths are matched exactly (case and drive letter included) - compare with how the library spells them, e.g. lokictl db query \"SELECT path FROM media WHERE path LIKE ? LIMIT 5\" --arg \"<scope>%\"")
		}

		_ = q.SetJobProgress(j.ID, 0, len(victims))
		for i, v := range victims {
			if i%thumbProgressEvery == 0 {
				if err := sw.pauseOrCancel(ctx, i, len(victims)); err != nil {
					return err
				}
				_ = q.SetJobProgress(j.ID, i, len(victims))
			}
			sw.visit(v.Index, v.Note)
		}
		_ = q.SetJobProgress(j.ID, len(victims), len(victims))
		log(sw.summary(len(ix.entries) - len(victims)))
		q.CompleteJob(j.ID)
		return nil
	}

	for _, lib := range libs {
		t0 = time.Now()
		ix.bit = lib.bit
		st, err := markLiveThumbnails(ctx, lib.DB, ix, true)
		if err != nil {
			return fail("building the set of valid thumbnails for "+lib.Label, err)
		}
		log(fmt.Sprintf("%s: %d media row(s), %d tagged timestamp(s), %d literal thumbnail reference(s) (matched in %s)",
			lib.Label, st.MediaRows, st.Timestamps, st.Literals, time.Since(t0).Round(time.Millisecond)))
	}
	for _, line := range thumbOwnershipReport(ix, libs) {
		log(line)
	}
	kept := 0
	for i := range ix.entries {
		if ix.isKept(i) {
			kept++
		}
	}
	if manifestPath != "" {
		if err := writeThumbManifest(manifestPath, ix, libs); err != nil {
			return fail("writing the manifest (nothing deleted)", err)
		}
		log("Manifest written to " + manifestPath)
	}

	total := len(ix.entries)
	_ = q.SetJobProgress(j.ID, 0, total)
	for i := range ix.entries {
		if i%thumbProgressEvery == 0 {
			if err := sw.pauseOrCancel(ctx, i, total); err != nil {
				return err
			}
			_ = q.SetJobProgress(j.ID, i, total)
		}
		if ix.isKept(i) {
			continue
		}
		sw.visit(i, "")
	}
	_ = q.SetJobProgress(j.ID, total, total)
	log(sw.summary(kept))
	q.CompleteJob(j.ID)
	return nil
}
