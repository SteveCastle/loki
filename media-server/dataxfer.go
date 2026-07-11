package main

// dataxfer.go — full-library export/import with cross-root path
// normalization. Exports the TAGGED items (curated content) and everything
// referencing them — tags, embeddings, faces, people — into a portable
// archive whose paths are storage-root-RELATIVE. Import rebases those
// relative keys onto the destination's root (local <-> s3 seamlessly),
// copies the file bytes, and merges by media item (skip items already
// present). Thumbnails are regenerated on the destination, not shipped.
//
// Archive layout (tar.gz):
//   manifest.json     — version, source roots, counts, timestamp
//   library.db        — SQLite with ONLY the selected rows, paths relative
//   files/<relkey>    — media bytes for each tagged file

import (
	"archive/tar"
	"compress/gzip"
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path"
	"sort"
	"strings"
	"time"

	"github.com/stevecastle/shrike/media"
	"github.com/stevecastle/shrike/storage"
)

const dataxferVersion = 1

type dataxferManifest struct {
	Version     int       `json:"version"`
	CreatedAt   time.Time `json:"created_at"`
	SourceRoots []string  `json:"source_roots"`
	MediaCount  int       `json:"media_count"`
	Note        string    `json:"note"`
}

// buildExportKeys assigns each selected media path a FLAT, collision-free
// archive key. The source's storage roots are deliberately ignored: every
// tagged file — whether it lived inside a configured root, on some other
// drive, or in an S3 bucket — gets a simple key (its basename, disambiguated
// with a counter). The destination rebases each key onto its own primary
// root, so a library from a local host imports cleanly onto an S3 server
// regardless of how the source was organized.
//
// Two different source paths NEVER share a key (even with the same basename),
// so nothing is silently overwritten. The input is sorted first so key
// assignment is stable across re-exports.
func buildExportKeys(selected []string) map[string]string {
	sorted := append([]string(nil), selected...)
	sort.Strings(sorted)

	keys := make(map[string]string, len(sorted))
	used := make(map[string]bool, len(sorted))
	for _, p := range sorted {
		base := flatBaseName(p)
		key := base
		if used[key] {
			ext := path.Ext(base)
			stem := strings.TrimSuffix(base, ext)
			for i := 1; used[key]; i++ {
				key = fmt.Sprintf("%s-%d%s", stem, i, ext)
			}
		}
		used[key] = true
		keys[p] = key
	}
	return keys
}

// flatBaseName returns the last path segment of any path (local Windows or
// POSIX, or s3://), sanitized to a safe flat filename with no separators.
func flatBaseName(p string) string {
	s := strings.ReplaceAll(p, "\\", "/")
	s = strings.TrimRight(s, "/")
	if i := strings.LastIndex(s, "/"); i >= 0 {
		s = s[i+1:]
	}
	s = strings.TrimSpace(s)
	if s == "" || s == "." || s == ".." {
		return "file"
	}
	return s
}

// safeFlatKey reports whether a key from an (untrusted) archive is a plain
// flat filename — no separators, no traversal. Import rejects anything else.
func safeFlatKey(key string) bool {
	if key == "" || key == "." || key == ".." {
		return false
	}
	return !strings.ContainsAny(key, "/\\") && !strings.Contains(key, "..")
}

// rebaseKey joins a destination root path with a flat archive key.
func rebaseKey(destRoot, key string) string {
	return strings.TrimRight(destRoot, "/\\") + "/" + key
}

// openMediaSource returns a reader for a media path, via its storage backend
// (local or s3) or a direct local open as a fallback.
func openMediaSource(ctx context.Context, reg *storage.Registry, p string) (io.ReadCloser, error) {
	if reg != nil {
		if b := reg.BackendFor(p); b != nil {
			return b.Download(ctx, p)
		}
	}
	if strings.HasPrefix(p, "s3://") || strings.HasPrefix(p, "http://") || strings.HasPrefix(p, "https://") {
		return nil, fmt.Errorf("no backend for %s", p)
	}
	return os.Open(p)
}

// exportHandler streams a .lokiexport (tar.gz) of the tagged library.
func exportHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "use GET", http.StatusMethodNotAllowed)
			return
		}
		ctx := r.Context()

		// Selected media = every path that has at least one tag.
		selected, err := selectTaggedPaths(deps.DB)
		if err != nil {
			http.Error(w, "select tagged: "+err.Error(), http.StatusInternalServerError)
			return
		}
		if len(selected) == 0 {
			http.Error(w, "no tagged media to export", http.StatusBadRequest)
			return
		}

		// Build the portable DB (relativized paths, thumbnails/previews nulled)
		// in a temp file on disk so we never hold it all in memory.
		tmpDB, err := os.CreateTemp("", "loki-export-*.db")
		if err != nil {
			http.Error(w, "temp: "+err.Error(), http.StatusInternalServerError)
			return
		}
		tmpDBPath := tmpDB.Name()
		tmpDB.Close()
		defer os.Remove(tmpDBPath)

		relByPath, err := buildExportDB(deps, tmpDBPath, selected)
		if err != nil {
			http.Error(w, "build export db: "+err.Error(), http.StatusInternalServerError)
			return
		}

		filename := fmt.Sprintf("lowkey-export-%s.lokiexport", time.Now().Format("20060102-150405"))
		w.Header().Set("Content-Type", "application/gzip")
		w.Header().Set("Content-Disposition", "attachment; filename=\""+filename+"\"")

		gz := gzip.NewWriter(w)
		defer gz.Close()
		tw := tar.NewWriter(gz)
		defer tw.Close()

		// manifest
		roots := []string{}
		for _, e := range deps.Storage.AllRoots() {
			roots = append(roots, e.Path)
		}
		manifest := dataxferManifest{
			Version:     dataxferVersion,
			CreatedAt:   time.Now().UTC(),
			SourceRoots: roots,
			MediaCount:  len(selected),
			Note:        "tagged media + referencing tags/embeddings/faces; flat root-agnostic keys; thumbnails regenerate on import",
		}
		manifestJSON, _ := json.MarshalIndent(manifest, "", "  ")
		if err := writeTarBytes(tw, "manifest.json", manifestJSON); err != nil {
			return
		}

		// library.db
		if err := writeTarFile(tw, "library.db", tmpDBPath); err != nil {
			return
		}

		// files/<relkey> — stream each tagged file's bytes.
		for _, p := range selected {
			rk := relByPath[p]
			if rk == "" {
				continue
			}
			if err := writeTarStream(ctx, tw, "files/"+rk, deps.Storage, p); err != nil {
				// Skip unreadable sources (moved/deleted) but keep the DB row —
				// the item still carries its tags/embeddings.
				continue
			}
		}
	}
}

// selectTaggedPaths returns the distinct media paths that carry at least one
// tag, restricted to rows that still exist in the media table.
func selectTaggedPaths(db *sql.DB) ([]string, error) {
	rows, err := db.Query(`
		SELECT DISTINCT m.path
		FROM media m
		JOIN media_tag_by_category t ON t.media_path = m.path`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []string
	for rows.Next() {
		var p string
		if err := rows.Scan(&p); err != nil {
			return nil, err
		}
		out = append(out, p)
	}
	return out, rows.Err()
}

// buildExportDB creates a fresh SQLite at dbPath containing only the selected
// media and their referencing rows, with every media path rewritten to a
// flat, root-agnostic archive key. Returns the path->key map for file staging.
func buildExportDB(deps *Dependencies, dbPath string, selected []string) (map[string]string, error) {
	src := deps.DB

	dst, err := sql.Open("sqlite", dbPath)
	if err != nil {
		return nil, err
	}
	defer dst.Close()
	if err := media.InitializeSchema(dst); err != nil {
		return nil, err
	}

	relByPath := buildExportKeys(selected)

	tx, err := dst.Begin()
	if err != nil {
		return nil, err
	}
	defer tx.Rollback()

	// Taxonomy (small; copy whole so tag/category refs resolve). Tag preview
	// thumbnails are nulled — regenerated on the destination.
	if err := copyRows(src, tx,
		`SELECT label, weight, description, tag_view_mode FROM category`,
		`INSERT OR IGNORE INTO category(label, weight, description, tag_view_mode) VALUES(?,?,?,?)`,
		4); err != nil {
		return nil, fmt.Errorf("category: %w", err)
	}
	if err := copyRows(src, tx,
		`SELECT label, category_label, weight FROM tag`,
		`INSERT OR IGNORE INTO tag(label, category_label, weight) VALUES(?,?,?)`,
		3); err != nil {
		return nil, fmt.Errorf("tag: %w", err)
	}

	// media (paths relativized, preview/thumbnails nulled)
	mrows, err := src.Query(`
		SELECT path, description, transcript, elo, views, wins, losses, size, hash, width, height
		FROM media WHERE path IN (`+placeholders(len(selected))+`)`, toArgs(selected)...)
	if err != nil {
		return nil, err
	}
	defer mrows.Close()
	for mrows.Next() {
		var p string
		var desc, transcript, hash sql.NullString
		var elo sql.NullFloat64
		var views, wins, losses, size, width, height sql.NullInt64
		if err := mrows.Scan(&p, &desc, &transcript, &elo, &views, &wins, &losses, &size, &hash, &width, &height); err != nil {
			return nil, err
		}
		if _, err := tx.Exec(`INSERT OR IGNORE INTO media
			(path, description, transcript, elo, views, wins, losses, size, hash, width, height)
			VALUES (?,?,?,?,?,?,?,?,?,?,?)`,
			relByPath[p], desc, transcript, elo, views, wins, losses, size, hash, width, height); err != nil {
			return nil, err
		}
	}
	mrows.Close()

	// Per-item referencing tables, media_path relativized.
	if err := copyRelKeyed(src, tx, selected, relByPath,
		`SELECT media_path, tag_label, category_label, weight, time_stamp, created_at FROM media_tag_by_category WHERE media_path IN (`,
		`INSERT OR IGNORE INTO media_tag_by_category(media_path, tag_label, category_label, weight, time_stamp, created_at) VALUES(?,?,?,?,?,?)`,
		6); err != nil {
		return nil, fmt.Errorf("mtc: %w", err)
	}
	if err := copyRelKeyed(src, tx, selected, relByPath,
		`SELECT media_path, model, dim, vector, created_at FROM media_embedding WHERE media_path IN (`,
		`INSERT OR IGNORE INTO media_embedding(media_path, model, dim, vector, created_at) VALUES(?,?,?,?,?)`,
		5); err != nil {
		return nil, fmt.Errorf("embedding: %w", err)
	}
	if err := copyRelKeyed(src, tx, selected, relByPath,
		`SELECT media_path, model, face_count, scanned_at FROM face_scan WHERE media_path IN (`,
		`INSERT OR IGNORE INTO face_scan(media_path, model, face_count, scanned_at) VALUES(?,?,?,?)`,
		4); err != nil {
		// face_scan.scanned_at column name may differ across versions; non-fatal.
		_ = err
	}

	// Faces: preserve original id so face_veto/face_cannot_link/person refs in
	// the export stay valid; import remaps ids on the destination.
	frows, err := src.Query(`
		SELECT id, media_path, model, frame_ts, bbox_x, bbox_y, bbox_w, bbox_h, det_score, vector, person_id, assigned_by, created_at
		FROM face WHERE media_path IN (`+placeholders(len(selected))+`)`, toArgs(selected)...)
	if err != nil {
		return nil, err
	}
	faceIDs := map[int64]bool{}
	personIDs := map[int64]bool{}
	for frows.Next() {
		var id int64
		var mp, model string
		var frameTs, bx, by, bw, bh, det float64
		var vec []byte
		var personID sql.NullInt64
		var assignedBy sql.NullString
		var createdAt sql.NullInt64
		if err := frows.Scan(&id, &mp, &model, &frameTs, &bx, &by, &bw, &bh, &det, &vec, &personID, &assignedBy, &createdAt); err != nil {
			frows.Close()
			return nil, err
		}
		if _, err := tx.Exec(`INSERT OR IGNORE INTO face
			(id, media_path, model, frame_ts, bbox_x, bbox_y, bbox_w, bbox_h, det_score, vector, person_id, assigned_by, created_at)
			VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)`,
			id, relByPath[mp], model, frameTs, bx, by, bw, bh, det, vec, personID, assignedBy, createdAt); err != nil {
			frows.Close()
			return nil, err
		}
		faceIDs[id] = true
		if personID.Valid {
			personIDs[personID.Int64] = true
		}
	}
	frows.Close()

	// People referenced by exported faces.
	if len(personIDs) > 0 {
		prows, err := src.Query(`SELECT id, name, cover_face_id, created_at FROM person`)
		if err != nil {
			return nil, err
		}
		for prows.Next() {
			var id int64
			var name sql.NullString
			var cover, createdAt sql.NullInt64
			if err := prows.Scan(&id, &name, &cover, &createdAt); err != nil {
				prows.Close()
				return nil, err
			}
			if !personIDs[id] {
				continue
			}
			if _, err := tx.Exec(`INSERT OR IGNORE INTO person(id, name, cover_face_id, created_at) VALUES(?,?,?,?)`,
				id, name, cover, createdAt); err != nil {
				prows.Close()
				return nil, err
			}
		}
		prows.Close()
	}

	// Curation assertions for exported faces.
	if len(faceIDs) > 0 {
		copyFaceRefs(src, tx, `SELECT face_id, person_id FROM face_veto`,
			`INSERT OR IGNORE INTO face_veto(face_id, person_id) VALUES(?,?)`, faceIDs)
		copyFaceRefs(src, tx, `SELECT face_a, face_b FROM face_cannot_link`,
			`INSERT OR IGNORE INTO face_cannot_link(face_a, face_b) VALUES(?,?)`, faceIDs)
	}

	return relByPath, tx.Commit()
}

// copyFaceRefs copies rows of a two-int face-reference table, keeping only
// rows whose first column is an exported face id.
func copyFaceRefs(src *sql.DB, tx *sql.Tx, query, insert string, faceIDs map[int64]bool) {
	rows, err := src.Query(query)
	if err != nil {
		return
	}
	defer rows.Close()
	for rows.Next() {
		var a, b int64
		if err := rows.Scan(&a, &b); err != nil {
			return
		}
		if !faceIDs[a] {
			continue
		}
		_, _ = tx.Exec(insert, a, b)
	}
}

// copyRows copies whole rows verbatim (no path rewriting).
func copyRows(src *sql.DB, tx *sql.Tx, query, insert string, ncol int) error {
	rows, err := src.Query(query)
	if err != nil {
		return err
	}
	defer rows.Close()
	for rows.Next() {
		vals := make([]any, ncol)
		ptrs := make([]any, ncol)
		for i := range vals {
			ptrs[i] = &vals[i]
		}
		if err := rows.Scan(ptrs...); err != nil {
			return err
		}
		if _, err := tx.Exec(insert, vals...); err != nil {
			return err
		}
	}
	return rows.Err()
}

// copyRelKeyed copies rows whose FIRST column is a media_path, rewriting it to
// its relative key. `query` must end with an open "(" for the IN placeholders.
func copyRelKeyed(src *sql.DB, tx *sql.Tx, selected []string, relByPath map[string]string, query, insert string, ncol int) error {
	rows, err := src.Query(query+placeholders(len(selected))+`)`, toArgs(selected)...)
	if err != nil {
		return err
	}
	defer rows.Close()
	for rows.Next() {
		vals := make([]any, ncol)
		ptrs := make([]any, ncol)
		for i := range vals {
			ptrs[i] = &vals[i]
		}
		if err := rows.Scan(ptrs...); err != nil {
			return err
		}
		if mp, ok := vals[0].(string); ok {
			vals[0] = relByPath[mp]
		} else if bs, ok := vals[0].([]byte); ok {
			vals[0] = relByPath[string(bs)]
		}
		if _, err := tx.Exec(insert, vals...); err != nil {
			return err
		}
	}
	return rows.Err()
}

// ---- tar helpers ----

func writeTarBytes(tw *tar.Writer, name string, data []byte) error {
	if err := tw.WriteHeader(&tar.Header{Name: name, Mode: 0o644, Size: int64(len(data))}); err != nil {
		return err
	}
	_, err := tw.Write(data)
	return err
}

func writeTarFile(tw *tar.Writer, name, srcPath string) error {
	fi, err := os.Stat(srcPath)
	if err != nil {
		return err
	}
	f, err := os.Open(srcPath)
	if err != nil {
		return err
	}
	defer f.Close()
	if err := tw.WriteHeader(&tar.Header{Name: name, Mode: 0o644, Size: fi.Size()}); err != nil {
		return err
	}
	_, err = io.Copy(tw, f)
	return err
}

func writeTarStream(ctx context.Context, tw *tar.Writer, name string, reg *storage.Registry, srcPath string) error {
	rc, err := openMediaSource(ctx, reg, srcPath)
	if err != nil {
		return err
	}
	defer rc.Close()
	// tar needs the size up front; buffer to a temp file to learn it without
	// holding the whole object in memory.
	tmp, err := os.CreateTemp("", "loki-xfer-*")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())
	n, err := io.Copy(tmp, rc)
	if err != nil {
		tmp.Close()
		return err
	}
	if _, err := tmp.Seek(0, io.SeekStart); err != nil {
		tmp.Close()
		return err
	}
	if err := tw.WriteHeader(&tar.Header{Name: name, Mode: 0o644, Size: n}); err != nil {
		tmp.Close()
		return err
	}
	_, err = io.Copy(tw, tmp)
	tmp.Close()
	return err
}

func placeholders(n int) string {
	if n <= 0 {
		return ""
	}
	return strings.TrimSuffix(strings.Repeat("?,", n), ",")
}

func toArgs(ss []string) []any {
	out := make([]any, len(ss))
	for i, s := range ss {
		out[i] = s
	}
	return out
}
