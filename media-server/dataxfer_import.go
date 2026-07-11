package main

// dataxfer_import.go — the import half of the export/import feature. Accepts
// a .lokiexport archive, rebases every storage-root-relative key onto the
// destination's root (local <-> s3), copies the file bytes, and merges by
// media item: an item already present in the destination is skipped whole;
// a new item brings in its media row, tags, embeddings, faces, and file.
// Face/person AUTOINCREMENT ids are remapped so curation refs stay valid.
// Thumbnails are left null and regenerate on first view.

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
	"path/filepath"
	"strings"

	"github.com/stevecastle/shrike/media"
	"github.com/stevecastle/shrike/storage"
	"github.com/stevecastle/shrike/tasks"
)

type importResult struct {
	Imported int      `json:"imported"`
	Skipped  int      `json:"skipped"`
	Files    int      `json:"files"`
	DestRoot string   `json:"dest_root"`
	Warnings []string `json:"warnings,omitempty"`
}

// importHandler receives a .lokiexport (multipart "file"), extracts it, and
// merges it into the destination library rooted at the default (or specified)
// storage root.
func importHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "use POST", http.StatusMethodNotAllowed)
			return
		}
		r.Body = http.MaxBytesReader(w, r.Body, 50<<30) // 50GB ceiling
		if err := r.ParseMultipartForm(32 << 20); err != nil {
			http.Error(w, "parse upload: "+err.Error(), http.StatusBadRequest)
			return
		}
		file, _, err := r.FormFile("file")
		if err != nil {
			http.Error(w, "missing 'file'", http.StatusBadRequest)
			return
		}
		defer file.Close()

		// Destination backend: an explicit root label, else the default.
		var dest storage.Backend
		if label := strings.TrimSpace(r.FormValue("destRoot")); label != "" {
			for _, b := range deps.Storage.AllBackends() {
				if b.Root().Path == label || b.Root().Name == label {
					dest = b
					break
				}
			}
		}
		if dest == nil {
			dest = deps.Storage.DefaultBackend()
		}
		if dest == nil {
			http.Error(w, "no destination storage root configured", http.StatusBadRequest)
			return
		}
		destRoot := dest.Root().Path

		// Extract to a temp dir on the data volume.
		workDir, err := os.MkdirTemp("", "loki-import-*")
		if err != nil {
			http.Error(w, "temp: "+err.Error(), http.StatusInternalServerError)
			return
		}
		defer os.RemoveAll(workDir)
		if err := extractArchive(file, workDir); err != nil {
			http.Error(w, "extract: "+err.Error(), http.StatusBadRequest)
			return
		}

		res, err := mergeImport(r.Context(), deps, workDir, dest, destRoot)
		if err != nil {
			http.Error(w, "import: "+err.Error(), http.StatusInternalServerError)
			return
		}

		// Rebuild the in-memory search indexes so imported items are
		// immediately searchable.
		if _, _, err := tasks.RebuildActiveIndex(deps.DB, nil); err != nil {
			res.Warnings = append(res.Warnings, "embedding index rebuild: "+err.Error())
		}
		if _, _, err := tasks.RebuildActiveFaceIndex(deps.DB, nil); err != nil {
			res.Warnings = append(res.Warnings, "face index rebuild: "+err.Error())
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(res)
	}
}

// extractArchive unpacks a tar.gz into dir, guarding against path traversal.
func extractArchive(r io.Reader, dir string) error {
	gz, err := gzip.NewReader(r)
	if err != nil {
		return err
	}
	defer gz.Close()
	tr := tar.NewReader(gz)
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		// Reject traversal / absolute members.
		clean := filepath.Clean(hdr.Name)
		if strings.HasPrefix(clean, "..") || filepath.IsAbs(clean) ||
			strings.Contains(clean, ".."+string(filepath.Separator)) {
			return fmt.Errorf("unsafe archive member: %s", hdr.Name)
		}
		target := filepath.Join(dir, clean)
		if !strings.HasPrefix(target, filepath.Clean(dir)+string(filepath.Separator)) {
			return fmt.Errorf("archive member escapes dir: %s", hdr.Name)
		}
		if hdr.FileInfo().IsDir() {
			os.MkdirAll(target, 0o755)
			continue
		}
		os.MkdirAll(filepath.Dir(target), 0o755)
		out, err := os.Create(target)
		if err != nil {
			return err
		}
		if _, err := io.Copy(out, tr); err != nil {
			out.Close()
			return err
		}
		out.Close()
	}
}

// mergeImport reads the extracted library.db and merges it into deps.DB,
// rebasing relative keys onto destRoot and uploading file bytes.
func mergeImport(ctx context.Context, deps *Dependencies, workDir string, dest storage.Backend, destRoot string) (*importResult, error) {
	res := &importResult{DestRoot: destRoot}

	// Validate manifest version.
	if mb, err := os.ReadFile(filepath.Join(workDir, "manifest.json")); err == nil {
		var m dataxferManifest
		if json.Unmarshal(mb, &m) == nil && m.Version > dataxferVersion {
			return nil, fmt.Errorf("archive version %d is newer than this server supports (%d)", m.Version, dataxferVersion)
		}
	}

	srcDBPath := filepath.Join(workDir, "library.db")
	if _, err := os.Stat(srcDBPath); err != nil {
		return nil, fmt.Errorf("archive missing library.db")
	}
	src, err := sql.Open("sqlite", srcDBPath)
	if err != nil {
		return nil, err
	}
	defer src.Close()

	dst := deps.DB
	if err := media.InitializeSchema(dst); err != nil {
		return nil, err
	}

	// --- taxonomy (merge; harmless if present) ---
	mergeSimple(src, dst,
		`SELECT label, weight, description, tag_view_mode FROM category`,
		`INSERT OR IGNORE INTO category(label, weight, description, tag_view_mode) VALUES(?,?,?,?)`, 4)
	mergeSimple(src, dst,
		`SELECT label, category_label, weight FROM tag`,
		`INSERT OR IGNORE INTO tag(label, category_label, weight) VALUES(?,?,?)`, 3)

	// --- media items (the unit of merge) ---
	newPaths := map[string]bool{}   // relkeys of items newly inserted this run
	absByRel := map[string]string{} // relkey -> rebased absolute dest path
	mrows, err := src.Query(`
		SELECT path, description, transcript, elo, views, wins, losses, size, hash, width, height FROM media`)
	if err != nil {
		return nil, err
	}
	for mrows.Next() {
		var rel string
		var desc, transcript, hash sql.NullString
		var elo sql.NullFloat64
		var views, wins, losses, size, width, height sql.NullInt64
		if err := mrows.Scan(&rel, &desc, &transcript, &elo, &views, &wins, &losses, &size, &hash, &width, &height); err != nil {
			mrows.Close()
			return nil, err
		}
		abs := rebaseKey(destRoot, rel)
		absByRel[rel] = abs
		// Skip items already present (merge = add missing).
		var exists int
		dst.QueryRow(`SELECT 1 FROM media WHERE path = ? LIMIT 1`, abs).Scan(&exists)
		if exists == 1 {
			res.Skipped++
			continue
		}
		if _, err := dst.Exec(`INSERT INTO media
			(path, description, transcript, elo, views, wins, losses, size, hash, width, height)
			VALUES (?,?,?,?,?,?,?,?,?,?,?)`,
			abs, desc, transcript, elo, views, wins, losses, size, hash, width, height); err != nil {
			mrows.Close()
			return nil, err
		}
		newPaths[rel] = true
		res.Imported++
	}
	mrows.Close()

	// --- copy file bytes for newly-imported items ---
	for rel := range newPaths {
		srcFile := filepath.Join(workDir, "files", filepath.FromSlash(rel))
		f, err := os.Open(srcFile)
		if err != nil {
			res.Warnings = append(res.Warnings, "missing file bytes for "+rel)
			continue
		}
		abs := absByRel[rel]
		if err := dest.Upload(ctx, abs, f, ""); err != nil {
			f.Close()
			res.Warnings = append(res.Warnings, "upload "+rel+": "+err.Error())
			continue
		}
		f.Close()
		res.Files++
	}

	// --- per-item referencing rows (only for newly-imported items) ---
	mergeRelKeyed(src, dst, newPaths, absByRel,
		`SELECT media_path, tag_label, category_label, weight, time_stamp, created_at FROM media_tag_by_category`,
		`INSERT OR IGNORE INTO media_tag_by_category(media_path, tag_label, category_label, weight, time_stamp, created_at) VALUES(?,?,?,?,?,?)`, 6)
	mergeRelKeyed(src, dst, newPaths, absByRel,
		`SELECT media_path, model, dim, vector, created_at FROM media_embedding`,
		`INSERT OR IGNORE INTO media_embedding(media_path, model, dim, vector, created_at) VALUES(?,?,?,?,?)`, 5)
	mergeRelKeyed(src, dst, newPaths, absByRel,
		`SELECT media_path, model, face_count, scanned_at FROM face_scan`,
		`INSERT OR IGNORE INTO face_scan(media_path, model, face_count, scanned_at) VALUES(?,?,?,?)`, 4)

	// --- people (upsert by name; remap ids) ---
	personMap := map[int64]int64{}
	prows, err := src.Query(`SELECT id, name, cover_face_id, created_at FROM person`)
	if err == nil {
		for prows.Next() {
			var oldID int64
			var name sql.NullString
			var cover, createdAt sql.NullInt64
			if err := prows.Scan(&oldID, &name, &cover, &createdAt); err != nil {
				break
			}
			var newID int64
			if name.Valid {
				if err := dst.QueryRow(`SELECT id FROM person WHERE name = ?`, name.String).Scan(&newID); err == nil {
					personMap[oldID] = newID
					continue
				}
			}
			r, err := dst.Exec(`INSERT INTO person(name, created_at) VALUES(?,?)`, name, createdAt)
			if err != nil {
				continue
			}
			newID, _ = r.LastInsertId()
			personMap[oldID] = newID
		}
		prows.Close()
	}

	// --- faces (only for newly-imported media; remap person_id, capture id map) ---
	faceMap := map[int64]int64{}
	frows, err := src.Query(`
		SELECT id, media_path, model, frame_ts, bbox_x, bbox_y, bbox_w, bbox_h, det_score, vector, person_id, assigned_by, created_at FROM face`)
	if err == nil {
		for frows.Next() {
			var oldID int64
			var rel, model string
			var frameTs, bx, by, bw, bh, det float64
			var vec []byte
			var personID sql.NullInt64
			var assignedBy sql.NullString
			var createdAt sql.NullInt64
			if err := frows.Scan(&oldID, &rel, &model, &frameTs, &bx, &by, &bw, &bh, &det, &vec, &personID, &assignedBy, &createdAt); err != nil {
				break
			}
			if !newPaths[rel] {
				continue // media pre-existed or wasn't imported
			}
			var newPerson sql.NullInt64
			if personID.Valid {
				if np, ok := personMap[personID.Int64]; ok {
					newPerson = sql.NullInt64{Int64: np, Valid: true}
				}
			}
			r, err := dst.Exec(`INSERT INTO face
				(media_path, model, frame_ts, bbox_x, bbox_y, bbox_w, bbox_h, det_score, vector, person_id, assigned_by, created_at)
				VALUES (?,?,?,?,?,?,?,?,?,?,?,?)`,
				absByRel[rel], model, frameTs, bx, by, bw, bh, det, vec, newPerson, assignedBy, createdAt)
			if err != nil {
				continue
			}
			newID, _ := r.LastInsertId()
			faceMap[oldID] = newID
		}
		frows.Close()
	}

	// --- person cover faces + curation assertions, remapped ---
	crows, err := src.Query(`SELECT id, cover_face_id FROM person WHERE cover_face_id IS NOT NULL`)
	if err == nil {
		for crows.Next() {
			var oldPerson int64
			var oldCover sql.NullInt64
			if err := crows.Scan(&oldPerson, &oldCover); err != nil {
				break
			}
			if !oldCover.Valid {
				continue
			}
			np, okP := personMap[oldPerson]
			nf, okF := faceMap[oldCover.Int64]
			if okP && okF {
				dst.Exec(`UPDATE person SET cover_face_id = ? WHERE id = ?`, nf, np)
			}
		}
		crows.Close()
	}
	remapFaceRefs(src, dst, faceMap, personMap,
		`SELECT face_id, person_id FROM face_veto`,
		`INSERT OR IGNORE INTO face_veto(face_id, person_id) VALUES(?,?)`, true)
	remapFaceRefs(src, dst, faceMap, personMap,
		`SELECT face_a, face_b FROM face_cannot_link`,
		`INSERT OR IGNORE INTO face_cannot_link(face_a, face_b) VALUES(?,?)`, false)

	return res, nil
}

func mergeSimple(src, dst *sql.DB, query, insert string, ncol int) {
	rows, err := src.Query(query)
	if err != nil {
		return
	}
	defer rows.Close()
	for rows.Next() {
		vals := make([]any, ncol)
		ptrs := make([]any, ncol)
		for i := range vals {
			ptrs[i] = &vals[i]
		}
		if err := rows.Scan(ptrs...); err != nil {
			return
		}
		dst.Exec(insert, vals...)
	}
}

// mergeRelKeyed inserts rows whose first column is a relative media key,
// rebased to its absolute dest path — but only for items newly imported this
// run (newPaths), so referencing data never attaches to a skipped item.
func mergeRelKeyed(src, dst *sql.DB, newPaths map[string]bool, absByRel map[string]string, query, insert string, ncol int) {
	rows, err := src.Query(query)
	if err != nil {
		return
	}
	defer rows.Close()
	for rows.Next() {
		vals := make([]any, ncol)
		ptrs := make([]any, ncol)
		for i := range vals {
			ptrs[i] = &vals[i]
		}
		if err := rows.Scan(ptrs...); err != nil {
			return
		}
		var rel string
		switch v := vals[0].(type) {
		case string:
			rel = v
		case []byte:
			rel = string(v)
		}
		if !newPaths[rel] {
			continue
		}
		vals[0] = absByRel[rel]
		dst.Exec(insert, vals...)
	}
}

// remapFaceRefs copies a two-id face-reference table, remapping ids old->new;
// rows referencing a face that wasn't imported are dropped. secondIsPerson
// selects whether the 2nd column is a person id (face_veto) or a face id
// (face_cannot_link).
func remapFaceRefs(src, dst *sql.DB, faceMap, personMap map[int64]int64, query, insert string, secondIsPerson bool) {
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
		na, ok := faceMap[a]
		if !ok {
			continue
		}
		var nb int64
		if secondIsPerson {
			if nb, ok = personMap[b]; !ok {
				continue
			}
		} else {
			if nb, ok = faceMap[b]; !ok {
				continue
			}
		}
		dst.Exec(insert, na, nb)
	}
}
