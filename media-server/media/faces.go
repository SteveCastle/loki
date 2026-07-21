package media

import (
	"database/sql"
	"fmt"

	"github.com/stevecastle/shrike/embedvec"
)

// Face is one stored face row. Bbox coordinates are relative ([0,1] of the
// image dimensions); Vec is the L2-normalized identity embedding. PersonID is
// 0 while unassigned; AssignedBy is "auto" (clustering) or "user" (manual
// label — ground truth that reclustering must never overwrite).
type Face struct {
	ID         int64
	MediaPath  string
	Model      string
	FrameTS    float64
	X, Y, W, H float64
	Score      float64
	Vec        []float32
	PersonID   int64
	AssignedBy string
	CreatedAt  int64
}

// NewFace is one detected face to persist (ID assigned by the DB).
type NewFace struct {
	FrameTS    float64
	X, Y, W, H float64
	Score      float64
	Vec        []float32
}

// ReplaceFaces atomically replaces all stored faces for path with faces
// under model and records the scan in face_scan. The new scan is
// authoritative for the WHOLE item, not just its own recognizer: rows and
// scan markers other models left on the path are cleared too. Domain routing
// sends an item to exactly one recognizer, so another model's rows are a
// superseded routing decision — leaving them behind strands ghost faces that
// no clustering pass will ever assign but that still count (and show) as
// "ungrouped" grouping workload. Replacing (not appending) makes a rescan
// idempotent; the scan marker distinguishes "scanned, no faces" from "never
// scanned". Person assignments on the old rows are dropped by design — a
// rescan means the old detections are stale. Returns the inserted row IDs
// (parallel to faces) so callers can update the in-memory face index.
func ReplaceFaces(db *sql.DB, path, model string, faces []NewFace, scannedAt int64) ([]int64, error) {
	tx, err := db.Begin()
	if err != nil {
		return nil, err
	}
	defer tx.Rollback()

	// The old rows are about to disappear — clear any person cover that points
	// at them, or the cover would dangle at a deleted face id (blank card in
	// the People UI). GetPeople falls back to the person's best face, so the
	// cover self-heals on the next listing.
	if _, err := tx.Exec(
		`UPDATE person SET cover_face_id = NULL
		 WHERE cover_face_id IN (SELECT id FROM face WHERE media_path=?)`,
		path,
	); err != nil {
		return nil, fmt.Errorf("clear stale covers: %w", err)
	}
	// The old rows' curation assertions (vetoes, cannot-links) go with them —
	// a rescan means the old detections (and anything asserted about them) are
	// stale, matching how person assignments are dropped here.
	if err := clearConstraintsForFacesTx(
		tx, `SELECT id FROM face WHERE media_path=?`, path,
	); err != nil {
		return nil, fmt.Errorf("clear stale face constraints: %w", err)
	}
	if _, err := tx.Exec(`DELETE FROM face WHERE media_path=?`, path); err != nil {
		return nil, fmt.Errorf("clear stale faces: %w", err)
	}
	// Other models' scan markers are superseded along with their rows; ours is
	// upserted below.
	if _, err := tx.Exec(`DELETE FROM face_scan WHERE media_path=? AND model<>?`, path, model); err != nil {
		return nil, fmt.Errorf("clear stale scan markers: %w", err)
	}
	ids := make([]int64, 0, len(faces))
	for _, f := range faces {
		res, err := tx.Exec(
			`INSERT INTO face (media_path, model, frame_ts, bbox_x, bbox_y, bbox_w, bbox_h, det_score, vector, created_at)
			 VALUES (?,?,?,?,?,?,?,?,?,?)`,
			path, model, f.FrameTS, f.X, f.Y, f.W, f.H, f.Score, embedvec.Encode(f.Vec), scannedAt,
		)
		if err != nil {
			return nil, fmt.Errorf("insert face: %w", err)
		}
		id, err := res.LastInsertId()
		if err != nil {
			return nil, fmt.Errorf("read face id: %w", err)
		}
		ids = append(ids, id)
	}
	if _, err := tx.Exec(
		`INSERT INTO face_scan (media_path, model, face_count, scanned_at)
		 VALUES (?,?,?,?)
		 ON CONFLICT(media_path, model)
		 DO UPDATE SET face_count=excluded.face_count, scanned_at=excluded.scanned_at`,
		path, model, len(faces), scannedAt,
	); err != nil {
		return nil, fmt.Errorf("mark face scan: %w", err)
	}
	if err := tx.Commit(); err != nil {
		return nil, err
	}
	return ids, nil
}

// CleanupSupersededFaces removes face rows (and their covers, curation
// assertions, and scan markers) whose scan was superseded by a NEWER scan of
// the same media under a different model. ReplaceFaces now enforces this
// invariant on every scan (one item, one recognizer), but rows stored before
// it did — e.g. an item scanned with the photo model, then re-routed to the
// anime model — linger as permanently "ungrouped" ghosts: no clustering pass
// touches them, yet they inflate the ungrouped count and top the manual
// review list. Runs at schema init; returns how many face rows it removed.
// Ties (identical scanned_at across models) are left alone — there is no
// winner to pick.
func CleanupSupersededFaces(db *sql.DB) (int, error) {
	const supersededFaceIDs = `
		SELECT f.id FROM face f
		JOIN face_scan fs ON fs.media_path = f.media_path AND fs.model = f.model
		WHERE COALESCE(fs.scanned_at, 0) < (
			SELECT MAX(COALESCE(s2.scanned_at, 0))
			FROM face_scan s2 WHERE s2.media_path = f.media_path)`
	tx, err := db.Begin()
	if err != nil {
		return 0, err
	}
	defer tx.Rollback()

	if _, err := tx.Exec(
		`UPDATE person SET cover_face_id = NULL WHERE cover_face_id IN (` + supersededFaceIDs + `)`,
	); err != nil {
		return 0, fmt.Errorf("clear superseded covers: %w", err)
	}
	if err := clearConstraintsForFacesTx(tx, supersededFaceIDs); err != nil {
		return 0, fmt.Errorf("clear superseded face constraints: %w", err)
	}
	res, err := tx.Exec(`DELETE FROM face WHERE id IN (` + supersededFaceIDs + `)`)
	if err != nil {
		return 0, fmt.Errorf("delete superseded faces: %w", err)
	}
	removed, _ := res.RowsAffected()
	if _, err := tx.Exec(
		`DELETE FROM face_scan
		 WHERE COALESCE(scanned_at, 0) < (
			SELECT MAX(COALESCE(s2.scanned_at, 0))
			FROM face_scan s2 WHERE s2.media_path = face_scan.media_path)`,
	); err != nil {
		return 0, fmt.Errorf("delete superseded scan markers: %w", err)
	}
	if err := tx.Commit(); err != nil {
		return 0, err
	}
	return int(removed), nil
}

// FaceScansForPaths reports which of the given paths already have a scan
// marker under ANY of the given models — the batch form of HasFaceScan, used
// by the routing scan to skip already-processed media without paying the
// per-item classification cost. Batched IN queries stay under SQLite's
// bind-variable cap.
func FaceScansForPaths(db *sql.DB, models []string, paths []string) (map[string]bool, error) {
	out := make(map[string]bool)
	if len(models) == 0 || len(paths) == 0 {
		return out, nil
	}
	const batch = 500
	for lo := 0; lo < len(paths); lo += batch {
		hi := lo + batch
		if hi > len(paths) {
			hi = len(paths)
		}
		chunk := paths[lo:hi]
		args := make([]any, 0, len(models)+len(chunk))
		mph := make([]byte, 0, len(models)*2)
		for i, m := range models {
			if i > 0 {
				mph = append(mph, ',')
			}
			mph = append(mph, '?')
			args = append(args, m)
		}
		pph := make([]byte, 0, len(chunk)*2)
		for i, p := range chunk {
			if i > 0 {
				pph = append(pph, ',')
			}
			pph = append(pph, '?')
			args = append(args, p)
		}
		rows, err := db.Query(
			`SELECT DISTINCT media_path FROM face_scan WHERE model IN (`+string(mph)+`) AND media_path IN (`+string(pph)+`)`,
			args...,
		)
		if err != nil {
			return nil, err
		}
		for rows.Next() {
			var p string
			if err := rows.Scan(&p); err != nil {
				rows.Close()
				return nil, err
			}
			out[p] = true
		}
		if err := rows.Err(); err != nil {
			rows.Close()
			return nil, err
		}
		rows.Close()
	}
	return out, nil
}

// HasFaceScan reports whether path was already scanned under model.
func HasFaceScan(db *sql.DB, path, model string) (bool, error) {
	var one int
	err := db.QueryRow(
		`SELECT 1 FROM face_scan WHERE media_path=? AND model=? LIMIT 1`,
		path, model,
	).Scan(&one)
	if err == sql.ErrNoRows {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	return true, nil
}

// FaceScanInfo reports whether path was scanned under model and when
// (unix seconds; 0 for legacy rows without a timestamp). The timestamp is
// what lets clients see an overwrite rescan complete — the scanned flag
// stays true across it, but scanned_at moves forward.
func FaceScanInfo(db *sql.DB, path, model string) (bool, int64, error) {
	var at int64
	err := db.QueryRow(
		`SELECT COALESCE(scanned_at, 0) FROM face_scan WHERE media_path=? AND model=? LIMIT 1`,
		path, model,
	).Scan(&at)
	if err == sql.ErrNoRows {
		return false, 0, nil
	}
	if err != nil {
		return false, 0, err
	}
	return true, at, nil
}

// GetFaces returns all stored faces for (path, model), detection-score
// descending.
func GetFaces(db *sql.DB, path, model string) ([]Face, error) {
	rows, err := db.Query(
		`SELECT id, media_path, model, frame_ts, bbox_x, bbox_y, bbox_w, bbox_h,
		        det_score, vector, COALESCE(person_id, 0), COALESCE(assigned_by, ''), COALESCE(created_at, 0)
		 FROM face WHERE media_path=? AND model=? ORDER BY det_score DESC`,
		path, model,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	return scanFaceRows(rows)
}

// LoadAllFaces returns every stored face for model (vector decoded). Used by
// the face index builder and clustering.
func LoadAllFaces(db *sql.DB, model string) ([]Face, error) {
	rows, err := db.Query(
		`SELECT id, media_path, model, frame_ts, bbox_x, bbox_y, bbox_w, bbox_h,
		        det_score, vector, COALESCE(person_id, 0), COALESCE(assigned_by, ''), COALESCE(created_at, 0)
		 FROM face WHERE model=?`,
		model,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	return scanFaceRows(rows)
}

// GetFaceByID returns one face row (vector decoded).
func GetFaceByID(db *sql.DB, id int64) (Face, bool, error) {
	rows, err := db.Query(
		`SELECT id, media_path, model, frame_ts, bbox_x, bbox_y, bbox_w, bbox_h,
		        det_score, vector, COALESCE(person_id, 0), COALESCE(assigned_by, ''), COALESCE(created_at, 0)
		 FROM face WHERE id=?`,
		id,
	)
	if err != nil {
		return Face{}, false, err
	}
	defer rows.Close()
	faces, err := scanFaceRows(rows)
	if err != nil || len(faces) == 0 {
		return Face{}, false, err
	}
	return faces[0], true, nil
}

// DeleteFacesForMedia removes all face rows and scan markers for a media path
// (all models). Called when media is deleted from the library.
func DeleteFacesForMedia(db *sql.DB, path string) error {
	if _, err := db.Exec(
		`UPDATE person SET cover_face_id = NULL
		 WHERE cover_face_id IN (SELECT id FROM face WHERE media_path=?)`, path,
	); err != nil {
		return err
	}
	if _, err := db.Exec(
		`DELETE FROM face_veto WHERE face_id IN (SELECT id FROM face WHERE media_path=?)`, path,
	); err != nil {
		return err
	}
	if _, err := db.Exec(
		`DELETE FROM face_cannot_link
		 WHERE face_a IN (SELECT id FROM face WHERE media_path=?)
		    OR face_b IN (SELECT id FROM face WHERE media_path=?)`, path, path,
	); err != nil {
		return err
	}
	if _, err := db.Exec(
		`DELETE FROM face_group_ban_member WHERE face_id IN (SELECT id FROM face WHERE media_path=?)`, path,
	); err != nil {
		return err
	}
	if _, err := db.Exec(`DELETE FROM face WHERE media_path=?`, path); err != nil {
		return err
	}
	_, err := db.Exec(`DELETE FROM face_scan WHERE media_path=?`, path)
	return err
}

// CountFaceScans returns how many media items have been scanned under model.
func CountFaceScans(db *sql.DB, model string) (int, error) {
	var n int
	err := db.QueryRow(`SELECT COUNT(*) FROM face_scan WHERE model=?`, model).Scan(&n)
	return n, err
}

func scanFaceRows(rows *sql.Rows) ([]Face, error) {
	var out []Face
	for rows.Next() {
		var f Face
		var blob []byte
		if err := rows.Scan(
			&f.ID, &f.MediaPath, &f.Model, &f.FrameTS,
			&f.X, &f.Y, &f.W, &f.H,
			&f.Score, &blob, &f.PersonID, &f.AssignedBy, &f.CreatedAt,
		); err != nil {
			return nil, err
		}
		vec, err := embedvec.Decode(blob)
		if err != nil {
			return nil, err
		}
		f.Vec = vec
		out = append(out, f)
	}
	return out, rows.Err()
}
