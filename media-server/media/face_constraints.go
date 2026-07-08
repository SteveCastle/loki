package media

import (
	"database/sql"
	"fmt"
	"time"
)

// Human curation assertions for face clustering.
//
// A veto is the direct negative assertion "face F is never person P". A
// cannot-link is its person-independent shadow: "faces A and B are never the
// same person". Rejecting a face records BOTH — the veto against the person it
// was removed from, and cannot-links against that person's exemplar faces —
// because they die at different times. The veto is cheap to check but points
// at a person row that a reset may dissolve (anonymous "Unknown #N" clusters
// are deleted and re-minted under new ids); the cannot-links are keyed by face
// ids, which persist across every recluster, so the rejection still binds when
// the same visual cluster re-forms as a brand-new person.
//
// Assertions are only removed by a contradicting USER action: assigning the
// face to the vetoed person, or merging the groups.

// rejectExemplarLimit caps how many of a person's faces a rejection records
// cannot-links against. User-assigned (ground truth) faces are taken first,
// then the clearest detections — the faces most likely to seed the cluster's
// next incarnation.
const rejectExemplarLimit = 8

// AddFaceVeto records "face is never this person". Idempotent.
func AddFaceVeto(db *sql.DB, faceID, personID int64) error {
	_, err := db.Exec(
		`INSERT OR IGNORE INTO face_veto (face_id, person_id, created_at) VALUES (?, ?, ?)`,
		faceID, personID, time.Now().Unix(),
	)
	return err
}

// FaceVetoes loads every veto whose face belongs to model, keyed
// face → set of forbidden persons. Loaded once per clustering pass.
func FaceVetoes(db *sql.DB, model string) (map[int64]map[int64]bool, error) {
	rows, err := db.Query(
		`SELECT v.face_id, v.person_id FROM face_veto v
		 JOIN face f ON f.id = v.face_id WHERE f.model = ?`, model,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := map[int64]map[int64]bool{}
	for rows.Next() {
		var faceID, personID int64
		if err := rows.Scan(&faceID, &personID); err != nil {
			return nil, err
		}
		if out[faceID] == nil {
			out[faceID] = map[int64]bool{}
		}
		out[faceID][personID] = true
	}
	return out, rows.Err()
}

// FaceCannotLinks loads every cannot-link pair whose faces belong to model,
// keyed symmetrically (both directions present). Loaded once per clustering
// pass.
func FaceCannotLinks(db *sql.DB, model string) (map[int64]map[int64]bool, error) {
	rows, err := db.Query(
		`SELECT l.face_a, l.face_b FROM face_cannot_link l
		 JOIN face fa ON fa.id = l.face_a WHERE fa.model = ?`, model,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := map[int64]map[int64]bool{}
	add := func(a, b int64) {
		if out[a] == nil {
			out[a] = map[int64]bool{}
		}
		out[a][b] = true
	}
	for rows.Next() {
		var a, b int64
		if err := rows.Scan(&a, &b); err != nil {
			return nil, err
		}
		add(a, b)
		add(b, a)
	}
	return out, rows.Err()
}

// addCannotLinkTx records one normalized (a<b) cannot-link pair.
func addCannotLinkTx(tx *sql.Tx, a, b int64) error {
	if a == b {
		return nil
	}
	if a > b {
		a, b = b, a
	}
	_, err := tx.Exec(
		`INSERT OR IGNORE INTO face_cannot_link (face_a, face_b, created_at) VALUES (?, ?, ?)`,
		a, b, time.Now().Unix(),
	)
	return err
}

// RejectFaceFromPerson is the "this is not them" action: it records a veto
// (face never joins personID again), records cannot-links against up to
// rejectExemplarLimit of the person's exemplar faces (user faces first, then
// clearest detections) so the assertion survives the person being dissolved
// and re-minted, and finally unassigns the face if it currently sits in that
// person. Returns how many cannot-links were recorded.
func RejectFaceFromPerson(db *sql.DB, faceID, personID int64) (int, error) {
	f, ok, err := GetFaceByID(db, faceID)
	if err != nil {
		return 0, err
	}
	if !ok {
		return 0, fmt.Errorf("no face with id %d", faceID)
	}
	if _, ok, err := GetPersonByID(db, personID); err != nil {
		return 0, err
	} else if !ok {
		return 0, fmt.Errorf("no person with id %d", personID)
	}

	tx, err := db.Begin()
	if err != nil {
		return 0, err
	}
	defer tx.Rollback()
	if _, err := tx.Exec(
		`INSERT OR IGNORE INTO face_veto (face_id, person_id, created_at) VALUES (?, ?, ?)`,
		faceID, personID, time.Now().Unix(),
	); err != nil {
		return 0, err
	}
	// Exemplars: the person's ground-truth faces first, then by detection
	// clarity — the faces most likely to anchor this identity's next
	// incarnation after a reset.
	rows, err := tx.Query(
		`SELECT id FROM face WHERE person_id = ? AND id != ?
		 ORDER BY (assigned_by = 'user') DESC, det_score * bbox_w * bbox_h DESC, id ASC
		 LIMIT ?`,
		personID, faceID, rejectExemplarLimit,
	)
	if err != nil {
		return 0, err
	}
	var exemplars []int64
	for rows.Next() {
		var id int64
		if err := rows.Scan(&id); err != nil {
			rows.Close()
			return 0, err
		}
		exemplars = append(exemplars, id)
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return 0, err
	}
	for _, ex := range exemplars {
		if err := addCannotLinkTx(tx, faceID, ex); err != nil {
			return 0, err
		}
	}
	if err := tx.Commit(); err != nil {
		return 0, err
	}

	if f.PersonID == personID {
		if err := UnassignFace(db, faceID); err != nil {
			return len(exemplars), err
		}
	}
	return len(exemplars), nil
}

// CuratePersonFaces applies a whole-group review in one call: every current
// face of personID listed in keepIDs is locked (promoted to a user
// assignment); every other face is rejected — veto + cannot-links + unassign,
// exactly as RejectFaceFromPerson. Keeps are locked FIRST so the rejections
// record their cannot-links against the surviving, user-confirmed faces (the
// exemplar query prefers user faces). keepIDs not belonging to the person are
// ignored. Returns how many faces were kept and rejected.
func CuratePersonFaces(db *sql.DB, personID int64, keepIDs []int64) (kept, rejected int, err error) {
	if _, ok, err := GetPersonByID(db, personID); err != nil {
		return 0, 0, err
	} else if !ok {
		return 0, 0, fmt.Errorf("no person with id %d", personID)
	}
	faces, err := PersonFacesByQuality(db, personID)
	if err != nil {
		return 0, 0, err
	}
	keep := make(map[int64]bool, len(keepIDs))
	for _, id := range keepIDs {
		keep[id] = true
	}
	for _, f := range faces {
		if !keep[f.ID] {
			continue
		}
		if f.AssignedBy != "user" {
			if err := AssignFace(db, f.ID, personID, "user"); err != nil {
				return kept, rejected, err
			}
		}
		kept++
	}
	for _, f := range faces {
		if keep[f.ID] {
			continue
		}
		if _, err := RejectFaceFromPerson(db, f.ID, personID); err != nil {
			return kept, rejected, err
		}
		rejected++
	}
	return kept, rejected, nil
}

// LockPersonFaces promotes every auto-assigned face of a person to a user
// assignment — the "yes, this whole group is right" endorsement. Locked faces
// survive every reset (--reset-all included) and act as high-weight seeds in
// subsequent clustering. Returns how many faces were promoted.
func LockPersonFaces(db *sql.DB, personID int64) (int, error) {
	if _, ok, err := GetPersonByID(db, personID); err != nil {
		return 0, err
	} else if !ok {
		return 0, fmt.Errorf("no person with id %d", personID)
	}
	res, err := db.Exec(
		`UPDATE face SET assigned_by = 'user' WHERE person_id = ? AND assigned_by = 'auto'`,
		personID,
	)
	if err != nil {
		return 0, err
	}
	n, err := res.RowsAffected()
	return int(n), err
}

// clearConstraintsForFacesTx drops vetoes and cannot-links whose face rows are
// being deleted, selected by the given face-id subquery (with args). Called
// inside the transaction that deletes the rows, BEFORE the delete.
func clearConstraintsForFacesTx(tx *sql.Tx, faceIDSubquery string, args ...any) error {
	if _, err := tx.Exec(
		`DELETE FROM face_veto WHERE face_id IN (`+faceIDSubquery+`)`, args...,
	); err != nil {
		return err
	}
	_, err := tx.Exec(
		`DELETE FROM face_cannot_link WHERE face_a IN (`+faceIDSubquery+`) OR face_b IN (`+faceIDSubquery+`)`,
		append(append([]any{}, args...), args...)...,
	)
	return err
}

// clearContradictedConstraintsTx removes assertions a user action has just
// contradicted: the face's veto against personID, and cannot-links between the
// face and any face currently assigned to personID. The newest user statement
// wins.
func clearContradictedConstraintsTx(tx *sql.Tx, faceID, personID int64) error {
	if _, err := tx.Exec(
		`DELETE FROM face_veto WHERE face_id = ? AND person_id = ?`, faceID, personID,
	); err != nil {
		return err
	}
	_, err := tx.Exec(
		`DELETE FROM face_cannot_link
		 WHERE (face_a = ? AND face_b IN (SELECT id FROM face WHERE person_id = ?))
		    OR (face_b = ? AND face_a IN (SELECT id FROM face WHERE person_id = ?))`,
		faceID, personID, faceID, personID,
	)
	return err
}

// FaceVetoExists reports whether face → person is vetoed.
func FaceVetoExists(db *sql.DB, faceID, personID int64) (bool, error) {
	var one int
	err := db.QueryRow(
		`SELECT 1 FROM face_veto WHERE face_id = ? AND person_id = ?`, faceID, personID,
	).Scan(&one)
	if err == sql.ErrNoRows {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	return true, nil
}
