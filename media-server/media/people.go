package media

import (
	"database/sql"
	"fmt"
	"strconv"
	"strings"
	"time"
)

// PeopleCategory is the taxonomy category that mirrors the person table.
// Naming a person creates a tag with the person's name in this category and
// media_tag_by_category rows for every media item their faces appear in
// (time_stamp = the face's frame_ts) — so person filters work in every
// existing tag-search surface for free. The category is face-managed: person
// rename/merge/delete cascades into these tag rows, and DeleteAllFaceData
// wipes the whole category.
const PeopleCategory = "People"

// Person is one identity: a named (or auto-named "Unknown #N") cluster of
// faces. Every person has a name so it can live in the taxonomy.
type Person struct {
	ID          int64  `json:"id"`
	Name        string `json:"name"`
	CoverFaceID int64  `json:"coverFaceId,omitempty"`
	FaceCount   int    `json:"faceCount"`
	MediaCount  int    `json:"mediaCount"`
	CreatedAt   int64  `json:"createdAt,omitempty"`
}

// personNameConflict returns an error when name is already used by a tag in a
// category other than People (tag labels are globally unique — a person may
// not silently steal a tag from another category).
func personNameConflict(q interface {
	QueryRow(string, ...any) *sql.Row
}, name string) error {
	var cat sql.NullString
	err := q.QueryRow(`SELECT category_label FROM tag WHERE label = ?`, name).Scan(&cat)
	if err == sql.ErrNoRows {
		return nil
	}
	if err != nil {
		return err
	}
	if cat.Valid && cat.String != "" && cat.String != PeopleCategory {
		return fmt.Errorf("name %q is already a tag in category %q", name, cat.String)
	}
	return nil
}

// ensurePersonTag creates the People category and the person's tag row inside
// an existing transaction.
func ensurePersonTag(tx *sql.Tx, name string) error {
	if _, err := tx.Exec(`INSERT OR IGNORE INTO category (label, weight) VALUES (?, 0)`, PeopleCategory); err != nil {
		return err
	}
	_, err := tx.Exec(
		`INSERT INTO tag (label, category_label) VALUES (?, ?)
		 ON CONFLICT(label) DO UPDATE SET category_label = excluded.category_label
		 WHERE tag.category_label IS NULL OR tag.category_label = '' OR tag.category_label = ?`,
		name, PeopleCategory, PeopleCategory,
	)
	return err
}

// CreatePerson creates a named person (plus their taxonomy tag) and returns
// the new ID. The name must be non-empty, unique among persons, and not in
// use by a tag outside the People category.
func CreatePerson(db *sql.DB, name string) (int64, error) {
	name = strings.TrimSpace(name)
	if name == "" {
		return 0, fmt.Errorf("person name required")
	}
	if err := personNameConflict(db, name); err != nil {
		return 0, err
	}
	if _, exists, err := GetPersonByName(db, name); err != nil {
		return 0, err
	} else if exists {
		return 0, fmt.Errorf("person %q already exists", name)
	}
	tx, err := db.Begin()
	if err != nil {
		return 0, err
	}
	defer tx.Rollback()
	res, err := tx.Exec(`INSERT INTO person (name, created_at) VALUES (?, ?)`, name, time.Now().Unix())
	if err != nil {
		return 0, fmt.Errorf("create person %q: %w", name, err)
	}
	id, err := res.LastInsertId()
	if err != nil {
		return 0, err
	}
	if err := ensurePersonTag(tx, name); err != nil {
		return 0, err
	}
	return id, tx.Commit()
}

// NextUnknownName returns the next free auto-cluster name ("Unknown #N").
func NextUnknownName(db *sql.DB) (string, error) {
	rows, err := db.Query(`SELECT name FROM person WHERE name LIKE 'Unknown #%'`)
	if err != nil {
		return "", err
	}
	defer rows.Close()
	max := 0
	for rows.Next() {
		var name string
		if err := rows.Scan(&name); err != nil {
			return "", err
		}
		if n, err := strconv.Atoi(strings.TrimPrefix(name, "Unknown #")); err == nil && n > max {
			max = n
		}
	}
	if err := rows.Err(); err != nil {
		return "", err
	}
	return fmt.Sprintf("Unknown #%d", max+1), nil
}

// GetPeople lists all persons with their face and distinct-media counts,
// ordered by face count descending (biggest clusters first).
func GetPeople(db *sql.DB) ([]Person, error) {
	rows, err := db.Query(`
		SELECT p.id, COALESCE(p.name, ''), COALESCE(p.cover_face_id, 0), COALESCE(p.created_at, 0),
		       COUNT(f.id), COUNT(DISTINCT f.media_path)
		FROM person p
		LEFT JOIN face f ON f.person_id = p.id
		GROUP BY p.id
		ORDER BY COUNT(f.id) DESC, p.id ASC
	`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Person
	for rows.Next() {
		var p Person
		if err := rows.Scan(&p.ID, &p.Name, &p.CoverFaceID, &p.CreatedAt, &p.FaceCount, &p.MediaCount); err != nil {
			return nil, err
		}
		out = append(out, p)
	}
	return out, rows.Err()
}

// GetPersonByID returns one person row (without counts).
func GetPersonByID(db *sql.DB, id int64) (Person, bool, error) {
	var p Person
	err := db.QueryRow(
		`SELECT id, COALESCE(name, ''), COALESCE(cover_face_id, 0), COALESCE(created_at, 0) FROM person WHERE id = ?`, id,
	).Scan(&p.ID, &p.Name, &p.CoverFaceID, &p.CreatedAt)
	if err == sql.ErrNoRows {
		return Person{}, false, nil
	}
	if err != nil {
		return Person{}, false, err
	}
	return p, true, nil
}

// GetPersonByName returns one person row by exact name.
func GetPersonByName(db *sql.DB, name string) (Person, bool, error) {
	var p Person
	err := db.QueryRow(
		`SELECT id, COALESCE(name, ''), COALESCE(cover_face_id, 0), COALESCE(created_at, 0) FROM person WHERE name = ?`, name,
	).Scan(&p.ID, &p.Name, &p.CoverFaceID, &p.CreatedAt)
	if err == sql.ErrNoRows {
		return Person{}, false, nil
	}
	if err != nil {
		return Person{}, false, err
	}
	return p, true, nil
}

// PersonMediaPaths returns the distinct media paths a person's faces appear
// in, best detection first.
func PersonMediaPaths(db *sql.DB, id int64) ([]string, error) {
	rows, err := db.Query(
		`SELECT media_path FROM face WHERE person_id = ? GROUP BY media_path ORDER BY MAX(det_score) DESC`, id,
	)
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

// retagPerson rewrites the person's taxonomy rows from oldName to newName
// inside a transaction: tag row and media_tag_by_category rows. Collisions
// with already-existing target rows are resolved by dropping the old row.
func retagPerson(tx *sql.Tx, oldName, newName string) error {
	if err := ensurePersonTag(tx, newName); err != nil {
		return err
	}
	if _, err := tx.Exec(
		`UPDATE OR IGNORE media_tag_by_category SET tag_label = ? WHERE tag_label = ? AND category_label = ?`,
		newName, oldName, PeopleCategory,
	); err != nil {
		return err
	}
	// Rows that collided with existing (path, newName, ts) rows remain under
	// the old name — they're duplicates now; drop them with the old tag.
	if _, err := tx.Exec(
		`DELETE FROM media_tag_by_category WHERE tag_label = ? AND category_label = ?`, oldName, PeopleCategory,
	); err != nil {
		return err
	}
	if _, err := tx.Exec(`DELETE FROM tag WHERE label = ? AND category_label = ?`, oldName, PeopleCategory); err != nil {
		return err
	}
	return nil
}

// RenamePerson renames a person and cascades to their taxonomy rows.
func RenamePerson(db *sql.DB, id int64, newName string) error {
	newName = strings.TrimSpace(newName)
	if newName == "" {
		return fmt.Errorf("person name required")
	}
	p, ok, err := GetPersonByID(db, id)
	if err != nil {
		return err
	}
	if !ok {
		return fmt.Errorf("no person with id %d", id)
	}
	if p.Name == newName {
		return nil
	}
	if err := personNameConflict(db, newName); err != nil {
		return err
	}
	if other, exists, err := GetPersonByName(db, newName); err != nil {
		return err
	} else if exists && other.ID != id {
		return fmt.Errorf("person %q already exists — merge instead", newName)
	}

	tx, err := db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()
	if _, err := tx.Exec(`UPDATE person SET name = ? WHERE id = ?`, newName, id); err != nil {
		return fmt.Errorf("rename person: %w", err)
	}
	if err := retagPerson(tx, p.Name, newName); err != nil {
		return err
	}
	return tx.Commit()
}

// MergePersons moves every face of fromID onto intoID (keeping each face's
// assigned_by), rewrites taxonomy rows, and deletes the source person.
func MergePersons(db *sql.DB, fromID, intoID int64) error {
	if fromID == intoID {
		return fmt.Errorf("cannot merge a person into itself")
	}
	from, ok, err := GetPersonByID(db, fromID)
	if err != nil {
		return err
	}
	if !ok {
		return fmt.Errorf("no person with id %d", fromID)
	}
	into, ok, err := GetPersonByID(db, intoID)
	if err != nil {
		return err
	}
	if !ok {
		return fmt.Errorf("no person with id %d", intoID)
	}

	tx, err := db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()
	if _, err := tx.Exec(`UPDATE face SET person_id = ? WHERE person_id = ?`, intoID, fromID); err != nil {
		return fmt.Errorf("merge faces: %w", err)
	}
	if err := retagPerson(tx, from.Name, into.Name); err != nil {
		return err
	}
	// Keep the target's cover face; adopt the source's if the target had none.
	if _, err := tx.Exec(
		`UPDATE person SET cover_face_id = COALESCE(NULLIF(cover_face_id, 0), ?) WHERE id = ?`,
		nullableID(from.CoverFaceID), intoID,
	); err != nil {
		return err
	}
	if _, err := tx.Exec(`DELETE FROM person WHERE id = ?`, fromID); err != nil {
		return err
	}
	return tx.Commit()
}

// DeletePerson unassigns the person's faces, removes their taxonomy rows, and
// deletes the person. Faces (and their vectors) are kept — they become
// unassigned again.
func DeletePerson(db *sql.DB, id int64) error {
	p, ok, err := GetPersonByID(db, id)
	if err != nil {
		return err
	}
	if !ok {
		return fmt.Errorf("no person with id %d", id)
	}
	tx, err := db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()
	if _, err := tx.Exec(`UPDATE face SET person_id = NULL, assigned_by = NULL WHERE person_id = ?`, id); err != nil {
		return err
	}
	if _, err := tx.Exec(
		`DELETE FROM media_tag_by_category WHERE tag_label = ? AND category_label = ?`, p.Name, PeopleCategory,
	); err != nil {
		return err
	}
	if _, err := tx.Exec(`DELETE FROM tag WHERE label = ? AND category_label = ?`, p.Name, PeopleCategory); err != nil {
		return err
	}
	if _, err := tx.Exec(`DELETE FROM person WHERE id = ?`, id); err != nil {
		return err
	}
	return tx.Commit()
}

// AssignFace assigns a face to a person (assignedBy = "user" or "auto") and
// maintains the taxonomy bridge row for the face's media item. Reassigning a
// face moves it (removing the old person's bridge row when this was their
// last face on that media/frame). Auto-assignment never overwrites a user
// assignment.
func AssignFace(db *sql.DB, faceID, personID int64, assignedBy string) error {
	if assignedBy != "user" && assignedBy != "auto" {
		return fmt.Errorf("assignedBy must be \"user\" or \"auto\"")
	}
	f, ok, err := GetFaceByID(db, faceID)
	if err != nil {
		return err
	}
	if !ok {
		return fmt.Errorf("no face with id %d", faceID)
	}
	if assignedBy == "auto" && f.AssignedBy == "user" {
		return nil // manual labels are ground truth
	}
	p, ok, err := GetPersonByID(db, personID)
	if err != nil {
		return err
	}
	if !ok {
		return fmt.Errorf("no person with id %d", personID)
	}

	tx, err := db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()

	// Moving off a previous person: drop their bridge row if this face was
	// their only one on this media/frame.
	if f.PersonID != 0 && f.PersonID != personID {
		if err := removeBridgeRowIfLastFace(tx, f, f.PersonID); err != nil {
			return err
		}
	}

	if _, err := tx.Exec(
		`UPDATE face SET person_id = ?, assigned_by = ? WHERE id = ?`, personID, assignedBy, faceID,
	); err != nil {
		return err
	}
	if err := ensurePersonTag(tx, p.Name); err != nil {
		return err
	}
	// The bridge row only exists for library media. Faces scanned on arbitrary
	// paths (explicit job input, query images) still get their assignment; the
	// tag row would violate the media_path foreign key present in older
	// databases and be unqueryable anyway.
	var inLibrary int
	if err := tx.QueryRow(`SELECT 1 FROM media WHERE path = ?`, f.MediaPath).Scan(&inLibrary); err != nil && err != sql.ErrNoRows {
		return err
	}
	if inLibrary == 1 {
		if _, err := tx.Exec(
			`INSERT OR IGNORE INTO media_tag_by_category (media_path, tag_label, category_label, weight, time_stamp, created_at)
			 VALUES (?, ?, ?, 0, ?, ?)`,
			f.MediaPath, p.Name, PeopleCategory, f.FrameTS, time.Now().Unix(),
		); err != nil {
			return err
		}
	}
	if _, err := tx.Exec(
		`UPDATE person SET cover_face_id = ? WHERE id = ? AND (cover_face_id IS NULL OR cover_face_id = 0)`,
		faceID, personID,
	); err != nil {
		return err
	}
	if err := tx.Commit(); err != nil {
		return err
	}
	// New People tag may make media eligible for tag-based pools.
	InvalidateRandomSampleCache()
	return nil
}

// UnassignFace clears a face's person assignment and removes the taxonomy
// bridge row when it was the person's last face on that media/frame.
func UnassignFace(db *sql.DB, faceID int64) error {
	f, ok, err := GetFaceByID(db, faceID)
	if err != nil {
		return err
	}
	if !ok {
		return fmt.Errorf("no face with id %d", faceID)
	}
	if f.PersonID == 0 {
		return nil
	}
	tx, err := db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()
	if _, err := tx.Exec(`UPDATE face SET person_id = NULL, assigned_by = NULL WHERE id = ?`, faceID); err != nil {
		return err
	}
	if err := removeBridgeRowIfLastFace(tx, f, f.PersonID); err != nil {
		return err
	}
	// Clear the cover if it pointed at this face.
	if _, err := tx.Exec(`UPDATE person SET cover_face_id = NULL WHERE id = ? AND cover_face_id = ?`, f.PersonID, faceID); err != nil {
		return err
	}
	return tx.Commit()
}

// removeBridgeRowIfLastFace deletes the (media, person, frame) taxonomy row
// when no OTHER face of personID remains on the face's media path + frame.
func removeBridgeRowIfLastFace(tx *sql.Tx, f Face, personID int64) error {
	var name string
	if err := tx.QueryRow(`SELECT COALESCE(name, '') FROM person WHERE id = ?`, personID).Scan(&name); err != nil {
		if err == sql.ErrNoRows {
			return nil // person already gone; nothing to unbridge
		}
		return err
	}
	var others int
	if err := tx.QueryRow(
		`SELECT COUNT(*) FROM face WHERE person_id = ? AND media_path = ? AND frame_ts = ? AND id != ?`,
		personID, f.MediaPath, f.FrameTS, f.ID,
	).Scan(&others); err != nil {
		return err
	}
	if others > 0 {
		return nil
	}
	_, err := tx.Exec(
		`DELETE FROM media_tag_by_category WHERE media_path = ? AND tag_label = ? AND category_label = ? AND time_stamp = ?`,
		f.MediaPath, name, PeopleCategory, f.FrameTS,
	)
	return err
}

// DeleteAllFaceData wipes every trace of the face-identity feature: faces,
// scan markers, persons, and the face-managed People taxonomy rows. The
// privacy escape hatch — everything is rebuildable by rescanning.
func DeleteAllFaceData(db *sql.DB) error {
	tx, err := db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()
	for _, stmt := range []string{
		`DELETE FROM face`,
		`DELETE FROM face_scan`,
		`DELETE FROM person`,
		`DELETE FROM media_tag_by_category WHERE category_label = '` + PeopleCategory + `'`,
		`DELETE FROM tag WHERE category_label = '` + PeopleCategory + `'`,
	} {
		if _, err := tx.Exec(stmt); err != nil {
			return err
		}
	}
	return tx.Commit()
}

func nullableID(id int64) any {
	if id == 0 {
		return nil
	}
	return id
}
