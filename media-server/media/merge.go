package media

import (
	"context"
	"database/sql"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Merging duplicate items.
//
// MergeInto is the one implementation behind every "merge these items" surface:
// the /api/media/merge-metadata endpoint (the viewer's context-palette Merge)
// and the dedupe task. Metadata merge is additive: tag rows and per-model
// embedding rows the target lacks are copied in (the target's own rows always
// win), an empty transcript is filled from the first source that has one, and
// that source's .vtt sidecar is moved next to the target. The sources are then
// DELETED — local file removed (plus leftover sidecar) and every database
// reference erased (tags, media row, embeddings, faces and their curation
// assertions, scan markers, battle-log rows). s3:// sources and files that
// fail to delete keep their rows and are reported in Failed so nothing
// silently orphans.

// MergeResult reports what a merge changed. Field names mirror the historical
// /api/media/merge-metadata response shape.
type MergeResult struct {
	Target  string   `json:"target"`
	Sources []string `json:"sources"`
	// Tags / Embeddings are the number of rows the target GAINED.
	Tags       int64 `json:"tags"`
	Embeddings int64 `json:"embeddings"`
	// Transcript is true when the target's transcript column was filled from a
	// source or a source's .vtt sidecar was moved next to the target.
	Transcript     bool   `json:"transcript"`
	TranscriptFile string `json:"transcriptFile"`
	// Deleted / Failed partition the sources: deleted ones are gone from disk
	// and the database; failed ones keep their file (if any) and their rows.
	Deleted []string `json:"deleted"`
	Failed  []string `json:"failed"`
	// FacesRemoved counts face rows erased with the sources — when it is
	// non-zero the caller should broadcast a people-updated event, since open
	// People views are now showing stale counts.
	FacesRemoved int64 `json:"facesRemoved"`
}

// MergeInto consolidates sources into target as described above. A database
// error before anything is deleted fails the whole merge; per-source deletion
// problems are reported in Failed rather than aborting the remaining sources.
func MergeInto(ctx context.Context, db *sql.DB, target string, sources []string) (*MergeResult, error) {
	if db == nil {
		return nil, fmt.Errorf("database connection not available")
	}
	target = strings.TrimSpace(target)
	var srcs []string
	for _, p := range sources {
		if p = strings.TrimSpace(p); p != "" && p != target {
			srcs = append(srcs, p)
		}
	}
	if target == "" || len(srcs) == 0 {
		return nil, fmt.Errorf("need a target and at least one source path")
	}
	res := &MergeResult{Target: target, Sources: srcs}

	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		return nil, err
	}
	defer tx.Rollback()

	countRows := func(query string) (int64, error) {
		var n int64
		err := tx.QueryRow(query, target).Scan(&n)
		return n, err
	}

	tagsBefore, err := countRows(`SELECT COUNT(*) FROM media_tag_by_category WHERE media_path = ?`)
	if err != nil {
		return nil, err
	}
	for _, src := range srcs {
		// NOT EXISTS with IS-comparison rather than ON CONFLICT: the PK
		// includes time_stamp and SQLite treats NULL PK values as distinct, so
		// conflict resolution alone would duplicate NULL-timestamp tag rows.
		if _, err := tx.Exec(
			`INSERT INTO media_tag_by_category
			   (media_path, tag_label, category_label, weight, time_stamp, created_at)
			 SELECT ?1, s.tag_label, s.category_label, s.weight, s.time_stamp, s.created_at
			 FROM media_tag_by_category s
			 WHERE s.media_path = ?2
			   AND NOT EXISTS (
			     SELECT 1 FROM media_tag_by_category t
			     WHERE t.media_path = ?1
			       AND t.tag_label = s.tag_label
			       AND t.category_label IS s.category_label
			       AND t.time_stamp IS s.time_stamp
			   )`,
			target, src,
		); err != nil {
			return nil, err
		}
	}
	tagsAfter, err := countRows(`SELECT COUNT(*) FROM media_tag_by_category WHERE media_path = ?`)
	if err != nil {
		return nil, err
	}
	res.Tags = tagsAfter - tagsBefore

	embBefore, err := countRows(`SELECT COUNT(*) FROM media_embedding WHERE media_path = ?`)
	if err != nil {
		return nil, err
	}
	for _, src := range srcs {
		// PK (media_path, model): the target's existing per-model vectors win;
		// earlier sources win over later ones for models it lacks.
		if _, err := tx.Exec(
			`INSERT OR IGNORE INTO media_embedding
			   (media_path, model, dim, vector, created_at)
			 SELECT ?, model, dim, vector, created_at
			 FROM media_embedding WHERE media_path = ?`,
			target, src,
		); err != nil {
			return nil, err
		}
	}
	embAfter, err := countRows(`SELECT COUNT(*) FROM media_embedding WHERE media_path = ?`)
	if err != nil {
		return nil, err
	}
	res.Embeddings = embAfter - embBefore

	// Transcript: fill only when the target has none — never overwrite.
	var targetTranscript sql.NullString
	if err := tx.QueryRow(
		`SELECT transcript FROM media WHERE path = ?`, target,
	).Scan(&targetTranscript); err != nil && err != sql.ErrNoRows {
		return nil, err
	}
	if !targetTranscript.Valid || targetTranscript.String == "" {
		for _, src := range srcs {
			var t sql.NullString
			err := tx.QueryRow(`SELECT transcript FROM media WHERE path = ?`, src).Scan(&t)
			if err == sql.ErrNoRows {
				continue
			}
			if err != nil {
				return nil, err
			}
			if t.Valid && t.String != "" {
				if _, err := tx.Exec(
					`UPDATE media SET transcript = ? WHERE path = ?`, t.String, target,
				); err != nil {
					return nil, err
				}
				res.Transcript = true
				break
			}
		}
	}

	if err := tx.Commit(); err != nil {
		return nil, err
	}

	// Transcript sidecar file: when the target has no .vtt of its own, move
	// the first source's sidecar next to the target — before the sources are
	// deleted below, so the file is never lost.
	if findVttSidecar(target) == "" {
		for _, src := range srcs {
			srcVtt := findVttSidecar(src)
			if srcVtt == "" {
				continue
			}
			destVtt := vttCandidates(target)[0]
			if err := os.Rename(srcVtt, destVtt); err == nil {
				res.Transcript = true
				res.TranscriptFile = destVtt
			}
			break
		}
	}

	// Delete the merged-away sources: local file plus leftover sidecar, then
	// every database reference. s3:// objects are not deleted here (no local
	// file to remove) — they keep their rows and are reported as failed.
	//
	// Files first, then ONE batched row erase for everything that came off
	// disk. Per-source erases (a transaction and a full reference sweep each)
	// made a 3,000-item merge take minutes; the batched removal commits 500
	// paths per transaction. A file that is already gone counts as deleted —
	// that is also how an interrupted merge is repaired: re-run it and the
	// rows of the files it removed last time are erased now.
	var toErase []string
	for _, src := range srcs {
		if strings.HasPrefix(src, "s3://") {
			res.Failed = append(res.Failed, src)
			continue
		}
		if err := os.Remove(src); err != nil && !os.IsNotExist(err) {
			res.Failed = append(res.Failed, src)
			continue
		}
		if leftover := findVttSidecar(src); leftover != "" {
			_ = os.Remove(leftover)
		}
		toErase = append(toErase, src)
	}
	if len(toErase) == 0 {
		return res, nil
	}
	// The files are gone; the erase MUST finish even if the caller's request
	// is abandoned (a browser giving up on a long merge used to cancel this
	// context mid-loop and leave deleted files with live rows).
	eraseCtx := context.WithoutCancel(ctx)
	committed := map[string]bool{}
	removal, err := RemoveItemsFromDBStream(eraseCtx, db, toErase, func(b RemovalBatch) {
		for _, p := range b.Paths {
			committed[p] = true
		}
	})
	if removal != nil {
		res.FacesRemoved = removal.FacesRemoved
	}
	for _, p := range toErase {
		if committed[p] {
			res.Deleted = append(res.Deleted, p)
		} else {
			res.Failed = append(res.Failed, p)
		}
	}
	if err != nil && len(res.Deleted) == 0 {
		return res, err
	}
	InvalidateRandomSampleCache()
	return res, nil
}

// Transcript sidecars live next to the media file: `<base>.vtt` (extension
// replaced — the transcribe convention) or `<path>.vtt` (appended). Mirrors
// the Electron viewer's transcript.ts.
func vttCandidates(mediaPath string) []string {
	ext := filepath.Ext(mediaPath)
	if ext == "" {
		return []string{mediaPath + ".vtt"}
	}
	return []string{strings.TrimSuffix(mediaPath, ext) + ".vtt", mediaPath + ".vtt"}
}

func findVttSidecar(mediaPath string) string {
	if strings.HasPrefix(mediaPath, "s3://") {
		return ""
	}
	for _, c := range vttCandidates(mediaPath) {
		if st, err := os.Stat(c); err == nil && !st.IsDir() {
			return c
		}
	}
	return ""
}
