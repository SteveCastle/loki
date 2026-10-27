package main

import (
	"bytes"
	"database/sql"
	"encoding/json"
	"log"
	"net/http"
	"sort"
)

// mediaItemJSON is one library item as the renderer's Item type expects it:
// { path, mtimeMs, elo, battles, height, width, weight, tagLabel, timeStamp,
// score }. Nullable columns are pointers so NULL is omitted while a real
// zero is still sent — the same shape the old per-row map[string]any had,
// at a fraction of the allocation (a map per row plus key sorting on encode
// made a million-row result cost gigabytes).
type mediaItemJSON struct {
	Path      string   `json:"path"`
	MtimeMs   int64    `json:"mtimeMs"`
	Elo       *float64 `json:"elo,omitempty"`
	Battles   *int64   `json:"battles,omitempty"`
	Height    *int64   `json:"height,omitempty"`
	Width     *int64   `json:"width,omitempty"`
	Weight    *float64 `json:"weight,omitempty"`
	TagLabel  *string  `json:"tagLabel,omitempty"`
	TimeStamp *float64 `json:"timeStamp,omitempty"`
	Score     *float32 `json:"score,omitempty"`
}

func nullF(v sql.NullFloat64) *float64 {
	if !v.Valid {
		return nil
	}
	return &v.Float64
}

func nullI(v sql.NullInt64) *int64 {
	if !v.Valid {
		return nil
	}
	return &v.Int64
}

func nullS(v sql.NullString) *string {
	if !v.Valid {
		return nil
	}
	return &v.String
}

// scanQueryItem scans BuildMediaQuery's 9 columns: path, elo, height, width,
// weight, tag_label, time_stamp, created_at, battles.
func scanQueryItem(rows *sql.Rows) (mediaItemJSON, error) {
	var it mediaItemJSON
	var elo, weight, timeStamp sql.NullFloat64
	var height, width, createdAt, battles sql.NullInt64
	var tagLabel sql.NullString
	if err := rows.Scan(&it.Path, &elo, &height, &width,
		&weight, &tagLabel, &timeStamp, &createdAt, &battles); err != nil {
		return it, err
	}
	it.MtimeMs = createdAt.Int64
	it.Elo, it.Battles = nullF(elo), nullI(battles)
	it.Height, it.Width = nullI(height), nullI(width)
	it.Weight, it.TagLabel, it.TimeStamp = nullF(weight), nullS(tagLabel), nullF(timeStamp)
	return it, nil
}

// scanTagJoinItem scans lokiMediaHandler's 10 columns: path, tag_label,
// category_label, weight, time_stamp, created_at, height, width, elo, battles.
func scanTagJoinItem(rows *sql.Rows) (mediaItemJSON, error) {
	var it mediaItemJSON
	var tagLabel, categoryLabel sql.NullString
	var weight, timeStamp, elo sql.NullFloat64
	var createdAt, height, width, battles sql.NullInt64
	if err := rows.Scan(&it.Path, &tagLabel, &categoryLabel, &weight, &timeStamp,
		&createdAt, &height, &width, &elo, &battles); err != nil {
		return it, err
	}
	it.MtimeMs = createdAt.Int64
	it.TagLabel, it.Weight, it.TimeStamp = nullS(tagLabel), nullF(weight), nullF(timeStamp)
	it.Height, it.Width = nullI(height), nullI(width)
	it.Elo, it.Battles = nullF(elo), nullI(battles)
	return it, nil
}

// sortMediaItemsByScore stamps each item's similarity score and orders the
// best match first (stable, so equal scores keep SQL order).
func sortMediaItemsByScore(items []mediaItemJSON, scoreByPath map[string]float32) {
	for i := range items {
		s := scoreByPath[items[i].Path]
		items[i].Score = &s
	}
	sort.SliceStable(items, func(a, b int) bool {
		return *items[a].Score > *items[b].Score
	})
}

// mediaArrayWriter streams a JSON array of items. Output is buffered and
// pushed to the client in chunks, so memory stays flat however large the
// result. Until the first chunk goes out a failure can still become a
// proper error response; after that, abort() kills the connection so the
// client sees a failed request rather than a valid-looking truncated list.
type mediaArrayWriter struct {
	w         http.ResponseWriter
	buf       bytes.Buffer
	enc       *json.Encoder
	n         int
	committed bool
}

const mediaArrayChunk = 256 << 10

func newMediaArrayWriter(w http.ResponseWriter) *mediaArrayWriter {
	aw := &mediaArrayWriter{w: w}
	aw.enc = json.NewEncoder(&aw.buf)
	aw.buf.WriteByte('[')
	return aw
}

func (aw *mediaArrayWriter) add(it *mediaItemJSON) error {
	if aw.n > 0 {
		aw.buf.WriteByte(',')
	}
	aw.n++
	if err := aw.enc.Encode(it); err != nil {
		return err
	}
	if aw.buf.Len() >= mediaArrayChunk {
		return aw.flush()
	}
	return nil
}

func (aw *mediaArrayWriter) flush() error {
	if !aw.committed {
		aw.w.Header().Set("Content-Type", "application/json")
		aw.committed = true
	}
	_, err := aw.w.Write(aw.buf.Bytes())
	aw.buf.Reset()
	return err
}

// finish closes the array and sends what remains.
func (aw *mediaArrayWriter) finish() {
	aw.buf.WriteByte(']')
	aw.buf.WriteByte('\n')
	_ = aw.flush()
}

// fail reports err: as a 500 if nothing was sent yet, else by aborting.
func (aw *mediaArrayWriter) fail(err error) {
	log.Printf("media list: %v", err)
	if !aw.committed {
		httpError(aw.w, err.Error(), http.StatusInternalServerError)
		return
	}
	panic(http.ErrAbortHandler)
}

// streamMediaRows writes every row as one JSON array, without holding the
// result in memory. Rows that fail to scan are skipped (as before); a query
// error, including one surfacing mid-iteration, fails the response.
func streamMediaRows(w http.ResponseWriter, rows *sql.Rows, scan func(*sql.Rows) (mediaItemJSON, error)) {
	aw := newMediaArrayWriter(w)
	skipped := 0
	for rows.Next() {
		it, err := scan(rows)
		if err != nil {
			skipped++
			continue
		}
		if err := aw.add(&it); err != nil {
			return // client went away
		}
	}
	if skipped > 0 {
		log.Printf("media list: skipped %d unscannable rows", skipped)
	}
	if err := rows.Err(); err != nil {
		aw.fail(err)
		return
	}
	aw.finish()
}

// collectMediaRows reads every row into memory (for results that must be
// re-ordered before sending), honouring rows.Err.
func collectMediaRows(rows *sql.Rows, scan func(*sql.Rows) (mediaItemJSON, error)) ([]mediaItemJSON, error) {
	items := []mediaItemJSON{}
	for rows.Next() {
		it, err := scan(rows)
		if err != nil {
			continue
		}
		items = append(items, it)
	}
	return items, rows.Err()
}

func writeMediaItems(w http.ResponseWriter, items []mediaItemJSON) {
	aw := newMediaArrayWriter(w)
	for i := range items {
		if err := aw.add(&items[i]); err != nil {
			return
		}
	}
	aw.finish()
}
