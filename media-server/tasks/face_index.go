package tasks

import (
	"database/sql"
	"strconv"
	"sync"

	"github.com/stevecastle/shrike/embedindex"
	"github.com/stevecastle/shrike/media"
)

// -----------------------------------------------------------------------------
// Package-level FACE vector index — a second exact-scan index alongside the
// media-embedding index, keyed by face row ID (as a string) instead of media
// path, because one media item holds many faces. A side map media_path →
// face-id keys lets a rescan (ReplaceFaces) or media deletion evict stale
// entries without a DB round-trip. All state is serialised behind one mutex.
// -----------------------------------------------------------------------------

var (
	faceIndexMu sync.Mutex
	faceIndex   embedindex.VectorIndex
	// faceIndexModel records which recognizer the installed index holds.
	// Searches for a different model fall back to brute-force over the DB.
	faceIndexModel string
	// facePathKeys maps media_path → the face-id keys currently indexed for it.
	facePathKeys map[string][]string
)

// faceKey is the index key for a face row ID.
func faceKey(id int64) string { return strconv.FormatInt(id, 10) }

// SetFaceIndexForModel installs the active face index (with its path→keys map)
// and records the recognizer it was built for. nil disables it → brute-force.
func SetFaceIndexForModel(idx embedindex.VectorIndex, model string, pathKeys map[string][]string) {
	faceIndexMu.Lock()
	faceIndex = idx
	faceIndexModel = model
	if pathKeys == nil {
		pathKeys = map[string][]string{}
	}
	facePathKeys = pathKeys
	faceIndexMu.Unlock()
}

// FaceIndexedModel returns the recognizer ID the installed face index was
// built for ("" when none).
func FaceIndexedModel() string {
	faceIndexMu.Lock()
	defer faceIndexMu.Unlock()
	return faceIndexModel
}

// FaceIndexSize returns the number of faces in the installed index (0 when
// none). Exported for the index-status API.
func FaceIndexSize() int {
	faceIndexMu.Lock()
	defer faceIndexMu.Unlock()
	if faceIndex == nil {
		return 0
	}
	return faceIndex.Len()
}

// faceIndexSearch runs a locked search for model. ok is false when no index is
// installed or it holds a different recognizer's vectors (caller brute-forces).
func faceIndexSearch(model string, query []float32, k int) ([]embedindex.SearchHit, bool) {
	faceIndexMu.Lock()
	defer faceIndexMu.Unlock()
	if faceIndex == nil || faceIndexModel != model {
		return nil, false
	}
	return faceIndex.Search(query, k), true
}

// faceIndexReplacePath replaces path's indexed faces with the given id/vector
// pairs (parallel slices), evicting whatever was indexed for it before. No-op
// when no index is installed or it holds a different model. Mirrors the
// ReplaceFaces DB semantics so index and table stay in step.
func faceIndexReplacePath(model, path string, ids []int64, faces []media.NewFace) {
	faceIndexMu.Lock()
	defer faceIndexMu.Unlock()
	if faceIndex == nil || faceIndexModel != model {
		return
	}
	for _, key := range facePathKeys[path] {
		faceIndex.Delete(key)
	}
	keys := make([]string, 0, len(ids))
	for i, id := range ids {
		key := faceKey(id)
		faceIndex.Add(key, faces[i].Vec) // Add L2-normalizes internally
		keys = append(keys, key)
	}
	if len(keys) == 0 {
		delete(facePathKeys, path)
	} else {
		facePathKeys[path] = keys
	}
}

// FaceIndexDeletePath evicts all of path's faces from the live index (no-op
// when none). Exported for the media-removal hook.
func FaceIndexDeletePath(path string) {
	faceIndexMu.Lock()
	defer faceIndexMu.Unlock()
	if faceIndex == nil {
		return
	}
	for _, key := range facePathKeys[path] {
		faceIndex.Delete(key)
	}
	delete(facePathKeys, path)
}

// RebuildActiveFaceIndex builds the face index for the currently-configured
// recognizer and installs it. Used at startup and after the active face model
// changes. Returns the installed model ID and face count, or an error (in
// which case the previous index is left untouched).
func RebuildActiveFaceIndex(db *sql.DB, onProgress IndexProgress) (string, int, error) {
	model := ActiveFaceModel()
	idx, pathKeys, err := BuildFaceIndexFromDB(db, model.ID, onProgress)
	if err != nil {
		return model.ID, 0, err
	}
	SetFaceIndexForModel(idx, model.ID, pathKeys)
	return model.ID, idx.Len(), nil
}

// BuildFaceIndexFromDB constructs a face index (and its path→keys map) from
// all stored faces for model.
func BuildFaceIndexFromDB(db *sql.DB, model string, onProgress IndexProgress) (embedindex.VectorIndex, map[string][]string, error) {
	all, err := media.LoadAllFaces(db, model)
	if err != nil {
		return nil, nil, err
	}
	idx := embedindex.New()
	pathKeys := make(map[string][]string)
	total := len(all)
	if onProgress != nil {
		onProgress(0, total)
	}
	for i, f := range all {
		key := faceKey(f.ID)
		idx.Add(key, f.Vec)
		pathKeys[f.MediaPath] = append(pathKeys[f.MediaPath], key)
		if onProgress != nil && ((i+1)%512 == 0 || i+1 == total) {
			onProgress(i+1, total)
		}
	}
	return idx, pathKeys, nil
}
