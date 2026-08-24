package main

import (
	"log"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/stevecastle/shrike/media"
	"github.com/stevecastle/shrike/tasks"
)

// Swipe vector-search modes, sharing the similar-mode contract (see
// swipe_similar.go): a deterministic ranked list per query, orphan and
// orientation filtering on the ranked prefix BEFORE the page slice so
// offsets index a stable sequence, and a grow-until-covered fetch loop.
//
//	mode=text&q=…      items ranked by SigLIP text→image similarity
//	mode=face&anchor=… items ranked by face similarity to the largest
//	                   face in the anchor item (anchor excluded — the
//	                   client keeps it at the top of its stack)

// swipeSearchCandidateSlack is the extra ranked candidates fetched beyond
// the requested page so orphan embeddings / off-orientation items can be
// filtered out without coming up short. Face mode needs more headroom than
// similar mode: hits are FACES, and collapsing several faces per item into
// one media entry shrinks the list before filtering even starts.
const swipeSearchCandidateSlack = 96

// swipeSearchPage parses the shared paging params (same bounds as the
// other swipe modes).
func swipeSearchPage(r *http.Request) (offset, limit int, orientation string) {
	offset = 0
	if s := r.URL.Query().Get("offset"); s != "" {
		if parsed, err := strconv.Atoi(s); err == nil && parsed >= 0 {
			offset = parsed
		}
	}
	limit = 20
	if s := r.URL.Query().Get("limit"); s != "" {
		if parsed, err := strconv.Atoi(s); err == nil && parsed > 0 && parsed <= 50 {
			limit = parsed
		}
	}
	return offset, limit, media.NormalizeOrientation(r.URL.Query().Get("orientation"))
}

// swipeSearchRespond slices the ranked list into the requested page and
// writes the standard swipe API response.
func swipeSearchRespond(w http.ResponseWriter, deps *Dependencies, ranked []string, offset, limit int) bool {
	hasMore := len(ranked) > offset+limit
	var picked []string
	if offset < len(ranked) {
		end := offset + limit
		if end > len(ranked) {
			end = len(ranked)
		}
		picked = ranked[offset:end]
	}
	items, err := media.GetItemsByPaths(deps.DB, picked)
	if err != nil {
		log.Printf("swipe search item fetch failed: %v", err)
		http.Error(w, "Error fetching media items", http.StatusInternalServerError)
		return true
	}
	writeJSON(w, media.APIResponse{Items: items, HasMore: hasMore})
	return true
}

// ---- text mode --------------------------------------------------------

// The text encoder is a subprocess (an ONNX run per call) — pages of one
// query must not pay it again. Entries are tiny (one 768-float vector), so
// a small TTL cache keyed by the exact query text is plenty.
type swipeTextVec struct {
	vec   []float32
	model string
	at    time.Time
}

var (
	swipeTextVecMu  sync.Mutex
	swipeTextVecs   = map[string]swipeTextVec{}
	swipeTextVecTTL = 15 * time.Minute
	swipeTextVecCap = 32
)

func swipeTextVector(r *http.Request, query string) ([]float32, string, error) {
	swipeTextVecMu.Lock()
	if e, ok := swipeTextVecs[query]; ok && time.Since(e.at) < swipeTextVecTTL {
		e.at = time.Now() // touch: repeated paging keeps the entry alive
		swipeTextVecs[query] = e
		swipeTextVecMu.Unlock()
		return e.vec, e.model, nil
	}
	swipeTextVecMu.Unlock()

	vec, m, err := tasks.TextQueryVector(r.Context(), query)
	if err != nil {
		return nil, "", err
	}

	swipeTextVecMu.Lock()
	defer swipeTextVecMu.Unlock()
	if len(swipeTextVecs) >= swipeTextVecCap {
		// Evict expired first, then the stalest — the cap is a backstop, not
		// an LRU contract.
		oldestKey := ""
		oldest := time.Now()
		for k, e := range swipeTextVecs {
			if time.Since(e.at) >= swipeTextVecTTL {
				delete(swipeTextVecs, k)
				continue
			}
			if e.at.Before(oldest) {
				oldest = e.at
				oldestKey = k
			}
		}
		if len(swipeTextVecs) >= swipeTextVecCap && oldestKey != "" {
			delete(swipeTextVecs, oldestKey)
		}
	}
	swipeTextVecs[query] = swipeTextVec{vec: vec, model: m.ID, at: time.Now()}
	return vec, m.ID, nil
}

// maybeHandleSwipeTextSearch serves /swipe/api requests with mode=text:
// the whole library ranked by SigLIP text→image similarity to q, best
// first. Returns false when the request isn't text-mode so the caller
// falls through. Shared by the per-platform swipeAPIHandler copies.
func maybeHandleSwipeTextSearch(w http.ResponseWriter, r *http.Request, deps *Dependencies) bool {
	if r.URL.Query().Get("mode") != "text" {
		return false
	}
	query := strings.TrimSpace(r.URL.Query().Get("q"))
	if query == "" {
		http.Error(w, "q is required for mode=text", http.StatusBadRequest)
		return true
	}
	offset, limit, orientation := swipeSearchPage(r)

	vec, modelID, err := swipeTextVector(r, query)
	if err != nil {
		// Almost always "text model / tokenizer / embed binary not
		// installed" — actionable, so surface the real message.
		log.Printf("swipe text search encode failed (q=%q): %v", query, err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return true
	}

	need := offset + limit + 1 // +1 so hasMore needs no extra query
	fetch := need + swipeSearchCandidateSlack
	var ranked []string
	for {
		hits, err := tasks.SearchByTextVector(deps.DB, modelID, vec, fetch, nil)
		if err != nil {
			log.Printf("swipe text search failed (q=%q model=%q): %v", query, modelID, err)
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return true
		}
		paths := make([]string, 0, len(hits))
		for _, h := range hits {
			paths = append(paths, h.Path)
		}
		ranked, err = media.FilterSwipePaths(deps.DB, paths, orientation)
		if err != nil {
			log.Printf("swipe text search orphan filter failed: %v", err)
			http.Error(w, "Error fetching media items", http.StatusInternalServerError)
			return true
		}
		exhausted := len(hits) < fetch
		if len(ranked) >= need || exhausted {
			break
		}
		fetch *= 2
	}
	return swipeSearchRespond(w, deps, ranked, offset, limit)
}

// ---- face mode --------------------------------------------------------

// maybeHandleSwipeFace serves /swipe/api requests with mode=face: items
// ranked by face similarity to the LARGEST face in the anchor item (the
// natural "more of this person" reading of a group shot). The anchor's own
// item is scanned on the fly when it has no stored faces — that result is
// persisted, so later pages read from the DB.
//
// Hits come back as FACES; they collapse to media paths keeping each
// path's best-ranked face, then the anchor is excluded and the ranked
// prefix is orphan/orientation filtered before slicing — the same stable
// offset contract as similar mode.
func maybeHandleSwipeFace(w http.ResponseWriter, r *http.Request, deps *Dependencies) bool {
	if r.URL.Query().Get("mode") != "face" {
		return false
	}
	anchor := r.URL.Query().Get("anchor")
	if anchor == "" {
		http.Error(w, "anchor is required for mode=face", http.StatusBadRequest)
		return true
	}
	offset, limit, orientation := swipeSearchPage(r)

	need := offset + limit + 1
	fetch := need + swipeSearchCandidateSlack
	var ranked []string
	for {
		hits, err := tasks.SearchFacesByMediaPath(r.Context(), deps.DB, anchor, fetch, nil)
		if err != nil {
			// Includes tasks.ErrNoFaceInQuery ("no face found in the query
			// image") — the client toasts the message and drops back.
			log.Printf("swipe face search failed (anchor=%q): %v", anchor, err)
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return true
		}
		// Faces → media paths: keep first (= best-scored) occurrence per
		// path, drop the anchor's own faces.
		seen := make(map[string]struct{}, len(hits))
		paths := make([]string, 0, len(hits))
		for _, h := range hits {
			if h.MediaPath == anchor {
				continue
			}
			if _, dup := seen[h.MediaPath]; dup {
				continue
			}
			seen[h.MediaPath] = struct{}{}
			paths = append(paths, h.MediaPath)
		}
		ranked, err = media.FilterSwipePaths(deps.DB, paths, orientation)
		if err != nil {
			log.Printf("swipe face search orphan filter failed: %v", err)
			http.Error(w, "Error fetching media items", http.StatusInternalServerError)
			return true
		}
		exhausted := len(hits) < fetch
		if len(ranked) >= need || exhausted {
			break
		}
		fetch *= 2
	}
	return swipeSearchRespond(w, deps, ranked, offset, limit)
}
