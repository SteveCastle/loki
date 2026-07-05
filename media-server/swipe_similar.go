package main

import (
	"log"
	"net/http"
	"strconv"

	"github.com/stevecastle/shrike/media"
	"github.com/stevecastle/shrike/tasks"
)

// swipeSimilarCandidateSlack is the extra ranked candidates fetched beyond the
// requested page so that orphan embeddings (paths with no media row) can be
// filtered out without coming up short of a full page.
const swipeSimilarCandidateSlack = 64

// maybeHandleSwipeSimilar serves /swipe/api requests with mode=similar: items
// ranked by embedding similarity to the anchor path (nearest first) instead of
// the seeded random shuffle. Returns false when the request isn't similar-mode
// so the caller falls through to the random path. Shared by the per-platform
// swipeAPIHandler copies.
//
// The anchor itself is excluded from the results — the client already has it
// on screen and keeps it at the top of its stack. Offsets index the ranked
// list AFTER anchor exclusion and orphan filtering; the ranking is
// deterministic for a fixed anchor, so pages compose consistently.
func maybeHandleSwipeSimilar(w http.ResponseWriter, r *http.Request, deps *Dependencies) bool {
	if r.URL.Query().Get("mode") != "similar" {
		return false
	}

	anchor := r.URL.Query().Get("anchor")
	if anchor == "" {
		http.Error(w, "anchor is required for mode=similar", http.StatusBadRequest)
		return true
	}

	offset := 0
	if s := r.URL.Query().Get("offset"); s != "" {
		if parsed, err := strconv.Atoi(s); err == nil && parsed >= 0 {
			offset = parsed
		}
	}
	limit := 20
	if s := r.URL.Query().Get("limit"); s != "" {
		if parsed, err := strconv.Atoi(s); err == nil && parsed > 0 && parsed <= 50 {
			limit = parsed
		}
	}

	modelID := tasks.ActiveEmbedModel().ID
	need := offset + limit + 1 // +1 so hasMore needs no extra query

	// Fetch ranked candidates, filter orphans, and grow the window until the
	// page is covered or the ranking is exhausted. +1 covers the anchor hit.
	fetch := need + swipeSimilarCandidateSlack
	var ranked []string
	for {
		hits, err := tasks.SimilarByPathOrEmbed(r.Context(), deps.DB, modelID, anchor, fetch+1)
		if err != nil {
			log.Printf("swipe similar search failed (anchor=%q model=%q): %v", anchor, modelID, err)
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return true
		}
		paths := make([]string, 0, len(hits))
		for _, h := range hits {
			if h.Path != anchor {
				paths = append(paths, h.Path)
			}
		}
		ranked, err = media.FilterExistingMediaPaths(deps.DB, paths)
		if err != nil {
			log.Printf("swipe similar orphan filter failed: %v", err)
			http.Error(w, "Error fetching media items", http.StatusInternalServerError)
			return true
		}
		exhausted := len(hits) < fetch+1
		if len(ranked) >= need || exhausted {
			break
		}
		fetch *= 2
	}

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
		log.Printf("swipe similar item fetch failed: %v", err)
		http.Error(w, "Error fetching media items", http.StatusInternalServerError)
		return true
	}

	writeJSON(w, media.APIResponse{Items: items, HasMore: hasMore})
	return true
}
