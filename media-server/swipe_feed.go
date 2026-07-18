package main

import (
	"log"
	"net/http"
	"strconv"
	"sync"

	"github.com/stevecastle/shrike/appconfig"
	"github.com/stevecastle/shrike/feed"
	"github.com/stevecastle/shrike/media"
	"github.com/stevecastle/shrike/tasks"
)

var (
	feedEngineOnce sync.Once
	feedEngine     *feed.Engine
)

// swipeFeedEngine lazily wires the For-You feed engine: vector search comes
// from the tasks package (ANN index / exact scan), tuning is re-read from
// config on every page so edits to the "swipeFeed" section apply live.
func swipeFeedEngine(deps *Dependencies) *feed.Engine {
	feedEngineOnce.Do(func() {
		feedEngine = feed.NewEngine(
			deps.DB,
			func() string { return tasks.ActiveEmbedModel().ID },
			func(model string, query []float32, limit int) ([]string, error) {
				hits, err := tasks.SearchByVector(deps.DB, model, query, limit)
				if err != nil {
					return nil, err
				}
				paths := make([]string, len(hits))
				for i, h := range hits {
					paths[i] = h.Path
				}
				return paths, nil
			},
			func() feed.Tuning { return appconfig.Get().SwipeFeed },
		)
	})
	return feedEngine
}

// maybeHandleSwipeFeed serves /swipe/api requests with mode=feed: the
// never-ending "For You" feed ranked from the user's swipe favorites (see
// the feed package for the algorithm). Returns false when the request isn't
// feed-mode so the caller falls through. Shared by the per-platform
// swipeAPIHandler copies.
//
// The session param (the client's per-load seed) keys the server-side feed
// sequence so offset/limit pages compose; lane-weight query params
// (exploit/fresh/bridge/wildcard) override the configured tuning for that
// request only — a quick way to experiment with the mix from the URL.
func maybeHandleSwipeFeed(w http.ResponseWriter, r *http.Request, deps *Dependencies) bool {
	if r.URL.Query().Get("mode") != "feed" {
		return false
	}
	handleSwipeFeed(w, r, deps, swipeFeedEngine(deps))
	return true
}

// handleSwipeFeed is the engine-injected core of maybeHandleSwipeFeed,
// split out so tests can run it against a per-test engine instead of the
// process-wide singleton.
func handleSwipeFeed(w http.ResponseWriter, r *http.Request, deps *Dependencies, engine *feed.Engine) {
	q := r.URL.Query()

	session := q.Get("session")
	if session == "" {
		session = "default"
	}
	offset := 0
	if s := q.Get("offset"); s != "" {
		if parsed, err := strconv.Atoi(s); err == nil && parsed >= 0 {
			offset = parsed
		}
	}
	limit := 20
	if s := q.Get("limit"); s != "" {
		if parsed, err := strconv.Atoi(s); err == nil && parsed > 0 && parsed <= 50 {
			limit = parsed
		}
	}

	var override func(*feed.Tuning)
	laneParams := map[string]*float64{}
	for _, name := range []string{"exploit", "fresh", "bridge", "wildcard"} {
		if s := q.Get(name); s != "" {
			if v, err := strconv.ParseFloat(s, 64); err == nil && v >= 0 {
				v := v
				laneParams[name] = &v
			}
		}
	}
	if len(laneParams) > 0 {
		override = func(t *feed.Tuning) {
			if v := laneParams["exploit"]; v != nil {
				t.ExploitWeight = *v
			}
			if v := laneParams["fresh"]; v != nil {
				t.FreshWeight = *v
			}
			if v := laneParams["bridge"]; v != nil {
				t.BridgeWeight = *v
			}
			if v := laneParams["wildcard"]; v != nil {
				t.WildcardWeight = *v
			}
		}
	}

	paths, hasMore, err := engine.Page(r.Context(), session, offset, limit, override)
	if err != nil {
		// A canceled context is the client navigating away mid-request —
		// routine, not an error worth logging loudly.
		if r.Context().Err() != nil {
			return
		}
		log.Printf("swipe feed page failed (session=%q offset=%d): %v", session, offset, err)
		http.Error(w, "Error building feed", http.StatusInternalServerError)
		return
	}

	items, err := media.GetItemsByPaths(deps.DB, paths)
	if err != nil {
		log.Printf("swipe feed item fetch failed: %v", err)
		http.Error(w, "Error fetching media items", http.StatusInternalServerError)
		return
	}

	writeJSON(w, media.APIResponse{Items: items, HasMore: hasMore})
}
