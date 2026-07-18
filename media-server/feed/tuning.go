// Package feed implements the /swipe "For You" mode: a never-ending
// algorithmic feed ranked from the user's swipe favorites via embedding
// similarity. The algorithm is deliberately factored into independently
// tunable "lanes" (exploit / fresh / bridge / wildcard) so the mix can be
// iterated on without touching the engine: every knob lives in Tuning,
// which is embedded in the server config under "swipeFeed" and can also be
// overridden per-request via query params for quick experiments.
package feed

// Tuning holds every adjustable parameter of the For-You feed algorithm.
// The zero value means "use the built-in default" for each field, so a
// config file may set any subset (or omit the section entirely).
type Tuning struct {
	// ---- Lane mix ----------------------------------------------------
	// Relative shares of each generated batch. They are normalized, so
	// only their ratios matter. Lanes that cannot produce items (e.g. no
	// likes yet) forfeit their share to the others.

	// ExploitWeight: nearest neighbors of the user's taste clusters —
	// the "more of what you like" backbone of the feed.
	ExploitWeight float64 `json:"exploitWeight"` // default 0.5

	// FreshWeight: neighbors of the single most recent like, so a new
	// like visibly bends the feed within one profile refresh.
	FreshWeight float64 `json:"freshWeight"` // default 0.15

	// BridgeWeight: the "surprising" lane. Midpoints between two
	// different taste clusters (things that connect separate interests)
	// and deep-band picks — similar-ish items well past the obvious
	// nearest neighbors.
	BridgeWeight float64 `json:"bridgeWeight"` // default 0.2

	// WildcardWeight: uniform random probes from the whole library —
	// genuine exposure outside every known interest.
	WildcardWeight float64 `json:"wildcardWeight"` // default 0.15

	// ---- Taste profile ------------------------------------------------

	// MaxLikes caps how many of the most recent favorites feed the
	// profile. Older likes age out entirely beyond this.
	MaxLikes int `json:"maxLikes"` // default 500

	// MaxClusters caps the number of taste clusters. Extra likes fold
	// into their nearest existing cluster.
	MaxClusters int `json:"maxClusters"` // default 12

	// ClusterThreshold is the cosine similarity a like must have to an
	// existing cluster centroid to join it; below this it founds a new
	// cluster (until MaxClusters). Higher = more, tighter clusters.
	ClusterThreshold float64 `json:"clusterThreshold"` // default 0.6

	// RecencyHalfLife is measured in likes: the Nth most recent like has
	// half the profile weight of the most recent one when N equals this.
	// Smaller = the feed chases recent taste harder.
	RecencyHalfLife int `json:"recencyHalfLife"` // default 80

	// ---- Candidate pools ----------------------------------------------

	// PoolSize is how many ranked neighbors are fetched (and cached) per
	// taste cluster per profile build.
	PoolSize int `json:"poolSize"` // default 600

	// DeepBandStart is the fraction of a cluster's neighbor pool where
	// the bridge lane's deep-band sampling begins: items ranked past
	// this point are "familiar but different". 0.25 means the first 25%
	// (the obvious picks) are skipped.
	DeepBandStart float64 `json:"deepBandStart"` // default 0.25

	// BatchSize is how many items one generation round appends to a
	// session's feed.
	BatchSize int `json:"batchSize"` // default 60

	// MaxSearchesPerPage caps how many vector searches (cluster pools,
	// fresh pool, bridge midpoints) one page request may trigger. On
	// multi-million-item libraries a single search is a full exact scan
	// (hundreds of ms), so this is the main latency guard: pools are
	// cached on the profile, so the feed warms up over the first few
	// pages instead of paying every scan on page one. Lanes whose pool
	// isn't built yet and can't afford a search fall back to already
	// built pools or the wildcard lane.
	MaxSearchesPerPage int `json:"maxSearchesPerPage"` // default 3

	// IncludeLiked re-shows items the user has already favorited. Off by
	// default: the feed is for discovery.
	IncludeLiked bool `json:"includeLiked"` // default false

	// ---- Freshness / caching -------------------------------------------

	// ProfileTTLSeconds is how long a built taste profile (clusters +
	// neighbor pools) is reused before an unconditional rebuild.
	ProfileTTLSeconds int `json:"profileTtlSeconds"` // default 300

	// LikeCheckSeconds is how often (at most) the engine re-checks the
	// favorites signature (count + newest timestamp); a change forces a
	// profile rebuild ahead of the TTL. This is what makes the feed
	// respond to new likes while staying cached: eventual consistency
	// within roughly this many seconds.
	LikeCheckSeconds int `json:"likeCheckSeconds"` // default 5

	// SessionTTLMinutes evicts feed sessions idle longer than this.
	SessionTTLMinutes int `json:"sessionTtlMinutes"` // default 120

	// MaxSessions caps concurrently tracked feed sessions (oldest-idle
	// evicted first).
	MaxSessions int `json:"maxSessions"` // default 64

	// ---- Source of truth for likes --------------------------------------

	// FavoritesTag / FavoritesCategory identify which tag counts as a
	// "like". Defaults match the swipe UI's heart button.
	FavoritesTag      string `json:"favoritesTag"`      // default "Favorites"
	FavoritesCategory string `json:"favoritesCategory"` // default "Swipe"
}

// WithDefaults returns a copy of t with every zero field replaced by its
// built-in default, so callers can hold a sparse Tuning (e.g. straight from
// config.json) and still get a fully usable parameter set.
func (t Tuning) WithDefaults() Tuning {
	def := func(v *float64, d float64) {
		if *v <= 0 {
			*v = d
		}
	}
	defi := func(v *int, d int) {
		if *v <= 0 {
			*v = d
		}
	}
	// Lane weights: only default when ALL are unset, so a config that
	// deliberately zeroes one lane (e.g. wildcardWeight: 0 with others set)
	// is respected.
	if t.ExploitWeight <= 0 && t.FreshWeight <= 0 && t.BridgeWeight <= 0 && t.WildcardWeight <= 0 {
		t.ExploitWeight, t.FreshWeight, t.BridgeWeight, t.WildcardWeight = 0.5, 0.15, 0.2, 0.15
	}
	defi(&t.MaxLikes, 500)
	defi(&t.MaxClusters, 12)
	def(&t.ClusterThreshold, 0.6)
	defi(&t.RecencyHalfLife, 80)
	defi(&t.PoolSize, 600)
	def(&t.DeepBandStart, 0.25)
	defi(&t.BatchSize, 60)
	defi(&t.MaxSearchesPerPage, 3)
	defi(&t.ProfileTTLSeconds, 300)
	defi(&t.LikeCheckSeconds, 5)
	defi(&t.SessionTTLMinutes, 120)
	defi(&t.MaxSessions, 64)
	if t.FavoritesTag == "" {
		t.FavoritesTag = "Favorites"
	}
	if t.FavoritesCategory == "" {
		t.FavoritesCategory = "Swipe"
	}
	return t
}
