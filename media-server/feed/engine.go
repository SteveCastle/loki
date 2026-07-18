package feed

import (
	"context"
	"database/sql"
	"fmt"
	"hash/fnv"
	"log"
	"math"
	"math/rand"
	"sort"
	"sync"
	"time"

	"github.com/stevecastle/shrike/embedvec"
	"github.com/stevecastle/shrike/media"
)

// SearchFn performs a vector-similarity search over the library and returns
// the ranked media paths (nearest first). Injected so the engine stays free
// of the tasks package (and its model/ONNX machinery) — handlers wire it to
// tasks.SearchByVector, tests to a fake.
type SearchFn func(model string, query []float32, limit int) ([]string, error)

// Engine serves never-ending For-You feed pages. It caches a taste profile
// built from the user's swipe favorites (eventually consistent with new
// likes — see Tuning.LikeCheckSeconds) and per-session feed sequences so
// offset/limit pages compose deterministically.
type Engine struct {
	db     *sql.DB
	model  func() string // active embed model ID
	search SearchFn
	tuning func() Tuning // current base tuning (defaults applied here)
	now    func() time.Time

	mu       sync.Mutex
	profile  *profile
	sessions map[string]*session
}

// NewEngine wires an Engine. tuning may return a sparse Tuning (zero fields);
// defaults are applied on every use so config edits take effect live.
func NewEngine(db *sql.DB, model func() string, search SearchFn, tuning func() Tuning) *Engine {
	return &Engine{
		db:       db,
		model:    model,
		search:   search,
		tuning:   tuning,
		now:      time.Now,
		sessions: make(map[string]*session),
	}
}

// cluster is one taste center: a recency-weighted mean of liked-item
// embeddings plus its lazily built (and profile-lifetime cached) ranked
// neighbor pool.
type cluster struct {
	sum      []float32 // unnormalized weighted vector sum
	centroid []float32 // Normalize(sum), maintained on add
	weight   float64   // total like weight; drives sampling share
	pool     []string  // ranked neighbor paths, nearest first
	poolOK   bool
}

// profile is the cached taste model. It is rebuilt when the embed model
// changes, its TTL lapses, or the favorites signature moves (checked at most
// once per LikeCheckSeconds — the eventual-consistency window for new likes).
type profile struct {
	model        string
	builtAt      time.Time
	signature    string
	sigCheckedAt time.Time
	clusters     []*cluster
	fresh        *cluster // pseudo-cluster around the single most recent like
	liked        map[string]bool
	bridgePools  map[[2]int][]string // midpoint neighbor pools, keyed by cluster index pair
}

// session is one client feed: the generated sequence so far plus the set of
// everything already dealt (including dropped orphans, so they are never
// retried).
type session struct {
	rng        *rand.Rand
	feed       []string
	seen       map[string]bool
	lastAccess time.Time
	exhausted  bool
}

// pageBudget bounds the expensive work one Page call may do. On large
// libraries every vector search is a full exact scan, so an unbounded page
// (dozens of pool/bridge searches) can pin the CPU for tens of seconds —
// pools instead warm up incrementally across pages.
type pageBudget struct {
	searches int // vector searches remaining
}

// maxRoundsPerPage caps generation rounds in one Page call: a page that
// still isn't full after this many batches returns short (hasMore stays
// true) instead of grinding on; the next request continues the work.
const maxRoundsPerPage = 6

// Page returns the feed slice [offset, offset+limit) for sessionID,
// generating further batches as needed. override (may be nil) mutates a copy
// of the tuning for this request only — the per-request experimentation
// hook. hasMore is false only once the library is exhausted for this session.
// Honors ctx cancellation between expensive steps so an abandoned request
// (client navigated away) stops consuming the library scan promptly.
func (e *Engine) Page(ctx context.Context, sessionID string, offset, limit int, override func(*Tuning)) ([]string, bool, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	start := e.now()

	t := e.tuning().WithDefaults()
	if override != nil {
		override(&t)
		t = t.WithDefaults()
	}

	p, err := e.freshProfile(ctx, t)
	if err != nil {
		return nil, false, err
	}

	s := e.session(sessionID, t)
	b := &pageBudget{searches: t.MaxSearchesPerPage}

	need := offset + limit + 1 // +1: cheap has-more probe
	rounds := 0
	for len(s.feed) < need && !s.exhausted && rounds < maxRoundsPerPage {
		if err := ctx.Err(); err != nil {
			return nil, false, err
		}
		rounds++
		added, err := e.generate(ctx, s, p, t, b)
		if err != nil {
			return nil, false, err
		}
		if added == 0 {
			s.exhausted = true
		}
	}

	if d := e.now().Sub(start); d > 2*time.Second {
		log.Printf("feed: slow page: %.1fs (session=%q offset=%d rounds=%d searches=%d clusters=%d feed=%d)",
			d.Seconds(), sessionID, offset, rounds, t.MaxSearchesPerPage-b.searches, len(p.clusters), len(s.feed))
	}

	var page []string
	if offset < len(s.feed) {
		end := offset + limit
		if end > len(s.feed) {
			end = len(s.feed)
		}
		page = append(page, s.feed[offset:end]...)
	}
	hasMore := len(s.feed) > offset+limit || !s.exhausted
	return page, hasMore, nil
}

// InvalidateProfile drops the cached taste profile so the next page rebuilds
// it immediately (e.g. after a bulk tag import). Normal like flow doesn't
// need this — the signature check picks changes up on its own.
func (e *Engine) InvalidateProfile() {
	e.mu.Lock()
	e.profile = nil
	e.mu.Unlock()
}

// ---------------------------------------------------------------------------
// Profile
// ---------------------------------------------------------------------------

// freshProfile returns the cached profile, rebuilding it when stale.
func (e *Engine) freshProfile(ctx context.Context, t Tuning) (*profile, error) {
	now := e.now()
	p := e.profile
	if p != nil && p.model == e.model() && now.Sub(p.builtAt) < time.Duration(t.ProfileTTLSeconds)*time.Second {
		// Cheap eventual-consistency check: has the favorites set moved?
		if now.Sub(p.sigCheckedAt) < time.Duration(t.LikeCheckSeconds)*time.Second {
			return p, nil
		}
		sig, err := e.likeSignature(ctx, t)
		if err != nil {
			return nil, err
		}
		p.sigCheckedAt = now
		if sig == p.signature {
			return p, nil
		}
	}
	return e.buildProfile(ctx, t)
}

func (e *Engine) likeSignature(ctx context.Context, t Tuning) (string, error) {
	var n int
	var maxAt int64
	err := e.db.QueryRowContext(ctx,
		`SELECT COUNT(*), COALESCE(MAX(created_at), 0) FROM media_tag_by_category
		 WHERE tag_label = ? AND category_label = ?`,
		t.FavoritesTag, t.FavoritesCategory,
	).Scan(&n, &maxAt)
	if err != nil {
		return "", fmt.Errorf("feed: like signature: %w", err)
	}
	return fmt.Sprintf("%d|%d", n, maxAt), nil
}

// recentLikes returns liked paths, most recent first, deduped.
func (e *Engine) recentLikes(ctx context.Context, t Tuning) ([]string, error) {
	rows, err := e.db.QueryContext(ctx,
		`SELECT media_path, MAX(COALESCE(created_at, 0)) AS ts
		 FROM media_tag_by_category
		 WHERE tag_label = ? AND category_label = ?
		 GROUP BY media_path
		 ORDER BY ts DESC
		 LIMIT ?`,
		t.FavoritesTag, t.FavoritesCategory, t.MaxLikes,
	)
	if err != nil {
		return nil, fmt.Errorf("feed: load likes: %w", err)
	}
	defer rows.Close()
	var paths []string
	for rows.Next() {
		var p string
		var ts int64
		if err := rows.Scan(&p, &ts); err != nil {
			return nil, err
		}
		paths = append(paths, p)
	}
	return paths, rows.Err()
}

// buildProfile clusters the recent likes into weighted taste centers.
// Greedy online pass in recency order: each like joins its nearest cluster
// when cosine ≥ ClusterThreshold, otherwise founds a new one (until
// MaxClusters, then it folds into the nearest regardless). Like weight
// decays by half every RecencyHalfLife likes so the profile tracks current
// taste. Neighbor pools are filled lazily by the lanes.
func (e *Engine) buildProfile(ctx context.Context, t Tuning) (*profile, error) {
	now := e.now()
	sig, err := e.likeSignature(ctx, t)
	if err != nil {
		return nil, err
	}
	likes, err := e.recentLikes(ctx, t)
	if err != nil {
		return nil, err
	}

	model := e.model()
	p := &profile{
		model:        model,
		builtAt:      now,
		signature:    sig,
		sigCheckedAt: now,
		liked:        make(map[string]bool, len(likes)),
		bridgePools:  make(map[[2]int][]string),
	}
	for _, l := range likes {
		p.liked[l] = true
	}

	vecs, err := media.GetEmbeddingsForPaths(e.db, model, likes)
	if err != nil {
		return nil, fmt.Errorf("feed: like embeddings: %w", err)
	}

	for i, path := range likes {
		v, ok := vecs[path]
		if !ok {
			continue // not embedded yet; it still counts as liked (excluded from feed)
		}
		v = embedvec.Normalize(v)
		if p.fresh == nil {
			p.fresh = newCluster(v, 1)
		}
		w := math.Pow(0.5, float64(i)/float64(t.RecencyHalfLife))

		best, bestSim := -1, float32(-2)
		for ci, c := range p.clusters {
			if sim := embedvec.Cosine(c.centroid, v); sim > bestSim {
				best, bestSim = ci, sim
			}
		}
		if best >= 0 && (float64(bestSim) >= t.ClusterThreshold || len(p.clusters) >= t.MaxClusters) {
			p.clusters[best].add(v, w)
		} else {
			p.clusters = append(p.clusters, newCluster(v, w))
		}
	}

	// Heaviest first so weighted sampling and bridge pairs favor the
	// dominant tastes deterministically.
	sort.SliceStable(p.clusters, func(i, j int) bool { return p.clusters[i].weight > p.clusters[j].weight })

	e.profile = p
	return p, nil
}

func newCluster(v []float32, w float64) *cluster {
	sum := make([]float32, len(v))
	for i, x := range v {
		sum[i] = float32(w) * x
	}
	return &cluster{sum: sum, centroid: embedvec.Normalize(sum), weight: w}
}

func (c *cluster) add(v []float32, w float64) {
	for i := range c.sum {
		c.sum[i] += float32(w) * v[i]
	}
	c.centroid = embedvec.Normalize(c.sum)
	c.weight += w
	c.pool, c.poolOK = nil, false // stale; rebuilt lazily (only relevant mid-build)
}

// ---------------------------------------------------------------------------
// Sessions
// ---------------------------------------------------------------------------

func (e *Engine) session(id string, t Tuning) *session {
	now := e.now()

	// Evict idle sessions; oldest-idle first when over the cap.
	ttl := time.Duration(t.SessionTTLMinutes) * time.Minute
	for k, s := range e.sessions {
		if k != id && now.Sub(s.lastAccess) > ttl {
			delete(e.sessions, k)
		}
	}
	if len(e.sessions) >= t.MaxSessions {
		oldest, oldestAt := "", now
		for k, s := range e.sessions {
			if k != id && s.lastAccess.Before(oldestAt) {
				oldest, oldestAt = k, s.lastAccess
			}
		}
		if oldest != "" {
			delete(e.sessions, oldest)
		}
	}

	s, ok := e.sessions[id]
	if !ok {
		h := fnv.New64a()
		h.Write([]byte(id))
		s = &session{
			rng:  rand.New(rand.NewSource(int64(h.Sum64()))),
			seen: make(map[string]bool),
		}
		e.sessions[id] = s
	}
	s.lastAccess = now
	return s
}

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------

// generate produces one batch: lane candidates in tuned proportions,
// shuffled together, orphan-filtered, appended to the session feed. Returns
// how many items were actually appended.
func (e *Engine) generate(ctx context.Context, s *session, p *profile, t Tuning, b *pageBudget) (int, error) {
	weights := []float64{t.ExploitWeight, t.FreshWeight, t.BridgeWeight, t.WildcardWeight}
	// Lanes without data forfeit their share.
	if len(p.clusters) == 0 {
		weights[0], weights[2] = 0, 0
	}
	if p.fresh == nil {
		weights[1] = 0
	}
	var total float64
	for _, w := range weights {
		total += w
	}
	if total <= 0 {
		// No taste data at all: the whole batch is wildcard so the feed
		// still works from a cold start.
		weights = []float64{0, 0, 0, 1}
		total = 1
	}

	counts := apportion(t.BatchSize, weights, total)
	var picked []string
	picked = append(picked, e.drawExploit(s, p, t, b, counts[0])...)
	picked = append(picked, e.drawFresh(s, p, t, b, counts[1])...)
	bridged, err := e.drawBridge(s, p, t, b, counts[2])
	if err != nil {
		return 0, err
	}
	picked = append(picked, bridged...)

	if err := ctx.Err(); err != nil {
		return 0, err
	}

	// Wildcard also absorbs any shortfall from starved lanes, keeping
	// batches full as taste pools drain (including lanes waiting on a
	// search budget that ran out this page).
	wildWant := counts[3]
	if short := t.BatchSize - len(picked); short > wildWant {
		wildWant = short
	}
	wild, err := e.drawWildcard(ctx, s, p, t, wildWant)
	if err != nil {
		return 0, err
	}
	picked = append(picked, wild...)

	// Interleave lanes so exploitation and surprise alternate organically
	// instead of arriving in blocks.
	s.rng.Shuffle(len(picked), func(i, j int) { picked[i], picked[j] = picked[j], picked[i] })

	existing, err := media.FilterExistingMediaPaths(e.db, picked)
	if err != nil {
		return 0, fmt.Errorf("feed: orphan filter: %w", err)
	}
	s.feed = append(s.feed, existing...)
	return len(existing), nil
}

// apportion splits n into integer lane counts proportional to weights
// (largest remainder method).
func apportion(n int, weights []float64, total float64) []int {
	counts := make([]int, len(weights))
	type rem struct {
		i int
		f float64
	}
	var rems []rem
	used := 0
	for i, w := range weights {
		exact := float64(n) * w / total
		counts[i] = int(exact)
		used += counts[i]
		rems = append(rems, rem{i, exact - float64(counts[i])})
	}
	sort.SliceStable(rems, func(a, b int) bool { return rems[a].f > rems[b].f })
	for k := 0; used < n && k < len(rems); k++ {
		if weights[rems[k].i] > 0 {
			counts[rems[k].i]++
			used++
		}
	}
	return counts
}

// take marks path as dealt and reports whether it should enter the feed.
func (e *Engine) take(s *session, p *profile, t Tuning, path string) bool {
	if s.seen[path] {
		return false
	}
	s.seen[path] = true
	if !t.IncludeLiked && p.liked[path] {
		return false
	}
	return true
}

// pool returns c's cached ranked neighbor list. The first use costs one
// vector search — a full-library scan — so it only fires while the page
// budget allows; otherwise ok=false and the caller falls back (the pool
// gets built by a later page).
func (e *Engine) pool(p *profile, t Tuning, b *pageBudget, c *cluster) ([]string, bool) {
	if !c.poolOK {
		if b.searches <= 0 {
			return nil, false
		}
		b.searches--
		paths, err := e.search(p.model, c.centroid, t.PoolSize)
		if err != nil {
			log.Printf("feed: cluster pool search failed: %v", err)
			return nil, false
		}
		c.pool, c.poolOK = paths, true
	}
	return c.pool, true
}

// drawExploit deals up to n unseen items from the tops of the taste-cluster
// pools, sampling clusters proportional to their like weight with a small
// positional jitter so the very same heads don't dominate every batch.
func (e *Engine) drawExploit(s *session, p *profile, t Tuning, b *pageBudget, n int) []string {
	if n <= 0 || len(p.clusters) == 0 {
		return nil
	}
	var out []string
	misses := 0
	for len(out) < n && misses < len(p.clusters)*2 {
		c := p.clusters[weightedPick(s.rng, p.clusters)]
		pool, ok := e.pool(p, t, b, c)
		if !ok || len(pool) == 0 {
			misses++
			continue
		}
		// Occasionally start a little deeper for variety.
		start := 0
		if s.rng.Float64() < 0.3 {
			start = s.rng.Intn(len(pool)/4 + 1)
		}
		found := false
		for _, path := range pool[start:] {
			if e.take(s, p, t, path) {
				out = append(out, path)
				found = true
				break
			}
		}
		if !found {
			misses++
		}
	}
	return out
}

// drawFresh deals up to n unseen items near the most recent like.
func (e *Engine) drawFresh(s *session, p *profile, t Tuning, b *pageBudget, n int) []string {
	if n <= 0 || p.fresh == nil {
		return nil
	}
	pool, ok := e.pool(p, t, b, p.fresh)
	if !ok {
		return nil
	}
	var out []string
	for _, path := range pool {
		if len(out) >= n {
			break
		}
		if e.take(s, p, t, path) {
			out = append(out, path)
		}
	}
	return out
}

// drawBridge is the surprise lane. With two or more taste clusters it
// searches the MIDPOINT between two of them — items that connect separate
// interests — sampling softly down the ranking rather than taking the very
// top. With a single cluster it samples that cluster's deep band
// (rank ≥ DeepBandStart·pool): familiar but past the obvious picks.
func (e *Engine) drawBridge(s *session, p *profile, t Tuning, b *pageBudget, n int) ([]string, error) {
	if n <= 0 || len(p.clusters) == 0 {
		return nil, nil
	}
	var out []string
	attempts := 0
	for len(out) < n && attempts < n*3 {
		attempts++
		var pool []string
		if len(p.clusters) >= 2 && s.rng.Float64() < 0.6 {
			i := weightedPick(s.rng, p.clusters)
			j := weightedPick(s.rng, p.clusters)
			for j == i {
				j = weightedPick(s.rng, p.clusters)
			}
			if i > j {
				i, j = j, i
			}
			key := [2]int{i, j}
			cached, ok := p.bridgePools[key]
			if !ok && b.searches > 0 {
				mid, err := embedvec.Combine(
					[][]float32{p.clusters[i].centroid, p.clusters[j].centroid},
					[]float32{1, 1},
				)
				if err != nil {
					continue
				}
				b.searches--
				cached, err = e.search(p.model, mid, t.PoolSize/2)
				if err != nil {
					return nil, fmt.Errorf("feed: bridge pool: %w", err)
				}
				p.bridgePools[key] = cached
				ok = true
			}
			if !ok {
				// No budget for a new midpoint this page: reuse any
				// already-built bridge pool instead of searching.
				for _, existing := range p.bridgePools {
					cached = existing
					ok = true
					break
				}
			}
			if ok {
				pool = cached
			}
		}
		if pool == nil {
			// Deep band of an already-built cluster pool: familiar but past
			// the obvious picks. Never spends budget — pool() only returns
			// built pools once the budget is gone, and exploit runs first.
			c := p.clusters[weightedPick(s.rng, p.clusters)]
			full, ok := e.pool(p, t, b, c)
			if !ok || len(full) == 0 {
				continue
			}
			start := int(float64(len(full)) * t.DeepBandStart)
			if start >= len(full) {
				start = len(full) - 1
			}
			pool = full[start:]
		}
		if len(pool) == 0 {
			continue
		}
		// Soft sample: random start, then first unseen from there.
		start := s.rng.Intn(len(pool))
		for _, path := range pool[start:] {
			if e.take(s, p, t, path) {
				out = append(out, path)
				break
			}
		}
	}
	return out, nil
}

// drawWildcard deals up to n uniform random library items via rowid probes —
// O(log n) each, no full-table ORDER BY RANDOM() scan, so it stays cheap on
// multi-million-row libraries.
func (e *Engine) drawWildcard(ctx context.Context, s *session, p *profile, t Tuning, n int) ([]string, error) {
	if n <= 0 {
		return nil, nil
	}
	// MIN and MAX are queried separately on purpose: SQLite's min/max
	// optimization (O(1) btree-end lookup) only applies to a lone
	// aggregate — combined "SELECT MIN(rowid), MAX(rowid)" full-scans the
	// table, which is seconds on a multi-million-row library.
	var minID, maxID sql.NullInt64
	if err := e.db.QueryRowContext(ctx, `SELECT MIN(rowid) FROM media`).Scan(&minID); err != nil {
		return nil, fmt.Errorf("feed: wildcard bounds: %w", err)
	}
	if err := e.db.QueryRowContext(ctx, `SELECT MAX(rowid) FROM media`).Scan(&maxID); err != nil {
		return nil, fmt.Errorf("feed: wildcard bounds: %w", err)
	}
	if !minID.Valid || !maxID.Valid {
		return nil, nil // empty library
	}
	span := maxID.Int64 - minID.Int64 + 1
	var out []string
	for probes := 0; len(out) < n && probes < n*6; probes++ {
		if ctx.Err() != nil {
			break
		}
		target := minID.Int64 + s.rng.Int63n(span)
		var path string
		err := e.db.QueryRowContext(ctx,
			`SELECT path FROM media WHERE rowid >= ? ORDER BY rowid LIMIT 1`, target,
		).Scan(&path)
		if err == sql.ErrNoRows {
			continue
		}
		if err != nil {
			return nil, fmt.Errorf("feed: wildcard probe: %w", err)
		}
		if e.take(s, p, t, path) {
			out = append(out, path)
		}
	}
	return out, nil
}

// weightedPick returns a cluster index sampled proportional to like weight.
func weightedPick(rng *rand.Rand, clusters []*cluster) int {
	var total float64
	for _, c := range clusters {
		total += c.weight
	}
	if total <= 0 {
		return rng.Intn(len(clusters))
	}
	x := rng.Float64() * total
	for i, c := range clusters {
		x -= c.weight
		if x <= 0 {
			return i
		}
	}
	return len(clusters) - 1
}
