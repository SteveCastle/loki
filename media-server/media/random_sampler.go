package media

import (
	"database/sql"
	"log"
	"math/rand"
	"strings"
	"sync"
	"time"

	"github.com/stevecastle/shrike/querylog"
)

// randomSampler caches the set of media paths that have at least one tag —
// the "swipeable" universe — so the swipe view can pull random pages without
// running ORDER BY RANDOM() over the whole table on every request.
//
// Build cost (one DISTINCT-scan over media_tag_by_category) is amortised
// across an entire session and the cache is invalidated explicitly on tag
// mutations, media deletion, and DB switch. A TTL provides a safety net in
// case any mutation site forgets to invalidate.
//
// Per-request cost is dominated by a single SELECT … WHERE path IN (?, ?, …)
// against the media PK — point lookups, sub-millisecond for typical page
// sizes (1–50).
type randomSampler struct {
	mu sync.Mutex

	// Universe: all distinct media_paths with at least one tag.
	paths   []string
	builtAt time.Time
	stale   bool

	// Single-flight build coordination. When non-nil, a build is in flight
	// and will close the channel on completion. Concurrent callers that
	// have a usable (even if stale) cache return immediately; callers that
	// have no cache wait on this channel.
	buildCh chan struct{}

	// Monotonic counter bumped by every Invalidate. The build snapshots
	// this at start; if it differs at end, the build is treated as
	// already-stale (a mutation arrived during the rebuild) and `stale` is
	// kept true so the next call schedules another build.
	invalidations uint64

	// One cached shuffle. The swipe client uses `offset` to paginate within
	// a single session, so a deterministic per-seed shuffle gives stable
	// pagination (no dupes across pages) without storing one shuffle per
	// caller.
	shuffleSeed   int64
	shuffledPaths []string
}

var defaultSampler = &randomSampler{}

// Cache lives at most this long even if no explicit invalidation arrives.
// Tag mutations call Invalidate directly; this is just a safety net.
const samplerTTL = 30 * time.Minute

// ensureBuilt makes sure a usable `paths` snapshot exists. Stale-while-
// revalidate: if any cache exists (even past TTL or marked stale by a
// recent mutation), this returns immediately and a background goroutine
// rebuilds. Only the very first call (no cache at all) blocks. This is
// what keeps the swipe view responsive after a tag like — without it, the
// next swipe request after AddTag has to scan the whole tag table while
// holding the mutex, freezing every concurrent request for seconds.
func (s *randomSampler) ensureBuilt(db *sql.DB) error {
	s.mu.Lock()
	fresh := s.paths != nil && !s.stale && time.Since(s.builtAt) < samplerTTL
	if fresh {
		s.mu.Unlock()
		return nil
	}

	// Build needed. If one is already in flight, ride along.
	if s.buildCh != nil {
		if s.paths != nil {
			// We have something to serve — don't wait.
			s.mu.Unlock()
			return nil
		}
		// No cache at all yet; wait for the in-flight build.
		ch := s.buildCh
		s.mu.Unlock()
		<-ch
		return nil
	}

	// Start a new build. Single-flight: any concurrent caller landing here
	// will see buildCh != nil and take the branches above.
	ch := make(chan struct{})
	s.buildCh = ch
	haveCache := s.paths != nil
	s.mu.Unlock()

	if haveCache {
		// Background rebuild — return immediately with the stale snapshot.
		go s.runBuild(db, ch)
		return nil
	}
	// First-ever build: must block, callers can't render anything yet.
	s.runBuild(db, ch)
	return nil
}

// runBuild performs the SQL scan and atomically swaps in the new paths.
// Must be called with s.buildCh set to `ch`. Closes `ch` on return.
func (s *randomSampler) runBuild(db *sql.DB, ch chan struct{}) {
	defer close(ch)

	s.mu.Lock()
	startInval := s.invalidations
	s.mu.Unlock()

	paths, err := s.queryPaths(db)
	if err != nil {
		log.Printf("[randomSampler] build failed: %v", err)
		s.mu.Lock()
		s.buildCh = nil
		s.mu.Unlock()
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	s.buildCh = nil
	s.paths = paths
	s.builtAt = time.Now()
	// If a mutation arrived during the rebuild, the snapshot we just took
	// is already out of date. Leave `stale` true so the next ensureBuilt
	// call kicks off another rebuild. Without this we'd swallow mid-build
	// invalidations.
	s.stale = s.invalidations != startInval
	// Underlying universe changed — drop any cached shuffle.
	s.shuffledPaths = nil
	s.shuffleSeed = 0
}

// queryPaths runs the SELECT DISTINCT scan without touching sampler state,
// so it can run outside the mutex. Slow on large databases — that's
// exactly why ensureBuilt no longer holds the lock around it.
func (s *randomSampler) queryPaths(db *sql.DB) ([]string, error) {
	const q = `SELECT DISTINCT media_path FROM media_tag_by_category`
	stop := querylog.Start("randomSampler.build", q, nil)
	rows, err := db.Query(q)
	if err != nil {
		stop(-1, err)
		return nil, err
	}
	defer rows.Close()

	paths := make([]string, 0, 65536)
	for rows.Next() {
		var p string
		if err := rows.Scan(&p); err != nil {
			stop(len(paths), err)
			return nil, err
		}
		paths = append(paths, p)
	}
	if err := rows.Err(); err != nil {
		stop(len(paths), err)
		return nil, err
	}
	stop(len(paths), nil)
	return paths, nil
}

// Invalidate marks the cache stale; the next sampler call schedules a
// background rebuild. Cheap (counter bump + flag set). Safe to call from
// mutation hot paths and from concurrent callers.
func (s *randomSampler) Invalidate() {
	s.mu.Lock()
	s.stale = true
	s.invalidations++
	s.mu.Unlock()
}

// sample returns a window into the per-seed shuffle. The returned slice is a
// fresh copy so the caller can hold it without blocking other requests.
func (s *randomSampler) sample(seed int64, offset, limit int) (paths []string, total int) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(s.paths) == 0 {
		return nil, 0
	}

	// (Re)shuffle if the requesting seed differs from the cached one.
	// seed=0 means "give me fresh randomness" — use wallclock as the source
	// so each session sees a different ordering.
	effectiveSeed := seed
	if effectiveSeed == 0 {
		effectiveSeed = time.Now().UnixNano()
	}
	if s.shuffledPaths == nil || s.shuffleSeed != effectiveSeed {
		s.shuffledPaths = make([]string, len(s.paths))
		copy(s.shuffledPaths, s.paths)
		rng := rand.New(rand.NewSource(effectiveSeed))
		rng.Shuffle(len(s.shuffledPaths), func(i, j int) {
			s.shuffledPaths[i], s.shuffledPaths[j] = s.shuffledPaths[j], s.shuffledPaths[i]
		})
		s.shuffleSeed = effectiveSeed
	}

	if offset >= len(s.shuffledPaths) {
		return nil, len(s.shuffledPaths)
	}
	end := offset + limit
	if end > len(s.shuffledPaths) {
		end = len(s.shuffledPaths)
	}
	out := make([]string, end-offset)
	copy(out, s.shuffledPaths[offset:end])
	return out, len(s.shuffledPaths)
}

// InvalidateRandomSampleCache marks the random-sampler cache stale. Call
// after any operation that changes the set of paths-with-tags: AddTag,
// RemoveTag, media deletion, or a DB swap.
func InvalidateRandomSampleCache() {
	defaultSampler.Invalidate()
}

// WarmRandomSampleCache builds the cache asynchronously so the first user
// request to /swipe/api doesn't pay the build cost. Safe to call multiple
// times — the sampler dedupes via its mutex.
func WarmRandomSampleCache(db *sql.DB) {
	go func() {
		if err := defaultSampler.ensureBuilt(db); err != nil {
			log.Printf("[randomSampler] background warm failed: %v", err)
		}
	}()
}

// getRandomItemsFromSampler is the fast path used when no searchQuery is
// active (the dominant swipe case). It samples N random paths from the
// in-memory cache and fetches their full rows via PK point lookups.
func getRandomItemsFromSampler(db *sql.DB, offset, limit int, seed int64) ([]MediaItem, bool, error) {
	if err := defaultSampler.ensureBuilt(db); err != nil {
		return nil, false, err
	}

	// Ask for one extra so we can compute hasMore without an extra query.
	pickN := limit + 1
	picked, total := defaultSampler.sample(seed, offset, pickN)
	if total == 0 || len(picked) == 0 {
		return nil, false, nil
	}

	hasMore := len(picked) > limit
	if hasMore {
		picked = picked[:limit]
	}

	// Fetch full rows for the chosen paths. The IN-list with the media PK
	// becomes N point lookups — fast even for limit=50.
	placeholders := strings.Repeat("?,", len(picked))
	placeholders = placeholders[:len(placeholders)-1]
	query := `SELECT m.path, m.description, m.size, m.hash, m.width, m.height ` +
		`FROM media m WHERE m.path IN (` + placeholders + `)`
	args := make([]interface{}, len(picked))
	for i, p := range picked {
		args[i] = p
	}

	stop := querylog.Start("GetRandomItems.cached", query, args)
	rows, err := db.Query(query, args...)
	if err != nil {
		stop(-1, err)
		return nil, false, err
	}
	defer rows.Close()

	// SQLite's IN-list returns rows in PK order, not the sampler's random
	// order. Re-sort so the client sees the items in the order the seed
	// produced — needed for stable offset-based pagination across calls.
	byPath := make(map[string]MediaItem, len(picked))
	rowCount := 0
	for rows.Next() {
		var item MediaItem
		if err := rows.Scan(&item.Path, &item.Description, &item.Size, &item.Hash, &item.Width, &item.Height); err != nil {
			stop(rowCount, err)
			return nil, false, err
		}
		rowCount++
		if item.Size.Valid {
			item.FormattedSize = FormatBytes(item.Size.Int64)
		} else {
			item.FormattedSize = "Unknown"
		}
		byPath[item.Path] = item
	}
	stop(rowCount, nil)

	items := make([]MediaItem, 0, len(picked))
	mediaPaths := make([]string, 0, len(picked))
	for _, p := range picked {
		if it, ok := byPath[p]; ok {
			items = append(items, it)
			mediaPaths = append(mediaPaths, p)
		}
	}

	// Tags. Mirrors the original GetRandomItems behaviour: skip the lookup
	// for limit==1 single-item fast loads.
	if limit > 1 || len(items) > 1 {
		tagMap, err := GetTags(db, mediaPaths)
		if err != nil {
			log.Printf("Error fetching media tags: %v", err)
		} else {
			for i := range items {
				if tags, exists := tagMap[items[i].Path]; exists {
					items[i].Tags = tags
				} else {
					items[i].Tags = []MediaTag{}
				}
			}
		}
	} else {
		for i := range items {
			items[i].Tags = []MediaTag{}
		}
	}

	// File-existence check happens concurrently in CheckFilesExistConcurrent.
	existenceMap := CheckFilesExistConcurrent(mediaPaths)
	for i := range items {
		if exists, found := existenceMap[items[i].Path]; found {
			items[i].Exists = exists
		} else {
			items[i].Exists = false
		}
	}

	return items, hasMore, nil
}
