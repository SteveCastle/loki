package feed

import (
	"context"
	"database/sql"
	"fmt"
	"sort"
	"testing"
	"time"

	"github.com/stevecastle/shrike/embedvec"
	"github.com/stevecastle/shrike/media"
	_ "modernc.org/sqlite"
)

const testModel = "test-model"

// fakeLibrary is an in-memory library + brute-force vector search the engine
// runs against in tests.
type fakeLibrary struct {
	db   *sql.DB
	vecs map[string][]float32
}

func newFakeLibrary(t *testing.T) *fakeLibrary {
	t.Helper()
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	if err := media.InitializeSchema(db); err != nil {
		t.Fatal(err)
	}
	return &fakeLibrary{db: db, vecs: map[string][]float32{}}
}

func (l *fakeLibrary) addItem(t *testing.T, path string, vec []float32) {
	t.Helper()
	if _, err := l.db.Exec(`INSERT INTO media (path) VALUES (?)`, path); err != nil {
		t.Fatal(err)
	}
	if vec != nil {
		l.vecs[path] = vec
		if err := media.UpsertEmbedding(l.db, path, testModel, vec, 0); err != nil {
			t.Fatal(err)
		}
	}
}

func (l *fakeLibrary) like(t *testing.T, path string, createdAt int64) {
	t.Helper()
	if _, err := l.db.Exec(
		`INSERT INTO media_tag_by_category (media_path, tag_label, category_label, weight, time_stamp, created_at)
		 VALUES (?, 'Favorites', 'Swipe', 0, 0, ?)`,
		path, createdAt,
	); err != nil {
		t.Fatal(err)
	}
}

func (l *fakeLibrary) search(model string, query []float32, limit int) ([]string, error) {
	if model != testModel {
		return nil, fmt.Errorf("unexpected model %q", model)
	}
	type hit struct {
		path  string
		score float32
	}
	var hits []hit
	for p, v := range l.vecs {
		hits = append(hits, hit{p, embedvec.CosineSim(query, v)})
	}
	sort.SliceStable(hits, func(i, j int) bool {
		if hits[i].score != hits[j].score {
			return hits[i].score > hits[j].score
		}
		return hits[i].path < hits[j].path
	})
	if limit > 0 && len(hits) > limit {
		hits = hits[:limit]
	}
	out := make([]string, len(hits))
	for i, h := range hits {
		out[i] = h.path
	}
	return out, nil
}

func newTestEngine(l *fakeLibrary, t Tuning) *Engine {
	return NewEngine(l.db, func() string { return testModel }, l.search, func() Tuning { return t })
}

// seedTwoTastes builds a library with two well-separated taste regions (A
// near x-axis, B near y-axis) plus off-taste items near z, and likes in both
// regions.
func seedTwoTastes(t *testing.T, l *fakeLibrary) {
	t.Helper()
	for i := 0; i < 20; i++ {
		f := float32(i) * 0.01
		l.addItem(t, fmt.Sprintf("/a/%d.jpg", i), embedvec.Normalize([]float32{1, f, 0}))
		l.addItem(t, fmt.Sprintf("/b/%d.jpg", i), embedvec.Normalize([]float32{f, 1, 0}))
		l.addItem(t, fmt.Sprintf("/z/%d.jpg", i), embedvec.Normalize([]float32{0, f, 1}))
	}
	l.like(t, "/a/0.jpg", 100)
	l.like(t, "/a/1.jpg", 101)
	l.like(t, "/b/0.jpg", 102)
	l.like(t, "/b/1.jpg", 103)
}

func TestPageComposesAndNeverRepeats(t *testing.T) {
	l := newFakeLibrary(t)
	seedTwoTastes(t, l)
	e := newTestEngine(l, Tuning{BatchSize: 10})

	seen := map[string]bool{}
	var all []string
	for offset := 0; offset < 40; offset += 10 {
		page, _, err := e.Page(context.Background(), "s1", offset, 10, nil)
		if err != nil {
			t.Fatal(err)
		}
		for _, p := range page {
			if seen[p] {
				t.Fatalf("path %q repeated in feed", p)
			}
			seen[p] = true
			all = append(all, p)
		}
	}
	if len(all) == 0 {
		t.Fatal("feed produced nothing")
	}
}

func TestPageIsDeterministicPerSession(t *testing.T) {
	l := newFakeLibrary(t)
	seedTwoTastes(t, l)

	page1, _, err := newTestEngine(l, Tuning{BatchSize: 12}).Page(context.Background(), "same-session", 0, 12, nil)
	if err != nil {
		t.Fatal(err)
	}
	page2, _, err := newTestEngine(l, Tuning{BatchSize: 12}).Page(context.Background(), "same-session", 0, 12, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(page1) != len(page2) {
		t.Fatalf("lengths differ: %d vs %d", len(page1), len(page2))
	}
	for i := range page1 {
		if page1[i] != page2[i] {
			t.Fatalf("pages diverge at %d: %q vs %q", i, page1[i], page2[i])
		}
	}
}

func TestLikedItemsExcludedByDefault(t *testing.T) {
	l := newFakeLibrary(t)
	seedTwoTastes(t, l)
	e := newTestEngine(l, Tuning{BatchSize: 30})

	page, _, err := e.Page(context.Background(), "s", 0, 30, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range page {
		switch p {
		case "/a/0.jpg", "/a/1.jpg", "/b/0.jpg", "/b/1.jpg":
			t.Fatalf("liked item %q appeared in feed", p)
		}
	}
}

func TestFeedCoversBothTastesAndWildcards(t *testing.T) {
	l := newFakeLibrary(t)
	seedTwoTastes(t, l)
	e := newTestEngine(l, Tuning{BatchSize: 40})

	page, _, err := e.Page(context.Background(), "s", 0, 40, nil)
	if err != nil {
		t.Fatal(err)
	}
	var a, b, z int
	for _, p := range page {
		switch p[1] {
		case 'a':
			a++
		case 'b':
			b++
		case 'z':
			z++
		}
	}
	if a == 0 || b == 0 {
		t.Fatalf("feed missed a taste cluster: a=%d b=%d", a, b)
	}
	if z == 0 {
		t.Fatalf("feed has no out-of-taste items (wildcard/bridge dead): a=%d b=%d z=%d", a, b, z)
	}
}

func TestOrphanEmbeddingsFilteredOut(t *testing.T) {
	l := newFakeLibrary(t)
	seedTwoTastes(t, l)
	// Orphan: embedded, extremely close to taste A, but no media row.
	l.vecs["/a/orphan.jpg"] = embedvec.Normalize([]float32{1, 0.005, 0})
	if err := media.UpsertEmbedding(l.db, "/a/orphan.jpg", testModel, l.vecs["/a/orphan.jpg"], 0); err != nil {
		t.Fatal(err)
	}
	e := newTestEngine(l, Tuning{BatchSize: 40})
	page, _, err := e.Page(context.Background(), "s", 0, 40, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range page {
		if p == "/a/orphan.jpg" {
			t.Fatal("orphan path leaked into the feed")
		}
	}
}

func TestColdStartFallsBackToWildcard(t *testing.T) {
	l := newFakeLibrary(t)
	for i := 0; i < 30; i++ {
		l.addItem(t, fmt.Sprintf("/m/%d.jpg", i), nil) // no embeddings, no likes
	}
	e := newTestEngine(l, Tuning{BatchSize: 10})
	page, hasMore, err := e.Page(context.Background(), "s", 0, 10, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(page) != 10 {
		t.Fatalf("cold-start feed returned %d items, want 10", len(page))
	}
	if !hasMore {
		t.Fatal("expected hasMore on a cold-start feed with items remaining")
	}
}

func TestExhaustionEndsFeed(t *testing.T) {
	l := newFakeLibrary(t)
	for i := 0; i < 5; i++ {
		l.addItem(t, fmt.Sprintf("/m/%d.jpg", i), nil)
	}
	e := newTestEngine(l, Tuning{BatchSize: 10})
	page, _, err := e.Page(context.Background(), "s", 0, 10, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(page) != 5 {
		t.Fatalf("got %d items, want the whole 5-item library", len(page))
	}
	page2, hasMore, err := e.Page(context.Background(), "s", 5, 10, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(page2) != 0 || hasMore {
		t.Fatalf("expected exhausted feed: items=%d hasMore=%v", len(page2), hasMore)
	}
}

func TestNewLikeRebuildsProfileWithinCheckWindow(t *testing.T) {
	l := newFakeLibrary(t)
	for i := 0; i < 20; i++ {
		f := float32(i) * 0.01
		l.addItem(t, fmt.Sprintf("/a/%d.jpg", i), embedvec.Normalize([]float32{1, f, 0}))
		l.addItem(t, fmt.Sprintf("/z/%d.jpg", i), embedvec.Normalize([]float32{0, f, 1}))
	}
	l.like(t, "/a/0.jpg", 100)

	e := newTestEngine(l, Tuning{BatchSize: 10, WildcardWeight: 0.0001, ExploitWeight: 1})
	clock := time.Unix(1000, 0)
	e.now = func() time.Time { return clock }

	if _, _, err := e.Page(context.Background(), "s", 0, 10, nil); err != nil {
		t.Fatal(err)
	}
	firstSig := e.profile.signature

	// A new like lands; within LikeCheckSeconds the cached profile is
	// still served, after it the signature check forces a rebuild.
	l.like(t, "/z/0.jpg", 200)
	if _, _, err := e.Page(context.Background(), "s", 10, 10, nil); err != nil {
		t.Fatal(err)
	}
	if e.profile.signature != firstSig {
		t.Fatal("profile rebuilt inside the like-check window; expected cached")
	}

	clock = clock.Add(10 * time.Second) // past LikeCheckSeconds (default 5)
	if _, _, err := e.Page(context.Background(), "s", 20, 10, nil); err != nil {
		t.Fatal(err)
	}
	if e.profile.signature == firstSig {
		t.Fatal("profile not rebuilt after favorites changed")
	}
	if !e.profile.liked["/z/0.jpg"] {
		t.Fatal("rebuilt profile missing the new like")
	}
}

func TestQueryOverrideChangesLaneMix(t *testing.T) {
	l := newFakeLibrary(t)
	seedTwoTastes(t, l)
	e := newTestEngine(l, Tuning{BatchSize: 20})

	// Force a pure-wildcard feed via the per-request override hook.
	page, _, err := e.Page(context.Background(), "s", 0, 20, func(t *Tuning) {
		t.ExploitWeight, t.FreshWeight, t.BridgeWeight, t.WildcardWeight = 0.000001, 0.000001, 0.000001, 1
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(page) == 0 {
		t.Fatal("override feed produced nothing")
	}
}

func TestSearchBudgetBoundsPerPageCost(t *testing.T) {
	l := newFakeLibrary(t)
	seedTwoTastes(t, l)

	searches := 0
	counting := func(model string, q []float32, limit int) ([]string, error) {
		searches++
		return l.search(model, q, limit)
	}
	tun := Tuning{BatchSize: 20, MaxSearchesPerPage: 2}
	e := NewEngine(l.db, func() string { return testModel }, counting, func() Tuning { return tun })

	page, _, err := e.Page(context.Background(), "s", 0, 20, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(page) == 0 {
		t.Fatal("budgeted page produced nothing")
	}
	if searches > 2 {
		t.Fatalf("page ran %d searches, budget is 2", searches)
	}

	// Later pages get their own budget, warming up more pools over time.
	if _, _, err := e.Page(context.Background(), "s", 20, 20, nil); err != nil {
		t.Fatal(err)
	}
	if searches > 4 {
		t.Fatalf("two pages ran %d searches, budget is 2 each", searches)
	}
}

func TestPageStopsOnCanceledContext(t *testing.T) {
	l := newFakeLibrary(t)
	seedTwoTastes(t, l)
	e := newTestEngine(l, Tuning{BatchSize: 10})

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, _, err := e.Page(ctx, "s", 0, 10, nil); err == nil {
		t.Fatal("expected an error from a canceled context")
	}
}

func TestTuningDefaults(t *testing.T) {
	d := Tuning{}.WithDefaults()
	if d.ExploitWeight != 0.5 || d.WildcardWeight != 0.15 {
		t.Fatalf("unexpected default lane weights: %+v", d)
	}
	if d.FavoritesTag != "Favorites" || d.FavoritesCategory != "Swipe" {
		t.Fatalf("unexpected favorites identity: %+v", d)
	}
	// A deliberately zeroed lane among set ones is preserved.
	p := Tuning{ExploitWeight: 1}.WithDefaults()
	if p.WildcardWeight != 0 {
		t.Fatalf("explicit lane mix was overridden: %+v", p)
	}
	// Sparse configs keep their explicit values.
	s := Tuning{BatchSize: 5}.WithDefaults()
	if s.BatchSize != 5 || s.MaxClusters != 12 {
		t.Fatalf("sparse tuning merged wrong: %+v", s)
	}
}
