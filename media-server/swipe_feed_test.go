package main

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stevecastle/shrike/embedvec"
	"github.com/stevecastle/shrike/feed"
	"github.com/stevecastle/shrike/media"
	"github.com/stevecastle/shrike/tasks"
	_ "modernc.org/sqlite"
)

// newSwipeFeedDeps seeds an in-memory library with a liked taste region
// (/a/*), an unrelated region (/z/*), and a favorite on /a/0.jpg.
func newSwipeFeedDeps(t *testing.T) (*Dependencies, *feed.Engine) {
	t.Helper()
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	if err := media.InitializeSchema(db); err != nil {
		t.Fatal(err)
	}

	model := tasks.ActiveEmbedModel().ID
	for i := 0; i < 10; i++ {
		f := float32(i) * 0.02
		for prefix, vec := range map[string][]float32{
			"/a": embedvec.Normalize([]float32{1, f}),
			"/z": embedvec.Normalize([]float32{f, 1}),
		} {
			p := fmt.Sprintf("%s/%d.jpg", prefix, i)
			if _, err := db.Exec(`INSERT INTO media (path) VALUES (?)`, p); err != nil {
				t.Fatal(err)
			}
			if err := media.UpsertEmbedding(db, p, model, vec, 0); err != nil {
				t.Fatal(err)
			}
		}
	}
	if _, err := db.Exec(
		`INSERT INTO media_tag_by_category (media_path, tag_label, category_label, weight, time_stamp, created_at)
		 VALUES ('/a/0.jpg', 'Favorites', 'Swipe', 0, 0, 100)`,
	); err != nil {
		t.Fatal(err)
	}

	deps := &Dependencies{DB: db}
	engine := feed.NewEngine(
		db,
		func() string { return model },
		func(m string, q []float32, limit int) ([]string, error) {
			hits, err := tasks.SearchByVector(db, m, q, limit)
			if err != nil {
				return nil, err
			}
			paths := make([]string, len(hits))
			for i, h := range hits {
				paths[i] = h.Path
			}
			return paths, nil
		},
		func() feed.Tuning { return feed.Tuning{BatchSize: 10} },
	)
	return deps, engine
}

func getSwipeFeed(t *testing.T, deps *Dependencies, engine *feed.Engine, url string) (media.APIResponse, *httptest.ResponseRecorder) {
	t.Helper()
	req := httptest.NewRequest(http.MethodGet, url, nil)
	rec := httptest.NewRecorder()
	handleSwipeFeed(rec, req, deps, engine)
	var resp media.APIResponse
	if rec.Code == http.StatusOK {
		if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
			t.Fatalf("decode: %v (body: %s)", err, rec.Body.String())
		}
	}
	return resp, rec
}

func TestSwipeFeedIgnoresOtherModes(t *testing.T) {
	deps, _ := newSwipeFeedDeps(t)
	req := httptest.NewRequest(http.MethodGet, "/swipe/api?offset=0&limit=5", nil)
	if maybeHandleSwipeFeed(httptest.NewRecorder(), req, deps) {
		t.Fatal("handled a request without mode=feed")
	}
}

func TestSwipeFeedServesPagesWithoutRepeats(t *testing.T) {
	deps, engine := newSwipeFeedDeps(t)

	seen := map[string]bool{}
	for offset := 0; offset < 15; offset += 5 {
		resp, rec := getSwipeFeed(t, deps, engine,
			fmt.Sprintf("/swipe/api?mode=feed&session=s1&offset=%d&limit=5", offset))
		if rec.Code != http.StatusOK {
			t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
		}
		for _, it := range resp.Items {
			if seen[it.Path] {
				t.Fatalf("path %q repeated across pages", it.Path)
			}
			seen[it.Path] = true
			if it.Path == "/a/0.jpg" {
				t.Fatal("liked item served back into the feed")
			}
		}
	}
	if len(seen) == 0 {
		t.Fatal("feed served nothing")
	}
}

func TestSwipeFeedExhaustsSmallLibrary(t *testing.T) {
	deps, engine := newSwipeFeedDeps(t)
	// 20 items minus 1 liked = 19 servable. Walk to the end.
	total := 0
	offset := 0
	for {
		resp, rec := getSwipeFeed(t, deps, engine,
			fmt.Sprintf("/swipe/api?mode=feed&session=s&offset=%d&limit=10", offset))
		if rec.Code != http.StatusOK {
			t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
		}
		total += len(resp.Items)
		offset += 10
		if !resp.HasMore {
			break
		}
		if offset > 100 {
			t.Fatal("feed never reported exhaustion")
		}
	}
	if total != 19 {
		t.Fatalf("served %d items, want 19 (library minus the liked item)", total)
	}
}

func TestSwipeFeedLaneOverrideParams(t *testing.T) {
	deps, engine := newSwipeFeedDeps(t)
	resp, rec := getSwipeFeed(t, deps, engine,
		"/swipe/api?mode=feed&session=w&offset=0&limit=10&exploit=0.0001&fresh=0.0001&bridge=0.0001&wildcard=1")
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	if len(resp.Items) == 0 {
		t.Fatal("override feed served nothing")
	}
}
