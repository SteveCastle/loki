package tasks

import (
	"database/sql"
	"fmt"
	"testing"

	"github.com/stevecastle/shrike/embedindex"
	"github.com/stevecastle/shrike/embedvec"
	"github.com/stevecastle/shrike/media"
	_ "modernc.org/sqlite"
)

// A SigLIP text vector is not drawn from the image distribution the library's
// mean describes, so centering it ranks by how ATYPICAL each image is instead
// of by how well it matches the text — measurably worse than random on a real
// library. These tests pin that text/blend queries take the plain-cosine route
// on both the indexed and brute-force paths, and that image queries still get
// the centered treatment.

// seedCommonComponentDB models the geometry that broke text search: a library
// whose vectors all share a strong common direction (dim 0, "is a photo"), one
// "match" item that additionally leans toward the query's direction (dim 1),
// and a handful of "odd" items that sit far from the library mean but have
// nothing to do with the query. Plain cosine ranks match.jpg first; centering
// promotes the odd ones, because a query from outside the image distribution
// leaves ‖v−mu‖ — atypicality — as the dominant term. Returns the query.
func seedCommonComponentDB(t *testing.T, db *sql.DB, idx embedindex.VectorIndex) []float32 {
	t.Helper()
	const dim = 8
	add := func(path string, v []float32) {
		if err := media.UpsertEmbedding(db, path, EmbedModelID, embedvec.Normalize(v), 0); err != nil {
			t.Fatal(err)
		}
		if idx != nil {
			idx.Add(path, v)
		}
	}
	for i := 0; i < embedindex.MinCenterCount*4; i++ {
		v := make([]float32, dim)
		v[0] = 1
		v[2+i%4] = 0.3
		add(fmt.Sprintf("bg-%03d.jpg", i), v)
	}
	// The match's edge over everything else is SMALL, as a real image↔text
	// cosine is (~0.10 vs ~0.08 across a library): enough to rank first under
	// plain cosine, not enough to survive division by ‖v−mu‖.
	match := make([]float32, dim)
	match[0] = 1
	match[1] = 0.03
	add("match.jpg", match)
	for i := 0; i < 8; i++ {
		v := make([]float32, dim)
		v[0] = 0.2
		v[1] = 0.01
		v[6] = 1
		add(fmt.Sprintf("odd-%d.jpg", i), v)
	}

	q := make([]float32, dim)
	q[1] = 1
	return q
}

func newCrossModalDB(t *testing.T) *sql.DB {
	t.Helper()
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	if err := media.InitializeSchema(db); err != nil {
		t.Fatal(err)
	}
	return db
}

// TestCrossModalSearchSkipsCenteringOnIndex is the regression test for text
// search returning unrelated content: against the centered index a text query
// must be scored with plain cosine, which puts the genuinely-matching item
// first.
func TestCrossModalSearchSkipsCenteringOnIndex(t *testing.T) {
	db := newCrossModalDB(t)
	idx := embedindex.NewCentered()
	q := seedCommonComponentDB(t, db, idx)
	SetVectorIndexForModel(idx, EmbedModelID)
	t.Cleanup(func() { SetVectorIndex(nil) })

	hits, err := searchByVectorCrossModal(db, EmbedModelID, q, 5, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) == 0 || hits[0].Path != "match.jpg" {
		t.Fatalf("cross-modal search should rank match.jpg first, got %+v", hits)
	}
	// The centered route is what regressed; if it happened to agree, this test
	// would not be proving anything.
	centered, err := searchByVectorWithin(db, EmbedModelID, q, 5, nil)
	if err != nil {
		t.Fatal(err)
	}
	if centered[0].Path == "match.jpg" {
		t.Fatalf("fixture no longer distinguishes the two scorings: %+v", centered)
	}
}

// TestCrossModalSearchSkipsCenteringBruteForce pins the same contract on the
// no-index path, so results don't depend on whether an index is installed.
func TestCrossModalSearchSkipsCenteringBruteForce(t *testing.T) {
	db := newCrossModalDB(t)
	q := seedCommonComponentDB(t, db, nil)

	hits, err := searchByVectorCrossModal(db, EmbedModelID, q, 5, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) == 0 || hits[0].Path != "match.jpg" {
		t.Fatalf("cross-modal brute force should rank match.jpg first, got %+v", hits)
	}
}

// TestScoringForMarksTextTermsCrossModal pins the rule that routes composite
// and shared-concept queries: one text term is enough to make the whole query
// cross-modal.
func TestScoringForMarksTextTermsCrossModal(t *testing.T) {
	if got := scoringFor(true); got != embedindex.ScorePlain {
		t.Errorf("scoringFor(true) = %v, want ScorePlain", got)
	}
	if got := scoringFor(false); got != embedindex.ScoreDefault {
		t.Errorf("scoringFor(false) = %v, want ScoreDefault", got)
	}
}
