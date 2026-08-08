package tasks

import (
	"context"
	"database/sql"
	"testing"

	"github.com/stevecastle/shrike/embedindex"
	"github.com/stevecastle/shrike/media"
	_ "modernc.org/sqlite"
)

// newSharedSearchDB seeds the football scenario: two query examples that share
// dim 0 with private scenery dims, the pure shared-concept item, and a
// scenery-only distractor.
func newSharedSearchDB(t *testing.T) *sql.DB {
	t.Helper()
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	if err := media.InitializeSchema(db); err != nil {
		t.Fatal(err)
	}
	_ = media.UpsertEmbedding(db, "q1.jpg", EmbedModelID, embedvecNormalize([]float32{1, 1, 0, 0}), 0)
	_ = media.UpsertEmbedding(db, "q2.jpg", EmbedModelID, embedvecNormalize([]float32{1, 0, 1, 0}), 0)
	_ = media.UpsertEmbedding(db, "football.jpg", EmbedModelID, embedvecNormalize([]float32{1, 0, 0, 0}), 0)
	_ = media.UpsertEmbedding(db, "scenery.jpg", EmbedModelID, embedvecNormalize([]float32{0, 1, 0, 0.2}), 0)
	return db
}

// TestSearchBySharedConceptBruteForce runs the full path-term pipeline with no
// index installed: the shared-concept item must beat both examples and the
// scenery distractor must come last.
func TestSearchBySharedConceptBruteForce(t *testing.T) {
	db := newSharedSearchDB(t)
	SetVectorIndex(nil)
	t.Cleanup(func() { SetVectorIndex(nil) })

	terms := []QueryTerm{
		{Kind: "path", Value: "q1.jpg", Weight: 1},
		{Kind: "path", Value: "q2.jpg", Weight: 1},
	}
	hits, err := SearchBySharedConcept(context.Background(), db, terms, 10, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 4 || hits[0].Path != "football.jpg" {
		t.Fatalf("expected football.jpg first, got %+v", hits)
	}
	if hits[len(hits)-1].Path != "scenery.jpg" {
		t.Fatalf("expected scenery.jpg last, got %+v", hits)
	}
}

// TestSearchBySharedConceptUsesIndex verifies the installed-index path and the
// allow-set restriction: results must come from the index alone.
func TestSearchBySharedConceptUsesIndex(t *testing.T) {
	db := newSharedSearchDB(t)
	idx := embedindex.New()
	idx.Add("q1.jpg", embedvecNormalize([]float32{1, 1, 0, 0}))
	idx.Add("football.jpg", embedvecNormalize([]float32{1, 0, 0, 0}))
	// q2/scenery deliberately NOT indexed: hits must come from the index.
	SetVectorIndexForModel(idx, EmbedModelID)
	t.Cleanup(func() { SetVectorIndex(nil) })

	terms := []QueryTerm{
		{Kind: "path", Value: "q1.jpg", Weight: 1},
		{Kind: "path", Value: "q2.jpg", Weight: 1},
	}
	hits, err := SearchBySharedConcept(context.Background(), db, terms, 10, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 2 || hits[0].Path != "football.jpg" {
		t.Fatalf("expected [football.jpg q1.jpg] from the index, got %+v", hits)
	}

	hits, err = SearchBySharedConcept(context.Background(), db, terms, 10, PathSet{"q1.jpg": {}})
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 1 || hits[0].Path != "q1.jpg" {
		t.Fatalf("allow set should restrict to q1.jpg, got %+v", hits)
	}
}

// TestSearchBySharedConceptNegativeTerms verifies negative-weight terms become
// steer-away penalties rather than members of the must-match set, and that a
// query with ONLY negative terms errors.
func TestSearchBySharedConceptNegativeTerms(t *testing.T) {
	db := newSharedSearchDB(t)
	SetVectorIndex(nil)
	t.Cleanup(func() { SetVectorIndex(nil) })

	// Positive on the shared concept, negative on q1's scenery dim: q1.jpg
	// must drop below q2.jpg.
	terms := []QueryTerm{
		{Kind: "path", Value: "football.jpg", Weight: 1},
		{Kind: "path", Value: "scenery.jpg", Weight: -1},
	}
	hits, err := SearchBySharedConcept(context.Background(), db, terms, 10, nil)
	if err != nil {
		t.Fatal(err)
	}
	rank := map[string]int{}
	for i, h := range hits {
		rank[h.Path] = i
	}
	if !(rank["q2.jpg"] < rank["q1.jpg"]) {
		t.Fatalf("negative scenery term should sink q1.jpg below q2.jpg, got %+v", hits)
	}

	if _, err := SearchBySharedConcept(context.Background(), db,
		[]QueryTerm{{Kind: "path", Value: "q1.jpg", Weight: -1}}, 10, nil); err == nil {
		t.Fatal("only-negative query: expected error")
	}
}
