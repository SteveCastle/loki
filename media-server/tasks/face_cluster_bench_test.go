package tasks

import (
	"database/sql"
	"fmt"
	"math/rand"
	"testing"

	"github.com/stevecastle/shrike/media"
	_ "modernc.org/sqlite"
)

// BenchmarkClusterFacesJunkBacklog measures the shape that made 80k-face
// libraries take 20+ minutes: a large unassigned backlog of mutually
// DISSIMILAR faces (random 512-dim vectors — nothing joins, nothing forms),
// so phase 2 degenerates into every face scanning a cluster list that grows
// toward n singletons. A slice of the backlog is pre-assigned to one person
// so phase 1's seed scoring is exercised too. The run is idempotent (junk
// never assigns), so every b.N iteration does identical work.
func BenchmarkClusterFacesJunkBacklog(b *testing.B) {
	for _, n := range []int{5000, 20000} {
		b.Run(fmt.Sprintf("faces=%d", n), func(b *testing.B) {
			db := benchFaceDB(b, n)
			model := FaceModel{ID: "bench", MatchThreshold: 0.42}
			p := clusterParams{joinThreshold: 0.42, formThreshold: 0.47, minQuality: 0.5, minCluster: 3, passes: 2}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := clusterFaces(db, model, p); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// benchFaceDB seeds n random-vector faces (512-dim, fixed RNG seed) plus one
// person holding n/20 assigned faces to act as phase-1 seeds.
func benchFaceDB(b *testing.B, n int) *sql.DB {
	b.Helper()
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { db.Close() })
	if err := media.InitializeSchema(db); err != nil {
		b.Fatal(err)
	}
	rng := rand.New(rand.NewSource(1))
	const perPath = 200
	var ids []int64
	for lo := 0; lo < n; lo += perPath {
		hi := min(lo+perPath, n)
		faces := make([]media.NewFace, 0, hi-lo)
		for i := lo; i < hi; i++ {
			vec := make([]float32, 512)
			for k := range vec {
				vec[k] = float32(rng.NormFloat64())
			}
			faces = append(faces, media.NewFace{X: 0.1, Y: 0.1, W: 0.2, H: 0.2, Score: 0.9, Vec: vec})
		}
		batch, err := media.ReplaceFaces(db, fmt.Sprintf("bench-%d.jpg", lo), "bench", faces, 1)
		if err != nil {
			b.Fatal(err)
		}
		ids = append(ids, batch...)
	}
	pid, err := media.CreatePerson(db, "Seeded")
	if err != nil {
		b.Fatal(err)
	}
	assignments := make([]media.FaceAssignment, 0, n/20)
	for _, id := range ids[:n/20] {
		assignments = append(assignments, media.FaceAssignment{FaceID: id, PersonID: pid})
	}
	if _, err := media.AssignFacesAuto(db, assignments); err != nil {
		b.Fatal(err)
	}
	return db
}
