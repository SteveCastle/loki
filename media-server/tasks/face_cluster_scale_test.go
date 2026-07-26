package tasks

import (
	"context"
	"database/sql"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"testing"
	"time"

	"github.com/stevecastle/shrike/media"
	_ "modernc.org/sqlite"
)

// TestClusterFacesScaleProfile is a manual profiling harness, not a CI test:
// it builds a synthetic library the shape of a real one (a minority of faces
// belonging to actual identities, the rest junk that never groups) and reports
// where a full pass spends its time, phase by phase. Run it with
//
//	LOKI_SCALE_FACES=100000 go test ./tasks -run ScaleProfile -v -timeout 2h
//
// Skipped unless LOKI_SCALE_FACES is set.
func TestClusterFacesScaleProfile(t *testing.T) {
	raw := os.Getenv("LOKI_SCALE_FACES")
	if raw == "" {
		t.Skip("set LOKI_SCALE_FACES=<n> to run the scale profile")
	}
	n, err := strconv.Atoi(raw)
	if err != nil {
		t.Fatalf("LOKI_SCALE_FACES: %v", err)
	}
	dims := 512
	if v := os.Getenv("LOKI_SCALE_DIMS"); v != "" {
		dims, _ = strconv.Atoi(v)
	}

	build := time.Now()
	db, assigned := buildScaleLibrary(t, n, dims)
	t.Logf("fixture: %d faces (%d dims), %d pre-assigned seeds, built in %s",
		n, dims, assigned, time.Since(build).Round(time.Millisecond))

	model := FaceModel{ID: "scale", MatchThreshold: 0.42}
	p := clusterParams{
		joinThreshold: 0.42, formThreshold: 0.47, minQuality: 0.5,
		minCluster: 3, passes: 2,
	}

	// Time each phase off the progress ticks: the first tick of a phase closes
	// out the previous one.
	type span struct {
		phase clusterPhase
		pass  int
		dur   time.Duration
	}
	var spans []span
	last := time.Now()
	var curPhase clusterPhase
	curPass := 0
	p.progress = func(pr clusterProgress) {
		if pr.Phase != curPhase || pr.Pass != curPass {
			if curPhase != "" {
				spans = append(spans, span{curPhase, curPass, time.Since(last)})
			}
			curPhase, curPass, last = pr.Phase, pr.Pass, time.Now()
		}
	}

	start := time.Now()
	stats, err := clusterFaces(context.Background(), db, model, p)
	total := time.Since(start)
	if err != nil {
		t.Fatal(err)
	}
	spans = append(spans, span{curPhase, curPass, time.Since(last)})

	t.Logf("TOTAL %s — %+v", total.Round(time.Millisecond), stats)
	for _, s := range spans {
		label := string(s.phase)
		if s.pass > 0 {
			label = fmt.Sprintf("%s pass %d", s.phase, s.pass)
		}
		t.Logf("  %-16s %8s  %5.1f%%", label, s.dur.Round(time.Millisecond),
			float64(s.dur)/float64(total)*100)
	}
}

// buildScaleLibrary seeds n faces: a minority drawn from real identity
// clusters (tight groups around random centers) and the rest junk (mutually
// near-orthogonal), which is the shape that makes phase 2 expensive. A slice
// of the identity faces is pre-assigned so phase 1 has seeds to score against.
func buildScaleLibrary(t *testing.T, n, dims int) (*sql.DB, int) {
	t.Helper()
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	if err := media.InitializeSchema(db); err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(11))

	randUnit := func() []float32 {
		v := make([]float32, dims)
		var norm float64
		for k := range v {
			x := rng.NormFloat64()
			v[k] = float32(x)
			norm += x * x
		}
		norm = math.Sqrt(norm)
		for k := range v {
			v[k] /= float32(norm)
		}
		return v
	}
	// A face of an identity: its center plus noise, mixed so members sit
	// around 0.6-0.8 cosine of each other (a realistic within-person spread).
	nearUnit := func(center []float32, spread float64) []float32 {
		v := make([]float32, dims)
		var norm float64
		for k := range v {
			x := float64(center[k]) + rng.NormFloat64()*spread/math.Sqrt(float64(dims))
			v[k] = float32(x)
			norm += x * x
		}
		norm = math.Sqrt(norm)
		for k := range v {
			v[k] /= float32(norm)
		}
		return v
	}

	// 40% of faces belong to identities averaging 12 faces each.
	identityFaces := n * 4 / 10
	nPeople := identityFaces / 12
	centers := make([][]float32, nPeople)
	for i := range centers {
		centers[i] = randUnit()
	}

	const perPath = 200
	var identityIDs []int64
	written := 0
	for written < n {
		count := min(perPath, n-written)
		faces := make([]media.NewFace, 0, count)
		fromIdentity := 0
		for i := range count {
			idx := written + i
			var vec []float32
			if idx < identityFaces && nPeople > 0 {
				vec = nearUnit(centers[idx%nPeople], 0.9)
				fromIdentity++
			} else {
				vec = randUnit()
			}
			faces = append(faces, media.NewFace{
				X: 0.1, Y: 0.1, W: 0.2, H: 0.2,
				Score: 0.6 + rng.Float64()*0.4, Vec: vec,
			})
		}
		ids, err := media.ReplaceFaces(db, fmt.Sprintf("scale-%d.jpg", written), "scale", faces, 1)
		if err != nil {
			t.Fatal(err)
		}
		for i := range fromIdentity {
			identityIDs = append(identityIDs, ids[i])
		}
		written += count
	}

	// Pre-assign a quarter of the identity faces across their people, the way
	// a partly-curated library looks.
	sort.Slice(identityIDs, func(i, j int) bool { return identityIDs[i] < identityIDs[j] })
	assigned := 0
	if nPeople > 0 {
		people := make([]int64, 0, nPeople)
		for i := range nPeople {
			pid, err := media.CreatePerson(db, fmt.Sprintf("Person %d", i))
			if err != nil {
				t.Fatal(err)
			}
			people = append(people, pid)
		}
		var batch []media.FaceAssignment
		for i, id := range identityIDs {
			if i%4 != 0 {
				continue
			}
			batch = append(batch, media.FaceAssignment{FaceID: id, PersonID: people[i%nPeople]})
		}
		applied, err := media.AssignFacesAuto(db, batch)
		if err != nil {
			t.Fatal(err)
		}
		assigned = len(applied)
	}
	return db, assigned
}
