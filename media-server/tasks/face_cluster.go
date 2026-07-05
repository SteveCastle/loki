package tasks

import (
	"database/sql"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"sync"

	"github.com/stevecastle/shrike/embedvec"
	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/media"
)

// minAutoClusterSize is how many mutually-similar unassigned faces it takes
// to mint an anonymous "Unknown #N" person. Below this, faces stay unassigned
// (singletons and pairs are usually noise). Overridable per job (--min-cluster).
const minAutoClusterSize = 3

// Corroboration: several independent near-matches against the same person are
// stronger evidence than one high match. Each extra match within
// corroborationSlack of the join threshold adds corroborationBonus to that
// person's effective score (capped at maxCorroborators extras), so a hard
// positive — profile shot, bad lighting, different age — can still join a
// person whose cluster already covers similar conditions. A single borderline
// match gets no bonus, so the bar isn't lowered for one-off chance hits.
const (
	corroborationSlack = float32(0.06)
	corroborationBonus = float32(0.02)
	maxCorroborators   = 3
)

// personScore accumulates the evidence one query face has for one person.
type personScore struct {
	best  float32 // best single cosine against the person's faces
	count int     // matches within corroborationSlack of the join threshold
}

// effectiveScore folds corroboration into a single comparable score.
func effectiveScore(s personScore) float32 {
	extra := s.count - 1
	if extra < 0 {
		extra = 0
	}
	if extra > maxCorroborators {
		extra = maxCorroborators
	}
	return s.best + float32(extra)*corroborationBonus
}

// acceptJoin decides whether accumulated evidence is enough to join a person:
// the corroborated score must clear the threshold AND the best raw match must
// be at least within slack of it (corroboration widens the gate, it never
// replaces a plausible direct match).
func acceptJoin(s personScore, threshold float32) bool {
	return effectiveScore(s) >= threshold && s.best >= threshold-corroborationSlack
}

// clusterParams are the knobs of one clustering pass.
type clusterParams struct {
	// joinThreshold gates joining an EXISTING person (phase 1 + the
	// incremental per-scan assignment).
	joinThreshold float32
	// formThreshold gates forming/growing a NEW anonymous cluster (phase 2).
	// Minting a brand-new identity needs stronger evidence than joining a
	// curated one, so it defaults to joinThreshold + 0.05.
	formThreshold float32
	// minQuality is the detection-confidence floor for phase-2 participants.
	// Blurry/occluded/background faces have unreliable embeddings; letting
	// them found clusters is where "random" unknown groups come from. They
	// can still join existing people in phase 1 and stay searchable.
	minQuality float64
	// minCluster is the minimum member count for a new anonymous person.
	minCluster int
	// passes is how many phase-1 iterations run. Faces joined in one pass
	// seed the next, giving bounded one-hop transitivity (a face that joins
	// Alice can pull in its own near-duplicates) without open-ended drift.
	passes int
}

func defaultClusterParams(model FaceModel) clusterParams {
	t := model.MatchThreshold
	return clusterParams{
		joinThreshold: t,
		formThreshold: t + 0.05,
		minQuality:    0.75,
		minCluster:    minAutoClusterSize,
		passes:        2,
	}
}

// autoAssignNewFaces incrementally assigns freshly-scanned faces to existing
// people using the same corroborated-join rule as the full clustering pass.
// Called by the faces collector after storing a batch, so growing a library
// keeps enriching existing people without a full recluster. Requires the live
// face index to hold this model (otherwise it's a no-op — the faces-cluster
// task will pick the faces up later). Returns how many faces were assigned.
func autoAssignNewFaces(db *sql.DB, model FaceModel, ids []int64, faces []media.NewFace) int {
	if FaceIndexedModel() != model.ID || model.MatchThreshold <= 0 {
		return 0
	}
	newIDs := make(map[int64]bool, len(ids))
	for _, id := range ids {
		newIDs[id] = true
	}
	threshold := model.MatchThreshold
	assigned := 0
	for i, id := range ids {
		hits, err := SearchFacesByVector(db, model.ID, faces[i].Vec, 12)
		if err != nil {
			continue
		}
		scores := map[int64]personScore{}
		for _, h := range hits {
			if newIDs[h.FaceID] || h.PersonID == 0 {
				continue // itself / a sibling from this batch / unassigned
			}
			ps := scores[h.PersonID]
			if h.Score > ps.best {
				ps.best = h.Score
			}
			if h.Score >= threshold-corroborationSlack {
				ps.count++
			}
			scores[h.PersonID] = ps
		}
		var bestPerson int64
		var best personScore
		for pid, ps := range scores {
			if bestPerson == 0 || effectiveScore(ps) > effectiveScore(best) {
				bestPerson, best = pid, ps
			}
		}
		if bestPerson != 0 && acceptJoin(best, threshold) {
			if err := media.AssignFace(db, id, bestPerson, "auto"); err == nil {
				assigned++
			}
		}
	}
	return assigned
}

// clusterStats summarises one full clustering pass.
type clusterStats struct {
	JoinedExisting int // unassigned faces that joined an existing person
	NewPeople      int // anonymous "Unknown #N" persons created
	NewlyClustered int // faces assigned into those new persons
	Unassigned     int // faces left unassigned (small/incoherent clusters)
	QualitySkipped int // faces below the quality floor (excluded from phase 2)
	Discarded      int // faces in clusters dropped by the coherence check
}

// seed is one already-assigned face acting as a join anchor.
type seed struct {
	vec      []float32
	personID int64
}

// clusterFaces runs a full clustering pass for model:
//
//  1. every unassigned face joins its best-matching already-assigned person
//     under the corroborated-join rule (multiple near-threshold matches to
//     the same person widen the gate). This repeats p.passes times, with the
//     faces joined in one pass seeding the next — bounded transitivity, so a
//     confident join can pull in its own near-duplicates without open-ended
//     chain drift. User assignments act as ground-truth seeds throughout.
//  2. the remaining unassigned faces with detection confidence ≥ p.minQuality
//     are greedily leader-clustered at the (stricter) formation threshold;
//     clusters of at least p.minCluster members whose final coherence — mean
//     member↔centroid cosine — still clears the formation threshold become
//     new anonymous "Unknown #N" people. Incoherent clusters are discarded
//     (their faces stay unassigned) rather than shipped as random groups.
//
// Existing assignments are never touched (auto assignments in anonymous
// clusters can be reset via the task's --reset flag before calling this).
func clusterFaces(db *sql.DB, model FaceModel, p clusterParams) (clusterStats, error) {
	var stats clusterStats
	all, err := media.LoadAllFaces(db, model.ID)
	if err != nil {
		return stats, err
	}
	var seeds []seed
	var unassigned []media.Face
	for _, f := range all {
		// Normalize defensively: cosine math below assumes unit vectors.
		f.Vec = embedvec.Normalize(f.Vec)
		if f.PersonID != 0 {
			seeds = append(seeds, seed{vec: f.Vec, personID: f.PersonID})
		} else {
			unassigned = append(unassigned, f)
		}
	}

	// Phase 1: corroborated joins against assigned seeds, p.passes rounds.
	for pass := 0; pass < p.passes && len(seeds) > 0 && len(unassigned) > 0; pass++ {
		matches := scoreAgainstSeeds(unassigned, seeds, p.joinThreshold)
		var leftovers []media.Face
		joinedThisPass := 0
		for i, personID := range matches {
			if personID == 0 {
				leftovers = append(leftovers, unassigned[i])
				continue
			}
			if err := media.AssignFace(db, unassigned[i].ID, personID, "auto"); err != nil {
				return stats, err
			}
			seeds = append(seeds, seed{vec: unassigned[i].Vec, personID: personID})
			stats.JoinedExisting++
			joinedThisPass++
		}
		unassigned = leftovers
		if joinedThisPass == 0 {
			break // converged early; further passes can't change anything
		}
	}

	// Phase 2: only confident detections may found new identities.
	var eligible []media.Face
	for _, f := range unassigned {
		if f.Score >= p.minQuality {
			eligible = append(eligible, f)
		} else {
			stats.QualitySkipped++
			stats.Unassigned++
		}
	}

	// Greedy leader clustering among the eligible leftovers, strongest
	// detections first so cluster centroids start from the clearest faces.
	sort.SliceStable(eligible, func(i, j int) bool { return eligible[i].Score > eligible[j].Score })
	type cluster struct {
		centroid []float32
		members  []media.Face
	}
	var clusters []cluster
	for _, f := range eligible {
		bestIdx := -1
		var bestScore float32
		for ci := range clusters {
			if sc := embedvec.CosineSim(f.Vec, clusters[ci].centroid); sc > bestScore {
				bestScore, bestIdx = sc, ci
			}
		}
		if bestIdx >= 0 && bestScore >= p.formThreshold {
			c := &clusters[bestIdx]
			// Running-mean centroid, renormalized.
			n := float32(len(c.members))
			for k := range c.centroid {
				c.centroid[k] = (c.centroid[k]*n + f.Vec[k]) / (n + 1)
			}
			c.centroid = embedvec.Normalize(c.centroid)
			c.members = append(c.members, f)
		} else {
			centroid := make([]float32, len(f.Vec))
			copy(centroid, f.Vec)
			clusters = append(clusters, cluster{centroid: centroid, members: []media.Face{f}})
		}
	}

	for _, c := range clusters {
		if len(c.members) < p.minCluster {
			stats.Unassigned += len(c.members)
			continue
		}
		// Coherence check: greedy growth lets a centroid drift; a "cluster"
		// whose members no longer agree with where it ended up is noise.
		var mean float32
		for _, m := range c.members {
			mean += embedvec.CosineSim(m.Vec, c.centroid)
		}
		mean /= float32(len(c.members))
		if mean < p.formThreshold {
			stats.Discarded += len(c.members)
			stats.Unassigned += len(c.members)
			continue
		}
		name, err := media.NextUnknownName(db)
		if err != nil {
			return stats, err
		}
		pid, err := media.CreatePerson(db, name)
		if err != nil {
			return stats, err
		}
		for _, m := range c.members {
			if err := media.AssignFace(db, m.ID, pid, "auto"); err != nil {
				return stats, err
			}
			stats.NewlyClustered++
		}
		stats.NewPeople++
	}
	return stats, nil
}

// scoreAgainstSeeds computes, for every unassigned face, the person it should
// join under the corroborated-join rule (0 = no join). Scoring is parallel;
// the caller applies the writes serially.
func scoreAgainstSeeds(unassigned []media.Face, seeds []seed, threshold float32) []int64 {
	matches := make([]int64, len(unassigned))
	if len(unassigned) == 0 || len(seeds) == 0 {
		return matches
	}
	workers := 8
	if len(unassigned) < workers {
		workers = len(unassigned)
	}
	chunk := (len(unassigned) + workers - 1) / workers
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		lo := w * chunk
		hi := lo + chunk
		if hi > len(unassigned) {
			hi = len(unassigned)
		}
		if lo >= hi {
			continue
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			scores := map[int64]personScore{}
			for i := lo; i < hi; i++ {
				for k := range scores {
					delete(scores, k)
				}
				for _, s := range seeds {
					sc := embedvec.CosineSim(unassigned[i].Vec, s.vec)
					if sc < threshold-corroborationSlack {
						continue // can neither win nor corroborate
					}
					ps := scores[s.personID]
					if sc > ps.best {
						ps.best = sc
					}
					ps.count++
					scores[s.personID] = ps
				}
				var bestPerson int64
				var best personScore
				for pid, ps := range scores {
					if bestPerson == 0 || effectiveScore(ps) > effectiveScore(best) {
						bestPerson, best = pid, ps
					}
				}
				if bestPerson != 0 && acceptJoin(best, threshold) {
					matches[i] = bestPerson
				}
			}
		}(lo, hi)
	}
	wg.Wait()
	return matches
}

// resetAutoAssignments clears the auto assignments of ANONYMOUS clusters for
// model and dissolves the emptied "Unknown #N" people. Anything the user has
// endorsed stays put: user-assigned faces (ground truth) and ALL faces of
// named people — renaming or merging a cluster is an endorsement of its
// contents, so a reset must not scatter it. Orphaned assignments (person row
// gone) are cleared too. Returns how many faces were unassigned.
func resetAutoAssignments(db *sql.DB, model string) (int, error) {
	rows, err := db.Query(`
		SELECT f.id FROM face f
		LEFT JOIN person p ON p.id = f.person_id
		WHERE f.model = ? AND f.assigned_by = 'auto'
		  AND (p.id IS NULL OR p.name LIKE 'Unknown #%')`, model)
	if err != nil {
		return 0, err
	}
	var ids []int64
	for rows.Next() {
		var id int64
		if err := rows.Scan(&id); err != nil {
			rows.Close()
			return 0, err
		}
		ids = append(ids, id)
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return 0, err
	}
	for _, id := range ids {
		if err := media.UnassignFace(db, id); err != nil {
			return len(ids), err
		}
	}
	// Dissolve now-empty anonymous clusters (named people are kept even when
	// empty — the user made them, the user deletes them).
	people, err := media.GetPeople(db)
	if err != nil {
		return len(ids), err
	}
	for _, p := range people {
		if p.FaceCount == 0 && strings.HasPrefix(p.Name, "Unknown #") {
			if err := media.DeletePerson(db, p.ID); err != nil {
				return len(ids), err
			}
		}
	}
	return len(ids), nil
}

// jobArgValue extracts `--key=value` (or `--key value`) from job arguments.
func jobArgValue(j *jobqueue.Job, key string) (string, bool) {
	prefix := key + "="
	for i := 0; i < len(j.Arguments); i++ {
		arg := j.Arguments[i]
		if strings.HasPrefix(arg, prefix) {
			if v := strings.TrimSpace(arg[len(prefix):]); v != "" {
				return v, true
			}
		}
		if arg == key && i+1 < len(j.Arguments) {
			if v := strings.TrimSpace(j.Arguments[i+1]); v != "" {
				return v, true
			}
		}
	}
	return "", false
}

func jobHasFlag(j *jobqueue.Job, flag string) bool {
	for _, arg := range j.Arguments {
		if arg == flag {
			return true
		}
	}
	return false
}

// facesClusterTask groups stored faces into people. Arguments:
//
//	--model=<id>           cluster a specific recognizer's faces (default: active)
//	--threshold=<0..1>     override the join threshold (model default)
//	--form-threshold=<0..1> override the new-cluster threshold (join + 0.05)
//	--min-quality=<0..1>   detection-confidence floor for new clusters (0.75)
//	--min-cluster=<n>      minimum faces for a new anonymous person (default 3)
//	--passes=<n>           phase-1 join iterations (default 2)
//	--reset                rebuild the anonymous "Unknown #N" clusters first.
//	                       User-assigned faces and everything inside NAMED
//	                       people are never touched — naming/merging a cluster
//	                       endorses its contents.
func facesClusterTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	// Which recognizers to cluster: an explicit --model pins one; otherwise,
	// with routing on, every known model that has stored faces gets its own
	// pass (per-model thresholds), so one job clusters photos AND anime.
	var clusterModels []FaceModel
	if id, ok := embedModelOverrideFromJob(j); ok {
		if m, known := FaceModelByID(id); known {
			clusterModels = []FaceModel{m}
		} else {
			m := ActiveFaceModel()
			q.PushJobStdout(j.ID, fmt.Sprintf("Unknown --model %q; using active model %q", id, m.ID))
			clusterModels = []FaceModel{m}
		}
	} else if FaceRoutingEnabled() {
		ids, err := faceModelsWithFaces(q.Db)
		if err != nil {
			q.PushJobStdout(j.ID, "Failed to list face models: "+err.Error())
			q.ErrorJob(j.ID)
			return err
		}
		for _, id := range ids {
			if m, known := FaceModelByID(id); known {
				clusterModels = append(clusterModels, m)
			} else {
				q.PushJobStdout(j.ID, fmt.Sprintf("Skipping faces stored under unknown model %q", id))
			}
		}
		if len(clusterModels) == 0 {
			clusterModels = []FaceModel{ActiveFaceModel()}
		}
	} else {
		clusterModels = []FaceModel{ActiveFaceModel()}
	}

	for _, model := range clusterModels {
		if err := clusterOneModel(j, q, model); err != nil {
			return err
		}
	}
	q.CompleteJob(j.ID)
	return nil
}

// faceModelsWithFaces lists the distinct recognizer IDs present in the face
// table.
func faceModelsWithFaces(db *sql.DB) ([]string, error) {
	rows, err := db.Query(`SELECT DISTINCT model FROM face`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []string
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, err
		}
		out = append(out, id)
	}
	return out, rows.Err()
}

// clusterOneModel runs one full clustering pass (flags + reset + stats
// logging) for a single recognizer. Errors the job on failure.
func clusterOneModel(j *jobqueue.Job, q *jobqueue.Queue, model FaceModel) error {
	p := defaultClusterParams(model)
	if v, ok := jobArgValue(j, "--threshold"); ok {
		if t, err := strconv.ParseFloat(v, 32); err == nil && t > 0 && t < 1 {
			p.joinThreshold = float32(t)
			p.formThreshold = float32(t) + 0.05
		}
	}
	if v, ok := jobArgValue(j, "--form-threshold"); ok {
		if t, err := strconv.ParseFloat(v, 32); err == nil && t > 0 && t < 1 {
			p.formThreshold = float32(t)
		}
	}
	if v, ok := jobArgValue(j, "--min-quality"); ok {
		if t, err := strconv.ParseFloat(v, 64); err == nil && t >= 0 && t < 1 {
			p.minQuality = t
		}
	}
	if v, ok := jobArgValue(j, "--min-cluster"); ok {
		if n, err := strconv.Atoi(v); err == nil && n >= 1 {
			p.minCluster = n
		}
	}
	if v, ok := jobArgValue(j, "--passes"); ok {
		if n, err := strconv.Atoi(v); err == nil && n >= 1 && n <= 5 {
			p.passes = n
		}
	}
	q.PushJobStdout(j.ID, fmt.Sprintf(
		"Clustering faces: model=%s join=%.3f form=%.3f min-quality=%.2f min-cluster=%d passes=%d",
		model.ID, p.joinThreshold, p.formThreshold, p.minQuality, p.minCluster, p.passes,
	))

	if jobHasFlag(j, "--reset") {
		n, err := resetAutoAssignments(q.Db, model.ID)
		if err != nil {
			q.PushJobStdout(j.ID, "Reset failed: "+err.Error())
			q.ErrorJob(j.ID)
			return err
		}
		q.PushJobStdout(j.ID, fmt.Sprintf("Reset %d auto assignment(s) in unnamed clusters; named people and user labels kept", n))
	}

	stats, err := clusterFaces(q.Db, model, p)
	if err != nil {
		q.PushJobStdout(j.ID, "Clustering failed: "+err.Error())
		q.ErrorJob(j.ID)
		return err
	}
	q.PushJobStdout(j.ID, fmt.Sprintf(
		"%s: %d joined existing people, %d new people (%d faces), %d left unassigned (%d below quality floor, %d in discarded incoherent clusters)",
		model.ID, stats.JoinedExisting, stats.NewPeople, stats.NewlyClustered, stats.Unassigned, stats.QualitySkipped, stats.Discarded,
	))
	return nil
}
