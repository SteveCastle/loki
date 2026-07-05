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

// autoAssignNewFaces incrementally assigns freshly-scanned faces to existing
// people: each new face joins the nearest already-ASSIGNED face at cosine ≥
// the model's match threshold. Called by the faces collector after storing a
// batch, so growing a library keeps enriching existing people without a full
// recluster. Requires the live face index to hold this model (otherwise it's
// a no-op — the faces-cluster task will pick the faces up later). Returns how
// many faces were assigned.
func autoAssignNewFaces(db *sql.DB, model FaceModel, ids []int64, faces []media.NewFace) int {
	if FaceIndexedModel() != model.ID || model.MatchThreshold <= 0 {
		return 0
	}
	newIDs := make(map[int64]bool, len(ids))
	for _, id := range ids {
		newIDs[id] = true
	}
	assigned := 0
	for i, id := range ids {
		hits, err := SearchFacesByVector(db, model.ID, faces[i].Vec, 8)
		if err != nil {
			continue
		}
		for _, h := range hits {
			if h.Score < model.MatchThreshold {
				break // hits are score-descending
			}
			if newIDs[h.FaceID] || h.PersonID == 0 {
				continue // itself / a sibling from this batch / unassigned
			}
			if err := media.AssignFace(db, id, h.PersonID, "auto"); err == nil {
				assigned++
			}
			break
		}
	}
	return assigned
}

// clusterStats summarises one full clustering pass.
type clusterStats struct {
	JoinedExisting int // unassigned faces that joined an existing person
	NewPeople      int // anonymous "Unknown #N" persons created
	NewlyClustered int // faces assigned into those new persons
	Unassigned     int // faces that remain unassigned (too small a cluster)
}

// clusterFaces runs a full clustering pass for model:
//
//  1. every unassigned face joins its nearest already-assigned neighbour at
//     cosine ≥ threshold (user assignments act as ground-truth seeds; seeds
//     are the assignments that existed BEFORE the pass so one borderline
//     match can't chain-drift a cluster);
//  2. the remaining unassigned faces are greedily leader-clustered among
//     themselves (centroid cosine ≥ threshold); clusters of at least
//     minCluster faces become new anonymous "Unknown #N" people.
//
// Existing assignments are never touched (auto assignments can be reset via
// the task's --reset flag before calling this).
func clusterFaces(db *sql.DB, model FaceModel, threshold float32, minCluster int) (clusterStats, error) {
	var stats clusterStats
	all, err := media.LoadAllFaces(db, model.ID)
	if err != nil {
		return stats, err
	}
	var seeds, unassigned []media.Face
	for _, f := range all {
		// Normalize defensively: cosine math below assumes unit vectors.
		f.Vec = embedvec.Normalize(f.Vec)
		if f.PersonID != 0 {
			seeds = append(seeds, f)
		} else {
			unassigned = append(unassigned, f)
		}
	}

	// Phase 1: join nearest assigned seed. Parallel scoring, serial writes.
	type match struct {
		faceID   int64
		personID int64
	}
	matches := make([]match, len(unassigned))
	var wg sync.WaitGroup
	workers := 8
	if len(unassigned) < workers {
		workers = len(unassigned)
	}
	if workers > 0 && len(seeds) > 0 {
		chunk := (len(unassigned) + workers - 1) / workers
		for w := 0; w < workers; w++ {
			lo := w * chunk
			hi := lo + chunk
			if hi > len(unassigned) {
				hi = len(unassigned)
			}
			wg.Add(1)
			go func(lo, hi int) {
				defer wg.Done()
				for i := lo; i < hi; i++ {
					var bestScore float32
					var bestPerson int64
					for _, s := range seeds {
						if sc := embedvec.CosineSim(unassigned[i].Vec, s.Vec); sc > bestScore {
							bestScore, bestPerson = sc, s.PersonID
						}
					}
					if bestScore >= threshold {
						matches[i] = match{faceID: unassigned[i].ID, personID: bestPerson}
					}
				}
			}(lo, hi)
		}
		wg.Wait()
	}
	var leftovers []media.Face
	for i, m := range matches {
		if m.personID != 0 {
			if err := media.AssignFace(db, m.faceID, m.personID, "auto"); err != nil {
				return stats, err
			}
			stats.JoinedExisting++
		} else {
			leftovers = append(leftovers, unassigned[i])
		}
	}

	// Phase 2: greedy leader clustering among the leftovers, strongest
	// detections first so cluster centroids start from the clearest faces.
	sort.SliceStable(leftovers, func(i, j int) bool { return leftovers[i].Score > leftovers[j].Score })
	type cluster struct {
		centroid []float32
		members  []int64
	}
	var clusters []cluster
	for _, f := range leftovers {
		bestIdx := -1
		var bestScore float32
		for ci := range clusters {
			if sc := embedvec.CosineSim(f.Vec, clusters[ci].centroid); sc > bestScore {
				bestScore, bestIdx = sc, ci
			}
		}
		if bestIdx >= 0 && bestScore >= threshold {
			c := &clusters[bestIdx]
			// Running-mean centroid, renormalized.
			n := float32(len(c.members))
			for k := range c.centroid {
				c.centroid[k] = (c.centroid[k]*n + f.Vec[k]) / (n + 1)
			}
			c.centroid = embedvec.Normalize(c.centroid)
			c.members = append(c.members, f.ID)
		} else {
			centroid := make([]float32, len(f.Vec))
			copy(centroid, f.Vec)
			clusters = append(clusters, cluster{centroid: centroid, members: []int64{f.ID}})
		}
	}

	for _, c := range clusters {
		if len(c.members) < minCluster {
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
		for _, faceID := range c.members {
			if err := media.AssignFace(db, faceID, pid, "auto"); err != nil {
				return stats, err
			}
			stats.NewlyClustered++
		}
		stats.NewPeople++
	}
	return stats, nil
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
//	--model=<id>       cluster a specific recognizer's faces (default: active)
//	--threshold=<0..1> override the model's match threshold
//	--min-cluster=<n>  minimum faces for a new anonymous person (default 3)
//	--reset            rebuild the anonymous "Unknown #N" clusters first.
//	                   User-assigned faces and everything inside NAMED people
//	                   are never touched — naming/merging a cluster endorses
//	                   its contents.
func facesClusterTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	model := ActiveFaceModel()
	if id, ok := embedModelOverrideFromJob(j); ok {
		if m, known := FaceModelByID(id); known {
			model = m
		} else {
			q.PushJobStdout(j.ID, fmt.Sprintf("Unknown --model %q; using active model %q", id, model.ID))
		}
	}
	threshold := model.MatchThreshold
	if v, ok := jobArgValue(j, "--threshold"); ok {
		if t, err := strconv.ParseFloat(v, 32); err == nil && t > 0 && t < 1 {
			threshold = float32(t)
		}
	}
	minCluster := minAutoClusterSize
	if v, ok := jobArgValue(j, "--min-cluster"); ok {
		if n, err := strconv.Atoi(v); err == nil && n >= 1 {
			minCluster = n
		}
	}
	q.PushJobStdout(j.ID, fmt.Sprintf("Clustering faces: model=%s threshold=%.3f min-cluster=%d", model.ID, threshold, minCluster))

	if jobHasFlag(j, "--reset") {
		n, err := resetAutoAssignments(q.Db, model.ID)
		if err != nil {
			q.PushJobStdout(j.ID, "Reset failed: "+err.Error())
			q.ErrorJob(j.ID)
			return err
		}
		q.PushJobStdout(j.ID, fmt.Sprintf("Reset %d auto assignment(s) in unnamed clusters; named people and user labels kept", n))
	}

	stats, err := clusterFaces(q.Db, model, threshold, minCluster)
	if err != nil {
		q.PushJobStdout(j.ID, "Clustering failed: "+err.Error())
		q.ErrorJob(j.ID)
		return err
	}
	q.PushJobStdout(j.ID, fmt.Sprintf(
		"Completed: %d joined existing people, %d new people (%d faces), %d left unassigned",
		stats.JoinedExisting, stats.NewPeople, stats.NewlyClustered, stats.Unassigned,
	))
	q.CompleteJob(j.ID)
	return nil
}
