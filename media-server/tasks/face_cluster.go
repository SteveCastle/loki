package tasks

import (
	"database/sql"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"sync"

	"github.com/stevecastle/shrike/appconfig"
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

// userSeedWeight is how much a USER-assigned face (hand-confirmed, "locked")
// counts relative to an auto-assigned one in corroboration counting: a single
// user-seed near-match carries the corroborating force of several auto
// matches. The mean-similarity guard goes further than weighting — when a
// person has ANY user seeds, the guard's center is computed from those alone
// (see scoreAgainstSeeds), so no volume of auto joins can drift the identity
// away from what the human confirmed.
const userSeedWeight = 3

// meanJoinSlack bounds how far a face's MEAN similarity to a person's faces
// may sit below the join threshold. The best-single-match rule alone is
// single-linkage: face A joins via member B, becomes a seed, pulls in C via
// itself, and so on — each hop needs only one good match, so over repeated
// passes (the incremental in-scan clustering runs every ~500 faces) a person
// degenerates into a transitive chain whose internal similarity is near
// random. Requiring the mean over ALL the person's faces to stay within this
// slack of the threshold blocks chain drift (a chained blob's mean is far
// below any plausible floor) while leaving room for genuinely multi-modal
// people (age/lighting/pose spread) where a hard positive matches half the
// cluster well and the other half loosely.
const meanJoinSlack = 2 * corroborationSlack

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
	// onlyFaceIDs, when non-nil, restricts the CANDIDATE set to these faces:
	// only they may join people or form clusters. Assigned faces still count
	// as seeds/match targets in full. This is what keeps the in-scan
	// incremental pass O(batch) instead of O(library): without it every pass
	// re-scored the entire unassigned backlog — thousands of hard/junk faces
	// that never join anything — against every seed, so passes got slower as
	// the library grew. nil = consider every unassigned face (full pass).
	onlyFaceIDs map[int64]bool
}

// defaultClusterParams starts from the recognizer's defaults and applies the
// SAVED grouping tuner (People panel Tune sliders, persisted in the server
// config) — so every clustering pass, including the plain Group new faces /
// Rebuild buttons and the incremental in-scan passes, runs with the tuned
// values. Explicit faces-cluster job flags override these per run (see
// clusterOneModel).
func defaultClusterParams(model FaceModel) clusterParams {
	cfg := appconfig.Get()
	t := model.MatchThreshold
	if o := cfg.FaceClusterThresholdOffset; o >= -0.2 && o <= 0.3 {
		t += float32(o)
	}
	minQuality := 0.75
	if q := cfg.FaceClusterMinQuality; q > 0 && q < 1 {
		minQuality = q
	}
	minCluster := minAutoClusterSize
	if n := cfg.FaceClusterMinCluster; n >= 1 {
		minCluster = n
	}
	return clusterParams{
		joinThreshold: t,
		formThreshold: t + 0.05,
		minQuality:    minQuality,
		minCluster:    minCluster,
		passes:        2,
	}
}

// incrementalClusterParams are the knobs for the frequent IN-SCAN passes.
// They are deliberately stricter than the defaults: a pass that runs every
// ~500 faces retests borderline candidates many times per scan (each
// retest is another chance for a false join), sees only small batches (easy
// for a few borderline faces to look coherent), and every join it makes
// becomes a full-strength seed for all later passes — so early mistakes
// snowball. Mid-scan grouping therefore takes only clearly-confident
// evidence; the FINAL pass at scan end runs the normal defaults once, which
// is where borderline cases are settled (same end quality as the old
// scan-then-cluster pipeline).
func incrementalClusterParams(model FaceModel) clusterParams {
	p := defaultClusterParams(model)
	p.joinThreshold += 0.03
	p.formThreshold += 0.03
	p.minCluster += 2 // small batches need more corroborating members
	// Only confident detections may found people mid-scan; a tuned floor
	// above 0.8 stays in force (stricter of the two).
	if p.minQuality < 0.8 {
		p.minQuality = 0.8
	}
	p.passes = 1 // no intra-pass transitivity between full passes
	return p
}

// autoAssignNewFaces incrementally assigns freshly-scanned faces to existing
// people using the same corroborated-join rule as the full clustering pass.
// Called by the faces collector after storing a batch, so growing a library
// keeps enriching existing people without a full recluster. Requires the live
// face index to hold this model (otherwise it's a no-op — the faces-cluster
// task will pick the faces up later). Returns how many faces were assigned.
// Curation constraints need no lookup here: these are brand-new face rows, so
// no veto or cannot-link can exist for them yet (and AssignFace re-checks
// vetoes anyway).
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
		if bestPerson != 0 && acceptJoin(best, threshold) &&
			personMeanSimAtLeast(db, model.ID, bestPerson, faces[i].Vec, threshold-meanJoinSlack) {
			if err := media.AssignFace(db, id, bestPerson, "auto"); err == nil {
				assigned++
			}
		}
	}
	return assigned
}

// personMeanSimAtLeast reports whether vec's mean cosine over the person's
// stored faces clears floor — the incremental-assignment form of the
// mean-similarity guard (see meanJoinSlack). Like scoreAgainstSeeds, the mean
// is taken over the person's USER-assigned faces alone when any exist (the
// confirmed center — it cannot drift as auto joins pile up mid-scan),
// otherwise over a random sample of all its faces. The per-face rule matches
// on the best of 12 index hits, i.e. single linkage; without this guard every
// assignment becomes a new match target and the person grows by transitive
// chaining for the rest of the scan. User faces sort first so they are always
// inside the 256-row cost cap.
func personMeanSimAtLeast(db *sql.DB, model string, personID int64, vec []float32, floor float32) bool {
	rows, err := db.Query(
		`SELECT vector, COALESCE(assigned_by, '') FROM face WHERE person_id=? AND model=?
		 ORDER BY (COALESCE(assigned_by, '') = 'user') DESC, RANDOM() LIMIT 256`, personID, model)
	if err != nil {
		return false
	}
	defer rows.Close()
	var total, userTotal float64
	var n, userN int
	for rows.Next() {
		var blob []byte
		var assignedBy string
		if rows.Scan(&blob, &assignedBy) != nil {
			return false
		}
		v, err := embedvec.Decode(blob)
		if err != nil {
			continue
		}
		sim := float64(embedvec.CosineSim(vec, v))
		total += sim
		n++
		if assignedBy == "user" {
			userTotal += sim
			userN++
		}
	}
	if rows.Err() != nil {
		return false
	}
	if userN > 0 {
		return float32(userTotal/float64(userN)) >= floor
	}
	if n == 0 {
		return true // no faces stored → nothing to contradict the join
	}
	return float32(total/float64(n)) >= floor
}

// clusterStats summarises one full clustering pass.
type clusterStats struct {
	JoinedExisting int // unassigned faces that joined an existing person
	NewPeople      int // anonymous "Unknown #N" persons created
	NewlyClustered int // faces assigned into those new persons
	Unassigned     int // faces left unassigned (small/incoherent clusters)
	QualitySkipped int // faces below the quality floor (excluded from phase 2)
	Discarded      int // faces in clusters dropped by the coherence check
	BanBlocked     int // faces in clusters blocked by dissolved-group bans
}

// seed is one already-assigned face acting as a join anchor.
type seed struct {
	id       int64 // face id — checked against cannot-link assertions
	vec      []float32
	personID int64
	user     bool // user-assigned (ground truth) → userSeedWeight
}

// pairSet is a symmetric/keyed constraint lookup: face → forbidden ids
// (persons for vetoes, other faces for cannot-links).
type pairSet = map[int64]map[int64]bool

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
//
// Human curation assertions constrain every step: a face never joins a person
// it has a veto against, never joins a person (or a forming cluster) holding a
// face it is cannot-linked to, and user-assigned seeds carry userSeedWeight in
// all evidence aggregation.
func clusterFaces(db *sql.DB, model FaceModel, p clusterParams) (clusterStats, error) {
	var stats clusterStats
	all, err := media.LoadAllFaces(db, model.ID)
	if err != nil {
		return stats, err
	}
	vetoes, err := media.FaceVetoes(db, model.ID)
	if err != nil {
		return stats, err
	}
	cannot, err := media.FaceCannotLinks(db, model.ID)
	if err != nil {
		return stats, err
	}
	bans, err := media.FaceGroupBans(db, model.ID)
	if err != nil {
		return stats, err
	}
	var seeds []seed
	var unassigned []media.Face
	for _, f := range all {
		// Restricted (incremental) pass: unassigned faces outside the batch
		// are not candidates — skip before the normalize so the backlog costs
		// nothing at all.
		if f.PersonID == 0 && p.onlyFaceIDs != nil && !p.onlyFaceIDs[f.ID] {
			continue
		}
		// Normalize defensively: cosine math below assumes unit vectors.
		f.Vec = embedvec.Normalize(f.Vec)
		if f.PersonID != 0 {
			seeds = append(seeds, seed{id: f.ID, vec: f.Vec, personID: f.PersonID, user: f.AssignedBy == "user"})
		} else {
			unassigned = append(unassigned, f)
		}
	}

	// Phase 1: corroborated joins against assigned seeds, p.passes rounds.
	for pass := 0; pass < p.passes && len(seeds) > 0 && len(unassigned) > 0; pass++ {
		matches := scoreAgainstSeeds(unassigned, seeds, p.joinThreshold, vetoes, cannot)
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
			seeds = append(seeds, seed{id: unassigned[i].ID, vec: unassigned[i].Vec, personID: personID})
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

	// Greedy AVERAGE-LINKAGE clustering among the eligible leftovers, strongest
	// detections first so clusters start from the clearest faces. A face joins
	// the cluster with the best MEAN cosine to its members (dot with the
	// unnormalized member sum / count — all vectors are unit). Comparing
	// against a NORMALIZED centroid instead is a trap: in high dimensions the
	// members' noise cancels in the mean, so cosine-to-normalized-centroid
	// ≈ sqrt(mean pairwise similarity) — a blob whose members agree at a
	// near-random 0.17 scores ~0.41 against its own centroid and sails past
	// the formation gate. Mean-to-members is the honest number.
	sort.SliceStable(eligible, func(i, j int) bool { return eligible[i].Score > eligible[j].Score })
	type cluster struct {
		sum     []float32 // unnormalized sum of (unit) member vectors
		members []media.Face
	}
	var clusters []cluster
	for _, f := range eligible {
		bestIdx := -1
		var bestScore float32
		cl := cannot[f.ID]
		for ci := range clusters {
			// A cannot-link to ANY member forbids the cluster: this is how a
			// rejection outlives the person it was recorded against — the same
			// visual group re-forming from its exemplars can't reabsorb the
			// rejected face.
			if len(cl) > 0 {
				blocked := false
				for _, m := range clusters[ci].members {
					if cl[m.ID] {
						blocked = true
						break
					}
				}
				if blocked {
					continue
				}
			}
			sc := dot32(f.Vec, clusters[ci].sum) / float32(len(clusters[ci].members))
			if sc > bestScore {
				bestScore, bestIdx = sc, ci
			}
		}
		if bestIdx >= 0 && bestScore >= p.formThreshold {
			c := &clusters[bestIdx]
			for k := range c.sum {
				c.sum[k] += f.Vec[k]
			}
			c.members = append(c.members, f)
		} else {
			sum := make([]float32, len(f.Vec))
			copy(sum, f.Vec)
			clusters = append(clusters, cluster{sum: sum, members: []media.Face{f}})
		}
	}

	for _, c := range clusters {
		if len(c.members) < p.minCluster {
			stats.Unassigned += len(c.members)
			continue
		}
		// Coherence check: the members' MEAN PAIRWISE cosine must itself clear
		// the formation threshold. Every join already required mean-to-members
		// ≥ formThreshold, so this holds by construction — it stays as a cheap
		// safety net against future rule changes and float drift.
		if meanPairwise(c.sum, len(c.members)) < p.formThreshold {
			stats.Discarded += len(c.members)
			stats.Unassigned += len(c.members)
			continue
		}
		// Dissolved-group bans: a cluster that would reunite the majority of
		// a group the user deleted is refused — its faces stay unassigned
		// (visible in the Ungrouped pool for manual triage). Genuine subsets
		// below the majority line still form freely.
		if clusterReunitesBan(c.members, bans) {
			stats.BanBlocked += len(c.members)
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

// clusterReunitesBan reports whether a candidate cluster recreates any
// dissolved group (see media.FaceGroupBan.Reunites for the majority rule).
func clusterReunitesBan(members []media.Face, bans []media.FaceGroupBan) bool {
	if len(bans) == 0 {
		return false
	}
	ids := make([]int64, len(members))
	for i, m := range members {
		ids[i] = m.ID
	}
	for _, b := range bans {
		if b.Reunites(ids) {
			return true
		}
	}
	return false
}

// dot32 is the raw float32 dot product (CosineSim renormalizes, which the
// sum-vector tricks here must avoid).
func dot32(a, b []float32) float32 {
	var d float64
	for i := range a {
		d += float64(a[i]) * float64(b[i])
	}
	return float32(d)
}

// meanPairwise computes the mean pairwise cosine of n unit vectors from their
// unnormalized sum: ||Σx||² = n + Σ_{i≠j} xᵢ·xⱼ, so the pairwise mean is
// (||Σx||² − n) / (n(n−1)). O(d) instead of O(n²d).
func meanPairwise(sum []float32, n int) float32 {
	if n < 2 {
		return 1
	}
	s2 := dot32(sum, sum)
	return (s2 - float32(n)) / float32(n*(n-1))
}

// scoreAgainstSeeds computes, for every unassigned face, the person it should
// join under the corroborated-join rule (0 = no join). A candidate person
// must ALSO pass the mean-similarity guard (see meanJoinSlack): the face's
// mean cosine over all the person's seed faces stays within meanJoinSlack of
// the threshold, or the join is a chain hop, not a match. Scoring is
// parallel; the caller applies the writes serially.
//
// Human assertions shape the outcome three ways: a vetoed person is never a
// candidate for that face; a person holding a seed the face is cannot-linked
// to is never a candidate; and USER seeds count userSeedWeight× as
// corroborators AND, when a person has any, they alone define the mean
// guard's center — the anchor a candidate is measured against never drifts,
// no matter how many auto faces the person accumulates.
func scoreAgainstSeeds(unassigned []media.Face, seeds []seed, threshold float32, vetoes, cannot pairSet) []int64 {
	matches := make([]int64, len(unassigned))
	if len(unassigned) == 0 || len(seeds) == 0 {
		return matches
	}
	// Per-person mean-guard centers (seed vectors are unit). A person with
	// user-confirmed seeds is anchored to THEIR mean only: a weighted
	// all-seed mean can still drift once wrong auto joins outnumber the
	// confirmed core's weight (8 confirmed × weight 3 lose to ~25 strays),
	// and each pass's joins seed the next, so drift snowballs. The confirmed
	// center is immutable within a run. Purely automatic clusters fall back
	// to the plain all-seed mean.
	type personAgg struct {
		sum     []float32 // every seed (fallback center)
		n       float32
		userSum []float32 // user-assigned seeds only (the anchor; nil if none)
		userN   float32
	}
	aggs := make(map[int64]*personAgg)
	for _, s := range seeds {
		a := aggs[s.personID]
		if a == nil {
			a = &personAgg{sum: make([]float32, len(s.vec))}
			aggs[s.personID] = a
		}
		for k, x := range s.vec {
			a.sum[k] += x
		}
		a.n++
		if s.user {
			if a.userSum == nil {
				a.userSum = make([]float32, len(s.vec))
			}
			for k, x := range s.vec {
				a.userSum[k] += x
			}
			a.userN++
		}
	}
	meanFloor := threshold - meanJoinSlack
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
			forbidden := map[int64]bool{}
			for i := lo; i < hi; i++ {
				for k := range scores {
					delete(scores, k)
				}
				for k := range forbidden {
					delete(forbidden, k)
				}
				veto := vetoes[unassigned[i].ID]
				cl := cannot[unassigned[i].ID]
				for pid := range veto {
					forbidden[pid] = true
				}
				for _, s := range seeds {
					if cl[s.id] {
						// Cannot-linked to a member → the whole person is off
						// the table, no matter how well other members match.
						forbidden[s.personID] = true
						continue
					}
					if forbidden[s.personID] {
						continue
					}
					sc := embedvec.CosineSim(unassigned[i].Vec, s.vec)
					if sc < threshold-corroborationSlack {
						continue // can neither win nor corroborate
					}
					ps := scores[s.personID]
					if sc > ps.best {
						ps.best = sc
					}
					if s.user {
						ps.count += userSeedWeight
					} else {
						ps.count++
					}
					scores[s.personID] = ps
				}
				var bestPerson int64
				var best personScore
				for pid, ps := range scores {
					if forbidden[pid] || !acceptJoin(ps, threshold) {
						continue
					}
					a := aggs[pid]
					if a == nil {
						continue
					}
					mean := dot32(unassigned[i].Vec, a.sum) / a.n
					if a.userN > 0 {
						mean = dot32(unassigned[i].Vec, a.userSum) / a.userN
					}
					if mean < meanFloor {
						continue // strong single match, but off the person's center
					}
					if bestPerson == 0 || effectiveScore(ps) > effectiveScore(best) {
						bestPerson, best = pid, ps
					}
				}
				matches[i] = bestPerson
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
	return resetAssignments(db, `
		SELECT f.id FROM face f
		LEFT JOIN person p ON p.id = f.person_id
		WHERE f.model = ? AND f.assigned_by = 'auto'
		  AND (p.id IS NULL OR p.name LIKE 'Unknown #%')`, model)
}

// resetAllAutoAssignments clears EVERY auto assignment for model — including
// faces sitting inside named people — keeping only user-assigned faces as
// ground truth. This is the recovery hatch for a poisoned library: once a bad
// auto cluster has been renamed (naming normally endorses its contents), the
// anonymous-only reset can never dislodge it, so reclustering deterministically
// rebuilds the same groups. Named person rows survive (possibly empty) so the
// following clustering pass can regrow them from their user-assigned seeds.
func resetAllAutoAssignments(db *sql.DB, model string) (int, error) {
	return resetAssignments(db, `SELECT id FROM face WHERE model = ? AND assigned_by = 'auto'`, model)
}

// resetAssignments unassigns the faces selected by query, then dissolves
// now-empty anonymous "Unknown #N" clusters (named people are kept even when
// empty — the user made them, the user deletes them). Returns how many faces
// were unassigned.
func resetAssignments(db *sql.DB, query, model string) (int, error) {
	rows, err := db.Query(query, model)
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
//	--threshold-offset=<±> shift join AND form thresholds relative to each
//	                       model's default (the tuning-slider form: one value
//	                       works across routed models with different scales)
//	--form-threshold=<0..1> override the new-cluster threshold (join + 0.05)
//	--min-quality=<0..1>   detection-confidence floor for new clusters (0.75)
//	--min-cluster=<n>      minimum faces for a new anonymous person (default 3)
//	--passes=<n>           phase-1 join iterations (default 2)
//	--reset                rebuild the anonymous "Unknown #N" clusters first.
//	                       User-assigned faces and everything inside NAMED
//	                       people are never touched — naming/merging a cluster
//	                       endorses its contents.
//	--reset-all            clear EVERY auto assignment first, including inside
//	                       named people (user labels alone survive) — the
//	                       from-scratch re-run for parameter tuning/recovery.
//
// Human curation always survives and constrains every run, whatever the flags:
// user-assigned faces stay put and seed clustering at userSeedWeight×, and
// rejections (face_veto + face_cannot_link) permanently keep a face out of the
// group it was removed from — even when that group is dissolved by a reset and
// re-forms under a new id.
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

// clusterModelIDs resolves the recognizer set a clustering pass would process:
// every known model with stored faces when routing is enabled, otherwise the
// active model only. Shared by the ungrouped-face count/list so the UI reports
// exactly the workload "Group new faces" would see.
func clusterModelIDs(db *sql.DB) ([]string, error) {
	var models []string
	if FaceRoutingEnabled() {
		ids, err := faceModelsWithFaces(db)
		if err != nil {
			return nil, err
		}
		for _, id := range ids {
			if _, known := FaceModelByID(id); known {
				models = append(models, id)
			}
		}
	}
	if len(models) == 0 {
		models = []string{ActiveFaceModel().ID}
	}
	return models, nil
}

func modelPlaceholders(models []string) (string, []any) {
	placeholders := strings.TrimSuffix(strings.Repeat("?,", len(models)), ",")
	args := make([]any, len(models))
	for i, m := range models {
		args[i] = m
	}
	return placeholders, args
}

// CountUngroupedFaces reports how many stored faces the next "Group new
// faces" pass would try to place: faces with no person assignment, counted
// over the recognizer set the pass itself would process. Powers the count
// shown next to the grouping button.
func CountUngroupedFaces(db *sql.DB) (int, error) {
	models, err := clusterModelIDs(db)
	if err != nil {
		return 0, err
	}
	placeholders, args := modelPlaceholders(models)
	var n int
	err = db.QueryRow(
		`SELECT COUNT(*) FROM face
		 WHERE COALESCE(person_id, 0) = 0 AND model IN (`+placeholders+`)`,
		args...).Scan(&n)
	return n, err
}

// UngroupedFace is one not-yet-assigned face, for the manual-review UI.
type UngroupedFace struct {
	ID        int64   `json:"id"`
	MediaPath string  `json:"path"`
	FrameTS   float64 `json:"frameTs"`
	DetScore  float64 `json:"detScore"`
	Model     string  `json:"model"`
}

// ListUngroupedFaces pages through the faces CountUngroupedFaces counts,
// best detections first — those are the faces that plausibly SHOULD have
// grouped, i.e. the interesting failures; the blurry tail sorts last.
func ListUngroupedFaces(db *sql.DB, limit, offset int) ([]UngroupedFace, error) {
	models, err := clusterModelIDs(db)
	if err != nil {
		return nil, err
	}
	placeholders, args := modelPlaceholders(models)
	args = append(args, limit, offset)
	rows, err := db.Query(
		`SELECT id, media_path, COALESCE(frame_ts, 0), det_score, model FROM face
		 WHERE COALESCE(person_id, 0) = 0 AND model IN (`+placeholders+`)
		 ORDER BY det_score DESC, id LIMIT ? OFFSET ?`,
		args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []UngroupedFace{}
	for rows.Next() {
		var f UngroupedFace
		if err := rows.Scan(&f.ID, &f.MediaPath, &f.FrameTS, &f.DetScore, &f.Model); err != nil {
			return nil, err
		}
		out = append(out, f)
	}
	return out, rows.Err()
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
	if v, ok := jobArgValue(j, "--threshold-offset"); ok {
		if t, err := strconv.ParseFloat(v, 32); err == nil && t >= -0.2 && t <= 0.3 {
			// Offset from the recognizer's OWN default — replaces the saved
			// tuner offset already folded into p, never stacks on it.
			p.joinThreshold = model.MatchThreshold + float32(t)
			p.formThreshold = model.MatchThreshold + 0.05 + float32(t)
		}
	}
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

	if jobHasFlag(j, "--reset-all") {
		n, err := resetAllAutoAssignments(q.Db, model.ID)
		if err != nil {
			q.PushJobStdout(j.ID, "Reset failed: "+err.Error())
			q.ErrorJob(j.ID)
			return err
		}
		q.PushJobStdout(j.ID, fmt.Sprintf("Reset ALL %d auto assignment(s); only user labels kept", n))
	} else if jobHasFlag(j, "--reset") {
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
		"%s: %d joined existing people, %d new people (%d faces), %d left unassigned (%d below quality floor, %d in discarded incoherent clusters, %d blocked by dissolved-group bans)",
		model.ID, stats.JoinedExisting, stats.NewPeople, stats.NewlyClustered, stats.Unassigned, stats.QualitySkipped, stats.Discarded, stats.BanBlocked,
	))
	// Live UIs (People grid) refetch on this instead of waiting for the
	// whole job to complete (multi-model runs cluster one model at a time).
	broadcastPeopleUpdated([]string{model.ID})
	return nil
}
