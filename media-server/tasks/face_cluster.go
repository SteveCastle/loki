package tasks

import (
	"context"
	"database/sql"
	"fmt"
	"math"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

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
// may sit below the join threshold — for people WITH user-confirmed faces
// only. The best-single-match rule alone is single-linkage: face A joins via
// member B, becomes a seed, pulls in C via itself, and so on — each hop
// needs only one good match, so over repeated passes (the incremental
// in-scan clustering runs every ~500 faces) a person degenerates into a
// transitive chain whose internal similarity is near random. Requiring the
// mean to stay within this slack of the threshold blocks chain drift while
// leaving room for genuinely multi-modal people (age/lighting/pose spread)
// where a hard positive matches half the cluster well and the other half
// loosely. Multi-modality is a property a human establishes by confirming
// faces, so purely automatic people get NO slack — their mean floor is the
// join threshold itself (see scoreAgainstSeeds).
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
	// flushEvery is how many candidate faces a pass may process before its
	// results are written. It is the granularity of everything the user sees
	// and keeps: joins land per chunk, forming clusters are committed (and
	// topped up) as they cross the member floor, and a cancel keeps every
	// flush that already happened. 0 = defaultClusterFlushEvery.
	flushEvery int
	// progress, when set, is called after every flush and periodically inside
	// the prescreen. It runs on the goroutine driving the pass, so it must not
	// block for long.
	progress func(clusterProgress)
}

// clusterPhase names the stage a progress tick came from. A pass's cost splits
// across three very different shapes of work, and "which one is running" is
// most of what a user watching a long job wants to know.
type clusterPhase string

const (
	phaseLoad      clusterPhase = "load"
	phaseJoin      clusterPhase = "join"
	phasePrescreen clusterPhase = "prescreen"
	phaseForm      clusterPhase = "form"
)

// clusterProgress is one progress tick from a clustering pass.
type clusterProgress struct {
	Phase clusterPhase
	Pass  int // 1-based phase-1 pass number (0 outside phase 1)
	// Done/Total measure the CURRENT phase's own scan (rows prescreened,
	// candidates scored...). They restart per phase.
	Done, Total int
	// Processed/Workload are the pass-wide bar, in candidate faces. Processed
	// counts work GOT THROUGH, not faces placed: a candidate that matches
	// nothing still cost a full scan against every seed and still advances the
	// bar. (Counting placements instead left the bar frozen near zero for the
	// whole of phase 1 on a library without many people yet — the exact case
	// where the run is longest.) Monotonic, and Processed == Workload exactly
	// when the pass finishes, whichever route it took to get there.
	Processed, Workload int
	// Stats are the running totals at this point — everything counted here is
	// already committed to the database.
	Stats clusterStats
}

// defaultClusterFlushEvery is the default candidate-faces-per-flush. Small
// enough that a 100k-face backlog reports and persists dozens of times a
// minute; large enough that the per-flush transaction and progress overhead
// stays lost in the scoring cost.
const defaultClusterFlushEvery = 2000

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
	// Tuned like every clustering pass: the saved tuner offset applies here
	// too. This gate runs on every commit batch — by volume it makes more
	// join decisions than any pass — so it must never sit below the
	// thresholds the passes themselves enforce (it used to run at the raw
	// model default while the passes ran tuned).
	threshold := defaultClusterParams(model).joinThreshold
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
			personMeanSimAtLeast(db, model.ID, bestPerson, faces[i].Vec, threshold-meanJoinSlack, threshold) {
			if err := media.AssignFace(db, id, bestPerson, "auto"); err == nil {
				assigned++
			}
		}
	}
	return assigned
}

// personMeanSimAtLeast reports whether vec's mean cosine over the person's
// stored faces clears the applicable floor — the incremental-assignment form
// of the mean-similarity guard (see meanJoinSlack). Like scoreAgainstSeeds,
// the mean is taken over the person's USER-assigned faces alone when any
// exist (the confirmed center — it cannot drift as auto joins pile up
// mid-scan) and compared against userFloor; a purely automatic person is
// measured over a random sample of all its faces against the stricter
// autoFloor — see scoreAgainstSeeds for why auto-only people get no slack.
// The per-face rule matches on the best of 12 index hits, i.e. single
// linkage; without this guard every assignment becomes a new match target
// and the person grows by transitive chaining for the rest of the scan.
// User faces sort first so they are always inside the 256-row cost cap.
func personMeanSimAtLeast(db *sql.DB, model string, personID int64, vec []float32, userFloor, autoFloor float32) bool {
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
		return float32(userTotal/float64(userN)) >= userFloor
	}
	if n == 0 {
		return true // no faces stored → nothing to contradict the join
	}
	return float32(total/float64(n)) >= autoFloor
}

// clusterWorkerBudget is the goroutine count for the clustering hot loops
// (seed scoring, prescreen, cluster scans). Clustering runs INSIDE the server
// process — not in worker subprocesses — so it must respect the faces
// performance preset's CPU budget the way the scan workers do; sizing to
// GOMAXPROCS pegged every core on the machine for the whole pass. The budget
// is workers×threads as resolved for the CPU provider (the DirectML worker
// cap is a GPU-contention rule and doesn't apply to pure-CPU math), clamped
// to the core count.
func clusterWorkerBudget() int {
	cfg := appconfig.Get()
	w, t := resolveResources(cfg.FacePerformance, cfg.FaceWorkers, cfg.FaceThreadsPerWorker, "cpu")
	budget := w * t
	if cpus := runtime.NumCPU(); budget > cpus {
		budget = cpus
	}
	if budget < 1 {
		budget = 1
	}
	return budget
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
	// FormFullScans counts faces too densely surrounded for the prescreen to
	// record their neighbours exactly (see maxFormCandidates), which put them
	// on the full every-cluster scan. It is the one input shape that keeps
	// phase 2 quadratic, and a large value means one identity dominates the
	// library — worth reporting rather than silently absorbing.
	FormFullScans int
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

// personAgg is one person's mean-guard center (seed vectors are unit). A
// person with user-confirmed seeds is anchored to THEIR mean only: a weighted
// all-seed mean can still drift once wrong auto joins outnumber the confirmed
// core's weight (8 confirmed × weight 3 lose to ~25 strays), and each pass's
// joins seed the next, so drift snowballs. The confirmed center is immutable
// within a run. Purely automatic clusters fall back to the plain all-seed mean.
type personAgg struct {
	sum     []float32 // every seed (fallback center)
	n       float32
	userSum []float32 // user-assigned seeds only (the anchor; nil if none)
	userN   float32
}

// buildPersonAggs folds a pass's seed set into per-person centers. Hoisted out
// of scoreAgainstSeeds because a chunked phase 1 calls the scorer once per
// chunk against the SAME frozen seed set — rebuilding the centers per chunk
// would be pure waste, and rebuilding them from a growing seed set would break
// the guarantee that chunking changes only write timing.
func buildPersonAggs(seeds []seed) map[int64]*personAgg {
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
	return aggs
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
//
// Human curation assertions constrain every step: a face never joins a person
// it has a veto against, never joins a person (or a forming cluster) holding a
// face it is cannot-linked to, and user-assigned seeds carry userSeedWeight in
// all evidence aggregation.
//
// A pass over a big backlog is minutes of every-core compute, so ctx is
// checked throughout (including inside the parallel hot loops): cancelling the
// owning job actually stops the burn instead of leaving the server pegged with
// nothing visibly running.
//
// Both phases persist CONTINUOUSLY, in p.flushEvery-sized steps, and report
// progress at each step. This is what makes a 100k-face backlog usable: people
// appear in the UI within seconds instead of after the whole run, and a cancel
// (or a crash) keeps every flush that already happened. Resuming is just
// running the task again — committed faces are no longer unassigned, so the
// next run starts from the remaining backlog and grows the people this one
// created. Flushing at the end only (flushEvery ≥ the candidate count)
// produces exactly the state the pass produced when it committed once at the
// end, so the incremental path is a pure change of write timing.
func clusterFaces(ctx context.Context, db *sql.DB, model FaceModel, p clusterParams) (clusterStats, error) {
	var stats clusterStats
	if err := ctx.Err(); err != nil {
		return stats, err
	}
	flushEvery := p.flushEvery
	if flushEvery <= 0 {
		flushEvery = defaultClusterFlushEvery
	}
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
	// Lay every vector out contiguously before the pairwise loops touch them —
	// both hot phases stream one of these arrays per comparison, and scattered
	// per-vector allocations defeat prefetch entirely. Must happen before the
	// seed/candidate split so both sides land in the same arena.
	arena := packFaceVectors(all)
	defer runtime.KeepAlive(arena)
	var seeds []seed
	var unassigned []media.Face
	for _, f := range all {
		// Restricted (incremental) pass: unassigned faces outside the batch
		// are not candidates.
		if f.PersonID == 0 && p.onlyFaceIDs != nil && !p.onlyFaceIDs[f.ID] {
			continue
		}
		if f.PersonID != 0 {
			seeds = append(seeds, seed{id: f.ID, vec: f.Vec, personID: f.PersonID, user: f.AssignedBy == "user"})
		} else {
			unassigned = append(unassigned, f)
		}
	}

	// The pass-wide bar measures WORK GOT THROUGH, not faces placed. A
	// candidate that joins nothing still costs a full scan against every seed,
	// so it has to move the bar.
	//
	// Every candidate is budgeted visitsPerFace visits: one per phase-1 join
	// pass, one for the prescreen, one for the greedy formation scan. visited
	// counts visits actually made; forfeited counts the ones a candidate will
	// never receive because it left the pool early — it joined a person, it
	// fell below the quality floor, or the join passes converged before using
	// their budget. Every route through the pass spends exactly visitsPerFace
	// per candidate (see the incremental tests), so the bar always lands
	// precisely on full instead of stalling short or being clamped there.
	//
	// Visits are weighted equally, which is a simplification — a prescreen row
	// and a seed scan are not the same amount of CPU — so the bar is a
	// throughput indicator, not a clock. The per-phase Done/Total carries the
	// detail.
	workload := len(unassigned)
	visitsPerFace := p.passes + 2
	visited, forfeited := 0, 0
	report := func(phase clusterPhase, pass, done, total int) {
		if p.progress == nil {
			return
		}
		processed := 0
		if visitsPerFace > 0 {
			processed = min((visited+forfeited)/visitsPerFace, workload)
		}
		p.progress(clusterProgress{
			Phase: phase, Pass: pass, Done: done, Total: total,
			Processed: processed, Workload: workload, Stats: stats,
		})
	}
	// Decoding every stored vector is the one stretch before the first flush
	// that produces nothing, and on a large library it is tens of seconds —
	// announce the size of the problem so the job doesn't look wedged.
	report(phaseLoad, 0, len(seeds), len(all))

	// Phase 1: corroborated joins against assigned seeds, p.passes rounds.
	// Each CHUNK's assignments go to the DB as one transaction — the per-face
	// AssignFace commit (plus its three point reads) used to dominate passes
	// that joined thousands of faces, and committing the whole pass at once
	// meant a 100k backlog wrote nothing for minutes. Only faces the bulk write
	// actually applied become seeds; a face skipped by its defense-in-depth
	// checks stays in the unassigned pool instead of masquerading as assigned.
	passesRun := 0
	for pass := 0; pass < p.passes && len(seeds) > 0 && len(unassigned) > 0; pass++ {
		passesRun++
		// The seed set is FROZEN for the whole pass: chunking is a write-timing
		// change, not a semantic one, so a candidate must see exactly the seeds
		// it would have seen scoring the pool in one go. The full slice
		// expression forces the appends below onto a new array, so this pass
		// can never observe its own joins. (Cross-pass seeding is the existing
		// bounded-transitivity rule — see the doc comment.)
		passSeeds := seeds[:len(seeds):len(seeds)]
		aggs := buildPersonAggs(passSeeds)
		var leftovers []media.Face
		appliedInPass := 0
		for lo := 0; lo < len(unassigned); lo += flushEvery {
			chunk := unassigned[lo:min(lo+flushEvery, len(unassigned))]
			matches := scoreAgainstSeeds(ctx, chunk, passSeeds, aggs, p.joinThreshold, vetoes, cannot)
			if err := ctx.Err(); err != nil {
				return stats, err // partial scores — don't apply them
			}
			var batch []media.FaceAssignment
			for i, personID := range matches {
				if personID != 0 {
					batch = append(batch, media.FaceAssignment{FaceID: chunk[i].ID, PersonID: personID})
				}
			}
			applied, err := media.AssignFacesAuto(db, batch)
			if err != nil {
				return stats, err
			}
			joined := make(map[int64]bool, len(applied))
			for _, a := range applied {
				joined[a.FaceID] = true
			}
			// Every candidate in the chunk was scored against every seed —
			// that is the visit, whether or not it found a person.
			visited += len(chunk)
			for i, f := range chunk {
				if joined[f.ID] {
					seeds = append(seeds, seed{id: f.ID, vec: f.Vec, personID: matches[i]})
					stats.JoinedExisting++
					// Placed: it will not be scored again, prescreened, or
					// scanned for formation.
					forfeited += (p.passes - pass - 1) + 2
				} else {
					leftovers = append(leftovers, f)
				}
			}
			appliedInPass += len(applied)
			report(phaseJoin, pass+1, lo+len(chunk), len(unassigned))
		}
		unassigned = leftovers
		if appliedInPass == 0 {
			break // converged early; further passes can't change anything
		}
	}
	// Whatever is still in the pool never spent the join passes that didn't
	// run — because phase 1 converged, or there were no seeds to score against
	// at all.
	forfeited += len(unassigned) * (p.passes - passesRun)

	// Phase 2: only confident detections may found new identities.
	var eligible []media.Face
	for _, f := range unassigned {
		if f.Score >= p.minQuality {
			eligible = append(eligible, f)
		} else {
			stats.QualitySkipped++
			stats.Unassigned++
			forfeited += 2 // never prescreened, never scanned for formation
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
	// Exact prescreen: face i only ever sees clusters built from faces before
	// it in processing order, and a cluster's mean-to-members can't exceed its
	// best single member match — so a face whose best similarity to EVERY
	// earlier face is below the formation threshold is guaranteed to found a
	// singleton, no cluster scan needed. A junk-heavy backlog (the common
	// shape of a large ungrouped pool) is almost entirely such faces, which
	// replaces the greedy loop's O(n²) growing-list traversal with this one
	// blocked, cache-friendly, fully parallel pairwise pass.
	prescreened := 0
	cand := prescreenCandidates(ctx, eligible, p.formThreshold, func(rows, total int) {
		visited += rows - prescreened // the callback reports a running total
		prescreened = rows
		report(phasePrescreen, 0, rows, total)
	})
	if err := ctx.Err(); err != nil {
		return stats, err // partial prescreen — the greedy loop can't trust it
	}
	var clusters []faceCluster
	// The greedy join must stay sequential (each join changes what later faces
	// see), and it used to be the hot spot of a big-backlog run: the cluster
	// list grows toward n singletons and every face streamed all of it, which
	// is memory-bandwidth bound long before it is compute bound. The prescreen
	// now hands over WHICH earlier faces each face can cluster with, so a face
	// only scans the clusters those neighbours landed in — see
	// bestClusterAmong for why that restriction is exact. Faces whose
	// neighbour list overflowed still take the full scan.
	//
	// clusterOf maps an eligible position to the cluster holding it, which is
	// what turns a neighbour list into a cluster list.
	clusterOf := make([]int32, len(eligible))
	scan := make([]int32, 0, maxFormCandidates)
	seenGen := make([]int32, len(eligible)+1)
	var gen int32
	overflowed := 0
	workers := clusterWorkerBudget()
	formVisitBase := visited
	for fi, f := range eligible {
		// Flush on the way past every flushEvery-th face AND on cancellation,
		// so what the run has already worked out is on disk either way.
		if fi > 0 && fi%flushEvery == 0 {
			if err := flushClusters(db, clusters, p, bans, &stats); err != nil {
				return stats, err
			}
			visited = formVisitBase + fi // fi faces scanned, placed or not
			report(phaseForm, 0, fi, len(eligible))
		}
		if fi&1023 == 0 {
			if err := ctx.Err(); err != nil {
				// Cancellation is the case incremental flushing exists for:
				// commit what the loop has built before unwinding.
				if ferr := flushClusters(db, clusters, p, bans, &stats); ferr != nil {
					return stats, ferr
				}
				return stats, err
			}
		}
		near, overflow := cand.near[fi], cand.overflow[fi]
		if len(near) == 0 && !overflow {
			clusterOf[fi] = int32(len(clusters))
			clusters = append(clusters, newFaceCluster(f))
			continue
		}

		var bestIdx int
		var bestScore float32
		if overflow {
			overflowed++
			bestIdx, bestScore = bestCluster(f, cannot[f.ID], clusters, workers)
		} else {
			// Neighbours → their clusters, deduplicated. Generation stamping
			// keeps that O(len(near)) without clearing a table per face.
			gen++
			scan = scan[:0]
			for _, j := range near {
				ci := clusterOf[j]
				if seenGen[ci] != gen {
					seenGen[ci] = gen
					scan = append(scan, ci)
				}
			}
			bestIdx, bestScore = bestClusterAmong(f, cannot[f.ID], clusters, scan)
		}

		if bestIdx >= 0 && bestScore >= p.formThreshold {
			c := &clusters[bestIdx]
			for k := range c.sum {
				c.sum[k] += f.Vec[k]
			}
			c.members = append(c.members, f)
			clusterOf[fi] = int32(bestIdx)
		} else {
			clusterOf[fi] = int32(len(clusters))
			clusters = append(clusters, newFaceCluster(f))
		}
	}
	stats.FormFullScans = overflowed
	if err := flushClusters(db, clusters, p, bans, &stats); err != nil {
		return stats, err
	}

	// Final accounting closes out the clusters that did NOT become people
	// (flushClusters has been keeping NewPeople/NewlyClustered current for the
	// ones that did), so the numbers are the same whether the run flushed once
	// or fifty times. Reasons are tested in the same order the single-shot
	// commit loop tested them.
	for _, c := range clusters {
		switch {
		case c.personID != 0:
			// Already counted at the flush that committed it.
		case len(c.members) < p.minCluster:
			stats.Unassigned += len(c.members)
		// Coherence check: the members' MEAN PAIRWISE cosine must itself clear
		// the formation threshold. Every join already required mean-to-members
		// ≥ formThreshold, so this holds by construction — it stays as a cheap
		// safety net against future rule changes and float drift.
		case meanPairwise(c.sum, len(c.members)) < p.formThreshold:
			stats.Discarded += len(c.members)
			stats.Unassigned += len(c.members)
		// Dissolved-group bans: a cluster that would reunite the majority of
		// a group the user deleted is refused — its faces stay unassigned
		// (visible in the Ungrouped pool for manual triage). Genuine subsets
		// below the majority line still form freely.
		case clusterReunitesBan(c.members, bans):
			stats.BanBlocked += len(c.members)
			stats.Unassigned += len(c.members)
		default:
			// Qualifies but uncommitted: only reachable when the context was
			// cancelled during the final flush.
			stats.Unassigned += len(c.members)
		}
	}
	visited = formVisitBase + len(eligible)
	report(phaseForm, 0, len(eligible), len(eligible))
	return stats, nil
}

// newFaceCluster starts a cluster from one face. The sum is a private copy —
// it accumulates as members join and must not alias the face's own vector.
func newFaceCluster(f media.Face) faceCluster {
	sum := make([]float32, len(f.Vec))
	copy(sum, f.Vec)
	return faceCluster{sum: sum, members: []media.Face{f}}
}

// flushClusters reconciles the in-memory phase-2 clusters with the database.
// A cluster that currently qualifies as a person — enough members, coherent,
// not reuniting a dissolved group — is created and then kept topped up as it
// grows; one that stops qualifying is retracted again. Both the coherence and
// the ban test can flip either way as members are added (Reunites in
// particular only ever gains overlap), so an early commit is provisional until
// the run ends: retraction is what keeps "commit as you go" from weakening the
// guarantees the single end-of-run commit gave.
//
// Retraction is rare in practice (every join already forces mean-to-members
// over the formation gate, and bans only exist in libraries where the user has
// dissolved a group), which is what makes the provisional writes worth it: the
// People grid fills in during the run, and a cancel keeps the result.
//
// Deliberately NOT cancellable: a flush is bounded database work whose whole
// purpose is to outlive the cancel that interrupted the pass.
//
// stats.NewPeople/NewlyClustered are kept in step with what is actually on
// disk (a retraction decrements them), so a progress tick taken between
// flushes reports committed reality rather than a running guess.
func flushClusters(db *sql.DB, clusters []faceCluster, p clusterParams, bans []media.FaceGroupBan, stats *clusterStats) error {
	for i := range clusters {
		c := &clusters[i]
		if c.written == len(c.members) {
			continue // unchanged since the last flush; the verdict can't have moved
		}
		qualifies := len(c.members) >= p.minCluster &&
			meanPairwise(c.sum, len(c.members)) >= p.formThreshold &&
			!clusterReunitesBan(c.members, bans)
		switch {
		case qualifies && c.personID == 0:
			name, err := media.NextUnknownName(db)
			if err != nil {
				return err
			}
			pid, err := media.CreatePerson(db, name)
			if err != nil {
				return err
			}
			c.personID = pid
			if err := c.assignFrom(db, 0, stats); err != nil {
				return err
			}
			stats.NewPeople++
		case qualifies:
			if err := c.assignFrom(db, c.written, stats); err != nil {
				return err
			}
		case c.personID != 0:
			ids := make([]int64, len(c.members))
			for k, m := range c.members {
				ids[k] = m.ID
			}
			if _, err := media.UnassignFacesBulk(db, ids); err != nil {
				return err
			}
			if err := media.DeletePerson(db, c.personID); err != nil {
				return err
			}
			stats.NewPeople--
			stats.NewlyClustered -= c.applied
			c.personID, c.written, c.applied = 0, 0, 0
		default:
			// Still short of the member floor — nothing to write, but record
			// that we looked so the next flush can skip it.
			c.written = len(c.members)
		}
	}
	return nil
}

// assignFrom writes members[from:] into the cluster's person in one
// transaction and advances the written watermark. applied tracks how many the
// bulk write actually took (user labels, vetoes and mid-run deletions are
// skipped), which is what the pass reports as newly clustered.
func (c *faceCluster) assignFrom(db *sql.DB, from int, stats *clusterStats) error {
	batch := make([]media.FaceAssignment, 0, len(c.members)-from)
	for _, m := range c.members[from:] {
		batch = append(batch, media.FaceAssignment{FaceID: m.ID, PersonID: c.personID})
	}
	applied, err := media.AssignFacesAuto(db, batch)
	if err != nil {
		return err
	}
	c.written = len(c.members)
	c.applied += len(applied)
	stats.NewlyClustered += len(applied)
	return nil
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

// dotf is the dot product for the two hot loops (phase-1 face×seed scoring,
// phase-2 face×cluster scans), which compare billions of pairs on a big
// backlog. All clustering vectors are unit (clusterFaces normalizes at load),
// so this replaces CosineSim's per-call renormalization — 3× the arithmetic
// for the same number — and skips dot32's per-element float64 conversion;
// four independent accumulators break the add dependency chain so the CPU can
// overlap the multiplies. Float32 accumulation over ≤1k-dim unit-scale inputs
// is exact to ~1e-6, noise against thresholds compared at 1e-2 granularity.
// Mismatched lengths return 0 (CosineSim's contract — mixed-model vectors
// must never match).
func dotf(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}
	b = b[:len(a)] // bounds-check elimination hint
	var s0, s1, s2, s3 float32
	i := 0
	for ; i+4 <= len(a); i += 4 {
		s0 += a[i] * b[i]
		s1 += a[i+1] * b[i+1]
		s2 += a[i+2] * b[i+2]
		s3 += a[i+3] * b[i+3]
	}
	for ; i < len(a); i++ {
		s0 += a[i] * b[i]
	}
	return (s0 + s1) + (s2 + s3)
}

// scoreGroup is how many candidates phase 1 scores against each seed at once
// — dot4's width.
const scoreGroup = 4

// dot4 computes the dot products of b against four rows at once. Same
// contract as dotf (unit vectors, mismatched lengths score 0), but it loads
// each b element once and reuses it across all four rows instead of streaming
// b again per pair — the pairwise sweeps are bandwidth-bound, so halving the
// loads per multiply is most of the win, and the four independent accumulator
// chains keep the FPU busy on top of that. Go does not auto-vectorize, so this
// register blocking is the available substitute for SIMD.
func dot4(a0, a1, a2, a3, b []float32) (float32, float32, float32, float32) {
	n := len(b)
	if len(a0) != n || len(a1) != n || len(a2) != n || len(a3) != n {
		return dotf(a0, b), dotf(a1, b), dotf(a2, b), dotf(a3, b)
	}
	// Re-slice to n so the compiler can drop the per-access bounds checks.
	a0, a1, a2, a3 = a0[:n], a1[:n], a2[:n], a3[:n]
	var s0, s1, s2, s3 float32
	var t0, t1, t2, t3 float32
	i := 0
	for ; i+2 <= n; i += 2 {
		x, y := b[i], b[i+1]
		s0 += a0[i] * x
		t0 += a0[i+1] * y
		s1 += a1[i] * x
		t1 += a1[i+1] * y
		s2 += a2[i] * x
		t2 += a2[i+1] * y
		s3 += a3[i] * x
		t3 += a3[i+1] * y
	}
	for ; i < n; i++ {
		x := b[i]
		s0 += a0[i] * x
		s1 += a1[i] * x
		s2 += a2[i] * x
		s3 += a3[i] * x
	}
	return s0 + t0, s1 + t1, s2 + t2, s3 + t3
}

// faceCluster is one forming phase-2 cluster: the unnormalized sum of its
// (unit) member vectors plus the members themselves, and the bookkeeping that
// lets it be committed while it is still growing (see flushClusters).
type faceCluster struct {
	sum     []float32
	members []media.Face
	// personID is the anonymous person this cluster has been committed as, or
	// 0 while it is still only in memory.
	personID int64
	// written is how many members the last flush accounted for — assigned when
	// personID is set, merely evaluated when it isn't. members beyond it are
	// what the next flush has to look at.
	written int
	// applied is how many assignments the bulk writes actually took (user
	// labels, vetoes and mid-run deletions are skipped), i.e. what the pass
	// reports as newly clustered.
	applied int
}

// bestClusterIn scans clusters[lo:hi] for the candidate with the highest mean
// cosine to f (dot with the unnormalized member sum / count — see the phase-2
// comment in clusterFaces for why the metric is mean-to-members, not
// centroid). A cluster holding any member f is cannot-linked to is skipped:
// this is how a rejection outlives the person it was recorded against — the
// same visual group re-forming from its exemplars can't reabsorb the rejected
// face. Returns (-1, 0) when nothing in the range scores above zero; ties
// resolve to the lowest index. Safe to run concurrently over disjoint ranges.
func bestClusterIn(f media.Face, cl map[int64]bool, clusters []faceCluster, lo, hi int) (int, float32) {
	bestIdx := -1
	var bestScore float32
	for ci := lo; ci < hi; ci++ {
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
		sc := dotf(f.Vec, clusters[ci].sum) / float32(len(clusters[ci].members))
		if sc > bestScore {
			bestScore, bestIdx = sc, ci
		}
	}
	return bestIdx, bestScore
}

// bestClusterAmong is bestClusterIn restricted to an explicit set of cluster
// indices — the clusters holding at least one of the query's prescreened near
// neighbours.
//
// This restriction is EXACT, not a heuristic. A cluster is only ever accepted
// when its mean-to-members clears formThreshold, and a mean can't exceed the
// best single member match, so any acceptable cluster must hold a member that
// is itself ≥ formThreshold to the query — and every such member, being an
// earlier face, is in the query's prescreen candidate list. Clusters with no
// candidate in them can therefore only score below the gate, where the caller
// discards them anyway.
//
// Ties resolve to the lowest cluster index, matching a full ascending scan,
// which is why the comparison carries the explicit index tie-break: candidates
// do not arrive in cluster order.
func bestClusterAmong(f media.Face, cl map[int64]bool, clusters []faceCluster, idxs []int32) (int, float32) {
	bestIdx := -1
	var bestScore float32
	for _, ci32 := range idxs {
		ci := int(ci32)
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
		sc := dotf(f.Vec, clusters[ci].sum) / float32(len(clusters[ci].members))
		if sc > bestScore || (bestIdx >= 0 && sc == bestScore && ci < bestIdx) {
			bestScore, bestIdx = sc, ci
		}
	}
	return bestIdx, bestScore
}

// bestCluster finds the best-matching cluster for f — bestClusterIn chunked
// across workers when the list is long enough to pay for the goroutines. The
// chunk merge iterates in index order with a strict >, so equal scores
// resolve to the lowest cluster index exactly like a single serial scan.
func bestCluster(f media.Face, cl map[int64]bool, clusters []faceCluster, workers int) (int, float32) {
	// Parallelism only pays once the list is long; short lists (and the hot
	// cache they imply) scan faster on one core.
	if len(clusters) < 4096 || workers < 2 {
		return bestClusterIn(f, cl, clusters, 0, len(clusters))
	}
	type chunkBest struct {
		idx   int
		score float32
	}
	chunk := (len(clusters) + workers - 1) / workers
	nChunks := (len(clusters) + chunk - 1) / chunk
	partial := make([]chunkBest, nChunks)
	var wg sync.WaitGroup
	for slot := 0; slot < nChunks; slot++ {
		lo := slot * chunk
		hi := min(lo+chunk, len(clusters))
		wg.Add(1)
		go func(slot, lo, hi int) {
			defer wg.Done()
			idx, score := bestClusterIn(f, cl, clusters, lo, hi)
			partial[slot] = chunkBest{idx: idx, score: score}
		}(slot, lo, hi)
	}
	wg.Wait()
	bestIdx, bestScore := -1, float32(0)
	for _, h := range partial {
		if h.idx >= 0 && h.score > bestScore {
			bestIdx, bestScore = h.idx, h.score
		}
	}
	return bestIdx, bestScore
}

// packFaceVectors copies every face's vector into one contiguous arena,
// L2-normalizing it on the way, and re-points the faces at the arena. The
// arena is returned so the caller keeps it alive.
//
// The pairwise loops are bandwidth-bound, and their blocking assumes a tile of
// vectors is a slab of memory. It wasn't: each vector arrives from its own
// embedvec.Decode allocation, so a tile was N scattered 2KB blocks — no
// sequential prefetch, and a TLB entry per vector. Packing makes a tile
// genuinely contiguous. Peak memory is unchanged: the originals become garbage
// as soon as the faces stop referencing them.
//
// Normalizing here rather than through embedvec.Normalize is not incidental —
// that returns a fresh slice per vector, which would scatter the arena again
// the moment the caller assigned it back. The cosine math downstream assumes
// unit vectors, so this is where the guarantee is established.
//
// Vectors of a different width than the first are left where they are; they
// can only be foreign-model rows, which dotf already scores 0 against
// everything.
func packFaceVectors(faces []media.Face) []float32 {
	if len(faces) == 0 {
		return nil
	}
	dim := len(faces[0].Vec)
	if dim == 0 {
		return nil
	}
	arena := make([]float32, 0, len(faces)*dim)
	for i := range faces {
		v := faces[i].Vec
		if len(v) != dim {
			faces[i].Vec = embedvec.Normalize(v)
			continue
		}
		var sum float64
		for _, x := range v {
			sum += float64(x) * float64(x)
		}
		off := len(arena)
		arena = append(arena, v...)
		dst := arena[off : off+dim : off+dim]
		if sum > 0 {
			norm := math.Sqrt(sum)
			for k, x := range dst {
				dst[k] = float32(float64(x) / norm)
			}
		}
		faces[i].Vec = dst
	}
	return arena
}

// maxFormCandidates bounds how many near neighbours the prescreen records per
// face. A face in a dense region blows through it, and rather than truncate
// (which would cost exactness) that face is marked overflowed and falls back
// to scanning every cluster. The cap therefore trades a little memory against
// how often the slow path runs; overflow is rare outside libraries with one
// hugely over-represented identity.
const maxFormCandidates = 256

// formCandidates is the prescreen's output: for each eligible face, which
// EARLIER faces it could possibly cluster with.
type formCandidates struct {
	// near[i] holds the positions of earlier eligible faces within the
	// formation threshold of face i, ascending. Empty (with overflow unset)
	// means face i is guaranteed to found a singleton.
	near [][]int32
	// overflow[i] marks a face whose neighbour list hit maxFormCandidates, so
	// near[i] is incomplete and only a full cluster scan is sound.
	overflow []bool
}

// prescreenCandidates records, per eligible face, the earlier faces (in
// processing order) at least threshold-similar to it. A face with none is
// guaranteed to found a singleton in the greedy phase-2 loop (a cluster's mean
// over members is bounded by its best member match, up to ~1e-6 of float
// rounding), so its cluster scan can be skipped entirely; a face WITH some
// only has to scan the clusters those neighbours ended up in, which is what
// keeps the formation loop from streaming the whole cluster array per face.
//
// Blocked so a tile of column vectors stays cache-hot across a tile of rows;
// row tiles are handed to workers via an atomic counter, and each face's entry
// is written only by the worker owning its row tile. Column tiles are visited
// in ascending order, so each list comes out sorted for free.
//
// onRows, when non-nil, is called with (rows finished, total rows) every so
// often. It is the only feedback available from this stage — the prescreen
// produces no assignments, so on a large backlog it is otherwise a silent
// minutes-long gap between the last join and the first new person. Rows are a
// linear measure of the work item, not of time: a row tile's cost grows with
// its index, so the count runs ahead of the elapsed fraction early on.
func prescreenCandidates(ctx context.Context, eligible []media.Face, threshold float32, onRows func(rows, total int)) formCandidates {
	n := len(eligible)
	out := formCandidates{near: make([][]int32, n), overflow: make([]bool, n)}
	const tile = 64
	// How many row tiles finish between progress reports. Reporting per tile
	// would fire thousands of times on a big backlog for no extra information.
	const reportEvery = 16
	rowTiles := (n + tile - 1) / tile
	workers := min(clusterWorkerBudget(), rowTiles)
	var next atomic.Int64
	var doneTiles atomic.Int64
	var reportMu sync.Mutex
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				rt := int(next.Add(1)) - 1
				if rt >= rowTiles || ctx.Err() != nil {
					return
				}
				iLo, iHi := rt*tile, min((rt+1)*tile, n)
				for jLo := 0; jLo < iHi; jLo += tile {
					jHi := min(jLo+tile, iHi)
					i := iLo
					// Fast path: four rows against each column at a time. Only
					// valid where the whole column tile is strictly below all
					// four rows, i.e. off the diagonal — which is nearly all of
					// the sweep once n is large.
					if jHi <= iLo {
						for ; i+4 <= iHi; i += 4 {
							if out.overflow[i] || out.overflow[i+1] || out.overflow[i+2] || out.overflow[i+3] {
								break // rare; let the scalar tail handle the group
							}
							v0, v1 := eligible[i].Vec, eligible[i+1].Vec
							v2, v3 := eligible[i+2].Vec, eligible[i+3].Vec
							for j := jLo; j < jHi; j++ {
								s0, s1, s2, s3 := dot4(v0, v1, v2, v3, eligible[j].Vec)
								if s0 >= threshold {
									out.near[i] = append(out.near[i], int32(j))
								}
								if s1 >= threshold {
									out.near[i+1] = append(out.near[i+1], int32(j))
								}
								if s2 >= threshold {
									out.near[i+2] = append(out.near[i+2], int32(j))
								}
								if s3 >= threshold {
									out.near[i+3] = append(out.near[i+3], int32(j))
								}
							}
							for k := i; k < i+4; k++ {
								if len(out.near[k]) > maxFormCandidates {
									out.overflow[k] = true
									out.near[k] = nil
								}
							}
						}
					}
					for ; i < iHi; i++ {
						if out.overflow[i] {
							continue // already conceded to the full-scan path
						}
						for j := jLo; j < min(jHi, i); j++ {
							if dotf(eligible[i].Vec, eligible[j].Vec) < threshold {
								continue
							}
							if len(out.near[i]) == maxFormCandidates {
								// Too dense to record exactly; drop the partial
								// list so nothing downstream mistakes it for
								// the complete set.
								out.overflow[i] = true
								out.near[i] = nil
								break
							}
							out.near[i] = append(out.near[i], int32(j))
						}
					}
				}
				if d := doneTiles.Add(1); onRows != nil && d%reportEvery == 0 {
					// Serialized: the callback reaches shared job state.
					reportMu.Lock()
					onRows(min(int(d)*tile, n), n)
					reportMu.Unlock()
				}
			}
		}()
	}
	wg.Wait()
	// Only claim the full sweep if it actually finished — a cancelled one
	// would otherwise credit rows no worker ever scanned.
	if onRows != nil && n > 0 && ctx.Err() == nil {
		onRows(n, n)
	}
	return out
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
// must ALSO pass the mean-similarity guard (see meanJoinSlack): for a person
// with user-confirmed faces the face's mean cosine over the confirmed center
// stays within meanJoinSlack of the threshold; a purely automatic person must
// clear the join threshold on average, no slack — the join is otherwise a
// chain hop, not a match. Scoring is parallel; the caller applies the writes
// serially. Face and seed vectors MUST be unit-normalized (clusterFaces
// normalizes everything at load) — similarity here is a raw dot product, not
// a full cosine.
//
// Human assertions shape the outcome three ways: a vetoed person is never a
// candidate for that face; a person holding a seed the face is cannot-linked
// to is never a candidate; and USER seeds count userSeedWeight× as
// corroborators AND, when a person has any, they alone define the mean
// guard's center — the anchor a candidate is measured against never drifts,
// no matter how many auto faces the person accumulates.
func scoreAgainstSeeds(ctx context.Context, unassigned []media.Face, seeds []seed, aggs map[int64]*personAgg, threshold float32, vetoes, cannot pairSet) []int64 {
	matches := make([]int64, len(unassigned))
	if len(unassigned) == 0 || len(seeds) == 0 {
		return matches
	}
	// The slack below the join threshold exists for genuinely multi-modal
	// people (age/lighting/pose spread) — and multi-modality is something a
	// HUMAN establishes by confirming faces. A purely automatic cluster gets
	// no slack: it must absorb at the join threshold ON AVERAGE (average
	// linkage), because "formed at the strict formation gate, then grew
	// forever at threshold − 0.12 against its own drifting mean" is exactly
	// how the mega-blob people formed (measured on the anime library:
	// 5,000-face blobs spanning dozens of characters, internal mean 0.61 vs
	// a 0.43 floor).
	userMeanFloor := threshold - meanJoinSlack
	autoMeanFloor := threshold
	workers := min(clusterWorkerBudget(), len(unassigned))
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
			// Candidates are scored FOUR AT A TIME against each seed. The seed
			// array is the streaming side of this loop — every candidate walks
			// all of it — so loading a seed once for four candidates instead of
			// once each is where the time goes. The dot products for a chunk of
			// seeds land in a small L1-resident buffer and the decision logic
			// below then reads them in exactly the order the one-at-a-time loop
			// did, so the constraint handling is untouched.
			const seedChunk = 256
			var buf [scoreGroup][seedChunk]float32
			var scores [scoreGroup]map[int64]personScore
			var forbidden [scoreGroup]map[int64]bool
			for g := range scores {
				scores[g] = map[int64]personScore{}
				forbidden[g] = map[int64]bool{}
			}

			for base := lo; base < hi; base += scoreGroup {
				// One ctx check per group ≈ one mutex op per full seed scan —
				// noise, but it lets a cancel stop a minutes-long pass.
				if ctx.Err() != nil {
					return
				}
				grp := min(scoreGroup, hi-base)
				for g := range grp {
					clear(scores[g])
					clear(forbidden[g])
					for pid := range vetoes[unassigned[base+g].ID] {
						forbidden[g][pid] = true
					}
				}

				for so := 0; so < len(seeds); so += seedChunk {
					sn := min(seedChunk, len(seeds)-so)
					// Raw dot: every clustering vector is unit (normalized at
					// load), so this IS the cosine without CosineSim's
					// per-call renormalization.
					if grp == scoreGroup {
						v0, v1 := unassigned[base].Vec, unassigned[base+1].Vec
						v2, v3 := unassigned[base+2].Vec, unassigned[base+3].Vec
						for k := range sn {
							buf[0][k], buf[1][k], buf[2][k], buf[3][k] =
								dot4(v0, v1, v2, v3, seeds[so+k].vec)
						}
					} else {
						for g := range grp {
							v := unassigned[base+g].Vec
							for k := range sn {
								buf[g][k] = dotf(v, seeds[so+k].vec)
							}
						}
					}
					for g := range grp {
						cl := cannot[unassigned[base+g].ID]
						fb, scg := forbidden[g], scores[g]
						for k := range sn {
							s := &seeds[so+k]
							if cl[s.id] {
								// Cannot-linked to a member → the whole person
								// is off the table, no matter how well other
								// members match.
								fb[s.personID] = true
								continue
							}
							if fb[s.personID] {
								continue
							}
							sc := buf[g][k]
							if sc < threshold-corroborationSlack {
								continue // can neither win nor corroborate
							}
							ps := scg[s.personID]
							if sc > ps.best {
								ps.best = sc
							}
							if s.user {
								ps.count += userSeedWeight
							} else {
								ps.count++
							}
							scg[s.personID] = ps
						}
					}
				}

				for g := range grp {
					i := base + g
					var bestPerson int64
					var best personScore
					for pid, ps := range scores[g] {
						if forbidden[g][pid] || !acceptJoin(ps, threshold) {
							continue
						}
						a := aggs[pid]
						if a == nil {
							continue
						}
						mean := dot32(unassigned[i].Vec, a.sum) / a.n
						floor := autoMeanFloor
						if a.userN > 0 {
							mean = dot32(unassigned[i].Vec, a.userSum) / a.userN
							floor = userMeanFloor
						}
						if mean < floor {
							continue // strong single match, but off the person's center
						}
						if bestPerson == 0 || effectiveScore(ps) > effectiveScore(best) {
							bestPerson, best = pid, ps
						}
					}
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
	// One transaction for the whole reset — per-face UnassignFace commits made
	// a full --reset-all pay minutes of fsyncs before clustering even started.
	n, err := media.UnassignFacesBulk(db, ids)
	if err != nil {
		return n, err
	}
	people, err := media.GetPeople(db)
	if err != nil {
		return n, err
	}
	for _, p := range people {
		if p.FaceCount == 0 && strings.HasPrefix(p.Name, "Unknown #") {
			if err := media.DeletePerson(db, p.ID); err != nil {
				return n, err
			}
		}
	}
	return n, nil
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
//	--flush-every=<n>      candidate faces per commit + progress report
//	                       (default 2000; see below)
//	--reset                rebuild the anonymous "Unknown #N" clusters first.
//	                       User-assigned faces and everything inside NAMED
//	                       people are never touched — naming/merging a cluster
//	                       endorses its contents.
//	--reset-all            clear EVERY auto assignment first, including inside
//	                       named people (user labels alone survive) — the
//	                       from-scratch re-run for parameter tuning/recovery.
//
// The job is INCREMENTAL: it commits and reports every --flush-every
// candidates rather than at the end, so on a six-figure backlog groups start
// appearing in People within seconds and keep arriving for the whole run. That
// makes the job interruptible in a useful way — cancel (or lose the process)
// and everything flushed so far is already saved; running it again resumes
// from what's left, because committed faces are no longer candidates. Progress
// drives the job's bar and a log line naming the current stage.
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

	// Resets run for every model FIRST so the workload the progress bar spans
	// is the real one: --reset-all can more than double the candidate pool,
	// and a total that grew halfway through the job would make the bar jump
	// backwards.
	for _, model := range clusterModels {
		if err := applyClusterResets(j, q, model); err != nil {
			return err
		}
	}
	rep := newClusterReporter(j, q, clusterModels)

	for _, model := range clusterModels {
		if err := clusterOneModel(j, q, model, rep); err != nil {
			return err
		}
		if j.Ctx.Err() != nil {
			q.PushJobStdout(j.ID, "Canceled — every flushed group is saved; run this again to continue from here")
			_ = q.CancelJob(j.ID)
			return j.Ctx.Err()
		}
	}
	q.CompleteJob(j.ID)
	return nil
}

// countUnassignedFaces reports how many of model's stored faces a clustering
// pass would treat as candidates.
func countUnassignedFaces(db *sql.DB, model string) (int, error) {
	var n int
	err := db.QueryRow(
		`SELECT COUNT(*) FROM face WHERE model = ? AND COALESCE(person_id, 0) = 0`, model).Scan(&n)
	return n, err
}

// clusterReporter turns a pass's progress ticks into the three things someone
// watching a long clustering job needs: a progress bar that spans every model
// in the run, a throttled log line naming the phase and what has been written
// so far, and a people-updated broadcast whenever a flush actually produced
// something — so the People grid fills in DURING the job rather than at the
// end. Ticks arrive on the pass's own goroutine, one at a time.
type clusterReporter struct {
	q       *jobqueue.Queue
	jobID   string
	total   int // candidate faces across every model in this run
	base    int // candidates resolved by models already finished
	unknown bool // workload could not be counted; fall back to log lines only

	lastLine  time.Time
	lastPhase clusterPhase
	lastPass  int

	lastBroadcast time.Time
	lastWritten   int // JoinedExisting+NewlyClustered at the last broadcast
	model         string
}

// clusterLogInterval throttles the per-flush log line. Flushes land every few
// seconds on a large backlog; one line each would bury the job log.
const clusterLogInterval = 10 * time.Second

// clusterBroadcastInterval throttles people-updated. The People grid refetches
// on it, so it must be slower than a person is worth looking at.
const clusterBroadcastInterval = 3 * time.Second

func newClusterReporter(j *jobqueue.Job, q *jobqueue.Queue, models []FaceModel) *clusterReporter {
	r := &clusterReporter{q: q, jobID: j.ID}
	for _, m := range models {
		n, err := countUnassignedFaces(q.Db, m.ID)
		if err != nil {
			r.unknown = true
			return r
		}
		r.total += n
	}
	_ = q.SetJobProgress(j.ID, 0, r.total)
	return r
}

// startModel points the reporter at the next recognizer's pass.
func (r *clusterReporter) startModel(model string) {
	r.model = model
	r.lastPhase, r.lastPass = "", 0
}

// finishModel rolls the finished model's candidates into the run-wide base so
// the next model's ticks continue the same bar.
func (r *clusterReporter) finishModel(workload int) {
	r.base += workload
	if !r.unknown {
		_ = r.q.SetJobProgress(r.jobID, min(r.base, r.total), r.total)
	}
}

func (r *clusterReporter) tick(pr clusterProgress) {
	if !r.unknown {
		_ = r.q.SetJobProgress(r.jobID, min(r.base+pr.Processed, r.total), r.total)
	}
	written := pr.Stats.JoinedExisting + pr.Stats.NewlyClustered
	now := time.Now()
	// Always announce a phase change; otherwise a long silent stage (the
	// prescreen especially) looks like a hang.
	if pr.Phase != r.lastPhase || pr.Pass != r.lastPass || now.Sub(r.lastLine) >= clusterLogInterval {
		r.lastPhase, r.lastPass, r.lastLine = pr.Phase, pr.Pass, now
		if pr.Phase == phaseLoad {
			r.q.PushJobStdout(r.jobID, fmt.Sprintf(
				"  [%s] loaded %d stored face(s): %d already grouped, %d to place",
				r.model, pr.Total, pr.Done, pr.Workload))
		} else {
			r.q.PushJobStdout(r.jobID, fmt.Sprintf("  [%s] %s %s: %s — %d joined, %d new people (%d faces) so far",
				r.model, phaseLabel(pr.Phase), passLabel(pr.Pass), ratio(pr.Done, pr.Total),
				pr.Stats.JoinedExisting, pr.Stats.NewPeople, pr.Stats.NewlyClustered))
		}
	}
	if written != r.lastWritten && now.Sub(r.lastBroadcast) >= clusterBroadcastInterval {
		r.lastWritten, r.lastBroadcast = written, now
		broadcastPeopleUpdated([]string{r.model})
	}
}

func phaseLabel(p clusterPhase) string {
	switch p {
	case phaseJoin:
		return "joining existing people"
	case phasePrescreen:
		return "prescreening for new groups"
	case phaseForm:
		return "forming new groups"
	}
	return string(p)
}

func passLabel(pass int) string {
	if pass == 0 {
		return ""
	}
	return fmt.Sprintf("(pass %d)", pass)
}

func ratio(done, total int) string {
	if total <= 0 {
		return fmt.Sprintf("%d", done)
	}
	return fmt.Sprintf("%d/%d (%d%%)", done, total, done*100/total)
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
func clusterOneModel(j *jobqueue.Job, q *jobqueue.Queue, model FaceModel, rep *clusterReporter) error {
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
	if v, ok := jobArgValue(j, "--flush-every"); ok {
		if n, err := strconv.Atoi(v); err == nil && n >= 1 {
			p.flushEvery = n
		}
	}
	q.PushJobStdout(j.ID, fmt.Sprintf(
		"Clustering faces: model=%s join=%.3f form=%.3f min-quality=%.2f min-cluster=%d passes=%d",
		model.ID, p.joinThreshold, p.formThreshold, p.minQuality, p.minCluster, p.passes,
	))

	workload := 0
	if rep != nil {
		rep.startModel(model.ID)
		p.progress = func(pr clusterProgress) {
			workload = pr.Workload
			rep.tick(pr)
		}
		defer func() { rep.finishModel(workload) }()
	}

	stats, err := clusterFaces(j.Ctx, q.Db, model, p)
	// Partial results are real results now — every flush this pass made is
	// already committed — so the stats are worth reporting on the way out of a
	// cancel too, and the next run picks up from what survived.
	summary := fmt.Sprintf(
		"%s: %d joined existing people, %d new people (%d faces), %d left unassigned (%d below quality floor, %d in discarded incoherent clusters, %d blocked by dissolved-group bans)",
		model.ID, stats.JoinedExisting, stats.NewPeople, stats.NewlyClustered, stats.Unassigned, stats.QualitySkipped, stats.Discarded, stats.BanBlocked,
	)
	if err != nil {
		if j.Ctx.Err() != nil {
			q.PushJobStdout(j.ID, "Clustering canceled — saved so far: "+summary)
			broadcastPeopleUpdated([]string{model.ID})
			_ = q.CancelJob(j.ID)
			return err
		}
		q.PushJobStdout(j.ID, "Clustering failed: "+err.Error())
		q.ErrorJob(j.ID)
		return err
	}
	q.PushJobStdout(j.ID, summary)
	// Live UIs (People grid) refetch on this instead of waiting for the
	// whole job to complete (multi-model runs cluster one model at a time).
	broadcastPeopleUpdated([]string{model.ID})
	return nil
}

// applyClusterResets performs the --reset / --reset-all clearing for one
// recognizer. Split out of clusterOneModel so the whole run's resets happen
// before any counting: they change how many candidates there are.
func applyClusterResets(j *jobqueue.Job, q *jobqueue.Queue, model FaceModel) error {
	var n int
	var err error
	var msg string
	switch {
	case jobHasFlag(j, "--reset-all"):
		n, err = resetAllAutoAssignments(q.Db, model.ID)
		msg = fmt.Sprintf("Reset ALL %d auto assignment(s); only user labels kept", n)
	case jobHasFlag(j, "--reset"):
		n, err = resetAutoAssignments(q.Db, model.ID)
		msg = fmt.Sprintf("Reset %d auto assignment(s) in unnamed clusters; named people and user labels kept", n)
	default:
		return nil
	}
	if err != nil {
		q.PushJobStdout(j.ID, "Reset failed: "+err.Error())
		q.ErrorJob(j.ID)
		return err
	}
	q.PushJobStdout(j.ID, msg)
	return nil
}
