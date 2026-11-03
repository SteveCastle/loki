package tasks

import (
	"database/sql"
	"errors"
	"fmt"

	"github.com/stevecastle/shrike/embedindex"
	"github.com/stevecastle/shrike/media"
)

// The per-item duplicate check: the viewer's metadata Duplicates tab asks
// "does THIS item have copies?" without waiting for a library-wide
// find-duplicates run. It is the same grouping rule applied to one item —
// one nearest-neighbour query over the vector index instead of an LSH pass —
// so whatever it records is exactly what a scan would have recorded:
//
//   - an item already in a group is answered by that group (handled by the
//     caller; nothing here regroups it);
//   - an ungrouped item that matches an existing group's ANCHOR at or above
//     the threshold joins that group (the scan's join step);
//   - otherwise the item becomes the anchor of a new star group with every
//     ungrouped neighbour at or above the threshold (the scan's seed step).
//
// PlanDuplicatesForPath only computes that outcome (read-only, so the tab can
// preview it for viewers who can't write); ApplyDuplicatePlan records it.

// duplicatesForPathNeighbours bounds the neighbour query. Neighbours beyond
// it are only missed when an item has hundreds of near-identical copies, and
// those clusters are what the library-wide scan already groups.
const duplicatesForPathNeighbours = 256

// ErrNoEmbedding means the item has no visual embedding for the active model.
var ErrNoEmbedding = errors.New("no visual embedding for this item")

// DuplicatePlan is the outcome of the per-item check.
type DuplicatePlan struct {
	Model     string
	Threshold float64 // similarity floor as a fraction, as stored on groups
	// JoinGroup > 0: the item joins that existing group, scoring
	// JoinScore against its anchor.
	JoinGroup int64
	JoinScore float64
	// Members of a new group anchored on the item (anchor first, score 1).
	// Empty with JoinGroup == 0 means no duplicate candidates.
	Members []media.DuplicateMember
}

// Found reports whether the check found anything to group.
func (p DuplicatePlan) Found() bool { return p.JoinGroup > 0 || len(p.Members) >= 2 }

// PlanDuplicatesForPath finds path's duplicate candidates at pct (percent,
// like find-duplicates' --threshold) among items not already grouped.
func PlanDuplicatesForPath(db *sql.DB, path string, pct float64) (DuplicatePlan, error) {
	if pct == 0 {
		pct = 100
	}
	minScore := duplicateThresholdScore(pct)
	model := ActiveEmbedModel().ID
	plan := DuplicatePlan{Model: model, Threshold: pct / 100}

	vec, ok, err := media.GetEmbedding(db, path, model)
	if err != nil {
		return plan, err
	}
	if !ok || len(vec) == 0 {
		return plan, ErrNoEmbedding
	}
	// Plain cosine — the scan verifies every pair with the plain dot of
	// normalized vectors, so the threshold means the same thing here.
	hits, err := searchByVectorScored(db, model, vec, duplicatesForPathNeighbours+1, nil, embedindex.ScorePlain)
	if err != nil {
		return plan, err
	}
	var near []SimilarHit
	for _, h := range hits {
		if h.Path != path && h.Score >= minScore {
			near = append(near, h)
		}
	}
	if len(near) == 0 {
		return plan, nil
	}
	paths := make([]string, len(near))
	for i, h := range near {
		paths[i] = h.Path
	}
	membership, err := media.DuplicateMembershipForPaths(db, paths)
	if err != nil {
		return plan, err
	}

	// Join: the best-scoring neighbour that anchors a group. Hits are sorted
	// by score, so the first anchor wins.
	for _, h := range near {
		if m, ok := membership[h.Path]; ok && m.Anchor {
			plan.JoinGroup, plan.JoinScore = m.GroupID, float64(h.Score)
			return plan, nil
		}
	}
	// Seed: a new star around the item from the ungrouped neighbours.
	members := []media.DuplicateMember{{Path: path, Score: 1}}
	for _, h := range near {
		if _, grouped := membership[h.Path]; !grouped {
			members = append(members, media.DuplicateMember{Path: h.Path, Score: float64(h.Score)})
		}
	}
	if len(members) >= 2 {
		plan.Members = members
	}
	return plan, nil
}

// ApplyDuplicatePlan records plan for path and returns the group id (0 when
// there was nothing to group, or a concurrent scan took the paths first).
func ApplyDuplicatePlan(db *sql.DB, path string, plan DuplicatePlan) (int64, error) {
	switch {
	case plan.JoinGroup > 0:
		added, err := media.AddDuplicateMembers(db, plan.JoinGroup,
			[]media.DuplicateMember{{Path: path, Score: plan.JoinScore}})
		if err != nil {
			return 0, fmt.Errorf("join group %d: %w", plan.JoinGroup, err)
		}
		if added == 0 {
			return media.DuplicateGroupForPath(db, path)
		}
		broadcastDuplicatesUpdated(0, added)
		return plan.JoinGroup, nil
	case len(plan.Members) >= 2:
		gid, err := media.CreateDuplicateGroup(db, plan.Model, plan.Threshold, path, plan.Members)
		if err != nil {
			return 0, err
		}
		if gid == 0 {
			return media.DuplicateGroupForPath(db, path)
		}
		broadcastDuplicatesUpdated(1, len(plan.Members))
		return gid, nil
	}
	return 0, nil
}
