package tasks

import (
	"fmt"
	"sync"

	"github.com/stevecastle/shrike/embedvec"
	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/media"
)

// Bulk person assignment.
//
// Dropping a person card onto a media item assigns that item's face to the
// person synchronously (/api/media/assign-person). Holding CTRL at drop time —
// the same widen-to-library gesture tags use — targets every item in the
// current view instead, and that has to be a job: items without stored face
// vectors are scanned on the fly (frame extraction + ONNX), which is seconds
// per item, not milliseconds.
//
// Input follows the bulk-task contract (a search query or a newline path
// list); each item gets the SAME semantics as the synchronous endpoint: the
// face most similar to the person's existing faces wins (people appear
// alongside others), falling back to the largest face when the person has
// none to compare against. The person's face set is re-read per item, so
// early assignments inform later picks.

var assignPersonOptions = []TaskOption{
	{Name: "person-id", Label: "Person ID", Type: "number", Required: true,
		Description: "The person (People grid card) every matched face is assigned to"},
}

// assignPersonBroadcastEvery throttles people-updated while the job runs, so
// open People views track progress without refetching on every single item.
const assignPersonBroadcastEvery = 25

func assignPersonTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	ctx := j.Ctx

	opts := ParseOptions(j, assignPersonOptions)
	personIDF, _ := opts["person-id"].(float64)
	personID := int64(personIDF)
	if personID <= 0 {
		q.PushJobStdout(j.ID, "Error: --person-id is required")
		q.ErrorJob(j.ID)
		return fmt.Errorf("--person-id is required")
	}
	person, found, err := media.GetPersonByID(q.Db, personID)
	if err != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("Error loading person: %v", err))
		q.ErrorJob(j.ID)
		return err
	}
	if !found {
		q.PushJobStdout(j.ID, fmt.Sprintf("Error: no person with id %d", personID))
		q.ErrorJob(j.ID)
		return fmt.Errorf("no person with id %d", personID)
	}

	items, err := resolveJobItems(j, q)
	if err != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("Error resolving input: %v", err))
		q.ErrorJob(j.ID)
		return err
	}
	paths := items.Paths
	if len(paths) == 0 {
		q.PushJobStdout(j.ID, "No items to process")
		q.CompleteJob(j.ID)
		return nil
	}

	q.PushJobStdout(j.ID, fmt.Sprintf("Assigning %q across %d item(s)", person.Name, len(paths)))
	_ = q.SetJobProgress(j.ID, 0, len(paths))

	var assigned, already, noFace, failed int
	sinceBroadcast := 0
	for i, p := range paths {
		select {
		case <-ctx.Done():
			q.PushJobStdout(j.ID, "Task was canceled")
			if sinceBroadcast > 0 {
				broadcastPeopleUpdated([]string{})
			}
			_ = q.CancelJob(j.ID)
			return ctx.Err()
		default:
		}
		if q.PauseRequested(j.ID) {
			q.PushJobStdout(j.ID, fmt.Sprintf("Paused at %d/%d - resume to continue", i, len(paths)))
			if sinceBroadcast > 0 {
				broadcastPeopleUpdated([]string{})
			}
			return jobqueue.ErrPaused
		}
		_ = q.SetJobProgress(j.ID, i, len(paths))

		faces, _, err := FacesForPathOrScan(ctx, q.Db, p)
		if err != nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("Warning: could not read faces for %s: %v", p, err))
			failed++
			continue
		}
		if len(faces) == 0 {
			noFace++
			continue
		}
		// Re-read per item (a cheap indexed query next to the scan above):
		// faces assigned earlier in this run sharpen the similarity pick for
		// the items that follow.
		personFaces, err := media.PersonFacesByQuality(q.Db, personID)
		if err != nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("Warning: could not load the person's faces: %v", err))
			failed++
			continue
		}
		best := PickFaceForPerson(faces, personFaces)
		if best.PersonID == personID {
			already++
			continue
		}
		if err := media.AssignFace(q.Db, best.ID, personID, "user"); err != nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("Warning: could not assign face in %s: %v", p, err))
			failed++
			continue
		}
		assigned++
		sinceBroadcast++
		if sinceBroadcast >= assignPersonBroadcastEvery {
			broadcastPeopleUpdated([]string{})
			sinceBroadcast = 0
		}
	}
	_ = q.SetJobProgress(j.ID, len(paths), len(paths))
	broadcastPeopleUpdated([]string{})

	q.PushJobStdout(j.ID, fmt.Sprintf(
		"Assignment complete: %d face(s) assigned to %q, %d already theirs, %d item(s) with no face, %d failure(s)",
		assigned, person.Name, already, noFace, failed))
	q.CompleteJob(j.ID)
	return nil
}

// PickFaceForPerson chooses which of a media item's faces to assign: the one
// most similar to the person's existing faces when the person has any
// (comparing only same-dimension vectors, i.e. the same recognizer), else the
// largest face in the item. Shared by the synchronous assign-person endpoint
// and the bulk assign-person task.
func PickFaceForPerson(faces, personFaces []media.Face) media.Face {
	best := faces[0]
	if len(personFaces) > 0 {
		var bestScore float32 = -2
		matched := false
		for _, f := range faces {
			for _, pf := range personFaces {
				if len(f.Vec) != len(pf.Vec) {
					continue
				}
				if sc := embedvec.CosineSim(f.Vec, pf.Vec); sc > bestScore {
					bestScore, best, matched = sc, f, true
				}
			}
		}
		if matched {
			return best
		}
	}
	for _, f := range faces[1:] {
		if f.W*f.H > best.W*best.H {
			best = f
		}
	}
	return best
}
