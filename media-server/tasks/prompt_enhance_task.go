package tasks

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/stevecastle/shrike/jobqueue"
)

// --- the enhance-prompt job ---
//
// The Enhance button's rewrite runs as a job: it takes the text-model bucket
// and the machine-wide local-compute slot like a describe job (registry.go,
// host.go), so the text model never loads beside a diffusion run, a face scan
// or another text job; it waits its turn in the queue and shows in the job
// list. POST /api/transform/enhance-prompt queues it (QueueEnhanceJob) and
// answers with the job id at once; the client follows the job's state on the
// /stream SSE events like any other job, and reads the rewrite with
// EnhanceJobStatus once it has finished. Cancelling is the job's own cancel.

const enhancePromptCommand = "enhance-prompt"

// enhanceKeep is how long a finished rewrite stays readable.
const enhanceKeep = 10 * time.Minute

type enhanceEntry struct {
	done     bool
	res      EnhanceResult
	err      error
	finished time.Time
}

// enhanceJobs holds every enhance job queued since the server started. A job
// not in it (still pending when the server restarted: its client is gone) is
// skipped instead of loading the text model for nothing.
var enhanceJobs = struct {
	sync.Mutex
	m map[string]*enhanceEntry
}{m: map[string]*enhanceEntry{}}

// QueueEnhanceJob validates the request (ErrEnhanceInput at once for no
// prompt / no guide) and queues the rewrite, returning the job id.
func QueueEnhanceJob(q *jobqueue.Queue, req EnhanceRequest) (string, error) {
	if _, err := validateEnhance(req); err != nil {
		return "", err
	}
	if q == nil {
		return "", errors.New("the job queue is not available")
	}
	raw, err := json.Marshal(req)
	if err != nil {
		return "", err
	}
	id := uuid.NewString()
	enhanceJobs.Lock()
	for k, e := range enhanceJobs.m {
		if e.done && time.Since(e.finished) > enhanceKeep {
			delete(enhanceJobs.m, k)
		}
	}
	enhanceJobs.m[id] = &enhanceEntry{}
	enhanceJobs.Unlock()
	if _, err := q.AddJob(id, enhancePromptCommand, []string{"--model", req.Model, "--request", string(raw)}, "", nil); err != nil {
		enhanceJobs.Lock()
		delete(enhanceJobs.m, id)
		enhanceJobs.Unlock()
		return "", err
	}
	return id, nil
}

// EnhanceJobState is GET /api/transform/enhance-prompt?job=ID's answer.
type EnhanceJobState struct {
	// State is the job's: pending | in_progress | completed | error | cancelled.
	State  string         `json:"state"`
	Result *EnhanceResult `json:"result,omitempty"`
	Error  string         `json:"error,omitempty"`
	// InputError marks a request the rewrite could not use (a 4xx kind of failure).
	InputError bool `json:"inputError,omitempty"`
}

// EnhanceJobStatus reports an enhance job; ok is false for an unknown id.
func EnhanceJobStatus(q *jobqueue.Queue, id string) (EnhanceJobState, bool) {
	enhanceJobs.Lock()
	e, known := enhanceJobs.m[id]
	var entry enhanceEntry
	if known {
		entry = *e
	}
	enhanceJobs.Unlock()
	if known && entry.done {
		if entry.err != nil {
			return EnhanceJobState{State: "error", Error: entry.err.Error(), InputError: errors.Is(entry.err, ErrEnhanceInput)}, true
		}
		res := entry.res
		return EnhanceJobState{State: "completed", Result: &res}, true
	}
	var j *jobqueue.Job
	if q != nil {
		j = q.GetJob(id)
	}
	if j == nil || j.Command != enhancePromptCommand {
		if known {
			// removed from the job list before it ran
			return EnhanceJobState{State: "cancelled"}, true
		}
		return EnhanceJobState{}, false
	}
	raw, _ := j.State.MarshalJSON()
	var st string
	_ = json.Unmarshal(raw, &st)
	if st == "completed" || st == "error" {
		// finished, but not by this server's run (it restarted in between)
		return EnhanceJobState{State: "error", Error: "the rewrite did not survive a server restart; enhance again"}, true
	}
	return EnhanceJobState{State: st}, true
}

func finishEnhance(id string, res EnhanceResult, err error) {
	enhanceJobs.Lock()
	defer enhanceJobs.Unlock()
	if e, ok := enhanceJobs.m[id]; ok {
		*e = enhanceEntry{done: true, res: res, err: err, finished: time.Now()}
	}
}

func enhancePromptTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	const tag = "enhance"
	ctx := j.Ctx
	if ctx == nil {
		ctx = context.Background()
	}
	fail := func(err error) error {
		q.PushJobStdout(j.ID, tag+": "+err.Error())
		finishEnhance(j.ID, EnhanceResult{}, err)
		q.ErrorJob(j.ID)
		return err
	}
	enhanceJobs.Lock()
	_, known := enhanceJobs.m[j.ID]
	enhanceJobs.Unlock()
	if !known {
		return fail(errors.New("queued before the server restarted, so nobody is waiting for it; skipped"))
	}
	var req EnhanceRequest
	for i := 0; i+1 < len(j.Arguments); i++ {
		if j.Arguments[i] == "--request" {
			if err := json.Unmarshal([]byte(j.Arguments[i+1]), &req); err != nil {
				return fail(fmt.Errorf("%w: bad request: %v", ErrEnhanceInput, err))
			}
		}
	}
	q.PushJobStdout(j.ID, fmt.Sprintf("%s: rewriting the prompt for %s", tag, req.Model))
	start := time.Now()
	prog := enhanceProgress{task: req.Task}
	_ = q.SetJobProgress(j.ID, 0, 100)
	res, err := enhancePrompt(ctx, req, func(line string) {
		q.PushJobStdout(j.ID, line)
		if p, ok := prog.update(line); ok {
			_ = q.SetJobProgress(j.ID, p, 100)
		}
	})
	if err != nil {
		if ctx.Err() != nil {
			// cancelled: the queue already moved the job to its final state, which is what the status reports
			return err
		}
		return fail(err)
	}
	q.PushJobStdout(j.ID, fmt.Sprintf("%s: done in %.1fs", tag, time.Since(start).Seconds()))
	// the result is readable before the completion event reaches the client
	finishEnhance(j.ID, res, nil)
	_ = q.SetJobProgress(j.ID, 100, 100)
	q.CompleteJob(j.ID)
	return nil
}

// enhanceProgress turns the text engine's log lines into the job's progress
// (0-100), which the Enhance button fills with: loading the model (layers i/n)
// to 30, reading the prompt and pictures done at 35, then the writing, whose
// length is unknown until it ends, easing from 35 toward 95 (every 32 tokens,
// "N tokens so far") against the length such a prompt usually runs to.
type enhanceProgress struct {
	task string
	last int
}

var (
	enhanceLayersRe = regexp.MustCompile(`layers (\d+)/(\d+)`)
	enhanceTokensRe = regexp.MustCompile(`(\d+) tokens so far`)
)

func (p *enhanceProgress) update(line string) (int, bool) {
	v := -1
	switch {
	case enhanceLayersRe.MatchString(line):
		m := enhanceLayersRe.FindStringSubmatch(line)
		i, _ := strconv.Atoi(m[1])
		n, _ := strconv.Atoi(m[2])
		if n > 0 {
			v = 30 * i / n
		}
	case strings.Contains(line, "-> generating"):
		v = 35
	case enhanceTokensRe.MatchString(line):
		n, _ := strconv.Atoi(enhanceTokensRe.FindStringSubmatch(line)[1])
		// an H3 video prompt (scene, camera, sound lines) runs several hundred
		// tokens; an image prompt or edit instruction far fewer
		typical := 180.0
		if p.task == "video" {
			typical = 450
		}
		v = 35 + int(60*(1-math.Exp(-float64(n)/typical)))
	}
	if v <= p.last {
		return 0, false // never backwards (a warm engine skips the load)
	}
	p.last = v
	return v, true
}
