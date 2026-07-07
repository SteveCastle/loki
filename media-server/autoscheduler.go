package main

// autoscheduler.go — the background idle scheduler. When enabled (mode
// "auto") it keeps metadata generation caught up by running the combined
// per-item job (`process --ops=... --scheduled --query64=path:*`) whenever
// the machine is free, and yielding — via the jobqueue's graceful pause —
// the moment the user needs the machine back.
//
// Idle is judged from layered, best-effort signals (sysmon package):
//   - session locked        → strongest "run" signal (top tier)
//   - fullscreen D3D app    → strongest "stop" signal (gaming/presenting)
//   - on battery            → hard veto
//   - total CPU load        → gates STARTS only (while our job runs, load is
//                             mostly ours; mid-run yielding relies on the
//                             OS-level background priority + the immediate
//                             triggers below)
//   - user input idleness   → picks conservative vs aggressive thresholds
//   - app-level activity    → media serving/search/swipe requests pause the
//                             job even when the session is locked (couch
//                             streaming must not stutter)
//   - queued user jobs      → any pending job that shares one of our buckets
//                             makes us yield immediately (soft priority)
//
// The job itself is idempotent (every op skips items that already have its
// output), commits per item, and frees its worker pools — including VRAM —
// while paused, so this loop is cheap and lossless. This file is untagged so
// all platform mains share it.

import (
	"encoding/base64"
	"encoding/json"
	"log"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/stevecastle/shrike/appconfig"
	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/stream"
	"github.com/stevecastle/shrike/sysmon"
	"github.com/stevecastle/shrike/tasks"
)

const (
	schedTick      = 15 * time.Second
	appQuietWindow = 2 * time.Minute  // app requests within this window = "in use"
	resumeDwell    = 2 * time.Minute  // after a pause, wait before resuming
	calmTicksToRun = 2                // consecutive calm ticks before starting
	relaunchDelay  = 30 * time.Minute // after a completed/errored run, before launching a new one

	cpuStartActive = 25.0 // unlocked + user at the desk: only start when very quiet
	cpuStartAway   = 60.0 // unlocked + input idle 15min+: moderate load is fine
	cpuStartLocked = 80.0 // locked: only a deliberately busy machine (render, build) blocks us
	awayInputIdleS = 15 * 60.0

	// Deployments with no machine sensors at all (Docker, stub platforms)
	// fall back to app-level quietness alone.
	headlessQuiet = 10 * time.Minute
)

// autoOpsDefault is every per-item op, in the order the combined job runs them.
var autoOpsDefault = []string{"hash", "dimensions", "describe", "transcribe", "llm-autotag", "embed", "faces"}

// ---------------------------------------------------------------------------
// App-level activity signal
// ---------------------------------------------------------------------------

var lastUserActivityUnix atomic.Int64

// touchUserActivity records "a person is using the app right now".
func touchUserActivity() { lastUserActivityUnix.Store(time.Now().Unix()) }

func appQuietFor() time.Duration {
	ts := lastUserActivityUnix.Load()
	if ts == 0 {
		return 24 * time.Hour // never seen activity since boot
	}
	return time.Since(time.Unix(ts, 0))
}

// userIntentPrefixes are request paths that represent a person actively using
// the app. Background chatter (/stream, /api/stats, /jobs/list polls,
// /api/scheduler, /health) is deliberately absent.
var userIntentPrefixes = []string{
	"/media", // browser pages, file/thumbnail/HLS serving, tagging
	"/api/media",
	"/swipe",
	"/create",
	"/job/",       // manual job control
	"/workflows",  // saved workflow runs
	"/api/search", // similarity/visual search
	"/api/image-search",
	"/api/faces",
	"/api/people",
	"/api/embeddings",
}

// withActivityTracking wraps the root handler and feeds the scheduler's
// app-activity signal from user-intent requests.
func withActivityTracking(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		p := r.URL.Path
		for _, prefix := range userIntentPrefixes {
			if strings.HasPrefix(p, prefix) {
				touchUserActivity()
				break
			}
		}
		next.ServeHTTP(w, r)
	})
}

// ---------------------------------------------------------------------------
// Policy
// ---------------------------------------------------------------------------

type policyInput struct {
	Sig            sysmon.State
	AppQuietFor    time.Duration
	UserJobWaiting bool
	Running        bool // our scheduled job is currently in progress
}

// wantRun is the pure policy decision: should the auto-job be running given
// these signals? The reason string is user-facing (shown on the dashboard
// widget). Start gates are stricter than keep-running gates: while running,
// CPU load is mostly ours, so only the immediate triggers stop us.
func wantRun(p policyInput) (bool, string) {
	if p.Sig.OnBattery == sysmon.Yes {
		return false, "on battery power"
	}
	if p.UserJobWaiting {
		return false, "yielding to a job you queued"
	}
	if p.AppQuietFor < appQuietWindow {
		return false, "the app is in use"
	}
	if p.Sig.SessionLocked == sysmon.Yes {
		// Locked = strongest idle signal. Only a deliberately busy machine
		// (overnight render, big build) blocks a start; a running auto-job
		// keeps going until unlock or another trigger.
		if !p.Running && p.Sig.CPUPercent >= 0 && p.Sig.CPUPercent > cpuStartLocked {
			return false, "machine is busy with other work"
		}
		return true, "session locked"
	}
	if p.Sig.FullscreenBusy == sysmon.Yes {
		return false, "a fullscreen app is active"
	}
	if p.Running {
		return true, "machine is free"
	}
	// Start gates (unlocked or lock state unknown).
	if p.Sig.CPUPercent >= 0 {
		away := p.Sig.InputIdleSeconds >= awayInputIdleS
		threshold := cpuStartActive
		reason := "machine is quiet"
		if away {
			threshold = cpuStartAway
			reason = "you're away from the keyboard"
		}
		if p.Sig.CPUPercent > threshold {
			return false, "machine is busy with other work"
		}
		return true, reason
	}
	// No machine sensors (Docker, unsupported platform): app quietness only.
	if p.AppQuietFor >= headlessQuiet {
		return true, "app has been idle"
	}
	return false, "waiting for a longer idle period"
}

// ---------------------------------------------------------------------------
// Scheduler
// ---------------------------------------------------------------------------

type schedulerStatus struct {
	Mode    string `json:"mode"`   // "off" | "auto"
	Forced  bool   `json:"forced"` // one-shot "run now" override active
	State   string `json:"state"`  // disabled | waiting | running | yielding | complete
	Reason  string `json:"reason"`
	JobID   string `json:"jobId,omitempty"`
	Done    int    `json:"done"`
	Total   int    `json:"total"`
	Ops     string `json:"ops"`
	Signals struct {
		CPUPercent       float64    `json:"cpuPercent"`
		SessionLocked    sysmon.Tri `json:"sessionLocked"`
		FullscreenBusy   sysmon.Tri `json:"fullscreenBusy"`
		OnBattery        sysmon.Tri `json:"onBattery"`
		InputIdleSeconds float64    `json:"inputIdleSeconds"`
		AppQuietSeconds  float64    `json:"appQuietSeconds"`
	} `json:"signals"`
}

type autoScheduler struct {
	deps *Dependencies
	mon  *sysmon.Monitor
	// sampleFn overrides sensor sampling in tests; nil = mon.Sample.
	sampleFn func() sysmon.State

	mu sync.Mutex
	// mode is runtime-only state ("off" | "auto") and deliberately NOT
	// persisted: every server start comes up stopped, so background compute
	// is always an affirmative per-session choice via the dashboard widget.
	mode         string
	forced       bool
	pausedByUs   bool
	manualHold   bool // user paused OUR job by hand; never auto-resume it
	jobID        string
	calmTicks    int
	prevLocked   sysmon.Tri
	lastPause    time.Time
	lastFinished time.Time // last time a scheduled run completed or errored

	lastPublished string
	lastStatus    schedulerStatus
}

var autoSched *autoScheduler
var autoSchedOnce sync.Once

// startAutoScheduler launches the scheduler loop. Called once from each
// platform main after the queue and runners are up.
func startAutoScheduler(deps *Dependencies) {
	autoSchedOnce.Do(func() {
		autoSched = &autoScheduler{deps: deps, mon: sysmon.NewMonitor(), mode: "off"}
		go func() {
			// Adopt any scheduled job that survived a restart: a paused one
			// was parked by us (or by the pre-restart user — we adopt it, the
			// supported "keep it stopped" control is mode=off).
			if j := autoSched.findScheduledJob(); j != nil && j.State == jobqueue.StatePaused {
				autoSched.mu.Lock()
				autoSched.pausedByUs = true
				autoSched.mu.Unlock()
			}
			ticker := time.NewTicker(schedTick)
			defer ticker.Stop()
			for {
				autoSched.tick()
				<-ticker.C
			}
		}()
	})
}

// findScheduledJob returns the live (pending/running/paused) job this
// scheduler manages, identified by the --scheduled marker argument.
func (s *autoScheduler) findScheduledJob() *jobqueue.Job {
	for _, j := range s.deps.Queue.GetJobs() {
		if j.Command != "process" {
			continue
		}
		if !hasScheduledMarker(j.Arguments) {
			continue
		}
		switch j.State {
		case jobqueue.StatePending, jobqueue.StateInProgress, jobqueue.StatePaused:
			jj := j
			return &jj
		}
	}
	return nil
}

func hasScheduledMarker(args []string) bool {
	for _, a := range args {
		if a == "--scheduled" {
			return true
		}
	}
	return false
}

// configuredOps returns the op list for scheduled runs, filtered to
// registered ops so a stale config entry can't wedge the job.
func configuredOps() []string {
	raw := strings.TrimSpace(appconfig.Get().AutoProcessOps)
	if raw == "" {
		return autoOpsDefault
	}
	known := map[string]bool{}
	for _, id := range tasks.ItemOpIDs() {
		known[id] = true
	}
	var out []string
	for _, op := range strings.Split(raw, ",") {
		if op = strings.ToLower(strings.TrimSpace(op)); op != "" && known[op] {
			out = append(out, op)
		}
	}
	if len(out) == 0 {
		return autoOpsDefault
	}
	return out
}

// gapsRemain consults the cached stats snapshot: is there anything left for
// the configured ops to do? nil stats (still counting) = assume yes.
func gapsRemain(stats *statsAPIResponse, ops []string) bool {
	if stats == nil {
		return true
	}
	if stats.TotalMedia == 0 {
		return false
	}
	for _, op := range ops {
		switch op {
		case "hash":
			if stats.WithHash < stats.TotalMedia || stats.WithSize < stats.TotalMedia {
				return true
			}
		case "dimensions":
			if stats.WithDimensions < stats.TotalMedia {
				return true
			}
		case "describe":
			if stats.WithDescription < stats.TotalMedia {
				return true
			}
		case "transcribe":
			if stats.VideosWithTranscript < stats.TotalVideos {
				return true
			}
		case "llm-autotag", "autotag":
			if stats.WithTags < stats.TotalMedia {
				return true
			}
		case "embed":
			if stats.WithEmbedding < stats.TotalMedia {
				return true
			}
		case "faces":
			if stats.WithFaceScan < stats.TotalMedia {
				return true
			}
		}
	}
	return false
}

// schedulerStatsSnapshot returns the merged stats snapshot, kicking off a
// recount when stale — the same contract the stats API uses.
func schedulerStatsSnapshot(deps *Dependencies) *statsAPIResponse {
	libStats.mu.Lock()
	defer libStats.mu.Unlock()
	merged := mergedStatsLocked()
	stale := libStats.snapshot == nil ||
		time.Since(time.Unix(libStats.snapshot.GeneratedAt, 0)) > libStatsTTL(libStats.computeMs)
	if stale && !libStats.computing {
		libStats.computing = true
		go computeLibraryStats(deps)
	}
	return merged
}

// userJobWaiting reports whether a pending NON-scheduled job shares a
// concurrency bucket with our running job — i.e. we are the reason it can't
// start.
func (s *autoScheduler) userJobWaiting(our *jobqueue.Job) bool {
	if our == nil || our.State != jobqueue.StateInProgress {
		return false
	}
	ourBuckets := map[string]bool{our.Host: true}
	for _, b := range our.Resources {
		ourBuckets[b] = true
	}
	for _, j := range s.deps.Queue.GetJobs() {
		if j.ID == our.ID || j.State != jobqueue.StatePending || hasScheduledMarker(j.Arguments) {
			continue
		}
		if ourBuckets[j.Host] {
			return true
		}
		for _, b := range j.Resources {
			if ourBuckets[b] {
				return true
			}
		}
	}
	return false
}

func (s *autoScheduler) createJob(ops []string) string {
	query := base64.StdEncoding.EncodeToString([]byte("path:*"))
	id, err := s.deps.Queue.AddJob("", "process",
		[]string{"--ops=" + strings.Join(ops, ","), "--scheduled"},
		"--query64="+query, nil)
	if err != nil {
		log.Printf("autoscheduler: failed to create job: %v", err)
		return ""
	}
	return id
}

// tick runs one scheduler evaluation. All transitions flow through here,
// under the scheduler lock so a concurrent SetMode/ForceRun-triggered tick
// can't race the ticker into creating duplicate jobs.
func (s *autoScheduler) tick() {
	s.mu.Lock()
	defer s.mu.Unlock()

	var sig sysmon.State
	if s.sampleFn != nil {
		sig = s.sampleFn()
	} else {
		sig = s.mon.Sample()
	}
	q := s.deps.Queue
	job := s.findScheduledJob()

	if job != nil {
		s.jobID = job.ID
	} else {
		if s.jobID != "" {
			// Our job reached a terminal state since the last tick.
			s.lastFinished = time.Now()
			s.forced = false
			s.pausedByUs = false
			s.manualHold = false
		}
		s.jobID = ""
	}

	// Detect a manual pause of our job (someone else parked it).
	if job != nil && job.State == jobqueue.StatePaused && !s.pausedByUs {
		s.manualHold = true
	}
	if job != nil && job.State != jobqueue.StatePaused {
		s.manualHold = false
	}

	status := schedulerStatus{Mode: s.mode, Forced: s.forced, Ops: strings.Join(configuredOps(), ",")}
	status.Signals.CPUPercent = sig.CPUPercent
	status.Signals.SessionLocked = sig.SessionLocked
	status.Signals.FullscreenBusy = sig.FullscreenBusy
	status.Signals.OnBattery = sig.OnBattery
	status.Signals.InputIdleSeconds = sig.InputIdleSeconds
	status.Signals.AppQuietSeconds = appQuietFor().Seconds()
	if job != nil {
		status.JobID = job.ID
		status.Done = job.ProgressDone
		status.Total = job.ProgressTotal
	}

	defer func() {
		s.publishLocked(status)
		s.prevLocked = sig.SessionLocked
	}()

	// ---- mode off: keep our job parked, do nothing else -------------------
	if s.mode != "auto" && !s.forced {
		if job != nil && (job.State == jobqueue.StateInProgress || job.State == jobqueue.StatePending) {
			if err := q.RequestPause(job.ID); err == nil {
				s.pausedByUs = true
			}
		}
		status.State = "disabled"
		status.Reason = "scheduler is off"
		if job != nil {
			status.Reason = "scheduler is off — run is paused"
		}
		return
	}

	// ---- forced: run regardless of signals ---------------------------------
	if s.forced {
		if job == nil {
			s.jobID = s.createJob(configuredOps())
			status.JobID = s.jobID
		} else if job.State == jobqueue.StatePaused {
			if err := q.ResumeJob(job.ID); err != nil {
				log.Printf("autoscheduler: resume failed: %v", err)
			}
			s.pausedByUs = false
			s.manualHold = false
		}
		status.State = "running"
		status.Reason = "forced run — ignoring idle signals"
		return
	}

	// ---- auto ---------------------------------------------------------------
	running := job != nil && job.State == jobqueue.StateInProgress
	p := policyInput{
		Sig:            sig,
		AppQuietFor:    appQuietFor(),
		UserJobWaiting: s.userJobWaiting(job),
		Running:        running,
	}
	run, reason := wantRun(p)

	// Unlock transition: the user just sat down — yield immediately once.
	if s.prevLocked == sysmon.Yes && sig.SessionLocked == sysmon.No && running {
		run, reason = false, "you're back at the machine"
	}

	if run {
		s.calmTicks++
	} else {
		s.calmTicks = 0
	}

	switch {
	case !run:
		if job != nil && (job.State == jobqueue.StateInProgress || job.State == jobqueue.StatePending) {
			if err := q.RequestPause(job.ID); err == nil {
				s.pausedByUs = true
				s.lastPause = time.Now()
			}
		}
		if job != nil {
			status.State = "yielding"
			status.Reason = reason
		} else {
			status.State = "waiting"
			status.Reason = reason
		}

	case s.manualHold:
		status.State = "yielding"
		status.Reason = "paused by you — press Run now or resume the job to continue"

	case running:
		status.State = "running"
		status.Reason = reason

	case job != nil && job.State == jobqueue.StatePending:
		status.State = "running"
		status.Reason = "queued"

	case job != nil && job.State == jobqueue.StatePaused:
		// Resume only after the dwell and a couple of calm ticks, so a brief
		// CPU spike doesn't thrash the worker pools (model reloads are the
		// expensive part).
		if time.Since(s.lastPause) >= resumeDwell && s.calmTicks >= calmTicksToRun {
			if err := q.ResumeJob(job.ID); err == nil {
				s.pausedByUs = false
				status.State = "running"
				status.Reason = reason
			} else {
				status.State = "yielding"
				status.Reason = "waiting to resume"
			}
		} else {
			status.State = "yielding"
			status.Reason = "resuming soon — " + reason
		}

	default: // no job: launch one if there's work and we're past the cooldown
		stats := schedulerStatsSnapshot(s.deps)
		switch {
		case !gapsRemain(stats, configuredOps()):
			status.State = "complete"
			status.Reason = "library is fully processed"
		case time.Since(s.lastFinished) < relaunchDelay && !s.lastFinished.IsZero():
			status.State = "waiting"
			status.Reason = "cooling down before the next run"
		case s.calmTicks >= calmTicksToRun:
			s.jobID = s.createJob(configuredOps())
			status.JobID = s.jobID
			status.State = "running"
			status.Reason = reason
		default:
			status.State = "waiting"
			status.Reason = reason
		}
	}
}

// publishLocked broadcasts the status over SSE when it changed. Caller holds mu.
func (s *autoScheduler) publishLocked(st schedulerStatus) {
	s.lastStatus = st
	payload, err := json.Marshal(st)
	if err != nil {
		return
	}
	// Progress counts change every item; only re-broadcast on state shifts
	// (the job's own "progress" SSE events carry the live numbers).
	key := st.Mode + "|" + st.State + "|" + st.Reason + "|" + st.JobID
	if key == s.lastPublished {
		return
	}
	s.lastPublished = key
	stream.Broadcast(stream.Message{Type: "scheduler", Msg: string(payload)})
}

// Status returns the latest published status (for the GET endpoint).
func (s *autoScheduler) Status() schedulerStatus {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.lastStatus
}

// SetMode switches between "off" and "auto" for THIS server session only —
// the mode is deliberately not persisted, so every restart comes up stopped
// and enabling background compute is an affirmative choice each time.
// Turning off also parks a running scheduled job. Either direction clears a
// forced run.
func (s *autoScheduler) SetMode(mode string) {
	if mode != "auto" {
		mode = "off"
	}
	s.mu.Lock()
	s.mode = mode
	s.forced = false
	s.mu.Unlock()
	s.tick() // apply immediately (pause/adopt) instead of waiting a tick
}

// ForceRun starts (or resumes) the scheduled job immediately, ignoring idle
// signals until it completes — the "old Run everything" behavior.
func (s *autoScheduler) ForceRun() {
	s.mu.Lock()
	s.forced = true
	s.manualHold = false
	s.mu.Unlock()
	s.tick()
}
