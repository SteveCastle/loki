package main

import (
	"database/sql"
	"path/filepath"
	"testing"
	"time"

	"github.com/stevecastle/shrike/appconfig"
	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/storage"
	"github.com/stevecastle/shrike/sysmon"
	_ "modernc.org/sqlite"
)

func sig(mut ...func(*sysmon.State)) sysmon.State {
	s := sysmon.State{CPUPercent: 5, InputIdleSeconds: 0, SampledAt: time.Now()}
	for _, m := range mut {
		m(&s)
	}
	return s
}

// TestWantRun pins the tier policy: locked > away > active, with the hard
// vetoes (battery, app activity, user jobs, fullscreen) on top.
func TestWantRun(t *testing.T) {
	quiet := 10 * time.Minute
	cases := []struct {
		name string
		in   policyInput
		want bool
	}{
		{"battery vetoes even locked", policyInput{
			Sig: sig(func(s *sysmon.State) { s.OnBattery = sysmon.Yes; s.SessionLocked = sysmon.Yes }), AppQuietFor: quiet}, false},
		{"user job waiting vetoes", policyInput{
			Sig: sig(func(s *sysmon.State) { s.SessionLocked = sysmon.Yes }), AppQuietFor: quiet, UserJobWaiting: true, Running: true}, false},
		{"app in use vetoes even locked", policyInput{
			Sig: sig(func(s *sysmon.State) { s.SessionLocked = sysmon.Yes }), AppQuietFor: 10 * time.Second}, false},
		{"locked runs despite moderate load", policyInput{
			Sig: sig(func(s *sysmon.State) { s.SessionLocked = sysmon.Yes; s.CPUPercent = 60 }), AppQuietFor: quiet}, true},
		{"locked start blocked by heavy load", policyInput{
			Sig: sig(func(s *sysmon.State) { s.SessionLocked = sysmon.Yes; s.CPUPercent = 92 }), AppQuietFor: quiet}, false},
		{"locked keeps running under heavy load (it's ours)", policyInput{
			Sig: sig(func(s *sysmon.State) { s.SessionLocked = sysmon.Yes; s.CPUPercent = 92 }), AppQuietFor: quiet, Running: true}, true},
		{"fullscreen app blocks when unlocked", policyInput{
			Sig: sig(func(s *sysmon.State) { s.FullscreenBusy = sysmon.Yes }), AppQuietFor: quiet, Running: true}, false},
		{"active user needs a very quiet machine", policyInput{
			Sig: sig(func(s *sysmon.State) { s.CPUPercent = 30 }), AppQuietFor: quiet}, false},
		{"active user + quiet machine runs light work", policyInput{
			Sig: sig(func(s *sysmon.State) { s.CPUPercent = 10 }), AppQuietFor: quiet}, true},
		{"away user tolerates moderate load", policyInput{
			Sig: sig(func(s *sysmon.State) { s.CPUPercent = 45; s.InputIdleSeconds = 20 * 60 }), AppQuietFor: quiet}, true},
		{"away user still blocked by heavy load", policyInput{
			Sig: sig(func(s *sysmon.State) { s.CPUPercent = 70; s.InputIdleSeconds = 20 * 60 }), AppQuietFor: quiet}, false},
		{"running keeps going despite our own load", policyInput{
			Sig: sig(func(s *sysmon.State) { s.CPUPercent = 95 }), AppQuietFor: quiet, Running: true}, true},
		{"no sensors: app quiet long enough", policyInput{
			Sig: sysmon.State{CPUPercent: -1, InputIdleSeconds: -1}, AppQuietFor: 15 * time.Minute}, true},
		{"no sensors: app quiet too short", policyInput{
			Sig: sysmon.State{CPUPercent: -1, InputIdleSeconds: -1}, AppQuietFor: 5 * time.Minute}, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, reason := wantRun(c.in)
			if got != c.want {
				t.Errorf("wantRun = %v (%q), want %v", got, reason, c.want)
			}
		})
	}
}

func TestGapsRemain(t *testing.T) {
	full := &statsAPIResponse{
		TotalMedia: 10, TotalVideos: 2, TotalAudio: 1, TotalTranscribable: 3,
		WithHash: 10, WithSize: 10, WithDimensions: 10, WithDescription: 10,
		WithTags: 10, WithEmbedding: 10, WithFaceScan: 10, WithTranscript: 3,
	}
	if gapsRemain(full, autoOpsDefault) {
		t.Error("fully covered library must report no gaps")
	}
	missingFaces := *full
	missingFaces.WithFaceScan = 7
	if !gapsRemain(&missingFaces, autoOpsDefault) {
		t.Error("missing face scans must count as a gap")
	}
	if gapsRemain(&missingFaces, []string{"hash", "embed"}) {
		t.Error("gaps outside the configured ops must not count")
	}
	if !gapsRemain(nil, autoOpsDefault) {
		t.Error("nil stats (still counting) must assume work remains")
	}
	if gapsRemain(&statsAPIResponse{TotalMedia: 0}, autoOpsDefault) {
		t.Error("empty library has nothing to do")
	}
}

// newSchedulerEnv builds a scheduler over a real queue with fake sensors.
func newSchedulerEnv(t *testing.T, mode string, s sysmon.State) (*autoScheduler, *jobqueue.Queue, *func(sysmon.State)) {
	t.Helper()
	// The scheduler no longer persists anything, but pin the config file to
	// a temp path anyway so no code path reached from these tests can ever
	// touch the developer's real config (it once reset a live dbPath and
	// jwtSecret; appconfig also self-protects under `go test`).
	t.Setenv("LOWKEY_CONFIG_PATH", filepath.Join(t.TempDir(), "config.json"))
	old := appconfig.Get()
	t.Cleanup(func() { appconfig.Set(old) })

	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	q := jobqueue.NewQueueWithDB(db)
	deps := &Dependencies{Queue: q, DB: db, Storage: storage.NewRegistry(nil)}

	current := s
	set := func(ns sysmon.State) { current = ns }
	sched := &autoScheduler{
		deps:     deps,
		mon:      sysmon.NewMonitor(),
		mode:     mode,
		sampleFn: func() sysmon.State { return current },
	}
	// Simulate long app quiet by backdating the activity clock.
	lastUserActivityUnix.Store(time.Now().Add(-time.Hour).Unix())
	t.Cleanup(func() { lastUserActivityUnix.Store(0) })
	return sched, q, &set
}

// TestSchedulerLifecycle drives the full loop: idle → job created; activity →
// job paused; forced → resumed; off → parked.
func TestSchedulerLifecycle(t *testing.T) {
	locked := sig(func(s *sysmon.State) { s.SessionLocked = sysmon.Yes })
	sched, q, setSig := newSchedulerEnv(t, "auto", locked)

	// Two calm ticks required before launch.
	sched.tick()
	if j := sched.findScheduledJob(); j != nil {
		t.Fatal("job must not launch on the first calm tick (hysteresis)")
	}
	sched.tick()
	job := sched.findScheduledJob()
	if job == nil {
		t.Fatal("expected a scheduled job after sustained calm")
	}
	if job.Command != "process" || !hasScheduledMarker(job.Arguments) {
		t.Fatalf("unexpected job shape: %s %v", job.Command, job.Arguments)
	}

	// User comes back to the app → the job pauses (it is pending here since
	// no runners are attached, so the pause lands immediately).
	touchUserActivity()
	sched.tick()
	job = sched.findScheduledJob()
	if job == nil || job.State != jobqueue.StatePaused {
		t.Fatalf("expected the scheduled job to pause on app activity, state=%v", job.State)
	}

	// Force run: resumes immediately regardless of activity.
	sched.ForceRun()
	job = sched.findScheduledJob()
	if job == nil || job.State != jobqueue.StatePending {
		t.Fatalf("forced run must resume the job, state=%v", job.State)
	}

	// Mode off: parks the job and reports disabled.
	sched.SetMode("off")
	job = sched.findScheduledJob()
	if job == nil || job.State != jobqueue.StatePaused {
		t.Fatalf("mode off must park the scheduled job, state=%v", job.State)
	}
	if st := sched.Status(); st.State != "disabled" {
		t.Fatalf("status state = %q, want disabled", st.State)
	}

	// Back to auto with a busy machine: stays parked.
	(*setSig)(sig(func(s *sysmon.State) { s.CPUPercent = 90 }))
	lastUserActivityUnix.Store(time.Now().Add(-time.Hour).Unix())
	sched.SetMode("auto")
	sched.tick()
	job = sched.findScheduledJob()
	if job == nil || job.State != jobqueue.StatePaused {
		t.Fatalf("busy machine must keep the job parked, state=%v", job.State)
	}
	_ = q
}

// TestSchedulerManualPauseRespected: a pause the scheduler didn't issue is
// never auto-resumed.
func TestSchedulerManualPauseRespected(t *testing.T) {
	locked := sig(func(s *sysmon.State) { s.SessionLocked = sysmon.Yes })
	sched, q, _ := newSchedulerEnv(t, "auto", locked)

	sched.tick()
	sched.tick()
	job := sched.findScheduledJob()
	if job == nil {
		t.Fatal("expected a scheduled job")
	}

	// The user pauses OUR job by hand (scheduler did not set pausedByUs).
	if err := q.RequestPause(job.ID); err != nil {
		t.Fatal(err)
	}

	// Plenty of calm ticks later it must still be paused.
	sched.lastPause = time.Now().Add(-time.Hour)
	for i := 0; i < 5; i++ {
		sched.tick()
	}
	job = sched.findScheduledJob()
	if job == nil || job.State != jobqueue.StatePaused {
		t.Fatalf("manually paused job must stay paused, state=%v", job.State)
	}
	if st := sched.Status(); st.State != "yielding" {
		t.Fatalf("status = %q, want yielding (manual hold)", st.State)
	}
}
