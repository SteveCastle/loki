package tasks

import (
	"database/sql"
	"testing"

	_ "modernc.org/sqlite"

	"github.com/stevecastle/shrike/appconfig"
	"github.com/stevecastle/shrike/jobqueue"
)

// routeScribe sends loki-scribe to "local" or a pool for one test.
func routeScribe(t *testing.T, route string) {
	t.Helper()
	cur := engineDisp.cfg
	t.Cleanup(func() { ConfigureEngines(nil, cur) })
	cfg := appconfig.EnginesConfig{Strategy: appconfig.EngineStrategyLocal}
	if route != appconfig.EngineStrategyLocal {
		cfg.Routes = map[string]string{scribeEngine: route}
	}
	ConfigureEngines(nil, cfg)
}

// TestInferenceHostMapping: every text-generation job lands in the scribe
// bucket, wherever the engine runs.
func TestInferenceHostMapping(t *testing.T) {
	for _, route := range []string{appconfig.EngineStrategyLocal, appconfig.EngineStrategyPool} {
		routeScribe(t, route)
		if got := InferenceHost(); got != HostBucketScribe {
			t.Errorf("InferenceHost() with route %s = %q; want %q", route, got, HostBucketScribe)
		}
	}
}

// TestResolveHostVisionTasks confirms the text-generation tasks share the
// scribe bucket, while the local ONNX tasks (autotag, embed) route to their
// own dedicated buckets.
func TestResolveHostVisionTasks(t *testing.T) {
	for _, cmd := range []string{"metadata", "describe"} {
		if got := ResolveHost(cmd, "/tmp/whatever.jpg"); got != HostBucketScribe {
			t.Errorf("ResolveHost(%s) = %q; want %q", cmd, got, HostBucketScribe)
		}
	}
	if got := ResolveHost("autotag", "/tmp/whatever.jpg"); got != HostBucketAutotag {
		t.Errorf("ResolveHost(autotag) = %q; want %q", got, HostBucketAutotag)
	}
	if got := ResolveHost("embed", "/tmp/whatever.jpg"); got != HostBucketEmbed {
		t.Errorf("ResolveHost(embed) = %q; want %q", got, HostBucketEmbed)
	}
}

// TestResolveHostIngest confirms ingest still uses URL hostname extraction
// (the policy moved out of jobqueue but the behavior is identical).
func TestResolveHostIngest(t *testing.T) {
	cases := []struct {
		input string
		want  string
	}{
		{"https://www.youtube.com/watch?v=abc", "youtube.com"},
		{"https://example.com/video.mp4", "example.com"},
		{"/local/path.mp4", "localhost"},
		{"not a url", "localhost"},
	}
	for _, c := range cases {
		if got := ResolveHost("ingest", c.input); got != c.want {
			t.Errorf("ResolveHost(ingest, %q) = %q; want %q", c.input, got, c.want)
		}
	}
}

// TestResolveHostUnknownCommand confirms commands without a registered
// resolver fall back to the default localhost bucket.
func TestResolveHostUnknownCommand(t *testing.T) {
	if got := ResolveHost("not-a-real-command", "x"); got != "localhost" {
		t.Errorf("ResolveHost(unknown) = %q; want \"localhost\"", got)
	}
}

// TestResolveResources pins the additional buckets each task occupies: every
// local model workload must hold the shared local-compute slot, composite
// jobs must hold the union of their ops' buckets, and remote inference must
// NOT consume local compute.
func TestResolveResources(t *testing.T) {
	has := func(rs []string, want string) bool {
		for _, r := range rs {
			if r == want {
				return true
			}
		}
		return false
	}

	routeScribe(t, appconfig.EngineStrategyLocal)

	cases := []struct {
		name      string
		command   string
		arguments []string
		input     string
		want      []string
		absent    []string
	}{
		{name: "embed", command: "embed",
			want: []string{HostBucketEmbed, HostBucketLocalCompute}},
		{name: "autotag", command: "autotag",
			want: []string{HostBucketAutotag, HostBucketLocalCompute}},
		{name: "faces", command: "faces",
			want: []string{HostBucketFaces, HostBucketLocalCompute}},
		{name: "transcribe", command: "transcribe",
			want: []string{HostBucketLocalCompute}},
		{name: "describe local LLM", command: "describe",
			want: []string{HostBucketScribe, HostBucketLocalCompute}},
		{name: "process cheap ops", command: "process",
			arguments: []string{"--ops=hash,dimensions"},
			absent:    []string{HostBucketLocalCompute, HostBucketEmbed}},
		{name: "process heavy union", command: "process",
			arguments: []string{"--ops=describe,embed,faces"},
			want:      []string{HostBucketScribe, HostBucketLocalCompute, HostBucketEmbed, HostBucketFaces}},
		{name: "process ops in input", command: "process",
			input: "--ops=embed --query64=eA==",
			want:  []string{HostBucketEmbed, HostBucketLocalCompute}},
		{name: "legacy metadata type", command: "metadata",
			arguments: []string{"--type", "transcript"},
			want:      []string{HostBucketLocalCompute},
			absent:    []string{HostBucketEmbed}},
		{name: "retouch", command: "retouch",
			want: []string{HostBucketLocalCompute}},
		{name: "reshoot", command: "reshoot",
			want: []string{HostBucketLocalCompute}},
		{name: "4kify alias", command: "4kify",
			want: []string{HostBucketLocalCompute}},
		{name: "cheap task", command: "hash",
			absent: []string{HostBucketLocalCompute}},
		{name: "unknown command", command: "wait",
			absent: []string{HostBucketLocalCompute}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := ResolveResources(c.command, c.arguments, c.input)
			for _, w := range c.want {
				if !has(got, w) {
					t.Errorf("ResolveResources(%s) = %v; missing %q", c.command, got, w)
				}
			}
			for _, a := range c.absent {
				if has(got, a) {
					t.Errorf("ResolveResources(%s) = %v; must not contain %q", c.command, got, a)
				}
			}
		})
	}

	// Text generation on a GPU pool must not hold the local-compute slot.
	routeScribe(t, appconfig.EngineStrategyPool)
	got := ResolveResources("describe", nil, "")
	if has(got, HostBucketLocalCompute) {
		t.Errorf("remote describe resources = %v; must not hold local-compute", got)
	}
	if !has(got, HostBucketScribe) {
		t.Errorf("remote describe resources = %v; missing the scribe bucket", got)
	}
}

// TestLocalComputeSerializesHeavyJobs is the scenario from the field: an
// embed job, an autotag job, and a faces job live in three different buckets
// and used to run simultaneously, stacking three full-machine worker pools
// onto one GPU. With the shared local-compute slot (limit 1) only one may
// run at a time.
func TestLocalComputeSerializesHeavyJobs(t *testing.T) {
	jobqueue.SetHostResolver(ResolveHost)
	jobqueue.SetResourceResolver(ResolveResources)
	t.Cleanup(func() {
		jobqueue.SetHostResolver(nil)
		jobqueue.SetResourceResolver(nil)
	})
	routeScribe(t, appconfig.EngineStrategyLocal)

	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	q := jobqueue.NewQueueWithDB(db)

	cfg := appconfig.Get()
	cfg.LocalComputeConcurrency = 1
	ApplyHostLimits(q, cfg)

	for _, cmd := range []string{"embed", "autotag", "faces"} {
		if _, err := q.AddJob("", cmd, nil, "--query64=eA==", nil); err != nil {
			t.Fatal(err)
		}
	}

	first, err := q.ClaimJob()
	if err != nil || first == nil {
		t.Fatalf("first claim: %v", err)
	}
	if second, _ := q.ClaimJob(); second != nil {
		t.Fatalf("second heavy job %s claimed while local-compute was held by %s", second.Command, first.Command)
	}

	// Releasing the first frees the machine for the next one.
	if err := q.CompleteJob(first.ID); err != nil {
		t.Fatal(err)
	}
	second, err := q.ClaimJob()
	if err != nil || second == nil {
		t.Fatalf("claim after release: %v", err)
	}

	// A combined process job holding embed+faces must exclude BOTH standalone
	// jobs even at local-compute limit 2 (it occupies their buckets too).
	if err := q.CompleteJob(second.ID); err != nil {
		t.Fatal(err)
	}
	if third, _ := q.ClaimJob(); third != nil {
		if err := q.CompleteJob(third.ID); err != nil {
			t.Fatal(err)
		}
	}
	cfg.LocalComputeConcurrency = 2
	ApplyHostLimits(q, cfg)
	procID, _ := q.AddJob("", "process", []string{"--ops=embed,faces"}, "--query64=eA==", nil)
	embedID, _ := q.AddJob("", "embed", nil, "--query64=eA==", nil)
	proc, err := q.ClaimJob()
	if err != nil || proc == nil || proc.ID != procID {
		t.Fatalf("claim process job: %v (%v)", err, proc)
	}
	if blocked, _ := q.ClaimJob(); blocked != nil {
		t.Fatalf("standalone embed %s claimed while a process job holds the embed bucket", blocked.ID)
	}
	if err := q.CompleteJob(procID); err != nil {
		t.Fatal(err)
	}
	claimed, err := q.ClaimJob()
	if err != nil || claimed == nil || claimed.ID != embedID {
		t.Fatalf("embed claim after process release: %v (%v)", err, claimed)
	}
}

// TestApplyHostLimitsConcurrentClaims is the integration check the design
// hangs on: configure RunPod with limit=2, submit two RunPod-bound jobs,
// verify both claim simultaneously. Then drop the limit to 1 and confirm
// only one claims at a time.
func TestApplyHostLimitsConcurrentClaims(t *testing.T) {
	// Install the same resolver wiring main_*.go uses at startup. The
	// process-global state is fine for a unit test — there's nothing else
	// asserting on a competing resolver.
	jobqueue.SetHostResolver(ResolveHost)
	t.Cleanup(func() { jobqueue.SetHostResolver(nil) })

	routeScribe(t, appconfig.EngineStrategyPool)

	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatalf("open sqlite: %v", err)
	}
	t.Cleanup(func() { db.Close() })
	q := jobqueue.NewQueueWithDB(db)
	ApplyHostLimits(q, appconfig.Get())

	// "metadata" (a text-generation task) routes to the scribe bucket, which
	// takes one job at a time: every job feeds the same batcher.
	for i := 0; i < 2; i++ {
		if _, err := q.AddJob("", "metadata", nil, "/tmp/whatever.jpg", nil); err != nil {
			t.Fatalf("AddJob %d: %v", i, err)
		}
	}
	first, err := q.ClaimJob()
	if err != nil || first == nil {
		t.Fatalf("ClaimJob #1 returned (%v, %v)", first, err)
	}
	if first.Host != HostBucketScribe {
		t.Errorf("first.Host = %q; want %q", first.Host, HostBucketScribe)
	}
	second, err := q.ClaimJob()
	if err != nil {
		t.Fatalf("ClaimJob #2 error: %v", err)
	}
	if second != nil {
		t.Errorf("ClaimJob #2 returned a job (id=%s) but the scribe bucket should be full", second.ID)
	}
	if err := q.CompleteJob(first.ID); err != nil {
		t.Fatalf("CompleteJob: %v", err)
	}
	next, err := q.ClaimJob()
	if err != nil || next == nil {
		t.Fatalf("ClaimJob after completion returned (%v, %v)", next, err)
	}
}

// TestResolveHostGPUDiffusion: the loki-retouch / loki-reshoot engines each
// need ~20 GB of VRAM, so every task that starts one (including the legacy
// 4kify alias) must share ONE host bucket.
func TestResolveHostGPUDiffusion(t *testing.T) {
	for _, cmd := range []string{"retouch", "reshoot", "4kify"} {
		if got := ResolveHost(cmd, "/tmp/x.png"); got != HostBucketGPUDiffusion {
			t.Errorf("ResolveHost(%s) = %q; want %q", cmd, got, HostBucketGPUDiffusion)
		}
	}
}

// TestGPUDiffusionJobsSerialize: a queued retouch and reshoot never run at
// the same time, even with a generous local-compute limit.
func TestGPUDiffusionJobsSerializeHost(t *testing.T) {
	jobqueue.SetHostResolver(ResolveHost)
	jobqueue.SetResourceResolver(ResolveResources)
	t.Cleanup(func() {
		jobqueue.SetHostResolver(nil)
		jobqueue.SetResourceResolver(nil)
	})
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	q := jobqueue.NewQueueWithDB(db)
	cfg := appconfig.Get()
	cfg.LocalComputeConcurrency = 4
	ApplyHostLimits(q, cfg)

	if _, err := q.AddJob("", "retouch", nil, "/tmp/a.png", nil); err != nil {
		t.Fatal(err)
	}
	if _, err := q.AddJob("", "reshoot", nil, "/tmp/b.png", nil); err != nil {
		t.Fatal(err)
	}
	first, err := q.ClaimJob()
	if err != nil || first == nil {
		t.Fatalf("first claim: %v %v", first, err)
	}
	if second, _ := q.ClaimJob(); second != nil {
		t.Fatalf("claimed %s while %s holds the GPU", second.Command, first.Command)
	}
	q.CompleteJob(first.ID)
	if next, _ := q.ClaimJob(); next == nil {
		t.Fatal("second diffusion job not claimable after the first finished")
	}
}
