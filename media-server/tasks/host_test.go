package tasks

import (
	"database/sql"
	"testing"

	_ "modernc.org/sqlite"

	"github.com/stevecastle/shrike/appconfig"
	"github.com/stevecastle/shrike/jobqueue"
)

func setProvider(t *testing.T, provider string) {
	t.Helper()
	cur := appconfig.Get()
	t.Cleanup(func() { appconfig.Set(cur) })
	cfg := cur
	cfg.InferenceProvider = provider
	appconfig.Set(cfg)
}

// TestInferenceHostMapping checks that the active vision provider drives the
// concurrency bucket name. Adding a new engine should require updating this
// table alongside the switch in InferenceHost.
func TestInferenceHostMapping(t *testing.T) {
	cases := []struct {
		provider string
		want     string
	}{
		{InferenceProviderOllama, HostBucketOllama},
		{InferenceProviderRunPod, HostBucketRunPod},
		{InferenceProviderLMStudio, HostBucketLMStudio},
		{InferenceProviderLlamaCpp, HostBucketLlamaCpp},
		{InferenceProviderOff, "localhost"},
		{"", "localhost"},
		{"something-new", "localhost"},
	}
	for _, c := range cases {
		t.Run(c.provider, func(t *testing.T) {
			setProvider(t, c.provider)
			if got := InferenceHost(); got != c.want {
				t.Errorf("InferenceHost() with provider=%q = %q; want %q", c.provider, got, c.want)
			}
		})
	}
}

// TestResolveHostVisionTasks confirms the LLM-backed metadata task delegates to
// InferenceHost (so the provider tab controls its bucket), while the local ONNX
// tasks (autotag, embed) route to their own dedicated buckets and are NOT
// affected by the LLM provider setting.
func TestResolveHostVisionTasks(t *testing.T) {
	setProvider(t, InferenceProviderRunPod)
	if got := ResolveHost("metadata", "/tmp/whatever.jpg"); got != HostBucketRunPod {
		t.Errorf("ResolveHost(metadata) = %q; want %q", got, HostBucketRunPod)
	}
	setProvider(t, InferenceProviderOllama)
	if got := ResolveHost("metadata", "/tmp/whatever.jpg"); got != HostBucketOllama {
		t.Errorf("ResolveHost(metadata) = %q; want %q", got, HostBucketOllama)
	}
	// Local ONNX tasks have fixed dedicated buckets regardless of provider.
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

	setProvider(t, InferenceProviderRunPod)

	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatalf("open sqlite: %v", err)
	}
	t.Cleanup(func() { db.Close() })
	q := jobqueue.NewQueueWithDB(db)

	// Limit RunPod bucket to 2; everything else stays at default 1.
	cfg := appconfig.Get()
	cfg.InferenceConcurrency.RunPod = 2
	cfg.InferenceConcurrency.Ollama = 1
	ApplyHostLimits(q, cfg)

	// Use "metadata" (an LLM-backed task) — it routes to the RunPod inference
	// bucket. autotag/embed now have their own buckets and would not exercise
	// the inference cap here.
	for i := 0; i < 3; i++ {
		if _, err := q.AddJob("", "metadata", nil, "/tmp/whatever.jpg", nil); err != nil {
			t.Fatalf("AddJob %d: %v", i, err)
		}
	}

	// First two claims should succeed (bucket capacity 2).
	first, err := q.ClaimJob()
	if err != nil || first == nil {
		t.Fatalf("ClaimJob #1 returned (%v, %v)", first, err)
	}
	if first.Host != HostBucketRunPod {
		t.Errorf("first.Host = %q; want %q", first.Host, HostBucketRunPod)
	}

	second, err := q.ClaimJob()
	if err != nil || second == nil {
		t.Fatalf("ClaimJob #2 returned (%v, %v)", second, err)
	}

	// Third claim should be blocked because RunPod is at capacity.
	third, err := q.ClaimJob()
	if err != nil {
		t.Fatalf("ClaimJob #3 error: %v", err)
	}
	if third != nil {
		t.Errorf("ClaimJob #3 returned a job (id=%s) but RunPod bucket should be full", third.ID)
	}

	// Lowering the bucket to 1 and finishing one job should still leave
	// the bucket overdrawn (1 running vs limit 1) — no new claim until
	// completion releases the slot.
	cfg.InferenceConcurrency.RunPod = 1
	ApplyHostLimits(q, cfg)
	if err := q.CompleteJob(first.ID); err != nil {
		t.Fatalf("CompleteJob: %v", err)
	}
	// One in-flight (second), limit 1 → still nothing claimable.
	blocked, err := q.ClaimJob()
	if err != nil {
		t.Fatalf("ClaimJob after lowering limit: %v", err)
	}
	if blocked != nil {
		t.Errorf("ClaimJob returned %s while bucket should be at capacity", blocked.ID)
	}
}
