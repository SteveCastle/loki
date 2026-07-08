package tasks

import (
	"net/url"
	"strings"
	"sync"

	"github.com/stevecastle/shrike/appconfig"
	"github.com/stevecastle/shrike/jobqueue"
)

// HostResolverFn returns the concurrency-bucket name for a job of the
// command this resolver is registered against. It receives the job input
// (typically a file path or URL) so resolvers that need to dispatch by
// input shape (e.g. ingest's URL parsing) can do so. Resolvers that don't
// care about the input can ignore it.
type HostResolverFn func(input string) string

var (
	hostResolversMu sync.RWMutex
	hostResolvers   = map[string]HostResolverFn{}
)

// RegisterHostResolver attaches a host-bucket resolver to a task command.
// Call alongside RegisterTask in registry.go (or any init()) so the policy
// for "where does this task's concurrency live" sits next to the task
// itself. Commands without a registered resolver fall back to "localhost".
func RegisterHostResolver(command string, fn HostResolverFn) {
	hostResolversMu.Lock()
	defer hostResolversMu.Unlock()
	hostResolvers[command] = fn
}

// ResolveHost is the entry point handed to jobqueue.SetHostResolver. It
// looks up the per-command resolver and delegates; unknown commands land
// in the default "localhost" bucket.
func ResolveHost(command, input string) string {
	hostResolversMu.RLock()
	fn, ok := hostResolvers[command]
	hostResolversMu.RUnlock()
	if ok {
		return fn(input)
	}
	return "localhost"
}

// opResources maps one per-item op to the concurrency buckets its work
// consumes. This is what makes multi-bucket claiming honest: a combined job
// holds the union of its ops' buckets, so it excludes the standalone jobs it
// overlaps with AND takes the machine-wide local-compute slot when any of
// its work runs on local hardware.
func opResources(op string) []string {
	switch op {
	case "describe":
		r := []string{InferenceHost()}
		if InferenceHostIsLocal() {
			r = append(r, HostBucketLocalCompute)
		}
		return r
	case "transcribe":
		// Transcription runs faster-whisper on the local GPU.
		return []string{HostBucketLocalCompute}
	case "embed":
		return []string{HostBucketEmbed, HostBucketLocalCompute}
	case "autotag":
		return []string{HostBucketAutotag, HostBucketLocalCompute}
	case "faces":
		return []string{HostBucketFaces, HostBucketLocalCompute}
	default: // hash, dimensions — cheap I/O work, no model resources
		return nil
	}
}

// legacyTypeToOp mirrors media_metadata.go's mapping for resource resolution
// of legacy `metadata --type ...` jobs.
var legacyTypeToOp = map[string]string{
	"description": "describe",
	"transcript":  "transcribe",
	"hash":        "hash",
	"dimensions":  "dimensions",
}

// flagListValue extracts a comma-separated flag value (e.g. --ops=a,b or
// "--ops a,b") from a job's argument list and input string.
func flagListValue(flag string, arguments []string, input string) []string {
	tokens := append(append([]string{}, arguments...), tokenizeCommandLine(input)...)
	prefix := "--" + flag + "="
	for i := 0; i < len(tokens); i++ {
		var raw string
		if strings.HasPrefix(tokens[i], prefix) {
			raw = tokens[i][len(prefix):]
		} else if tokens[i] == "--"+flag && i+1 < len(tokens) {
			raw = tokens[i+1]
		} else {
			continue
		}
		var out []string
		for _, v := range strings.Split(raw, ",") {
			if v = strings.ToLower(strings.TrimSpace(v)); v != "" {
				out = append(out, v)
			}
		}
		return out
	}
	return nil
}

// ResolveResources is the entry point handed to jobqueue.SetResourceResolver:
// the ADDITIONAL buckets (beyond Host) a job occupies while running. Composite
// jobs resolve per selected op, so `process --ops=hash,dimensions` holds
// nothing extra while `process --ops=describe,embed,faces` holds the
// inference, embed, faces, and local-compute buckets.
func ResolveResources(command string, arguments []string, input string) []string {
	var ops []string
	switch command {
	case "describe", "transcribe", "embed", "autotag", "faces":
		ops = []string{command}
	case "faces-cluster":
		// Clustering shares the faces bucket (its Host) and crunches vectors
		// locally.
		return []string{HostBucketLocalCompute}
	case "process":
		ops = flagListValue("ops", arguments, input)
	case "metadata":
		for _, t := range flagListValue("type", arguments, input) {
			if op, ok := legacyTypeToOp[t]; ok {
				ops = append(ops, op)
			}
		}
		if len(ops) == 0 {
			// The legacy task's implicit default (describe,hash,dimensions).
			ops = []string{"describe", "hash", "dimensions"}
		}
	default:
		return nil
	}

	seen := map[string]struct{}{}
	var out []string
	for _, op := range ops {
		for _, b := range opResources(op) {
			if _, dup := seen[b]; dup {
				continue
			}
			seen[b] = struct{}{}
			out = append(out, b)
		}
	}
	return out
}

// ApplyHostLimits writes per-bucket concurrency caps from config into the
// queue. Safe to call repeatedly — limit changes only affect future
// ClaimJob calls, in-flight jobs keep running. Called once at startup and
// again after every config save so the UI's numbers take effect without a
// server restart.
//
// Buckets without a positive config value are left at the queue's default
// (1) so a misconfigured/zero value doesn't accidentally stop a provider
// from running anything at all.
func ApplyHostLimits(q *jobqueue.Queue, cfg appconfig.Config) {
	if q == nil {
		return
	}
	if n := cfg.InferenceConcurrency.Ollama; n > 0 {
		q.SetHostLimit(HostBucketOllama, n)
	}
	if n := cfg.InferenceConcurrency.RunPod; n > 0 {
		q.SetHostLimit(HostBucketRunPod, n)
	}
	if n := cfg.InferenceConcurrency.LMStudio; n > 0 {
		q.SetHostLimit(HostBucketLMStudio, n)
	}
	if n := cfg.InferenceConcurrency.LlamaCpp; n > 0 {
		q.SetHostLimit(HostBucketLlamaCpp, n)
	}
	// One embed job at a time; the job parallelizes internally via its worker
	// pool, so additional concurrent embed jobs would just oversubscribe.
	q.SetHostLimit(HostBucketEmbed, 1)
	// Likewise one autotag job at a time (internally parallel via its pool).
	q.SetHostLimit(HostBucketAutotag, 1)
	// Likewise one faces job at a time (internally parallel via its pool).
	q.SetHostLimit(HostBucketFaces, 1)
	// Machine-wide cap on concurrent heavy LOCAL jobs (see the bucket's doc
	// in llm_vision.go). Config value <= 0 falls back to the safe default of
	// one heavy local workload at a time.
	localLimit := cfg.LocalComputeConcurrency
	if localLimit <= 0 {
		localLimit = 1
	}
	q.SetHostLimit(HostBucketLocalCompute, localLimit)
}

// urlHostResolver pulls the URL hostname out of an input string, falling
// back to "localhost" when the input isn't a parseable URL. Used as the
// ingest command's host resolver — moved here from jobqueue so the
// jobqueue layer no longer knows about specific task commands.
func urlHostResolver(input string) string {
	u, err := url.Parse(input)
	if err == nil && u.Host != "" {
		return strings.TrimPrefix(u.Hostname(), "www.")
	}
	return "localhost"
}
