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
