// Package workerapi is the wire contract between the Lowkey Media Server and
// loki-worker, the node agent that runs the loki-* GPU engines (loki-retouch,
// loki-reshoot, loki-dream, loki-segment) as resident processes on a GPU
// host and serves them over HTTP (persistent nodes) or as a RunPod
// queue-based serverless worker. It holds the shared types, the clients both
// sides use, and the argument staging that lets a command line written for
// local files run on a machine that has none of them.
//
// A job is a command line for one engine. Every local file it names is sent
// along (uploaded with the job, or fetched by the worker from a presigned
// URL) and placed under the job's directory on the worker; the command line
// refers to that directory as $JOB. The outputs the engine writes there come
// back the same way, mapped to the local paths the command line asked for.
package workerapi

import (
	"strings"
	"time"
)

// Protocol is the version of this contract; bumped only on incompatible change.
const Protocol = 1

// JobDir is the placeholder for the job's directory on the worker, in job
// arguments and in the stdout/log lines a worker returns.
const JobDir = "$JOB"

// SessionDir is the same placeholder for a session's directory.
const SessionDir = "$SESSION"

// Engines are the tools a worker can host, by executable name.
var Engines = []string{"loki-retouch", "loki-reshoot", "loki-dream", "loki-segment"}

// EngineName normalizes an engine name: "retouch" and "loki-retouch" are the
// same engine.
func EngineName(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	s = strings.TrimSuffix(s, ".exe")
	if s != "" && !strings.HasPrefix(s, "loki-") {
		s = "loki-" + s
	}
	return s
}

// Input is one file a job needs, placed at Path (relative to the job
// directory, e.g. "in/0/photo.jpg"). Its bytes come with the multipart
// request (the part named Path), from URL (a presigned GET the worker
// fetches), or inline in Data.
type Input struct {
	Path string `json:"path"`
	URL  string `json:"url,omitempty"`
	Data []byte `json:"data,omitempty"`
	Size int64  `json:"size,omitempty"`
}

// OutputSink says where a job's outputs go when the caller cannot download
// them from the worker (serverless workers are not addressable): each output
// file is PUT to the next presigned URL in order, or returned inline in the
// result when Inline is set and they fit MaxInline bytes in total.
type OutputSink struct {
	PutURLs   []string `json:"putUrls,omitempty"`
	Inline    bool     `json:"inline,omitempty"`
	MaxInline int64    `json:"maxInline,omitempty"`
}

// JobSpec is one engine run.
type JobSpec struct {
	// Tool is the engine (loki-retouch, ...).
	Tool string `json:"tool"`
	// Args is the command line after the executable; $JOB stands for the
	// job directory.
	Args []string `json:"args"`
	// Inputs are placed in the job directory before the run.
	Inputs []Input `json:"inputs,omitempty"`
	// Dirs are created in the job directory before the run (an output
	// folder the engine expects to exist).
	Dirs []string `json:"dirs,omitempty"`
	// Outputs, when set, pushes the results out instead of keeping them for
	// download.
	Outputs *OutputSink `json:"outputs,omitempty"`
	// Light marks a quick call that loads no model (--show-prompt, --help):
	// it runs as its own short process instead of queueing behind GPU work.
	Light bool `json:"light,omitempty"`
	// Env adds environment variables for this run (engine tuning knobs,
	// LOKI_*); only LOKI_ names are honored.
	Env map[string]string `json:"env,omitempty"`
}

// JobState is a job's lifecycle state.
type JobState string

const (
	JobQueued    JobState = "queued"
	JobRunning   JobState = "running"
	JobDone      JobState = "done"
	JobFailed    JobState = "failed"
	JobCancelled JobState = "cancelled"
)

// Finished reports whether the state is terminal.
func (s JobState) Finished() bool {
	return s == JobDone || s == JobFailed || s == JobCancelled
}

// OutputFile is one file a job produced, relative to the job directory. Slot
// is the index of the PutURL it went to (-1: kept on the worker); Data holds
// it when returned inline.
type OutputFile struct {
	Path string `json:"path"`
	Size int64  `json:"size"`
	Slot int    `json:"slot"`
	Data []byte `json:"data,omitempty"`
}

// JobStatus is a job as the worker reports it. Logs holds the stderr lines
// from index LogStart on (a poll passes ?logs=N with the next index it
// wants, LogNext).
type JobStatus struct {
	ID       string       `json:"id"`
	Tool     string       `json:"tool"`
	State    JobState     `json:"state"`
	Error    string       `json:"error,omitempty"`
	OOM      bool         `json:"oom,omitempty"`
	Stdout   []string     `json:"stdout,omitempty"`
	Outputs  []OutputFile `json:"outputs,omitempty"`
	Logs     []string     `json:"logs,omitempty"`
	LogStart int          `json:"logStart"`
	LogNext  int          `json:"logNext"`
	// Step / Steps mirror the engine's last "step i/n" line.
	Step  int `json:"step,omitempty"`
	Steps int `json:"steps,omitempty"`
	// QueuePos is the number of jobs ahead of this one (queued only).
	QueuePos int       `json:"queuePos,omitempty"`
	Worker   string    `json:"worker,omitempty"`
	GPU      int       `json:"gpu"`
	Created  time.Time `json:"created"`
	Started  time.Time `json:"started,omitzero"`
	Ended    time.Time `json:"ended,omitzero"`
	// Secs is the engine's own time for the run (excludes staging).
	Secs float64 `json:"secs,omitempty"`
}

// Resident is one model held on a worker's GPU.
type Resident struct {
	Key string `json:"key"`
	MB  int    `json:"mb"`
}

// ToolInfo describes one engine on a worker.
type ToolInfo struct {
	Name    string `json:"name"`
	Version string `json:"version,omitempty"`
	Model   string `json:"model,omitempty"`
	// Path is the executable on the worker; empty when not installed.
	Path string `json:"path,omitempty"`
	// State per GPU slot: "stopped" (no process), "starting", "idle", "busy".
	States   []string   `json:"states,omitempty"`
	Resident []Resident `json:"resident,omitempty"`
	// Help is the engine's --help text (only with GET /v1/info?help=1); a
	// caller uses it to tell which flags this build takes.
	Help string `json:"help,omitempty"`
}

// GPU is one device of a worker.
type GPU struct {
	Index   int    `json:"index"`
	Name    string `json:"name,omitempty"`
	TotalMB int    `json:"totalMb,omitempty"`
	FreeMB  int    `json:"freeMb,omitempty"`
	// Busy is the id of the job running on it.
	Busy string `json:"busy,omitempty"`
}

// Info is a worker's self-description (GET /v1/info).
type Info struct {
	Worker   string     `json:"worker"`
	Version  string     `json:"version"`
	Protocol int        `json:"protocol"`
	Mode     string     `json:"mode"`
	Ready    bool       `json:"ready"`
	GPUs     []GPU      `json:"gpus"`
	Tools    []ToolInfo `json:"tools"`
	// Slots is how many GPU jobs run at once (one per GPU).
	Slots   int     `json:"slots"`
	Running int     `json:"running"`
	Queued  int     `json:"queued"`
	Uptime  float64 `json:"uptime"`
	// Sessions is the number of open interactive sessions.
	Sessions int `json:"sessions"`
}

// Tool returns the named engine's entry.
func (i *Info) Tool(name string) *ToolInfo {
	name = EngineName(name)
	for k := range i.Tools {
		if i.Tools[k].Name == name {
			return &i.Tools[k]
		}
	}
	return nil
}

// Warm reports whether a model of the named engine is resident on a GPU.
func (i *Info) Warm(name string) bool {
	t := i.Tool(name)
	return t != nil && len(t.Resident) > 0
}

// WarmRequest asks a worker to fetch and load engines' models ahead of use.
type WarmRequest struct {
	Tools []string `json:"tools"`
	// Model picks a variant ("fast" for loki-reshoot's FastH3 Omni).
	Model string `json:"model,omitempty"`
}

// SessionCall is one request in an interactive session (loki-segment's
// --serve vocabulary: open, segment, track, close). Paths in Req use
// $SESSION for the session directory.
type SessionCall struct {
	Req map[string]any `json:"req"`
}

// SessionReply is a session request's answer: the engine's reply and the
// files the session directory gained or changed with it.
type SessionReply struct {
	Reply map[string]any `json:"reply"`
	Files []OutputFile   `json:"files,omitempty"`
	Error string         `json:"error,omitempty"`
}

// RunPodInput is the "input" of a RunPod serverless job for loki-worker:
// exactly one of Spec (run an engine), Warm or Info.
type RunPodInput struct {
	Spec *JobSpec     `json:"spec,omitempty"`
	Warm *WarmRequest `json:"warm,omitempty"`
	Info bool         `json:"info,omitempty"`
}

// RunPodStream is one chunk a serverless job streams while it runs.
type RunPodStream struct {
	Logs  []string `json:"logs,omitempty"`
	Step  int      `json:"step,omitempty"`
	Steps int      `json:"steps,omitempty"`
}
