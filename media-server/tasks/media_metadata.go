package tasks

// media_metadata.go — the legacy "metadata" task, kept as a compatibility
// alias. Its subtasks now live as standalone ItemOps (describe, transcribe,
// hash, dimensions, llm-autotag); this task just maps the old --type values
// onto those ops and runs them through the unified per-item runner, so
// existing saved workflows, SPA payloads, and scripts keep working while
// gaining progress reporting, pause/resume, and per-item durability.

import (
	"strings"
	"sync"

	"github.com/stevecastle/shrike/jobqueue"
)

var metadataOptions = []TaskOption{
	{Name: "type", Label: "Metadata Types", Type: "multi-enum", Choices: []string{"description", "transcript", "hash", "dimensions", "autotag"}, Default: "description,hash,dimensions", Description: "Comma-separated list of metadata types to generate"},
	{Name: "overwrite", Label: "Overwrite Existing", Type: "bool", Description: "Overwrite existing metadata values"},
	{Name: "apply", Label: "Apply Scope", Type: "enum", Choices: []string{"new", "all"}, Default: "new", Description: "Deprecated - has no effect (kept for compatibility)"},
	{Name: "model", Label: "Vision Model", Type: "string", Description: "Vision model to use for descriptions and tag selection"},
	{Name: "prompt", Label: "Custom Description Prompt", Type: "string", Description: "Override the configured describe prompt for this run"},
}

// legacyMetadataTypeToOp maps the old --type values to their ItemOp IDs.
var legacyMetadataTypeToOp = map[string]string{
	"description": "describe",
	"transcript":  "transcribe",
	"hash":        "hash",
	"dimensions":  "dimensions",
	"autotag":     "llm-autotag",
}

// metadataTask translates the legacy --type option into an op list and
// delegates to the unified runner. The describe/llm-autotag ops read the
// unprefixed --model/--prompt flags directly, matching the old behavior.
func metadataTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	opts := ParseOptions(j, metadataOptions)
	typesStr, _ := opts["type"].(string)

	var opIDs []string
	for _, t := range strings.Split(typesStr, ",") {
		t = strings.ToLower(strings.TrimSpace(t))
		if t == "" {
			continue
		}
		if opID, ok := legacyMetadataTypeToOp[t]; ok {
			opIDs = append(opIDs, opID)
		} else {
			q.PushJobStdout(j.ID, "Warning: unknown metadata type '"+t+"' - valid types are: description, transcript, hash, dimensions, autotag")
		}
	}
	if len(opIDs) == 0 {
		opIDs = []string{"describe", "hash", "dimensions"}
	}
	return runItemOps(j, q, opIDs, false)
}
