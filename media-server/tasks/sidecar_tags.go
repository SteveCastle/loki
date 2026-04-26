package tasks

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/media"
)

// sidecarTagCategory is the category under which tags lifted from gallery-dl
// JSON sidecars are stored.
const sidecarTagCategory = "Suggested"

// extractSidecarTags reads a gallery-dl --write-metadata JSON sidecar and
// returns the deduplicated list of tag tokens. Tokens are taken from the
// "tags" field if present (used by gelbooru, rule34, etc.), otherwise from
// "tag_string" (used by danbooru). Both are space-separated strings.
//
// Returns (nil, nil) when the file exists and parses but has no usable tag
// field — a soft no-op for callers.
func extractSidecarTags(jsonPath string) ([]string, error) {
	data, err := os.ReadFile(jsonPath)
	if err != nil {
		return nil, err
	}
	var obj map[string]any
	if err := json.Unmarshal(data, &obj); err != nil {
		return nil, fmt.Errorf("parse %s: %w", jsonPath, err)
	}

	raw, _ := obj["tags"].(string)
	if raw == "" {
		raw, _ = obj["tag_string"].(string)
	}
	if raw == "" {
		return nil, nil
	}

	seen := make(map[string]struct{})
	var tags []string
	for _, t := range strings.Fields(raw) {
		if _, dup := seen[t]; dup {
			continue
		}
		seen[t] = struct{}{}
		tags = append(tags, t)
	}
	return tags, nil
}

// applySidecarTagsToMedia parses each sidecar in finalToSidecar and inserts
// the extracted tags against the corresponding final media path under the
// "Suggested" category. Errors on individual files are logged to the job
// stdout and do not abort the loop.
func applySidecarTagsToMedia(db *sql.DB, q *jobqueue.Queue, jobID string, finalToSidecar map[string]string) {
	if len(finalToSidecar) == 0 {
		return
	}
	totalTags := 0
	taggedFiles := 0
	for finalPath, sidecar := range finalToSidecar {
		tags, err := extractSidecarTags(sidecar)
		if err != nil {
			q.PushJobStdout(jobID, fmt.Sprintf("Warning: sidecar parse failed for %s: %v", sidecar, err))
			continue
		}
		if len(tags) == 0 {
			continue
		}
		inserted := 0
		for _, label := range tags {
			if err := media.AddTag(db, finalPath, label, sidecarTagCategory); err != nil {
				q.PushJobStdout(jobID, fmt.Sprintf("Warning: failed to add tag %s to %s: %v", label, finalPath, err))
				continue
			}
			inserted++
		}
		if inserted > 0 {
			taggedFiles++
			totalTags += inserted
		}
	}
	q.PushJobStdout(jobID, fmt.Sprintf("Applied %d sidecar tag(s) across %d file(s)", totalTags, taggedFiles))
}
