package tasks

import (
	"encoding/base64"
	"fmt"
	"path/filepath"
	"strings"

	"database/sql"

	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/media"
)

// extractQueryFromJob checks args and input for a query string.
func extractQueryFromJob(j *jobqueue.Job) (string, bool) {
	for i := 0; i < len(j.Arguments); i++ {
		arg := j.Arguments[i]
		lower := strings.ToLower(arg)
		if strings.HasPrefix(lower, "--query64=") {
			enc := arg[len("--query64="):]
			if dec, err := base64.StdEncoding.DecodeString(enc); err == nil {
				return string(dec), true
			}
		}
		if lower == "--query64" {
			if i+1 < len(j.Arguments) {
				enc := j.Arguments[i+1]
				if dec, err := base64.StdEncoding.DecodeString(enc); err == nil {
					return string(dec), true
				}
			}
		}
		if lower == "--query" {
			if i+1 < len(j.Arguments) {
				start := i + 1
				end := start
				for end < len(j.Arguments) {
					next := j.Arguments[end]
					if strings.HasPrefix(next, "-") && end > start {
						break
					}
					end++
				}
				return strings.Join(j.Arguments[start:end], " "), true
			}
		}
		if strings.HasPrefix(lower, "--query=") {
			return arg[len("--query="):], true
		}
	}

	input := strings.TrimSpace(j.Input)
	if input != "" {
		tokens := tokenizeCommandLine(input)
		for i := 0; i < len(tokens); i++ {
			lt := strings.ToLower(tokens[i])
			if lt == "--query64" {
				if i+1 < len(tokens) {
					start := i + 1
					end := start
					for end < len(tokens) {
						next := tokens[end]
						if strings.HasPrefix(next, "-") && end > start {
							break
						}
						end++
					}
					joined := strings.Join(tokens[start:end], " ")
					if dec, err := base64.StdEncoding.DecodeString(joined); err == nil {
						return string(dec), true
					}
				}
			}
			if lt == "--query" {
				if i+1 < len(tokens) {
					start := i + 1
					end := start
					for end < len(tokens) {
						next := tokens[end]
						if strings.HasPrefix(next, "-") && end > start {
							break
						}
						end++
					}
					return strings.Join(tokens[start:end], " "), true
				}
			}
			if strings.HasPrefix(lt, "--query=") {
				return tokens[i][len("--query="):], true
			}
			if strings.HasPrefix(lt, "--query64=") {
				enc := tokens[i][len("--query64="):]
				if dec, err := base64.StdEncoding.DecodeString(enc); err == nil {
					return string(dec), true
				}
			}
		}
	}

	return "", false
}

// tokenizeCommandLine splits a command line string into tokens, respecting simple quotes
func tokenizeCommandLine(s string) []string {
	var tokens []string
	var b strings.Builder
	inQuotes := false
	quoteChar := byte(0)

	flush := func() {
		if b.Len() > 0 {
			tokens = append(tokens, b.String())
			b.Reset()
		}
	}

	for i := 0; i < len(s); i++ {
		c := s[i]
		if inQuotes {
			if c == quoteChar {
				inQuotes = false
				continue
			}
			b.WriteByte(c)
			continue
		}
		if c == '"' || c == '\'' {
			inQuotes = true
			quoteChar = c
			continue
		}
		if c == ' ' || c == '\t' || c == '\n' || c == '\r' {
			flush()
			continue
		}
		b.WriteByte(c)
	}
	flush()
	return tokens
}

// parseInputPaths parses newline and comma separated file paths from j.Input
func parseInputPaths(raw string) []string {
	var paths []string
	for _, line := range strings.Split(raw, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		for _, part := range strings.Split(line, ",") {
			p := strings.Trim(strings.TrimSpace(part), `"'`)
			if p != "" {
				paths = append(paths, p)
			}
		}
	}
	return paths
}

// isMediaFile checks if a file is a supported media file based on its
// extension. The audio set mirrors the client's FileTypes.Audio so
// transcript jobs can target audio libraries; image/video mirror
// media/search.go's extensionsForFiletype.
func isMediaFile(path string) bool {
	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".heic", ".tif", ".tiff",
		".mp4", ".mov", ".avi", ".mkv", ".webm", ".wmv", ".m4v",
		".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".opus", ".wma", ".aiff", ".ape":
		return true
	}
	return false
}

// filterMediaPaths drops non-media paths from a query result. Library scans
// ingest sidecar files too (e.g. the .json metadata some downloaders pair
// with every image), so media rows are NOT guaranteed to be media files —
// without this, a batch task fed by a query does a job's worth of work per
// sidecar as well, doubling its target count.
func filterMediaPaths(paths []string) []string {
	out := make([]string, 0, len(paths))
	for _, p := range paths {
		if isMediaFile(p) {
			out = append(out, p)
		}
	}
	return out
}

// getMediaPathsByQuery retrieves all media paths matching a search query using the media package
func getMediaPathsByQuery(db *sql.DB, query string) ([]string, error) {
	const pageSize = 1000
	offset := 0
	var paths []string
	for {
		items, _, hasMore, err := media.GetItems(db, offset, pageSize, query)
		if err != nil {
			return nil, err
		}
		for _, it := range items {
			paths = append(paths, it.Path)
		}
		if !hasMore {
			break
		}
		offset += pageSize
	}
	return filterMediaPaths(paths), nil
}

// getMediaPathsByQueryFast retrieves all media paths matching a search query efficiently
// This only fetches paths (no tags, no file existence checks, no extra columns)
func getMediaPathsByQueryFast(db *sql.DB, query string) ([]string, error) {
	paths, err := media.GetPathsByQuery(db, query)
	if err != nil {
		return nil, err
	}
	return filterMediaPaths(paths), nil
}

// resolvedItems is the outcome of resolveJobItems: the concrete file list a
// per-item task will run over, plus where it came from.
type resolvedItems struct {
	Paths     []string
	FromQuery bool
	Query     string // the decoded query when FromQuery
}

// resolveJobItems is the single input-resolution path for per-item tasks.
// Precedence matches the historical behavior of every task: a search query
// (--query / --query64, in arguments or input) wins; otherwise the input is
// parsed as a newline/comma-separated path list (flag tokens dropped, paths
// absolutized, non-media files filtered out).
func resolveJobItems(j *jobqueue.Job, q *jobqueue.Queue) (resolvedItems, error) {
	return resolveJobItemsFiltered(j, q, true)
}

// resolveJobItemsRaw is resolveJobItems without the non-media filter on
// explicit path lists — for transform tasks (ffmpeg, hls) whose workflow
// inputs legitimately include non-library files (.m3u8 playlists, temp
// frames, subtitle files).
func resolveJobItemsRaw(j *jobqueue.Job, q *jobqueue.Queue) (resolvedItems, error) {
	return resolveJobItemsFiltered(j, q, false)
}

func resolveJobItemsFiltered(j *jobqueue.Job, q *jobqueue.Queue, mediaOnly bool) (resolvedItems, error) {
	if qstr, ok := extractQueryFromJob(j); ok {
		paths, err := getMediaPathsByQueryFast(q.Db, qstr)
		if err != nil {
			return resolvedItems{}, fmt.Errorf("load media paths for query: %w", err)
		}
		return resolvedItems{Paths: paths, FromQuery: true, Query: qstr}, nil
	}

	raw := strings.TrimSpace(j.Input)
	if raw == "" {
		return resolvedItems{}, nil
	}
	var out []string
	for _, p := range parseInputPaths(raw) {
		// Option flags share the input string with paths on some job shapes
		// (e.g. "--overwrite" landing in input via workflow chaining); they
		// are never paths.
		if strings.HasPrefix(p, "--") {
			continue
		}
		// Absolutize LOCAL paths only — s3:// identities must survive
		// verbatim (Abs/FromSlash would mangle the scheme into a bogus
		// local path and every op would fail with "not found on disk").
		if !strings.HasPrefix(p, "s3://") {
			if abs, err := filepath.Abs(p); err == nil {
				p = filepath.FromSlash(abs)
			}
		}
		out = append(out, p)
	}
	if mediaOnly {
		out = filterMediaPaths(out)
	}
	return resolvedItems{Paths: out}, nil
}
