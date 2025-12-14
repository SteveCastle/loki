package tasks

import (
	"encoding/base64"
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

// isMediaFile checks if a file is a supported media file based on its extension
func isMediaFile(path string) bool {
	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".heic", ".tif", ".tiff",
		".mp4", ".mov", ".avi", ".mkv", ".webm", ".wmv":
		return true
	}
	return false
}

// getMediaPathsByQuery retrieves all media paths matching a search query using the media package
func getMediaPathsByQuery(db *sql.DB, query string) ([]string, error) {
	const pageSize = 1000
	offset := 0
	var paths []string
	for {
		items, hasMore, err := media.GetItems(db, offset, pageSize, query)
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
	return paths, nil
}
