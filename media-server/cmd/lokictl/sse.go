package main

import (
	"bufio"
	"context"
	"io"
	"strings"
)

type sseEvent struct {
	Name string
	Data string
}

// readSSE parses a Server-Sent-Events stream, invoking fn for each event.
// Multiple data: lines are joined with \n. fn returning false stops reading.
// Returns nil on clean EOF or when fn stopped the stream.
func readSSE(ctx context.Context, r io.Reader, fn func(sseEvent) bool) error {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	var ev sseEvent
	var dataLines []string
	flush := func() bool {
		if ev.Name == "" && len(dataLines) == 0 {
			return true
		}
		ev.Data = strings.Join(dataLines, "\n")
		ok := fn(ev)
		ev = sseEvent{}
		dataLines = nil
		return ok
	}
	for scanner.Scan() {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		line := scanner.Text()
		switch {
		case line == "":
			if !flush() {
				return nil
			}
		case strings.HasPrefix(line, "event:"):
			ev.Name = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
		case strings.HasPrefix(line, "data:"):
			dataLines = append(dataLines, strings.TrimPrefix(strings.TrimPrefix(line, "data:"), " "))
		case strings.HasPrefix(line, ":"):
			// comment / keep-alive
		}
	}
	flush()
	if err := scanner.Err(); err != nil && ctx.Err() == nil {
		return err
	}
	return nil
}
