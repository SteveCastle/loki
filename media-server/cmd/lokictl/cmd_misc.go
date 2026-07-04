package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
)

func init() {
	register(command{
		group: "health", summary: "Server health + queue snapshot (GET /health, no auth)",
		run: func(a *App, args []string) int {
			var out any
			if err := a.Client.DoJSON("GET", "/health", nil, &out); err != nil {
				return a.Fail(err)
			}
			return a.PrintJSON(out)
		},
	})
	register(command{
		group: "stats", summary: "Library statistics (GET /api/stats; Windows servers only)",
		run: func(a *App, args []string) int {
			var out any
			if err := a.Client.DoJSON("GET", "/api/stats", nil, &out); err != nil {
				var apiErr *APIError
				if errors.As(err, &apiErr) && apiErr.Status == http.StatusNotFound {
					apiErr.Hint = "the stats API is only registered on Windows servers"
				}
				return a.Fail(err)
			}
			return a.PrintJSON(out)
		},
	})
	register(command{
		group: "api", args: "<METHOD> <path> [--body JSON|@file|-]",
		summary: "Raw escape hatch — call any server endpoint with auth attached",
		run:     cmdAPI,
	})
	register(command{
		group: "whoami", summary: "Who the current token authenticates as (GET /auth/status)",
		run: func(a *App, args []string) int {
			var out any
			if err := a.Client.DoJSON("GET", "/auth/status", nil, &out); err != nil {
				return a.Fail(err)
			}
			return a.PrintJSON(out)
		},
	})
}

// readBodyArg resolves --body values: literal JSON, @file, or - for stdin.
func readBodyArg(v string, stdin io.Reader) ([]byte, error) {
	switch {
	case v == "":
		return nil, nil
	case v == "-":
		return io.ReadAll(stdin)
	case strings.HasPrefix(v, "@"):
		return os.ReadFile(v[1:])
	default:
		return []byte(v), nil
	}
}

func cmdAPI(a *App, args []string) int {
	fs := flag.NewFlagSet("api", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	body := fs.String("body", "", "request body: literal JSON, @file, or - for stdin")
	if len(args) < 2 || strings.HasPrefix(args[0], "-") || strings.HasPrefix(args[1], "-") {
		return a.Usage(fs, "usage: lokictl api <METHOD> <path> [--body JSON|@file|-]")
	}
	if err := fs.Parse(args[2:]); err != nil {
		return a.Usage(fs, err.Error())
	}
	method := strings.ToUpper(args[0])
	path := args[1]
	if !strings.HasPrefix(path, "/") {
		path = "/" + path
	}

	b, err := readBodyArg(*body, os.Stdin)
	if err != nil {
		return a.Fail(fmt.Errorf("failed to read body: %w", err))
	}
	var reader io.Reader
	if b != nil {
		reader = bytes.NewReader(b)
	}
	resp, err := a.Client.DoRaw(method, path, "application/json", reader)
	if err != nil {
		return a.Fail(err)
	}
	defer resp.Body.Close()
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return a.Fail(fmt.Errorf("failed to read response: %w", err))
	}
	trimmed := bytes.TrimSpace(respBody)
	if len(trimmed) == 0 {
		return a.PrintJSON(map[string]any{"status": resp.StatusCode})
	}
	var pretty bytes.Buffer
	if json.Indent(&pretty, trimmed, "", "  ") == nil {
		fmt.Fprintln(a.Out, pretty.String())
		return 0
	}
	// Non-JSON response (HTML admin pages etc.) — pass through raw.
	fmt.Fprintln(a.Out, string(respBody))
	return 0
}
