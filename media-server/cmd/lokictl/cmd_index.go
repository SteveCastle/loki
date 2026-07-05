package main

import (
	"flag"
	"fmt"
	"io"
	"net/url"
	"strconv"
	"strings"
)

func init() {
	register(command{group: "index", name: "status",
		summary: "Embedding index + stored-vector stats, orphans, coverage (GET /api/index/status)",
		run:     cmdIndexStatus})
	register(command{group: "index", name: "models",
		summary: "Embedding model registry with active/indexed flags (GET /api/index/models)",
		run:     cmdIndexModels})
	register(command{group: "index", name: "rebuild",
		summary: "Rebuild the vector index for the active model (POST /api/index/rebuild); large libraries need --timeout",
		run:     cmdIndexRebuild})
	register(command{group: "index", name: "missing", args: "[--model M] [--limit N]",
		summary: "Media paths lacking an embedding (GET /api/index/missing)",
		run:     cmdIndexMissing})
	register(command{group: "index", name: "get", args: "<path> [--model M] [--vector]",
		summary: "Stored embedding rows for one media item (GET /api/embeddings)",
		run:     cmdIndexGet})
	register(command{group: "index", name: "delete", args: "<path> [--model M] --yes",
		summary: "Delete stored embeddings for one media item (DELETE /api/embeddings)",
		run:     cmdIndexDelete})
	register(command{group: "index", name: "prune", args: "--yes",
		summary: "Delete orphaned embeddings whose media row is gone (POST /api/embeddings/prune)",
		run:     cmdIndexPrune})
	register(command{group: "index", name: "embed",
		args:    "[paths/tokens...] [--field k=v]... [--wait] [--follow] [--timeout D]",
		summary: `Alias for "job run embed" — generate embeddings via the job queue`,
		run:     cmdIndexEmbed})
}

func cmdIndexStatus(a *App, args []string) int {
	if len(args) > 0 {
		return a.Usage(nil, "index status takes no arguments")
	}
	var out any
	if err := a.Client.DoJSON("GET", "/api/index/status", nil, &out); err != nil {
		return a.Fail(err)
	}
	return a.PrintJSON(out)
}

func cmdIndexModels(a *App, args []string) int {
	if len(args) > 0 {
		return a.Usage(nil, "index models takes no arguments")
	}
	var resp struct {
		Models []map[string]any `json:"models"`
	}
	if err := a.Client.DoJSON("GET", "/api/index/models", nil, &resp); err != nil {
		return a.Fail(err)
	}
	return a.PrintJSON(resp.Models)
}

func cmdIndexRebuild(a *App, args []string) int {
	if len(args) > 0 {
		return a.Usage(nil, "index rebuild takes no arguments (raise --timeout for large libraries)")
	}
	var out any
	if err := a.Client.DoJSON("POST", "/api/index/rebuild", nil, &out); err != nil {
		return a.Fail(err)
	}
	return a.PrintJSON(out)
}

func cmdIndexMissing(a *App, args []string) int {
	fs := flag.NewFlagSet("index missing", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	model := fs.String("model", "", "embedding model id (default: the active model)")
	limit := fs.Int("limit", 100, "max paths to return (0 = count only)")
	if err := fs.Parse(args); err != nil {
		return a.Usage(fs, err.Error())
	}
	q := url.Values{"limit": {strconv.Itoa(*limit)}}
	if *model != "" {
		q.Set("model", *model)
	}
	var out any
	if err := a.Client.DoJSON("GET", "/api/index/missing?"+q.Encode(), nil, &out); err != nil {
		return a.Fail(err)
	}
	return a.PrintJSON(out)
}

func cmdIndexGet(a *App, args []string) int {
	fs := flag.NewFlagSet("index get", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	model := fs.String("model", "", "only this model's row")
	vector := fs.Bool("vector", false, "include the raw float vector")
	if len(args) == 0 || strings.HasPrefix(args[0], "-") {
		return a.Usage(fs, "usage: lokictl index get <path> [--model M] [--vector]")
	}
	path := args[0]
	if err := fs.Parse(args[1:]); err != nil {
		return a.Usage(fs, err.Error())
	}
	q := url.Values{"path": {path}}
	if *model != "" {
		q.Set("model", *model)
	}
	if *vector {
		q.Set("vector", "true")
	}
	var out any
	if err := a.Client.DoJSON("GET", "/api/embeddings?"+q.Encode(), nil, &out); err != nil {
		return a.Fail(err)
	}
	return a.PrintJSON(out)
}

func cmdIndexDelete(a *App, args []string) int {
	fs := flag.NewFlagSet("index delete", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	model := fs.String("model", "", "only this model's row (default: all models)")
	fs.Bool("yes", false, "confirm destructive action")
	if len(args) == 0 || strings.HasPrefix(args[0], "-") {
		return a.Usage(fs, "usage: lokictl index delete <path> [--model M] --yes")
	}
	path := args[0]
	if err := fs.Parse(args[1:]); err != nil {
		return a.Usage(fs, err.Error())
	}
	if !hasYesFlag(args) {
		return a.Usage(fs, fmt.Sprintf("this deletes stored embeddings for %q — re-run with --yes to confirm", path))
	}
	q := url.Values{"path": {path}}
	if *model != "" {
		q.Set("model", *model)
	}
	var out any
	if err := a.Client.DoJSON("DELETE", "/api/embeddings?"+q.Encode(), nil, &out); err != nil {
		return a.Fail(err)
	}
	return a.PrintJSON(out)
}

func cmdIndexPrune(a *App, args []string) int {
	if !hasYesFlag(args) {
		return a.Usage(nil, "this deletes every embedding whose media row is gone — re-run with --yes to confirm")
	}
	var out any
	if err := a.Client.DoJSON("POST", "/api/embeddings/prune", nil, &out); err != nil {
		return a.Fail(err)
	}
	return a.PrintJSON(out)
}

// cmdIndexEmbed is a discoverability alias: identical semantics to
// "lokictl job run embed <args...>".
func cmdIndexEmbed(a *App, args []string) int {
	return cmdJobRun(a, append([]string{"embed"}, args...))
}
