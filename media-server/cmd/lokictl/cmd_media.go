package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/url"
	"os"
	"strconv"
	"strings"
)

// predicate mirrors the server's media_query.go Predicate wire shape.
type predicate struct {
	Type    string `json:"type"`
	Value   string `json:"value"`
	Exclude bool   `json:"exclude,omitempty"`
	Join    string `json:"join,omitempty"`
}

// stringList collects repeatable flags (--tag a --tag b).
type stringList []string

func (s *stringList) String() string     { return strings.Join(*s, ",") }
func (s *stringList) Set(v string) error { *s = append(*s, v); return nil }

func init() {
	register(command{group: "media", name: "query",
		args:    "[--tag T]... [--exclude-tag T]... [--path P] [--description D] [--hash H] [--similar PATH] [--visual TEXT] [--mode AND|OR] [--predicates FILE|-]",
		summary: "Structured library query (POST /api/media/query)", run: cmdMediaQuery})
	register(command{group: "media", name: "search", args: "<text>",
		summary: "Search descriptions (POST /api/media/search)", run: cmdMediaSearch})
	register(command{group: "media", name: "similar", args: "<path> [--limit N]",
		summary: "Visually similar media (GET /api/media/similar)", run: cmdMediaSimilar})
	register(command{group: "media", name: "visual", args: "<text> [--limit N]",
		summary: "Text-to-image search (GET /api/media/search/visual)", run: cmdMediaVisual})
	register(command{group: "media", name: "metadata", args: "<path>",
		summary: "File metadata + description + hash (POST /api/media/metadata)",
		run:     mediaPathCommand("/api/media/metadata")})
	register(command{group: "media", name: "tags", args: "<path>",
		summary: "Tags on one media item (POST /api/media/tags)",
		run:     mediaPathCommand("/api/media/tags")})
	register(command{group: "media", name: "describe", args: "<path> --text D",
		summary: "Set a media item's description (POST /api/media/description)", run: cmdMediaDescribe})
	register(command{group: "media", name: "delete", args: "<path> --yes",
		summary: "Delete a media item (POST /api/media/delete)", run: cmdMediaDelete})
}

// buildPredicates converts the query flags into the predicate array.
func buildPredicates(tags, exTags stringList, path, description, hash, similar, visual, mode string) []predicate {
	var preds []predicate
	add := func(ptype, value string, exclude bool) {
		if value != "" {
			preds = append(preds, predicate{Type: ptype, Value: value, Exclude: exclude, Join: mode})
		}
	}
	for _, t := range tags {
		add("tag", t, false)
	}
	for _, t := range exTags {
		add("tag", t, true)
	}
	add("path", path, false)
	add("description", description, false)
	add("hash", hash, false)
	add("similar", similar, false)
	add("visual", visual, false)
	return preds
}

func cmdMediaQuery(a *App, args []string) int {
	fs := flag.NewFlagSet("media query", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	var tags, exTags stringList
	fs.Var(&tags, "tag", "require this tag (repeatable)")
	fs.Var(&exTags, "exclude-tag", "exclude this tag (repeatable)")
	path := fs.String("path", "", "path substring/glob filter")
	description := fs.String("description", "", "description substring filter")
	hash := fs.String("hash", "", "exact content-hash filter")
	similar := fs.String("similar", "", "visually similar to this media path")
	visual := fs.String("visual", "", "text-to-image visual search")
	mode := fs.String("mode", "AND", "combine predicates with AND or OR")
	predFile := fs.String("predicates", "", "raw predicate JSON (file or - for stdin); overrides the flags")
	if err := fs.Parse(args); err != nil {
		return a.Usage(fs, err.Error())
	}
	*mode = strings.ToUpper(*mode)
	if *mode != "AND" && *mode != "OR" {
		return a.Usage(fs, "--mode must be AND or OR")
	}

	var preds []predicate
	if *predFile != "" {
		b, err := readBodyArg(nullSafeAt(*predFile), os.Stdin)
		if err != nil {
			return a.Fail(err)
		}
		if err := json.Unmarshal(b, &preds); err != nil {
			return a.Fail(fmt.Errorf("--predicates must be a JSON array of {type,value,exclude,join}: %w", err))
		}
	} else {
		preds = buildPredicates(tags, exTags, *path, *description, *hash, *similar, *visual, *mode)
	}
	if len(preds) == 0 {
		return a.Usage(fs, "no predicates — pass at least one filter flag or --predicates")
	}

	var out any
	if err := a.Client.DoJSON("POST", "/api/media/query", map[string]any{"predicates": preds, "mode": *mode}, &out); err != nil {
		return a.Fail(err)
	}
	return a.PrintJSON(out)
}

// nullSafeAt lets --predicates accept both bare paths and @path / - like --body.
func nullSafeAt(v string) string {
	if v == "-" || strings.HasPrefix(v, "@") {
		return v
	}
	return "@" + v
}

func cmdMediaSearch(a *App, args []string) int {
	if len(args) != 1 {
		return a.Usage(nil, "usage: lokictl media search <text>")
	}
	var out any
	body := map[string]any{"description": args[0], "tags": []string{}, "filteringMode": ""}
	if err := a.Client.DoJSON("POST", "/api/media/search", body, &out); err != nil {
		return a.Fail(err)
	}
	return a.PrintJSON(out)
}

func limitFlagCommand(usage, urlPath, queryKey string) func(a *App, args []string) int {
	return func(a *App, args []string) int {
		fs := flag.NewFlagSet(usage, flag.ContinueOnError)
		fs.SetOutput(io.Discard)
		limit := fs.Int("limit", 50, "max results")
		if len(args) == 0 || strings.HasPrefix(args[0], "-") {
			return a.Usage(fs, "usage: lokictl "+usage)
		}
		value := args[0]
		if err := fs.Parse(args[1:]); err != nil {
			return a.Usage(fs, err.Error())
		}
		q := url.Values{queryKey: {value}, "limit": {strconv.Itoa(*limit)}}
		var out any
		if err := a.Client.DoJSON("GET", urlPath+"?"+q.Encode(), nil, &out); err != nil {
			return a.Fail(err)
		}
		return a.PrintJSON(out)
	}
}

func cmdMediaSimilar(a *App, args []string) int {
	return limitFlagCommand("media similar <path> [--limit N]", "/api/media/similar", "path")(a, args)
}

func cmdMediaVisual(a *App, args []string) int {
	return limitFlagCommand("media visual <text> [--limit N]", "/api/media/search/visual", "q")(a, args)
}

func mediaPathCommand(urlPath string) func(a *App, args []string) int {
	return func(a *App, args []string) int {
		if len(args) != 1 {
			return a.Usage(nil, fmt.Sprintf("usage: lokictl media %s <path>", strings.TrimPrefix(urlPath, "/api/media/")))
		}
		var out any
		if err := a.Client.DoJSON("POST", urlPath, map[string]string{"path": args[0]}, &out); err != nil {
			return a.Fail(err)
		}
		return a.PrintJSON(out)
	}
}

func cmdMediaDescribe(a *App, args []string) int {
	fs := flag.NewFlagSet("media describe", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	text := fs.String("text", "", "the description to set (required)")
	if len(args) == 0 || strings.HasPrefix(args[0], "-") {
		return a.Usage(fs, "usage: lokictl media describe <path> --text D")
	}
	path := args[0]
	if err := fs.Parse(args[1:]); err != nil {
		return a.Usage(fs, err.Error())
	}
	if *text == "" {
		return a.Usage(fs, "--text is required")
	}
	if err := a.Client.DoJSON("POST", "/api/media/description", map[string]string{"path": path, "description": *text}, nil); err != nil {
		return a.Fail(err)
	}
	return a.PrintJSON(map[string]string{"status": "ok", "path": path})
}

func cmdMediaDelete(a *App, args []string) int {
	var path string
	for _, arg := range args {
		if arg != "--yes" && arg != "-y" {
			path = arg
		}
	}
	if path == "" {
		return a.Usage(nil, "usage: lokictl media delete <path> --yes")
	}
	if !hasYesFlag(args) {
		return a.Usage(nil, fmt.Sprintf("this permanently deletes %q from the library — re-run with --yes to confirm", path))
	}
	if err := a.Client.DoJSON("POST", "/api/media/delete", map[string]string{"path": path}, nil); err != nil {
		return a.Fail(err)
	}
	return a.PrintJSON(map[string]string{"status": "deleted", "path": path})
}
