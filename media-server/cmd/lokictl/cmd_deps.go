package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"strings"
	"time"
)

// depItem mirrors deps/status.Item. Detail is whatever the server puts there
// (an install's progress, a binary's install hints, a user-supplied copy's
// source...), so it is kept as raw JSON: typing it broke every status read the
// day a dependency's detail became an object.
type depItem struct {
	ID          string          `json:"id"`
	Category    string          `json:"category"`
	Name        string          `json:"name"`
	Feature     string          `json:"feature,omitempty"`
	Description string          `json:"description,omitempty"`
	State       string          `json:"state"`
	Version     string          `json:"version,omitempty"`
	SizeBytes   int64           `json:"size_bytes,omitempty"`
	Path        string          `json:"path,omitempty"`
	Error       string          `json:"error,omitempty"`
	Detail      json.RawMessage `json:"detail,omitempty"`
	Source      string          `json:"source,omitempty"`
}

// progress is an install's bytes (Detail of an item with an install in
// flight: deps/models.Install), when it has them.
func (d depItem) progress() (done, total int64) {
	var p struct {
		BytesDone  int64 `json:"bytes_done"`
		BytesTotal int64 `json:"bytes_total"`
	}
	if len(d.Detail) > 0 && json.Unmarshal(d.Detail, &p) == nil {
		return p.BytesDone, p.BytesTotal
	}
	return 0, 0
}

// Waiting on a download (deps download --wait). Variables so tests run fast.
var (
	// depsPoll is how often the status is read.
	depsPoll = 2 * time.Second
	// depsStatusGrace is how long reading the status may keep failing (the
	// server restarting mid-download) before the wait gives up.
	depsStatusGrace = 2 * time.Minute
	// depsRestartAfter: a model that reads "missing" for this long while we
	// wait lost its install (the server restarted; its tracker is in memory),
	// so the download is asked for again. Downloads resume where they stopped.
	depsRestartAfter = 10 * time.Second
)

func fetchDeps(a *App) ([]depItem, error) {
	var items []depItem
	if err := a.Client.DoJSON("GET", "/api/deps/status", nil, &items); err != nil {
		return nil, err
	}
	return items, nil
}

func init() {
	register(command{group: "deps", name: "status",
		summary: "Binary and model dependency status (GET /api/deps/status)",
		run: func(a *App, args []string) int {
			items, err := fetchDeps(a)
			if err != nil {
				return a.Fail(err)
			}
			return a.PrintJSON(items)
		}})
	register(command{group: "deps", name: "download", args: "<model-id> [--wait] [--timeout D]",
		summary: "Download a model (POST /api/deps/models/{id}/download); --wait polls until installed",
		run:     cmdDepsDownload})
	register(command{group: "deps", name: "verify", args: "<model-id>",
		summary: "Verify a model's files (POST /api/deps/models/{id}/verify)",
		run: func(a *App, args []string) int {
			if len(args) != 1 {
				return a.Usage(nil, "usage: lokictl deps verify <model-id>")
			}
			var out any
			if err := a.Client.DoJSON("POST", "/api/deps/models/"+args[0]+"/verify", nil, &out); err != nil {
				return a.Fail(err)
			}
			return a.PrintJSON(out)
		}})
	register(command{group: "deps", name: "delete", args: "<model-id> --yes",
		summary: "Delete a downloaded model (DELETE /api/deps/models/{id})",
		run: func(a *App, args []string) int {
			var id string
			for _, arg := range args {
				if arg != "--yes" && arg != "-y" {
					id = arg
				}
			}
			if id == "" {
				return a.Usage(nil, "usage: lokictl deps delete <model-id> --yes")
			}
			if !hasYesFlag(args) {
				return a.Usage(nil, fmt.Sprintf("this deletes model %q from disk — re-run with --yes to confirm", id))
			}
			if err := a.Client.DoJSON("DELETE", "/api/deps/models/"+id, nil, nil); err != nil {
				return a.Fail(err)
			}
			return a.PrintJSON(map[string]string{"status": "deleted", "id": id})
		}})
}

func cmdDepsDownload(a *App, args []string) int {
	fs := flag.NewFlagSet("deps download", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	wait := fs.Bool("wait", false, "poll /api/deps/status until the model is installed")
	timeout := fs.Duration("timeout", 30*time.Minute, "give up after this long (with --wait)")
	if len(args) == 0 || strings.HasPrefix(args[0], "-") {
		return a.Usage(fs, "usage: lokictl deps download <model-id> [--wait] [--timeout D]")
	}
	id := args[0]
	if err := fs.Parse(args[1:]); err != nil {
		return a.Usage(fs, err.Error())
	}

	start := func() (any, error) {
		var started any
		err := a.Client.DoJSON("POST", "/api/deps/models/"+id+"/download", nil, &started)
		return started, err
	}
	started, err := start()
	if err != nil {
		return a.Fail(err)
	}
	if !*wait {
		return a.PrintJSON(started)
	}

	deadline := time.Now().Add(*timeout)
	lastState, lastPct := "", -1
	var failingSince, missingSince time.Time
	for {
		if time.Now().After(deadline) {
			fmt.Fprintf(a.ErrOut, `{"error":"timed out waiting for model %s (state: %s)"}`+"\n", id, lastState)
			return 3
		}
		items, err := fetchDeps(a)
		if err != nil {
			// The server may be restarting: keep trying for a while.
			if failingSince.IsZero() {
				failingSince = time.Now()
			}
			if time.Since(failingSince) > depsStatusGrace {
				return a.Fail(fmt.Errorf("reading the dependency status kept failing for %s: %w", depsStatusGrace, err))
			}
			time.Sleep(depsPoll)
			continue
		}
		failingSince = time.Time{}
		var item *depItem
		for i := range items {
			if items[i].ID == id {
				item = &items[i]
				break
			}
		}
		if item == nil {
			return a.Fail(fmt.Errorf("model %q not in /api/deps/status — see: lokictl deps status", id))
		}
		done, total := item.progress()
		pct := -1
		if total > 0 {
			pct = int(done * 100 / total)
		}
		if !a.Quiet && (item.State != lastState || pct/10 != lastPct/10) {
			if total > 0 {
				fmt.Fprintf(a.ErrOut, `{"progress":"%s","model":"%s","bytes_done":%d,"bytes_total":%d}`+"\n", item.State, id, done, total)
			} else {
				fmt.Fprintf(a.ErrOut, `{"progress":"%s","model":"%s"}`+"\n", item.State, id)
			}
		}
		lastState, lastPct = item.State, pct
		switch {
		case item.State == "installed":
			return a.PrintJSON(item)
		case item.Error != "" || item.State == "failed" || item.State == "cancelled" || strings.Contains(item.State, "error"):
			a.PrintJSON(item)
			return 1
		case item.State == "missing":
			// Nothing is installing it: the server lost the install (restart).
			if missingSince.IsZero() {
				missingSince = time.Now()
			} else if time.Since(missingSince) >= depsRestartAfter {
				if _, err := start(); err != nil {
					return a.Fail(fmt.Errorf("restarting the download: %w", err))
				}
				missingSince = time.Time{}
			}
		default:
			missingSince = time.Time{}
		}
		time.Sleep(depsPoll)
	}
}
