package main

import (
	"flag"
	"fmt"
	"io"
	"strings"
	"time"
)

// depItem mirrors deps/status.Item.
type depItem struct {
	ID        string `json:"id"`
	Category  string `json:"category"`
	Name      string `json:"name"`
	State     string `json:"state"`
	Version   string `json:"version,omitempty"`
	SizeBytes int64  `json:"size_bytes,omitempty"`
	Path      string `json:"path,omitempty"`
	Error     string `json:"error,omitempty"`
	Detail    string `json:"detail,omitempty"`
}

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

	var started any
	if err := a.Client.DoJSON("POST", "/api/deps/models/"+id+"/download", nil, &started); err != nil {
		return a.Fail(err)
	}
	if !*wait {
		return a.PrintJSON(started)
	}

	deadline := time.Now().Add(*timeout)
	lastState := ""
	for {
		items, err := fetchDeps(a)
		if err != nil {
			return a.Fail(err)
		}
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
		if item.State != lastState {
			lastState = item.State
			if !a.Quiet {
				fmt.Fprintf(a.ErrOut, `{"progress":"%s","model":"%s"}`+"\n", item.State, id)
			}
		}
		switch {
		case item.State == "installed":
			return a.PrintJSON(item)
		case item.Error != "" || strings.Contains(item.State, "error") || strings.Contains(item.State, "failed"):
			a.PrintJSON(item)
			return 1
		}
		if time.Now().After(deadline) {
			fmt.Fprintf(a.ErrOut, `{"error":"timed out waiting for model %s (state: %s)"}`+"\n", id, item.State)
			return 3
		}
		time.Sleep(2 * time.Second)
	}
}
