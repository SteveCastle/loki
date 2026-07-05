package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

// readDAG loads a workflow DAG from a file (or stdin with "-"), accepting a
// bare JSON array or an object wrapping it as {"dag":[...]} / {"tasks":[...]}.
func readDAG(pathOrDash string, stdin io.Reader) ([]map[string]any, error) {
	var b []byte
	var err error
	if pathOrDash == "-" {
		b, err = io.ReadAll(stdin)
	} else {
		b, err = os.ReadFile(pathOrDash)
	}
	if err != nil {
		return nil, err
	}
	var arr []map[string]any
	if json.Unmarshal(b, &arr) == nil {
		return arr, nil
	}
	var wrapped struct {
		DAG   []map[string]any `json:"dag"`
		Tasks []map[string]any `json:"tasks"`
	}
	if err := json.Unmarshal(b, &wrapped); err != nil {
		return nil, fmt.Errorf("DAG must be a JSON array of tasks (or {\"dag\":[...]}): %w", err)
	}
	if len(wrapped.DAG) > 0 {
		return wrapped.DAG, nil
	}
	if len(wrapped.Tasks) > 0 {
		return wrapped.Tasks, nil
	}
	return nil, fmt.Errorf("no tasks found in DAG input")
}

// oldServerHint upgrades 404s on /workflows* with a version hint.
func oldServerHint(err error) error {
	var apiErr *APIError
	if errors.As(err, &apiErr) && apiErr.Status == http.StatusNotFound {
		apiErr.Hint = "saved-workflow routes need a current server build — rebuild media-server (npm run build:server)"
	}
	return err
}

// waitForJobs waits on several job ids; exit 0 only if all completed.
func waitForJobs(a *App, ids []string, timeout time.Duration) int {
	worst := 0
	results := make([]any, 0, len(ids))
	for _, id := range ids {
		job, code := waitForJob(a, id, timeout)
		if code > worst {
			worst = code
		}
		if job != nil {
			results = append(results, job)
		}
	}
	if pc := a.PrintJSON(results); worst == 0 && pc != 0 {
		return pc
	}
	return worst
}

func init() {
	register(command{group: "workflow", name: "list",
		summary: "List saved workflows (GET /workflows)",
		run: func(a *App, args []string) int {
			var out any
			if err := a.Client.DoJSON("GET", "/workflows", nil, &out); err != nil {
				return a.Fail(oldServerHint(err))
			}
			return a.PrintJSON(out)
		}})
	register(command{group: "workflow", name: "get", args: "<id>",
		summary: "Show a saved workflow (GET /workflows/{id})",
		run: func(a *App, args []string) int {
			if len(args) != 1 {
				return a.Usage(nil, "usage: lokictl workflow get <id>")
			}
			var out any
			if err := a.Client.DoJSON("GET", "/workflows/"+args[0], nil, &out); err != nil {
				return a.Fail(oldServerHint(err))
			}
			return a.PrintJSON(out)
		}})
	register(command{group: "workflow", name: "create", args: "--name N --dag FILE|-",
		summary: "Save a workflow (POST /workflows/create)", run: cmdWorkflowCreate})
	register(command{group: "workflow", name: "update", args: "<id> [--name N] [--dag FILE|-]",
		summary: "Update a saved workflow (PUT /workflows/{id})", run: cmdWorkflowUpdate})
	register(command{group: "workflow", name: "delete", args: "<id> --yes",
		summary: "Delete a saved workflow (DELETE /workflows/{id})", run: cmdWorkflowDelete})
	register(command{group: "workflow", name: "run", args: "<id> [--input S] [--wait] [--timeout D]",
		summary: "Run a saved workflow (POST /workflows/{id}/run)", run: cmdWorkflowRun})
	register(command{group: "workflow", name: "run-adhoc", args: "--dag FILE|- [--wait] [--timeout D]",
		summary: "Run a one-off DAG without saving it (POST /workflow)", run: cmdWorkflowRunAdhoc})
}

func cmdWorkflowCreate(a *App, args []string) int {
	fs := flag.NewFlagSet("workflow create", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	name := fs.String("name", "", "workflow name (required)")
	dagPath := fs.String("dag", "", "DAG JSON file, or - for stdin (required)")
	if err := fs.Parse(args); err != nil {
		return a.Usage(fs, err.Error())
	}
	if *name == "" || *dagPath == "" {
		return a.Usage(fs, "usage: lokictl workflow create --name N --dag FILE|-")
	}
	dag, err := readDAG(*dagPath, os.Stdin)
	if err != nil {
		return a.Fail(err)
	}
	var out any
	if err := a.Client.DoJSON("POST", "/workflows/create", map[string]any{"name": *name, "dag": dag}, &out); err != nil {
		return a.Fail(oldServerHint(err))
	}
	return a.PrintJSON(out)
}

func cmdWorkflowUpdate(a *App, args []string) int {
	fs := flag.NewFlagSet("workflow update", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	name := fs.String("name", "", "new name (keeps current if omitted)")
	dagPath := fs.String("dag", "", "new DAG JSON file, or - for stdin (keeps current if omitted)")
	if len(args) == 0 || strings.HasPrefix(args[0], "-") {
		return a.Usage(fs, "usage: lokictl workflow update <id> [--name N] [--dag FILE|-]")
	}
	id := args[0]
	if err := fs.Parse(args[1:]); err != nil {
		return a.Usage(fs, err.Error())
	}

	// Read-modify-write: fetch the current workflow, overlay the changes.
	var current struct {
		ID   string           `json:"id"`
		Name string           `json:"name"`
		DAG  []map[string]any `json:"dag"`
	}
	if err := a.Client.DoJSON("GET", "/workflows/"+id, nil, &current); err != nil {
		return a.Fail(oldServerHint(err))
	}
	if *name != "" {
		current.Name = *name
	}
	if *dagPath != "" {
		dag, err := readDAG(*dagPath, os.Stdin)
		if err != nil {
			return a.Fail(err)
		}
		current.DAG = dag
	}
	var out any
	if err := a.Client.DoJSON("PUT", "/workflows/"+id, map[string]any{"name": current.Name, "dag": current.DAG}, &out); err != nil {
		return a.Fail(oldServerHint(err))
	}
	return a.PrintJSON(out)
}

func cmdWorkflowDelete(a *App, args []string) int {
	var id string
	for _, arg := range args {
		if arg != "--yes" && arg != "-y" {
			id = arg
		}
	}
	if id == "" {
		return a.Usage(nil, "usage: lokictl workflow delete <id> --yes")
	}
	if !hasYesFlag(args) {
		return a.Usage(nil, fmt.Sprintf("this permanently deletes workflow %s — re-run with --yes to confirm", id))
	}
	if err := a.Client.DoJSON("DELETE", "/workflows/"+id, nil, nil); err != nil {
		return a.Fail(oldServerHint(err))
	}
	return a.PrintJSON(map[string]string{"status": "deleted", "id": id})
}

func cmdWorkflowRun(a *App, args []string) int {
	fs := flag.NewFlagSet("workflow run", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	input := fs.String("input", "", "input injected into the workflow's root tasks")
	wait := fs.Bool("wait", false, "wait for all spawned jobs to finish")
	timeout := fs.Duration("timeout", 0, "per-job wait timeout (0 = forever)")
	if len(args) == 0 || strings.HasPrefix(args[0], "-") {
		return a.Usage(fs, "usage: lokictl workflow run <id> [--input S] [--wait] [--timeout D]")
	}
	id := args[0]
	if err := fs.Parse(args[1:]); err != nil {
		return a.Usage(fs, err.Error())
	}
	var out struct {
		IDs []string `json:"ids"`
	}
	if err := a.Client.DoJSON("POST", "/workflows/"+id+"/run", map[string]string{"input": *input}, &out); err != nil {
		return a.Fail(oldServerHint(err))
	}
	if !*wait {
		return a.PrintJSON(out)
	}
	return waitForJobs(a, out.IDs, *timeout)
}

func cmdWorkflowRunAdhoc(a *App, args []string) int {
	fs := flag.NewFlagSet("workflow run-adhoc", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	dagPath := fs.String("dag", "", "DAG JSON file, or - for stdin (required)")
	wait := fs.Bool("wait", false, "wait for all spawned jobs to finish")
	timeout := fs.Duration("timeout", 0, "per-job wait timeout (0 = forever)")
	if err := fs.Parse(args); err != nil {
		return a.Usage(fs, err.Error())
	}
	if *dagPath == "" {
		return a.Usage(fs, "usage: lokictl workflow run-adhoc --dag FILE|- [--wait] [--timeout D]")
	}
	dag, err := readDAG(*dagPath, os.Stdin)
	if err != nil {
		return a.Fail(err)
	}
	var out struct {
		IDs []string `json:"ids"`
	}
	if err := a.Client.DoJSON("POST", "/workflow", map[string]any{"tasks": dag}, &out); err != nil {
		return a.Fail(err)
	}
	if !*wait {
		return a.PrintJSON(out)
	}
	return waitForJobs(a, out.IDs, *timeout)
}
