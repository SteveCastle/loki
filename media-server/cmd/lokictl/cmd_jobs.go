package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"strings"
	"time"
)

// jobRow mirrors jobqueue.Job's JSON shape.
type jobRow struct {
	ID            string   `json:"id"`
	Command       string   `json:"command"`
	Arguments     []string `json:"arguments"`
	Input         string   `json:"input"`
	OriginalInput string   `json:"original_input"`
	Host          string   `json:"host"`
	Dependencies  []string `json:"dependencies"`
	State         string   `json:"state"`
	CreatedAt     any      `json:"created_at"`
	ClaimedAt     any      `json:"claimed_at"`
	CompletedAt   any      `json:"completed_at"`
	ErroredAt     any      `json:"errored_at"`
	OutputFiles   []string `json:"output_files"`
	SourceFiles   []string `json:"source_files"`
	WorkflowID    string   `json:"workflow_id"`
}

func isTerminalState(s string) bool {
	return s == "completed" || s == "cancelled" || s == "error"
}

// pollInterval is a variable so tests can shrink it.
var pollInterval = 1 * time.Second

const pollMaxInterval = 3 * time.Second

func fetchJobs(a *App) ([]jobRow, error) {
	var jobs []jobRow
	if err := a.Client.DoJSON("GET", "/jobs/list", nil, &jobs); err != nil {
		return nil, err
	}
	return jobs, nil
}

func fetchJob(a *App, id string) (*jobRow, error) {
	jobs, err := fetchJobs(a)
	if err != nil {
		return nil, err
	}
	for i := range jobs {
		if jobs[i].ID == id {
			return &jobs[i], nil
		}
	}
	return nil, fmt.Errorf("job %q not found", id)
}

// waitForJob polls until the job reaches a terminal state. Returns the final
// job and an exit code: 0 completed, 3 error/cancelled/timed out, 1 API error.
func waitForJob(a *App, id string, timeout time.Duration) (*jobRow, int) {
	deadline := time.Time{}
	if timeout > 0 {
		deadline = time.Now().Add(timeout)
	}
	interval := pollInterval
	for {
		job, err := fetchJob(a, id)
		if err != nil {
			a.Fail(err)
			return nil, 1
		}
		if isTerminalState(job.State) {
			if job.State == "completed" {
				return job, 0
			}
			return job, 3
		}
		if !deadline.IsZero() && time.Now().After(deadline) {
			fmt.Fprintf(a.ErrOut, `{"error":"timed out waiting for job %s (state: %s)"}`+"\n", id, job.State)
			return job, 3
		}
		time.Sleep(interval)
		if interval < pollMaxInterval {
			interval += pollInterval
		}
	}
}

// buildJobInput assembles the /create input string. ParseCommand on the
// server splits on spaces honoring double quotes, with no escape syntax —
// so tokens containing spaces are quoted and embedded quotes are rejected.
func buildJobInput(task string, tokens []string) (string, error) {
	parts := []string{task}
	for _, tok := range tokens {
		if strings.Contains(tok, `"`) {
			return "", fmt.Errorf("token %q contains a double quote, which the server cannot parse — pass it via --field name=value instead", tok)
		}
		if strings.ContainsAny(tok, " \t") {
			tok = `"` + tok + `"`
		}
		parts = append(parts, tok)
	}
	return strings.Join(parts, " "), nil
}

// fieldFlags collects repeated --field k=v flags.
type fieldFlags map[string]string

func (f fieldFlags) String() string { return "" }
func (f fieldFlags) Set(v string) error {
	k, val, ok := strings.Cut(v, "=")
	if !ok || k == "" {
		return fmt.Errorf("--field wants name=value, got %q", v)
	}
	f[k] = val
	return nil
}

func init() {
	register(command{group: "job", name: "run",
		args:    "<task> [task args...] [--field k=v]... [--wait] [--follow] [--timeout D]",
		summary: "Create a job (POST /create); --wait polls until done, --follow streams stdout",
		run:     cmdJobRun})
	register(command{group: "job", name: "list", args: "[--state S]",
		summary: "List jobs (GET /jobs/list)", run: cmdJobList})
	register(command{group: "job", name: "get", args: "<id>",
		summary: "Show one job", run: cmdJobGet})
	register(command{group: "job", name: "wait", args: "<id> [--timeout D]",
		summary: "Block until a job reaches a terminal state", run: cmdJobWait})
	register(command{group: "job", name: "logs", args: "<id>",
		summary: "Stream a running job's stdout via SSE (live jobs only; output is not stored)",
		run:     cmdJobLogs})
	register(command{group: "job", name: "cancel", args: "<id>",
		summary: "Cancel a job (POST /job/{id}/cancel)", run: jobAction("cancel")})
	register(command{group: "job", name: "copy", args: "<id>",
		summary: "Clone a job into a new pending job (POST /job/{id}/copy)", run: cmdJobCopy})
	register(command{group: "job", name: "remove", args: "<id>",
		summary: "Remove a job from the queue (POST /job/{id}/remove)", run: jobAction("remove")})
	register(command{group: "job", name: "clear", args: "--yes",
		summary: "Clear all non-running jobs (POST /jobs/clear)", run: cmdJobClear})
}

// splitControlFlags separates lokictl's own flags from task tokens: any of
// --wait/--follow/--field/--timeout is consumed wherever it appears; all
// other tokens (including task option flags like --type) pass through in
// order to the job input string.
func splitControlFlags(args []string) (tokens []string, fields fieldFlags, wait, follow bool, timeout time.Duration, err error) {
	fields = fieldFlags{}
	i := 0
	for i < len(args) {
		arg := args[i]
		name, val, hasEq := strings.Cut(arg, "=")
		takeVal := func() (string, error) {
			if hasEq {
				return val, nil
			}
			if i+1 < len(args) {
				i++
				return args[i], nil
			}
			return "", fmt.Errorf("%s requires a value", name)
		}
		switch name {
		case "--wait":
			wait = true
		case "--follow":
			follow = true
		case "--field":
			v, verr := takeVal()
			if verr != nil {
				err = verr
				return
			}
			if serr := fields.Set(v); serr != nil {
				err = serr
				return
			}
		case "--timeout":
			v, verr := takeVal()
			if verr != nil {
				err = verr
				return
			}
			d, derr := time.ParseDuration(v)
			if derr != nil {
				err = fmt.Errorf("invalid --timeout: %v", derr)
				return
			}
			timeout = d
		default:
			tokens = append(tokens, arg)
		}
		i++
	}
	return
}

func cmdJobRun(a *App, args []string) int {
	if len(args) == 0 {
		return a.Usage(nil, "usage: lokictl job run <task> [task args...] [--field k=v]... [--wait] [--follow] [--timeout D]")
	}
	task := args[0]
	tokens, fields, wait, follow, timeout, err := splitControlFlags(args[1:])
	if err != nil {
		return a.Usage(nil, err.Error())
	}
	input, err := buildJobInput(task, tokens)
	if err != nil {
		return a.Usage(nil, err.Error())
	}

	body := map[string]any{"input": input}
	if len(fields) > 0 {
		body["fields"] = fields
	}
	var created struct {
		ID string `json:"id"`
	}
	if err := a.Client.DoJSON("POST", "/create", body, &created); err != nil {
		return a.Fail(err)
	}
	if created.ID == "" {
		return a.Fail(fmt.Errorf("server did not return a job id"))
	}
	if !wait && !follow {
		if a.Quiet {
			fmt.Fprintln(a.Out, created.ID)
			return 0
		}
		return a.PrintJSON(map[string]string{"id": created.ID, "state": "pending"})
	}

	var stopFollow context.CancelFunc
	if follow {
		ctx, cancel := context.WithCancel(context.Background())
		stopFollow = cancel
		go followJobStdout(ctx, a, created.ID)
	}
	job, code := waitForJob(a, created.ID, timeout)
	if stopFollow != nil {
		stopFollow()
	}
	if job != nil {
		if pc := a.PrintJSON(job); code == 0 && pc != 0 {
			return pc
		}
	}
	return code
}

// followJobStdout attaches to /stream and relays stdout-<id> events to
// stderr (stdout is reserved for the final JSON result).
func followJobStdout(ctx context.Context, a *App, id string) {
	resp, err := a.Client.DoStream("GET", "/stream")
	if err != nil {
		fmt.Fprintf(a.ErrOut, `{"warning":"could not attach to job output stream: %s"}`+"\n", err)
		return
	}
	defer resp.Body.Close()
	go func() {
		<-ctx.Done()
		resp.Body.Close()
	}()
	want := "stdout-" + id
	_ = readSSE(ctx, resp.Body, func(ev sseEvent) bool {
		if ev.Name == want && ev.Data != "" {
			fmt.Fprintln(a.ErrOut, ev.Data)
		}
		return true
	})
}

func cmdJobList(a *App, args []string) int {
	fs := flag.NewFlagSet("job list", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	state := fs.String("state", "", "filter by state (pending|inProgress|completed|cancelled|error)")
	if err := fs.Parse(args); err != nil {
		return a.Usage(fs, err.Error())
	}
	jobs, err := fetchJobs(a)
	if err != nil {
		return a.Fail(err)
	}
	if *state != "" {
		filtered := jobs[:0]
		for _, j := range jobs {
			if j.State == *state {
				filtered = append(filtered, j)
			}
		}
		jobs = filtered
	}
	if jobs == nil {
		jobs = []jobRow{}
	}
	return a.PrintJSON(jobs)
}

func cmdJobGet(a *App, args []string) int {
	if len(args) != 1 {
		return a.Usage(nil, "usage: lokictl job get <id>")
	}
	job, err := fetchJob(a, args[0])
	if err != nil {
		return a.Fail(err)
	}
	return a.PrintJSON(job)
}

func cmdJobWait(a *App, args []string) int {
	fs := flag.NewFlagSet("job wait", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	timeout := fs.Duration("timeout", 0, "give up after this long (0 = wait forever)")
	if err := fs.Parse(args); err != nil {
		return a.Usage(fs, err.Error())
	}
	if fs.NArg() != 1 {
		return a.Usage(fs, "usage: lokictl job wait <id> [--timeout D]")
	}
	job, code := waitForJob(a, fs.Arg(0), *timeout)
	if job != nil {
		if pc := a.PrintJSON(job); code == 0 && pc != 0 {
			return pc
		}
	}
	return code
}

func cmdJobLogs(a *App, args []string) int {
	if len(args) != 1 {
		return a.Usage(nil, "usage: lokictl job logs <id>")
	}
	id := args[0]
	// Surface a clear error if the job is already done (its stdout is gone).
	if job, err := fetchJob(a, id); err == nil && isTerminalState(job.State) {
		return a.Fail(fmt.Errorf("job %s is %s — stdout is only streamed live and is not stored", id, job.State))
	}
	resp, err := a.Client.DoStream("GET", "/stream")
	if err != nil {
		return a.Fail(err)
	}
	defer resp.Body.Close()
	want := "stdout-" + id
	err = readSSE(context.Background(), resp.Body, func(ev sseEvent) bool {
		if ev.Name == want && ev.Data != "" {
			fmt.Fprintln(a.Out, ev.Data)
		}
		return true
	})
	if err != nil {
		return a.Fail(err)
	}
	return 0
}

func jobAction(action string) func(a *App, args []string) int {
	return func(a *App, args []string) int {
		if len(args) != 1 {
			return a.Usage(nil, fmt.Sprintf("usage: lokictl job %s <id>", action))
		}
		if err := a.Client.DoJSON("POST", "/job/"+args[0]+"/"+action, nil, nil); err != nil {
			return a.Fail(err)
		}
		return a.PrintJSON(map[string]string{"status": "ok", "action": action, "id": args[0]})
	}
}

func cmdJobCopy(a *App, args []string) int {
	if len(args) != 1 {
		return a.Usage(nil, "usage: lokictl job copy <id>")
	}
	var out any
	if err := a.Client.DoJSON("POST", "/job/"+args[0]+"/copy", nil, &out); err != nil {
		return a.Fail(err)
	}
	return a.PrintJSON(out)
}

func cmdJobClear(a *App, args []string) int {
	if !hasYesFlag(args) {
		return a.Usage(nil, "job clear removes ALL non-running jobs — re-run with --yes to confirm")
	}
	var out any
	if err := a.Client.DoJSON("POST", "/jobs/clear", nil, &out); err != nil {
		return a.Fail(err)
	}
	return a.PrintJSON(out)
}

func hasYesFlag(args []string) bool {
	for _, a := range args {
		if a == "--yes" || a == "-y" {
			return true
		}
	}
	return false
}
