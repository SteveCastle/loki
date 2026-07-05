// lokictl is a command-line client for the Lowkey Media Server HTTP API,
// designed for AI agents first: deterministic JSON on stdout, JSON errors on
// stderr, stable exit codes (0 ok, 1 server/network error, 2 usage error,
// 3 awaited job failed/cancelled/timed out).
package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/stevecastle/shrike/appconfig"
	"github.com/stevecastle/shrike/platform"
)

// App carries everything a command needs: the HTTP client and the output
// contract. Commands never touch os.Stdout/os.Stderr directly so tests can
// capture output.
type App struct {
	Client *Client
	Out    io.Writer
	ErrOut io.Writer
	Quiet  bool
	Table  bool
}

// command is one entry in the dispatch table. Two-level commands set both
// group and name ("job run"); single-word commands leave name empty
// ("health"). Each cmd_*.go file registers its commands in init().
type command struct {
	group   string
	name    string
	args    string
	summary string
	run     func(a *App, args []string) int
}

var commands []command

func register(c command) { commands = append(commands, c) }

func main() {
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr))
}

// defaultServer is the fallback base URL when neither the --server flag,
// LOKICTL_SERVER, nor the CLI config file specify one. The port is discovered
// from the local media-server's own configuration so the CLI keeps working
// after the server port is changed: LOWKEY_PORT env > the server's
// config.json "port" > the compiled-in default (10111, "L0K1").
func defaultServer() string {
	return fmt.Sprintf("http://localhost:%d", localServerPort())
}

func localServerPort() int {
	if v := os.Getenv("LOWKEY_PORT"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 && n <= 65535 {
			return n
		}
	}
	if b, err := os.ReadFile(filepath.Join(platform.GetDataDir(), "config.json")); err == nil {
		var c struct {
			Port int `json:"port"`
		}
		if json.Unmarshal(b, &c) == nil && c.Port > 0 && c.Port <= 65535 {
			return c.Port
		}
	}
	return appconfig.DefaultPort
}

func run(argv []string, stdout, stderr io.Writer) int {
	var (
		serverFlag string
		tokenFlag  string
		outMode    = "json"
		timeout    = 30 * time.Second
		quiet      bool
	)

	// Global flags are only recognized before the first positional token;
	// everything after belongs to the subcommand.
	rest := []string{}
	i := 0
	for i < len(argv) {
		arg := argv[i]
		if !strings.HasPrefix(arg, "-") {
			rest = append(rest, argv[i:]...)
			break
		}
		name, val, hasEq := strings.Cut(arg, "=")
		next := func() (string, bool) {
			if hasEq {
				return val, true
			}
			if i+1 < len(argv) {
				i++
				return argv[i], true
			}
			return "", false
		}
		switch name {
		case "--server":
			v, ok := next()
			if !ok {
				fmt.Fprintln(stderr, `{"error":"--server requires a value"}`)
				return 2
			}
			serverFlag = v
		case "--token":
			v, ok := next()
			if !ok {
				fmt.Fprintln(stderr, `{"error":"--token requires a value"}`)
				return 2
			}
			tokenFlag = v
		case "-o", "--output":
			v, ok := next()
			if !ok || (v != "json" && v != "table") {
				fmt.Fprintln(stderr, `{"error":"-o must be json or table"}`)
				return 2
			}
			outMode = v
		case "--timeout":
			v, ok := next()
			if !ok {
				fmt.Fprintln(stderr, `{"error":"--timeout requires a duration (e.g. 60s)"}`)
				return 2
			}
			d, err := time.ParseDuration(v)
			if err != nil {
				fmt.Fprintf(stderr, `{"error":"invalid --timeout: %s"}`+"\n", err)
				return 2
			}
			timeout = d
		case "-q", "--quiet":
			quiet = true
		case "-h", "--help":
			printHelp(stdout)
			return 0
		default:
			fmt.Fprintf(stderr, `{"error":"unknown global flag %q","hint":"lokictl help"}`+"\n", arg)
			return 2
		}
		i++
	}

	if len(rest) == 0 || rest[0] == "help" {
		printHelp(stdout)
		return 0
	}

	fileCfg := loadCLIConfig()
	server := resolve(serverFlag, "LOKICTL_SERVER", fileCfg.Server, defaultServer())
	token := resolve(tokenFlag, "LOKICTL_TOKEN", fileCfg.Token, "")

	app := &App{
		Client: NewClient(server, token, timeout),
		Out:    stdout,
		ErrOut: stderr,
		Quiet:  quiet,
		Table:  outMode == "table",
	}

	group := rest[0]
	// Single-word command?
	for _, c := range commands {
		if c.group == group && c.name == "" {
			return c.run(app, rest[1:])
		}
	}
	// Two-level command.
	if len(rest) >= 2 {
		for _, c := range commands {
			if c.group == group && c.name == rest[1] {
				return c.run(app, rest[2:])
			}
		}
	}
	// Unknown — if the group exists, list its subcommands.
	var subs []string
	for _, c := range commands {
		if c.group == group && c.name != "" {
			subs = append(subs, c.name)
		}
	}
	if len(subs) > 0 {
		sort.Strings(subs)
		fmt.Fprintf(stderr, `{"error":"unknown %s command","hint":"available: %s"}`+"\n", group, strings.Join(subs, ", "))
		return 2
	}
	fmt.Fprintf(stderr, `{"error":"unknown command %q","hint":"lokictl help"}`+"\n", group)
	return 2
}

func printHelp(w io.Writer) {
	fmt.Fprint(w, `lokictl — CLI for the Lowkey Media Server

Usage: lokictl [global flags] <command> [args]

Global flags (before the command):
  --server URL    server base URL (env LOKICTL_SERVER, config file; default `+defaultServer()+`,
                  port auto-detected from the local server config / LOWKEY_PORT)
  --token TOKEN   bearer token (env LOKICTL_TOKEN, config file; set by "lokictl login")
  -o json|table   output format (default json)
  --timeout DUR   HTTP timeout (default 30s; streaming commands ignore it)
  -q, --quiet     minimal output
  -h, --help      this help

Exit codes: 0 ok · 1 server/network error · 2 usage error · 3 awaited job failed/cancelled/timed out.
Errors are JSON on stderr: {"error":..., "status":..., "hint":...}

Commands:
`)
	sorted := make([]command, len(commands))
	copy(sorted, commands)
	sort.Slice(sorted, func(i, j int) bool {
		if sorted[i].group != sorted[j].group {
			return sorted[i].group < sorted[j].group
		}
		return sorted[i].name < sorted[j].name
	})
	for _, c := range sorted {
		full := c.group
		if c.name != "" {
			full += " " + c.name
		}
		if c.args != "" {
			full += " " + c.args
		}
		fmt.Fprintf(w, "  lokictl %-72s %s\n", full, c.summary)
	}
}
