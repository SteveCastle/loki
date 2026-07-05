package main

import (
	"flag"
	"io"
	"strings"
)

// dbQueryBody mirrors the server's dbquery_api.go request shape.
type dbQueryBody struct {
	SQL       string `json:"sql"`
	Args      []any  `json:"args,omitempty"`
	Limit     int    `json:"limit,omitempty"`
	TimeoutMS int    `json:"timeout_ms,omitempty"`
}

func init() {
	register(command{group: "db", name: "query",
		args:    `"SQL" [--arg V]... [--limit N] [--timeout-ms N]`,
		summary: "Read-only SQL over the library DB (POST /api/db/query; single SELECT/WITH, ? bind args)",
		run:     cmdDBQuery})
	register(command{group: "db", name: "tables",
		summary: "List tables and views (sugar over db query)",
		run: func(a *App, args []string) int {
			return runDBQuery(a, dbQueryBody{
				SQL: "SELECT name, type FROM sqlite_master WHERE type IN ('table','view') ORDER BY name",
			})
		}})
	register(command{group: "db", name: "schema", args: "[table]",
		summary: "Show CREATE statements (sugar over db query)",
		run: func(a *App, args []string) int {
			body := dbQueryBody{SQL: "SELECT name, sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY name"}
			if len(args) == 1 {
				body.SQL = "SELECT name, sql FROM sqlite_master WHERE sql IS NOT NULL AND name = ?"
				body.Args = []any{args[0]}
			} else if len(args) > 1 {
				return a.Usage(nil, "usage: lokictl db schema [table]")
			}
			return runDBQuery(a, body)
		}})
}

func runDBQuery(a *App, body dbQueryBody) int {
	var out any
	if err := a.Client.DoJSON("POST", "/api/db/query", body, &out); err != nil {
		return a.Fail(err)
	}
	return a.PrintJSON(out)
}

func cmdDBQuery(a *App, args []string) int {
	fs := flag.NewFlagSet("db query", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	var bindArgs stringList
	fs.Var(&bindArgs, "arg", "positional bind value for ? placeholders (repeatable; sent as string, SQLite affinity coerces)")
	limit := fs.Int("limit", 0, "row cap (server default 1000, max 10000)")
	timeoutMS := fs.Int("timeout-ms", 0, "server-side query timeout (default 5000, max 30000)")
	if len(args) == 0 || strings.HasPrefix(args[0], "-") {
		return a.Usage(fs, `usage: lokictl db query "SELECT ..." [--arg V]... [--limit N] [--timeout-ms N]`)
	}
	sqlText := args[0]
	if err := fs.Parse(args[1:]); err != nil {
		return a.Usage(fs, err.Error())
	}
	body := dbQueryBody{SQL: sqlText, Limit: *limit, TimeoutMS: *timeoutMS}
	for _, v := range bindArgs {
		body.Args = append(body.Args, v)
	}
	return runDBQuery(a, body)
}
