// Package querylog appends a JSONL record per executed media query so that
// post-hoc analysis can identify slow queries. The log file lives next to the
// other server data (platform.GetDataDir()/query-log.jsonl). Writes are
// serialized through a mutex; failures are silent — logging must never break
// a query.
package querylog

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/stevecastle/shrike/platform"
)

var (
	mu       sync.Mutex
	file     *os.File
	once     sync.Once
	logPath  string
	disabled bool
)

func ensureFile() *os.File {
	once.Do(func() {
		dir := platform.GetDataDir()
		if err := os.MkdirAll(dir, 0o755); err != nil {
			disabled = true
			return
		}
		logPath = filepath.Join(dir, "query-log.jsonl")
		f, err := os.OpenFile(logPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
		if err != nil {
			disabled = true
			return
		}
		file = f
	})
	return file
}

// Entry is one log record. Source defaults to "go-server" when empty.
type Entry struct {
	Ts         string        `json:"ts"`
	Source     string        `json:"source"`
	Name       string        `json:"name,omitempty"`
	SQL        string        `json:"sql"`
	Params     []any `json:"params,omitempty"`
	DurationMs float64       `json:"duration_ms"`
	Rows       *int          `json:"rows,omitempty"`
	Error      string        `json:"error,omitempty"`
}

// Log writes a single entry. Caller is responsible for setting fields except
// Ts and Source (which are filled here when empty).
func Log(e Entry) {
	if disabled {
		return
	}
	f := ensureFile()
	if f == nil {
		return
	}
	if e.Ts == "" {
		e.Ts = time.Now().UTC().Format(time.RFC3339Nano)
	}
	if e.Source == "" {
		e.Source = "go-server"
	}
	e.SQL = collapse(e.SQL)
	b, err := json.Marshal(e)
	if err != nil {
		return
	}
	mu.Lock()
	defer mu.Unlock()
	_, _ = file.Write(b)
	_, _ = file.Write([]byte("\n"))
}

// Start returns a stop function that callers invoke once the query (and any
// row iteration) is complete. The closure captures the start time so the
// caller doesn't need to track it manually:
//
//	stop := querylog.Start("GetItems", sql, args)
//	rows, err := db.Query(sql, args...)
//	... iterate ...
//	stop(rowCount, err)
//
// rowCount may be -1 if the caller doesn't know it.
func Start(name, sql string, params []any) func(rowCount int, err error) {
	start := time.Now()
	return func(rowCount int, err error) {
		elapsed := float64(time.Since(start).Microseconds()) / 1000.0
		e := Entry{
			Name:       name,
			SQL:        sql,
			Params:     params,
			DurationMs: elapsed,
		}
		if rowCount >= 0 {
			r := rowCount
			e.Rows = &r
		}
		if err != nil {
			e.Error = err.Error()
		}
		Log(e)
	}
}

func collapse(s string) string {
	return strings.Join(strings.Fields(s), " ")
}

// Path returns the active log file path (or "" if logging is disabled).
func Path() string {
	ensureFile()
	if disabled {
		return ""
	}
	return logPath
}
