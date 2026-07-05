package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"sort"
	"strings"
)

// PrintJSON writes v pretty-printed to stdout and returns exit code 0. In
// table mode, list-shaped values render as TSV instead (falling back to JSON
// for anything that isn't an array of objects).
func (a *App) PrintJSON(v any) int {
	if a.Table && printTable(a.Out, v) {
		return 0
	}
	enc := json.NewEncoder(a.Out)
	enc.SetIndent("", "  ")
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		fmt.Fprintf(a.ErrOut, `{"error":"failed to encode output: %s"}`+"\n", err)
		return 1
	}
	return 0
}

// Fail writes a single JSON error object to stderr and returns exit code 1.
func (a *App) Fail(err error) int {
	out := map[string]any{"error": err.Error()}
	var apiErr *APIError
	if errors.As(err, &apiErr) {
		out["status"] = apiErr.Status
		if apiErr.Hint != "" {
			out["hint"] = apiErr.Hint
		}
		var detail any
		if json.Unmarshal([]byte(apiErr.Body), &detail) == nil {
			out["detail"] = detail
		}
	}
	enc := json.NewEncoder(a.ErrOut)
	enc.SetEscapeHTML(false)
	_ = enc.Encode(out)
	return 1
}

// Usage reports a usage error (exit 2): message to stderr as JSON plus the
// flag set's defaults when available.
func (a *App) Usage(fs *flag.FlagSet, msg string) int {
	enc := json.NewEncoder(a.ErrOut)
	enc.SetEscapeHTML(false)
	_ = enc.Encode(map[string]string{"error": msg})
	if fs != nil {
		var b strings.Builder
		fs.SetOutput(&b)
		fs.PrintDefaults()
		if b.Len() > 0 {
			fmt.Fprint(a.ErrOut, b.String())
		}
	}
	return 2
}

// printTable renders an array of JSON objects as TSV: header = union of keys
// (first row's keys in document order, then any extras sorted). Returns false
// when v is not an array of objects so the caller falls back to JSON.
func printTable(w io.Writer, v any) bool {
	raw, err := json.Marshal(v)
	if err != nil {
		return false
	}
	var rows []map[string]any
	if err := json.Unmarshal(raw, &rows); err != nil || len(rows) == 0 {
		return false
	}
	cols := firstRowKeyOrder(raw)
	seen := map[string]bool{}
	for _, c := range cols {
		seen[c] = true
	}
	var extras []string
	for _, row := range rows {
		for k := range row {
			if !seen[k] {
				seen[k] = true
				extras = append(extras, k)
			}
		}
	}
	sort.Strings(extras)
	cols = append(cols, extras...)
	if len(cols) == 0 {
		return false
	}

	fmt.Fprintln(w, strings.Join(cols, "\t"))
	for _, row := range rows {
		cells := make([]string, len(cols))
		for i, c := range cols {
			cells[i] = renderCell(row[c])
		}
		fmt.Fprintln(w, strings.Join(cells, "\t"))
	}
	return true
}

// firstRowKeyOrder token-scans the first object of a JSON array to recover
// its key order (maps lose it).
func firstRowKeyOrder(raw []byte) []string {
	dec := json.NewDecoder(bytes.NewReader(raw))
	tok, err := dec.Token() // [
	if err != nil || tok != json.Delim('[') {
		return nil
	}
	tok, err = dec.Token() // {
	if err != nil || tok != json.Delim('{') {
		return nil
	}
	var keys []string
	for dec.More() {
		kTok, err := dec.Token()
		if err != nil {
			return keys
		}
		k, ok := kTok.(string)
		if !ok {
			return keys
		}
		keys = append(keys, k)
		// Skip the value (possibly nested).
		var skip json.RawMessage
		if err := dec.Decode(&skip); err != nil {
			return keys
		}
	}
	return keys
}

func renderCell(v any) string {
	switch t := v.(type) {
	case nil:
		return ""
	case string:
		return strings.ReplaceAll(strings.ReplaceAll(t, "\t", " "), "\n", " ")
	case float64, bool:
		return fmt.Sprint(t)
	default:
		b, err := json.Marshal(t)
		if err != nil {
			return fmt.Sprint(t)
		}
		return string(b)
	}
}
