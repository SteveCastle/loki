package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"mime/multipart"
	"os"
	"path/filepath"
)

func init() {
	register(command{group: "config", name: "get",
		summary: "Active server config, secrets redacted (GET /api/config)",
		run: func(a *App, args []string) int {
			var out any
			if err := a.Client.DoJSON("GET", "/api/config", nil, &out); err != nil {
				return a.Fail(err)
			}
			return a.PrintJSON(out)
		}})
	register(command{group: "config", name: "set", args: "--json '{...}'|@file|-",
		summary: "Merge fields into the server config (GET /api/config + POST /config)",
		run:     cmdConfigSet})

	register(command{group: "fs", name: "list", args: "[path]",
		summary: "Browse storage roots (POST /api/fs/list; empty path lists roots)",
		run: func(a *App, args []string) int {
			path := ""
			if len(args) == 1 {
				path = args[0]
			} else if len(args) > 1 {
				return a.Usage(nil, "usage: lokictl fs list [path]")
			}
			var out any
			if err := a.Client.DoJSON("POST", "/api/fs/list", map[string]string{"path": path}, &out); err != nil {
				return a.Fail(err)
			}
			return a.PrintJSON(out)
		}})
	register(command{group: "fs", name: "scan", args: "<path> [--recursive]",
		summary: "Scan a directory within a storage root (POST /api/fs/scan)",
		run:     cmdFSScan})

	register(command{group: "upload", args: "<file>... [--dest DIR] [--no-ingest]",
		summary: "Upload files (multipart POST /api/upload)",
		run:     cmdUpload})
}

func cmdConfigSet(a *App, args []string) int {
	fs := flag.NewFlagSet("config set", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	jsonArg := fs.String("json", "", "fields to merge: literal JSON, @file, or - for stdin (required)")
	if err := fs.Parse(args); err != nil {
		return a.Usage(fs, err.Error())
	}
	if *jsonArg == "" {
		return a.Usage(fs, `usage: lokictl config set --json '{"ollamaModel":"llava"}'`)
	}
	b, err := readBodyArg(*jsonArg, os.Stdin)
	if err != nil {
		return a.Fail(err)
	}
	var updates map[string]any
	if err := json.Unmarshal(b, &updates); err != nil {
		return a.Fail(fmt.Errorf("--json must be a JSON object of config fields: %w", err))
	}

	// Read-modify-write: POST /config requires dbPath and treats missing
	// fields as "keep current" only for some — send current + overrides.
	var current map[string]any
	if err := a.Client.DoJSON("GET", "/api/config", nil, &current); err != nil {
		return a.Fail(err)
	}
	// Never echo redaction placeholders back into the config.
	for k, v := range current {
		if s, ok := v.(string); ok && s == "<redacted>" {
			delete(current, k)
		}
	}
	delete(current, "roots") // roots carry redacted keys; update them via the web UI
	for k, v := range updates {
		current[k] = v
	}
	if err := a.Client.DoJSON("POST", "/config", current, nil); err != nil {
		return a.Fail(err)
	}
	return a.PrintJSON(map[string]any{"status": "ok", "updated": updates})
}

func cmdFSScan(a *App, args []string) int {
	fs := flag.NewFlagSet("fs scan", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	recursive := fs.Bool("recursive", false, "recurse into subdirectories")
	if len(args) == 0 || args[0] == "" || args[0][0] == '-' {
		return a.Usage(fs, "usage: lokictl fs scan <path> [--recursive]")
	}
	path := args[0]
	if err := fs.Parse(args[1:]); err != nil {
		return a.Usage(fs, err.Error())
	}
	var out any
	if err := a.Client.DoJSON("POST", "/api/fs/scan", map[string]any{"path": path, "recursive": *recursive}, &out); err != nil {
		return a.Fail(err)
	}
	return a.PrintJSON(out)
}

func cmdUpload(a *App, args []string) int {
	fs := flag.NewFlagSet("upload", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	dest := fs.String("dest", "", "destination directory (must be inside a storage root)")
	noIngest := fs.Bool("no-ingest", false, "skip the automatic ingest job")
	var files []string
	for len(args) > 0 && len(args[0]) > 0 && args[0][0] != '-' {
		files = append(files, args[0])
		args = args[1:]
	}
	if err := fs.Parse(args); err != nil {
		return a.Usage(fs, err.Error())
	}
	if len(files) == 0 {
		return a.Usage(fs, "usage: lokictl upload <file>... [--dest DIR] [--no-ingest]")
	}

	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)
	for _, f := range files {
		part, err := mw.CreateFormFile("files", filepath.Base(f))
		if err != nil {
			return a.Fail(err)
		}
		src, err := os.Open(f)
		if err != nil {
			return a.Fail(err)
		}
		_, err = io.Copy(part, src)
		src.Close()
		if err != nil {
			return a.Fail(err)
		}
	}
	if *dest != "" {
		_ = mw.WriteField("destination", *dest)
	}
	if *noIngest {
		_ = mw.WriteField("autoIngest", "false")
	}
	if err := mw.Close(); err != nil {
		return a.Fail(err)
	}

	resp, err := a.Client.DoRaw("POST", "/api/upload", mw.FormDataContentType(), &buf)
	if err != nil {
		return a.Fail(err)
	}
	defer resp.Body.Close()
	var out any
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return a.Fail(fmt.Errorf("invalid upload response: %w", err))
	}
	return a.PrintJSON(out)
}
