package main

import (
	"fmt"
	"strings"
)

func init() {
	register(command{group: "media", name: "thumbnail-cleanup",
		args:    "[--dir D] [--dry-run] [--db PATH]... [--no-discover] [--manifest FILE] [--detach] [--timeout D]",
		summary: `Delete cached thumbnails no library sharing the cache still names (job "thumbnail-cleanup"); streams the task log, --dir scopes to a folder`,
		run:     cmdMediaThumbnailCleanup})
}

// cmdMediaThumbnailCleanup is sugar over "job run thumbnail-cleanup". The
// task's own log is what you want to read (which files, how much space,
// what it skipped, which libraries need what), so by default the command
// follows the job and streams its stdout, then prints the final job record
// — whose "stdout" array holds the whole log — as JSON. --detach just
// queues the job and prints its id.
//
// Every SQLite database beside the configured one is treated as a library
// sharing the cache (that is how both apps place the cache); --db adds
// libraries elsewhere and --no-discover trusts only the configured one plus
// --db. --manifest writes every cache file with the libraries naming it.
//
//	lokictl media thumbnail-cleanup --dry-run                          # whole cache, report only
//	lokictl media thumbnail-cleanup --dry-run --manifest C:/tmp/thumbs.tsv
//	lokictl media thumbnail-cleanup --dir "D:/photos/2019" --dry-run   # bounded test
//	lokictl media thumbnail-cleanup --dir "D:/photos/2019"             # really delete, that folder only
//	lokictl media thumbnail-cleanup --db "E:/other/dream.sqlite"       # a library outside the folder
func cmdMediaThumbnailCleanup(a *App, args []string) int {
	usage := "usage: lokictl media thumbnail-cleanup [--dir D] [--dry-run] [--db PATH]... [--no-discover] [--manifest FILE] [--detach] [--timeout D]"
	var (
		dir        string
		dryRun     bool
		detach     bool
		noDiscover bool
		manifest   string
		dbs        []string
		passthr    []string
	)
	takeValue := func(i *int, val string, hasEq bool) (string, bool) {
		if hasEq {
			return val, true
		}
		if *i+1 < len(args) {
			*i++
			return args[*i], true
		}
		return "", false
	}
	for i := 0; i < len(args); i++ {
		name, val, hasEq := strings.Cut(args[i], "=")
		switch name {
		case "--dir":
			v, ok := takeValue(&i, val, hasEq)
			if !ok {
				return a.Usage(nil, "--dir requires a directory")
			}
			dir = v
		case "--db":
			v, ok := takeValue(&i, val, hasEq)
			if !ok || strings.TrimSpace(v) == "" {
				return a.Usage(nil, "--db requires a database path")
			}
			dbs = append(dbs, v)
		case "--manifest":
			v, ok := takeValue(&i, val, hasEq)
			if !ok || strings.TrimSpace(v) == "" {
				return a.Usage(nil, "--manifest requires a file path")
			}
			manifest = v
		case "--no-discover":
			noDiscover = true
		case "--dry-run":
			dryRun = true
		case "--detach", "--no-wait":
			detach = true
		case "--timeout", "--wait", "--follow":
			// job run's own control flags; forward as-is (with a value
			// where one follows).
			passthr = append(passthr, args[i])
			if name == "--timeout" && !hasEq && i+1 < len(args) {
				i++
				passthr = append(passthr, args[i])
			}
		default:
			if strings.HasPrefix(args[i], "-") {
				return a.Usage(nil, fmt.Sprintf("unknown flag %s\n%s", args[i], usage))
			}
			// A bare path is the scope directory, as the task itself accepts.
			if dir != "" {
				return a.Usage(nil, usage)
			}
			dir = args[i]
		}
	}
	for _, v := range append([]string{dir, manifest}, dbs...) {
		if strings.Contains(v, `"`) {
			return a.Usage(nil, "paths cannot contain a double quote (the server's job parser has no escape syntax)")
		}
	}
	for _, v := range dbs {
		if strings.Contains(v, ";") {
			return a.Usage(nil, "a --db path cannot contain ';' (it separates paths on the server side)")
		}
	}

	jobArgs := []string{"thumbnail-cleanup"}
	if dryRun {
		jobArgs = append(jobArgs, "--dry-run")
	}
	if noDiscover {
		jobArgs = append(jobArgs, "--discover=false")
	}
	if len(dbs) > 0 {
		jobArgs = append(jobArgs, "--db", strings.Join(dbs, ";"))
	}
	if manifest != "" {
		jobArgs = append(jobArgs, "--manifest", manifest)
	}
	if !detach {
		jobArgs = append(jobArgs, "--follow")
	}
	jobArgs = append(jobArgs, passthr...)
	if dir != "" {
		// Last on purpose: the server treats the final token of a job's
		// input as the job's Input and everything before it as arguments,
		// and the task reads both, so the path is safe in either slot — but
		// a path with spaces is quoted by job run and must not be followed
		// by tokens the quoting could swallow.
		jobArgs = append(jobArgs, "--dir", dir)
	}
	return cmdJobRun(a, jobArgs)
}
