package main

import (
	"fmt"
	"strconv"
	"strings"
)

func init() {
	register(command{group: "media", name: "cleanup",
		args:    "[--dir D] [--dry-run] [--max-missing-percent N] [--skip-orphans] [--detach] [--timeout D]",
		summary: `Forget media whose files no longer exist, with every tag/embedding/face/battle row naming them (job "cleanup"); streams the task log`,
		run:     cmdMediaCleanup})
}

// cmdMediaCleanup is sugar over "job run cleanup". Like thumbnail-cleanup it
// follows the job by default so the phase-by-phase log is visible, then
// prints the final job record (its "stdout" array is the whole log).
//
//	lokictl media cleanup --dry-run                              # whole library, report only
//	lokictl media cleanup --dir "D:/photos/2019" --dry-run          # bounded test
//	lokictl media cleanup                                        # really forget missing items
//	lokictl media cleanup --max-missing-percent 100              # a volume is gone for good: purge it
func cmdMediaCleanup(a *App, args []string) int {
	usage := "usage: lokictl media cleanup [--dir D] [--dry-run] [--max-missing-percent N] [--skip-orphans] [--detach] [--timeout D]"
	var (
		dir         string
		dryRun      bool
		detach      bool
		skipOrphans bool
		maxPct      string
		passthr     []string
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
		case "--max-missing-percent":
			v, ok := takeValue(&i, val, hasEq)
			if !ok {
				return a.Usage(nil, "--max-missing-percent requires a number")
			}
			n, err := strconv.ParseFloat(strings.TrimSpace(v), 64)
			if err != nil || n < 0 || n > 100 {
				return a.Usage(nil, "--max-missing-percent must be a number from 0 to 100")
			}
			maxPct = strconv.FormatFloat(n, 'f', -1, 64)
		case "--skip-orphans":
			skipOrphans = true
		case "--dry-run":
			dryRun = true
		case "--detach", "--no-wait":
			detach = true
		case "--timeout", "--wait", "--follow":
			passthr = append(passthr, args[i])
			if name == "--timeout" && !hasEq && i+1 < len(args) {
				i++
				passthr = append(passthr, args[i])
			}
		default:
			if strings.HasPrefix(args[i], "-") {
				return a.Usage(nil, fmt.Sprintf("unknown flag %s\n%s", args[i], usage))
			}
			if dir != "" {
				return a.Usage(nil, usage)
			}
			dir = args[i]
		}
	}
	if strings.Contains(dir, `"`) {
		return a.Usage(nil, "the scope directory cannot contain a double quote (the server's job parser has no escape syntax)")
	}

	jobArgs := []string{"cleanup"}
	if dryRun {
		jobArgs = append(jobArgs, "--dry-run")
	}
	if skipOrphans {
		jobArgs = append(jobArgs, "--skip-orphans")
	}
	if maxPct != "" {
		jobArgs = append(jobArgs, "--max-missing-percent", maxPct)
	}
	if !detach {
		jobArgs = append(jobArgs, "--follow")
	}
	jobArgs = append(jobArgs, passthr...)
	if dir != "" {
		// Last: the server takes the final input token as the job Input and
		// the task reads both slots, so a quoted path is safe here.
		jobArgs = append(jobArgs, "--dir", dir)
	}
	return cmdJobRun(a, jobArgs)
}
