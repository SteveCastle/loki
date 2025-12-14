package tasks

import (
	"bufio"
	"fmt"
	"path/filepath"
	"strconv"
	"strings"
	"sync"

	"github.com/stevecastle/shrike/embedexec"
	"github.com/stevecastle/shrike/jobqueue"
)

// ffmpegTask runs ffmpeg per selected file with placeholder expansion.
func ffmpegTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	ctx := j.Ctx

	var files []string
	if qstr, ok := extractQueryFromJob(j); ok {
		q.PushJobStdout(j.ID, fmt.Sprintf("ffmpeg: using query to select files: %s", qstr))
		mediaPaths, err := getMediaPathsByQuery(q.Db, qstr)
		if err != nil {
			q.PushJobStdout(j.ID, "ffmpeg: failed to load paths from query: "+err.Error())
			q.ErrorJob(j.ID)
			return err
		}
		files = mediaPaths
	} else {
		raw := strings.TrimSpace(j.Input)
		if raw == "" {
			q.PushJobStdout(j.ID, "ffmpeg: no input paths or query provided")
			q.CompleteJob(j.ID)
			return nil
		}
		files = parseInputPaths(raw)
	}

	if len(files) == 0 {
		q.PushJobStdout(j.ID, "ffmpeg: no files to process")
		q.CompleteJob(j.ID)
		return nil
	}

	templateArgs := append([]string{}, j.Arguments...)
	if len(templateArgs) == 0 {
		q.PushJobStdout(j.ID, "ffmpeg: no arguments provided for ffmpeg")
		q.CompleteJob(j.ID)
		return nil
	}

	for idx, src := range files {
		select {
		case <-ctx.Done():
			q.PushJobStdout(j.ID, "ffmpeg: task canceled")
			q.ErrorJob(j.ID)
			return ctx.Err()
		default:
		}

		abs := src
		if a, err := filepath.Abs(src); err == nil {
			abs = filepath.FromSlash(a)
		}
		dir := filepath.Dir(abs)
		base := filepath.Base(abs)
		ext := filepath.Ext(abs)
		name := strings.TrimSuffix(base, ext)
		idxStr := strconv.Itoa(idx + 1)

		expanded := make([]string, len(templateArgs))
		for i, ta := range templateArgs {
			s := ta
			s = strings.ReplaceAll(s, "{input}", abs)
			s = strings.ReplaceAll(s, "{dir}", dir)
			s = strings.ReplaceAll(s, "{base}", base)
			s = strings.ReplaceAll(s, "{name}", name)
			s = strings.ReplaceAll(s, "{ext}", ext)
			s = strings.ReplaceAll(s, "{idx}", idxStr)
			expanded[i] = s
		}

		hasInput := false
		for _, e := range expanded {
			if e == "-i" || strings.Contains(e, "-i=") || strings.Contains(e, "-i:") {
				hasInput = true
				break
			}
		}
		finalArgs := expanded
		if !hasInput {
			finalArgs = append([]string{"-i", abs}, expanded...)
		}

		q.PushJobStdout(j.ID, "ffmpeg: running on "+base)

		cmd, cleanup, err := embedexec.GetExec(ctx, "ffmpeg", finalArgs...)
		if err != nil {
			q.PushJobStdout(j.ID, "ffmpeg: failed to prepare: "+err.Error())
			q.ErrorJob(j.ID)
			return err
		}
		if cleanup != nil {
			defer cleanup()
		}

		stdout, err := cmd.StdoutPipe()
		if err != nil {
			q.PushJobStdout(j.ID, "ffmpeg: stdout pipe error: "+err.Error())
			q.ErrorJob(j.ID)
			return err
		}
		stderr, err := cmd.StderrPipe()
		if err != nil {
			q.PushJobStdout(j.ID, "ffmpeg: stderr pipe error: "+err.Error())
			q.ErrorJob(j.ID)
			return err
		}

		doneErr := make(chan struct{})
		go func() {
			s := bufio.NewScanner(stderr)
			for s.Scan() {
				_ = q.PushJobStdout(j.ID, "ffmpeg: "+s.Text())
			}
			close(doneErr)
		}()

		if err := cmd.Start(); err != nil {
			q.PushJobStdout(j.ID, "ffmpeg: failed to start: "+err.Error())
			q.ErrorJob(j.ID)
			return err
		}

		scan := bufio.NewScanner(stdout)
		for scan.Scan() {
			_ = q.PushJobStdout(j.ID, scan.Text())
		}
		_ = cmd.Wait()
		<-doneErr

		q.PushJobStdout(j.ID, "ffmpeg: completed for "+base)
	}

	q.CompleteJob(j.ID)
	return nil
}
