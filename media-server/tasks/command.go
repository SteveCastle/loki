package tasks

import (
	"bufio"
	"fmt"
	"io"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"

	"github.com/stevecastle/shrike/embedexec"
	"github.com/stevecastle/shrike/jobqueue"
)

func normalizeArg(a string) string {
	// Heuristic for path detection
	if strings.ContainsAny(a, "/\\") {
		return normalizePath(a)
	}
	if strings.HasPrefix(a, ".") || strings.HasPrefix(a, "~") {
		return normalizePath(a)
	}
	if len(a) >= 2 && a[1] == ':' && ((a[0] >= 'A' && a[0] <= 'Z') || (a[0] >= 'a' && a[0] <= 'z')) {
		return normalizePath(a)
	}
	if strings.Contains(a, ".") && !strings.Contains(a, "=") && !strings.HasPrefix(a, "-") {
		ext := filepath.Ext(a)
		if len(ext) >= 2 && len(ext) <= 5 {
			base := strings.TrimSuffix(a, ext)
			if base != "" && !strings.Contains(base, ".") {
				return normalizePath(a)
			}
		}
	}
	return a
}

func normalizePath(a string) string {
	c := filepath.Clean(a)
	if abs, err := filepath.Abs(c); err == nil {
		c = abs
	}
	return filepath.FromSlash(c)
}

func executeCommand(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	ctx := j.Ctx
	fmt.Println("Executing job:", j.ID, j.Command, j.Arguments, j.Input)

	args := make([]string, 0, len(j.Arguments)+1)
	for _, a := range j.Arguments {
		args = append(args, normalizeArg(a))
	}
	if j.Input != "" {
		args = append(args, j.Input)
	}

	cmd, cleanup, err := embedexec.GetExec(ctx, j.Command, args...)
	if err != nil {
		_ = q.PushJobStdout(j.ID, fmt.Sprintf("Error starting job: %s", err))
		_ = q.ErrorJob(j.ID)
		return fmt.Errorf("start %q: %w", j.Command, err)
	}
	if cleanup != nil {
		defer cleanup()
	}

	go func() {
		<-ctx.Done()
		if cmd.Process != nil {
			if runtime.GOOS == "windows" {
				_ = exec.Command("taskkill", "/F", "/T", "/PID", fmt.Sprintf("%d", cmd.Process.Pid)).Run()
			} else {
				_ = cmd.Process.Kill()
			}
		}
	}()

	if j.StdIn != nil {
		cmd.Stdin = j.StdIn
	}

	stdoutPipe, err := cmd.StdoutPipe()
	if err != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("Error getting stdout pipe: %s", err))
		_ = q.ErrorJob(j.ID)
		return fmt.Errorf("stdout pipe: %w", err)
	}
	stderrPipe, err := cmd.StderrPipe()
	if err != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("Error getting stderr pipe: %s", err))
		_ = q.ErrorJob(j.ID)
		return fmt.Errorf("stderr pipe: %w", err)
	}

	if err := cmd.Start(); err != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("Error starting command: %s", err))
		_ = q.ErrorJob(j.ID)
		return fmt.Errorf("start: %w", err)
	}

	doneReading := make(chan struct{})
	totalReaders := 2
	doneCount := 0

	scanAndPush := func(pipe io.ReadCloser) {
		s := bufio.NewScanner(pipe)
		for s.Scan() {
			_ = q.PushJobStdout(j.ID, s.Text())
		}
		if err := s.Err(); err != nil && err != io.EOF {
			_ = q.PushJobStdout(j.ID, fmt.Sprintf("Error reading pipe: %s", err))
			_ = q.ErrorJob(j.ID)
		}
		mu.Lock()
		doneCount++
		if doneCount == totalReaders {
			close(doneReading)
		}
		mu.Unlock()
	}

	go scanAndPush(stdoutPipe)
	go scanAndPush(stderrPipe)

	err = cmd.Wait()
	<-doneReading

	select {
	case <-ctx.Done():
		q.PushJobStdout(j.ID, "Task was canceled")
		_ = q.CancelJob(j.ID)
		return ctx.Err()
	default:
	}

	if err != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("Error waiting for command: %s", err))
		_ = q.ErrorJob(j.ID)
		return err
	}

	_ = q.CompleteJob(j.ID)
	return nil
}
