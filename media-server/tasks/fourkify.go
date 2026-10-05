package tasks

import (
	"bufio"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"

	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/platform"
)

// --- 4kify ---
//
// Wraps the external `4kify` binary (Qwen Image 2.1 restore + outpaint onto a
// 4K canvas). Shaped like the ffmpeg media-altering tasks (rotate, crop): one
// new `<name>_4k.png` per image beside the original (or in the workflow temp
// dir when chained), registered as an output file so downstream steps and the
// library pick it up. Images only; the binary is resolved from PATH.

var fourKifyOptions = []TaskOption{
	{Name: "phone", Label: "Phone Wallpaper", Type: "bool", Default: false, Description: "Make a vertical 1296x2800 phone wallpaper instead of a 3840x2160 desktop one"},
	{Name: "steps", Label: "Steps", Type: "number", Default: 25.0, Description: "Sampling steps (more is slower)"},
}

func fourKifyTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	ctx := j.Ctx
	opts := ParseOptions(j, fourKifyOptions)
	phone, _ := opts["phone"].(bool)
	steps, _ := opts["steps"].(float64)
	if steps <= 0 {
		steps = 25
	}

	bin, err := exec.LookPath("4kify")
	if err != nil {
		q.PushJobStdout(j.ID, "4kify: binary not found on PATH")
		q.ErrorJob(j.ID)
		return fmt.Errorf("4kify not found on PATH: %w", err)
	}

	res, rerr := resolveJobItemsRaw(j, q)
	if rerr != nil {
		q.PushJobStdout(j.ID, "4kify: failed to resolve input: "+rerr.Error())
		q.ErrorJob(j.ID)
		return rerr
	}
	if res.FromQuery {
		q.PushJobStdout(j.ID, fmt.Sprintf("4kify: query: %s", res.Query))
	}
	files := res.Paths

	if len(files) == 0 {
		q.PushJobStdout(j.ID, "4kify: no files to process")
		q.CompleteJob(j.ID)
		return nil
	}
	_ = q.SetJobProgress(j.ID, 0, len(files))

	// Every job is a workflow, so "in a workflow" can't mean "a save step
	// follows". When nothing downstream consumes this job's output (a plain
	// context-menu run, or the last step of a chain), write the finished file
	// beside the original and, like the save step, announce it so the library
	// reloads and the cursor jumps to the first new file. The announcement
	// fires once at the end (also on a partial run) so a batch doesn't hop the
	// cursor per image. With a later step, output stays in .loki-temp for it.
	terminal := !q.HasDependents(j.ID)
	var created []string
	defer func() {
		if terminal {
			broadcastMediaCreated(created)
		}
	}()

	for idx, src := range files {
		select {
		case <-ctx.Done():
			q.PushJobStdout(j.ID, "4kify: task canceled")
			q.ErrorJob(j.ID)
			return ctx.Err()
		default:
		}
		if q.PauseRequested(j.ID) {
			q.PushJobStdout(j.ID, fmt.Sprintf("4kify: paused at %d/%d - resume to continue", idx, len(files)))
			return jobqueue.ErrPaused
		}

		abs := src
		if a, err := filepath.Abs(src); err == nil {
			abs = filepath.FromSlash(a)
		}
		dir := filepath.Dir(abs)
		base := filepath.Base(abs)
		ext := filepath.Ext(abs)
		name := strings.TrimSuffix(base, ext)

		if !isImageExt(ext) {
			q.PushJobStdout(j.ID, "4kify: skipping non-image "+base)
			_ = q.SetJobProgress(j.ID, idx+1, len(files))
			continue
		}

		// Intermediate steps write to a temp dir; strip any existing .loki-temp
		// so chained jobs stay at the same depth instead of nesting.
		outputDir := dir
		if j.WorkflowID != "" && !terminal {
			originalDir := stripLokiTemp(abs)
			outputDir = filepath.Join(originalDir, ".loki-temp", j.ID)
			if err := os.MkdirAll(outputDir, 0755); err != nil {
				q.PushJobStdout(j.ID, "4kify: failed to create temp dir: "+err.Error())
				q.ErrorJob(j.ID)
				return err
			}
		}
		output := filepath.Join(outputDir, name+"_4k.png")
		if terminal {
			if _, err := os.Stat(output); err == nil {
				output = resolveConflict(output)
			}
		}

		args := []string{"--steps", fmt.Sprintf("%d", int(steps)), "-o", output}
		if phone {
			args = append(args, "--phone")
		}
		args = append(args, abs)

		q.PushJobStdout(j.ID, "4kify: running on "+base+" -> "+filepath.Base(output))

		cmd := exec.CommandContext(ctx, bin, args...)
		platform.HideSubprocessWindow(cmd)

		stdout, err := cmd.StdoutPipe()
		if err != nil {
			q.PushJobStdout(j.ID, "4kify: stdout pipe error: "+err.Error())
			q.ErrorJob(j.ID)
			return err
		}
		stderr, err := cmd.StderrPipe()
		if err != nil {
			q.PushJobStdout(j.ID, "4kify: stderr pipe error: "+err.Error())
			q.ErrorJob(j.ID)
			return err
		}

		doneErr := make(chan struct{})
		go func() {
			s := bufio.NewScanner(stderr)
			for s.Scan() {
				_ = q.PushJobStdout(j.ID, "4kify: "+s.Text())
			}
			close(doneErr)
		}()

		if err := cmd.Start(); err != nil {
			q.PushJobStdout(j.ID, "4kify: failed to start: "+err.Error())
			q.ErrorJob(j.ID)
			return err
		}

		scan := bufio.NewScanner(stdout)
		for scan.Scan() {
			_ = q.PushJobStdout(j.ID, scan.Text())
		}
		waitErr := cmd.Wait()
		<-doneErr

		if ctx.Err() != nil {
			q.PushJobStdout(j.ID, "4kify: task canceled")
			q.ErrorJob(j.ID)
			return ctx.Err()
		}
		if waitErr != nil {
			q.PushJobStdout(j.ID, "4kify: failed for "+base+": "+waitErr.Error())
			q.ErrorJob(j.ID)
			return waitErr
		}
		info, statErr := os.Stat(output)
		if statErr != nil {
			q.PushJobStdout(j.ID, "4kify: no output produced for "+base)
			q.ErrorJob(j.ID)
			return fmt.Errorf("4kify produced no output for %s", base)
		}
		// A finished file the chain ends on belongs in the library right away:
		// without a media row it is invisible to path/tag queries until
		// someone re-ingests the folder. Intermediate (.loki-temp) output is
		// never added — the step that places it owns that.
		if terminal {
			if err := insertMediaRecord(q.Db, output, info.Size()); err != nil {
				q.PushJobStdout(j.ID, "4kify: failed to add to library: "+err.Error())
			}
		}

		q.PushJobStdout(j.ID, "4kify: completed for "+base)
		q.RegisterOutputFile(j.ID, output, abs)
		created = append(created, output)
		_ = q.SetJobProgress(j.ID, idx+1, len(files))
	}

	q.CompleteJob(j.ID)
	return nil
}
