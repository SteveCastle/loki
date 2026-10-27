package workerapi

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"time"
)

// Hooks receive a remote run's progress.
type Hooks struct {
	// Log gets every stderr line of the engine (paths already local), in order.
	Log func(line string)
	// Status gets the dispatcher's own notes ("uploading 2 files", "queued
	// behind 1 job"), each once.
	Status func(msg string)
}

func (h Hooks) log(s string) {
	if h.Log != nil {
		h.Log(s)
	}
}

func (h Hooks) status(s string) {
	if h.Status != nil {
		h.Status(s)
	}
}

// Result is a finished remote run.
type Result struct {
	// Stdout are the engine's result lines with worker paths mapped to local.
	Stdout []string
	// Files are the local files written from the worker's outputs.
	Files []string
	// Secs is the engine's own run time on the worker.
	Secs float64
}

// RemoteError is a run that reached the engine and failed there (as opposed
// to a transport failure, which a caller may retry elsewhere).
type RemoteError struct {
	Msg string
	OOM bool
}

func (e *RemoteError) Error() string { return e.Msg }

// ErrNotSubmitted wraps failures before the worker accepted the job: the job
// never ran, so it is safe to try another worker.
var ErrNotSubmitted = errors.New("job not submitted")

func mib(n int64) string { return fmt.Sprintf("%.1f MB", float64(n)/(1<<20)) }

func totalSize(in []LocalInput) int64 {
	var t int64
	for _, i := range in {
		t += i.Size
	}
	return t
}

// RunOnNode runs a planned command line on a persistent worker and brings
// its outputs back to the local paths the command line named.
func RunOnNode(ctx context.Context, n *Node, plan *Plan, tool string, light bool, h Hooks) (*Result, error) {
	spec := JobSpec{Tool: EngineName(tool), Args: plan.Args, Dirs: plan.Dirs, Light: light}
	for _, in := range plan.Inputs {
		spec.Inputs = append(spec.Inputs, Input{Path: in.Remote, Size: in.Size})
	}
	if len(plan.Inputs) > 0 && !light {
		h.status(fmt.Sprintf("sending %d file(s) (%s) to %s", len(plan.Inputs), mib(totalSize(plan.Inputs)), n.Name))
	}
	st, err := n.Submit(ctx, spec, plan.Inputs)
	if err != nil {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		return nil, fmt.Errorf("%w: %s: %v", ErrNotSubmitted, n.Name, err)
	}
	id := st.ID
	defer func() {
		dctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		_ = n.Delete(dctx, id)
	}()

	next, lastPos, failures := 0, -1, 0
	wait := 250 * time.Millisecond
	for {
		if st.State == JobQueued && st.QueuePos != lastPos {
			lastPos = st.QueuePos
			if st.QueuePos > 0 {
				h.status(fmt.Sprintf("queued on %s behind %d job(s)", n.Name, st.QueuePos))
			}
		}
		for _, l := range st.Logs {
			h.log(plan.RewriteLine(l))
		}
		if st.LogNext > next {
			next = st.LogNext
		}
		if st.State.Finished() {
			break
		}
		select {
		case <-ctx.Done():
			cctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
			_ = n.Cancel(cctx, id)
			cancel()
			return nil, ctx.Err()
		case <-time.After(wait):
		}
		if wait < time.Second && !light {
			wait += 250 * time.Millisecond
		}
		s, err := n.Status(ctx, id, next)
		if err != nil {
			if ctx.Err() != nil {
				continue
			}
			var he *HTTPError
			if errors.As(err, &he) && he.Status == 404 {
				return nil, fmt.Errorf("%s lost the job (the worker restarted?)", n.Name)
			}
			// network blips and proxy hiccups: keep polling for a while
			if failures++; failures > 60 {
				return nil, fmt.Errorf("%s stopped answering: %v", n.Name, err)
			}
			continue
		}
		failures = 0
		st = s
	}
	if st.State != JobDone {
		msg := st.Error
		if msg == "" {
			msg = string(st.State)
		}
		return nil, &RemoteError{Msg: RewritePaths(msg, plan.LocalPath), OOM: st.OOM}
	}
	res := &Result{Secs: st.Secs}
	if len(st.Outputs) > 0 {
		err := n.Outputs(ctx, id, func(rel string, r io.Reader) error {
			local, ok := plan.LocalPath(rel)
			if !ok {
				return nil
			}
			if _, err := WriteFileAtomic(local, r); err != nil {
				return fmt.Errorf("writing %s: %w", local, err)
			}
			res.Files = append(res.Files, local)
			return nil
		})
		if err != nil {
			return nil, fmt.Errorf("fetching the results from %s: %w", n.Name, err)
		}
	}
	for _, l := range st.Stdout {
		res.Stdout = append(res.Stdout, plan.RewriteLine(l))
	}
	return res, nil
}

// Transfer moves files between this machine and serverless workers through
// object storage the workers can reach (presigned URLs).
type Transfer interface {
	// Upload stores a local file under key and returns a URL to GET it.
	Upload(ctx context.Context, key, local string) (string, error)
	// PresignPut returns a URL a worker can PUT an object to under key.
	PresignPut(ctx context.Context, key string) (string, error)
	// Fetch streams the object under key to fn.
	Fetch(ctx context.Context, key string, fn func(io.Reader) error) error
	// Delete removes objects (best effort).
	Delete(ctx context.Context, keys []string)
}

// InlineLimit caps what goes inline in a serverless request when no transfer
// bucket is configured (RunPod accepts ~10 MB per request, base64 adds 1/3).
const InlineLimit = 7 << 20

// OutputSlots is how many output files a serverless run can return through
// a transfer bucket.
const OutputSlots = 32

// RunOnRunPod runs a planned command line as a RunPod serverless job. Files
// travel through tr (or inline when tr is nil and they are small).
func RunOnRunPod(ctx context.Context, rp *RunPod, plan *Plan, tool string, light bool, tr Transfer, h Hooks) (*Result, error) {
	spec := JobSpec{Tool: EngineName(tool), Args: plan.Args, Dirs: plan.Dirs, Light: light}
	prefix := fmt.Sprintf("loki-worker/%d-%s", time.Now().UnixNano(), randHex(4))
	var keys []string
	defer func() {
		if tr != nil && len(keys) > 0 {
			dctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			tr.Delete(dctx, keys)
			cancel()
		}
	}()
	if tr == nil && totalSize(plan.Inputs) > InlineLimit {
		return nil, fmt.Errorf("%w: the inputs (%s) are too large to send inline to a serverless worker; give the pool a transfer bucket", ErrNotSubmitted, mib(totalSize(plan.Inputs)))
	}
	if len(plan.Inputs) > 0 && !light {
		h.status(fmt.Sprintf("sending %d file(s) (%s) to RunPod", len(plan.Inputs), mib(totalSize(plan.Inputs))))
	}
	for i, in := range plan.Inputs {
		if tr == nil {
			b, err := os.ReadFile(in.Local)
			if err != nil {
				return nil, err
			}
			spec.Inputs = append(spec.Inputs, Input{Path: in.Remote, Data: b, Size: in.Size})
			continue
		}
		key := fmt.Sprintf("%s/in/%d-%s", prefix, i, SafeName(in.Local[strings.LastIndexAny(in.Local, `/\`)+1:]))
		u, err := tr.Upload(ctx, key, in.Local)
		if err != nil {
			return nil, fmt.Errorf("%w: uploading %s: %v", ErrNotSubmitted, in.Local, err)
		}
		keys = append(keys, key)
		spec.Inputs = append(spec.Inputs, Input{Path: in.Remote, URL: u, Size: in.Size})
	}
	outKeys := make([]string, 0, OutputSlots)
	if tr != nil {
		sink := &OutputSink{}
		for i := 0; i < OutputSlots; i++ {
			key := fmt.Sprintf("%s/out/%d", prefix, i)
			u, err := tr.PresignPut(ctx, key)
			if err != nil {
				return nil, fmt.Errorf("%w: %v", ErrNotSubmitted, err)
			}
			sink.PutURLs = append(sink.PutURLs, u)
			outKeys = append(outKeys, key)
		}
		spec.Outputs = sink
	} else {
		spec.Outputs = &OutputSink{Inline: true, MaxInline: InlineLimit}
	}

	id, err := rp.Run(ctx, RunPodInput{Spec: &spec})
	if err != nil {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		return nil, fmt.Errorf("%w: RunPod: %v", ErrNotSubmitted, err)
	}
	queuedNoted, failures := false, 0
	var st *JobStatus
	for st == nil {
		select {
		case <-ctx.Done():
			cctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
			_ = rp.Cancel(cctx, id)
			cancel()
			return nil, ctx.Err()
		case <-time.After(time.Second):
		}
		s, err := rp.Stream(ctx, id)
		if err != nil {
			if ctx.Err() != nil {
				continue
			}
			if failures++; failures > 120 {
				return nil, fmt.Errorf("RunPod stopped answering: %v", err)
			}
			continue
		}
		failures = 0
		for _, c := range s.Chunks {
			for _, l := range c.Logs {
				h.log(plan.RewriteLine(l))
			}
		}
		switch s.Status {
		case "IN_QUEUE":
			if !queuedNoted {
				queuedNoted = true
				h.status("queued on RunPod (a cold worker takes a minute or more to start)")
			}
		case "COMPLETED":
			out, err := rp.Result(ctx, id)
			if err != nil {
				return nil, err
			}
			st = out
		case "FAILED", "CANCELLED", "TIMED_OUT":
			msg := s.Error
			if msg == "" {
				msg = "RunPod job " + strings.ToLower(s.Status)
			}
			return nil, &RemoteError{Msg: RewritePaths(msg, plan.LocalPath)}
		}
	}
	for _, l := range st.Logs {
		h.log(plan.RewriteLine(l))
	}
	if st.State != JobDone {
		return nil, &RemoteError{Msg: RewritePaths(st.Error, plan.LocalPath), OOM: st.OOM}
	}
	res := &Result{Secs: st.Secs}
	for _, o := range st.Outputs {
		local, ok := plan.LocalPath(o.Path)
		if !ok {
			continue
		}
		var err error
		switch {
		case o.Data != nil:
			_, err = WriteFileAtomic(local, bytes.NewReader(o.Data))
		case o.Slot >= 0 && o.Slot < len(outKeys) && tr != nil:
			keys = append(keys, outKeys[o.Slot])
			err = tr.Fetch(ctx, outKeys[o.Slot], func(r io.Reader) error {
				_, werr := WriteFileAtomic(local, r)
				return werr
			})
		default:
			err = fmt.Errorf("the worker kept %s (no way back)", o.Path)
		}
		if err != nil {
			return nil, fmt.Errorf("fetching %s: %w", o.Path, err)
		}
		res.Files = append(res.Files, local)
	}
	for _, l := range st.Stdout {
		res.Stdout = append(res.Stdout, plan.RewriteLine(l))
	}
	return res, nil
}
