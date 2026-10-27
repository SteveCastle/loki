package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/stevecastle/shrike/workerapi"
)

// RunPod queue-based serverless: the worker pulls jobs from RunPod
// (RUNPOD_WEBHOOK_GET_JOB), streams progress (RUNPOD_WEBHOOK_POST_STREAM),
// posts the result (RUNPOD_WEBHOOK_POST_OUTPUT) and heartbeats
// (RUNPOD_WEBHOOK_PING) — the protocol of RunPod's own worker SDK. The job
// input is a workerapi.RunPodInput; inputs arrive by presigned URL or
// inline, outputs leave the same way (a serverless worker is not addressable).

type runpodEnv struct {
	getJob, postOutput, postStream, ping string
	key, podID                           string
	pingEvery                            time.Duration
}

func runpodFromEnv() (runpodEnv, bool) {
	e := runpodEnv{
		getJob:     os.Getenv("RUNPOD_WEBHOOK_GET_JOB"),
		postOutput: os.Getenv("RUNPOD_WEBHOOK_POST_OUTPUT"),
		postStream: os.Getenv("RUNPOD_WEBHOOK_POST_STREAM"),
		ping:       os.Getenv("RUNPOD_WEBHOOK_PING"),
		key:        os.Getenv("RUNPOD_AI_API_KEY"),
		podID:      os.Getenv("RUNPOD_POD_ID"),
		pingEvery:  10 * time.Second,
	}
	if e.podID == "" {
		e.podID = "local-" + randID(3)
	}
	if ms := os.Getenv("RUNPOD_PING_INTERVAL"); ms != "" {
		var n int
		if _, err := fmt.Sscan(ms, &n); err == nil && n >= 1000 {
			e.pingEvery = time.Duration(n) * time.Millisecond
		}
	}
	return e, e.getJob != "" && e.postOutput != ""
}

func withQuery(u, q string) string {
	if strings.Contains(u, "?") {
		return u + "&" + q
	}
	return u + "?" + q
}

func (e runpodEnv) request(ctx context.Context, method, u string, body any) (*http.Response, error) {
	var r io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return nil, err
		}
		r = bytes.NewReader(b)
	}
	req, err := http.NewRequestWithContext(ctx, method, u, r)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", e.key)
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	return http.DefaultClient.Do(req)
}

// post sends a job's output or a stream chunk, retrying a few times.
func (e runpodEnv) post(tmpl, jobID string, isStream bool, body any) {
	u := strings.ReplaceAll(strings.ReplaceAll(tmpl, "$ID", jobID), "$RUNPOD_POD_ID", e.podID)
	u = withQuery(u, "isStream="+map[bool]string{true: "true", false: "false"}[isStream])
	for attempt := 0; attempt < 4; attempt++ {
		ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
		resp, err := e.request(ctx, http.MethodPost, u, body)
		cancel()
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode/100 == 2 {
				return
			}
			err = fmt.Errorf("status %d", resp.StatusCode)
		}
		log.Printf("runpod: posting %s for job %s: %v", map[bool]string{true: "stream", false: "result"}[isStream], jobID, err)
		time.Sleep(time.Duration(attempt+1) * time.Second)
	}
}

// serveRunPod pulls and runs jobs until ctx ends.
func (w *worker) serveRunPod(ctx context.Context, e runpodEnv) {
	var mu sync.Mutex
	current := ""
	go func() {
		if e.ping == "" {
			return
		}
		u := strings.ReplaceAll(e.ping, "$RUNPOD_POD_ID", e.podID)
		for {
			mu.Lock()
			q := url.Values{"job_id": {current}, "runpod_version": {"loki-worker-" + version}}
			mu.Unlock()
			pctx, cancel := context.WithTimeout(ctx, 10*time.Second)
			if resp, err := e.request(pctx, http.MethodGet, withQuery(u, q.Encode()), nil); err == nil {
				resp.Body.Close()
			}
			cancel()
			select {
			case <-ctx.Done():
				return
			case <-time.After(e.pingEvery):
			}
		}
	}()
	take := strings.ReplaceAll(e.getJob, "$ID", e.podID)
	log.Printf("runpod: serverless worker %s polling for jobs", e.podID)
	for ctx.Err() == nil {
		mu.Lock()
		busy := current != ""
		mu.Unlock()
		gctx, cancel := context.WithTimeout(ctx, 90*time.Second)
		resp, err := e.request(gctx, http.MethodGet, withQuery(take, "job_in_progress="+map[bool]string{true: "1", false: "0"}[busy]), nil)
		if err != nil {
			cancel()
			if ctx.Err() == nil {
				log.Printf("runpod: taking a job: %v", err)
				time.Sleep(3 * time.Second)
			}
			continue
		}
		b, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		cancel()
		if resp.StatusCode == http.StatusNoContent || resp.StatusCode == http.StatusBadRequest || len(bytes.TrimSpace(b)) == 0 {
			continue
		}
		if resp.StatusCode/100 != 2 {
			log.Printf("runpod: job-take answered %d", resp.StatusCode)
			time.Sleep(3 * time.Second)
			continue
		}
		var rj struct {
			ID    string                `json:"id"`
			Input workerapi.RunPodInput `json:"input"`
		}
		if err := json.Unmarshal(b, &rj); err != nil || rj.ID == "" {
			// the batch endpoint answers with a list
			var list []json.RawMessage
			if json.Unmarshal(b, &list) == nil && len(list) > 0 {
				_ = json.Unmarshal(list[0], &rj)
			}
		}
		if rj.ID == "" {
			log.Printf("runpod: unreadable job: %.200s", b)
			continue
		}
		mu.Lock()
		current = rj.ID
		mu.Unlock()
		w.runPodJob(e, rj.ID, rj.Input)
		mu.Lock()
		current = ""
		mu.Unlock()
	}
}

func (w *worker) runPodJob(e runpodEnv, id string, in workerapi.RunPodInput) {
	switch {
	case in.Info:
		e.post(e.postOutput, id, false, map[string]any{"output": w.info(true)})
		return
	case in.Warm != nil:
		var errs []string
		for _, t := range in.Warm.Tools {
			if err := w.warmNow(context.Background(), workerapi.EngineName(t), in.Warm.Model); err != nil {
				errs = append(errs, err.Error())
			}
		}
		if len(errs) > 0 {
			e.post(e.postOutput, id, false, map[string]any{"error": strings.Join(errs, "; ")})
			return
		}
		e.post(e.postOutput, id, false, map[string]any{"output": map[string]bool{"ok": true}})
		return
	case in.Spec == nil:
		e.post(e.postOutput, id, false, map[string]any{"error": "the input needs \"spec\", \"warm\" or \"info\""})
		return
	}
	j, err := w.newJob(*in.Spec)
	if err != nil {
		e.post(e.postOutput, id, false, map[string]any{"error": err.Error()})
		return
	}
	defer os.RemoveAll(j.dir)
	// stream the log every 2 s
	var smu sync.Mutex
	var pending []string
	j.onLog = func(l string) {
		smu.Lock()
		pending = append(pending, l)
		smu.Unlock()
	}
	flush := func() {
		smu.Lock()
		chunk := pending
		pending = nil
		smu.Unlock()
		if len(chunk) == 0 || e.postStream == "" {
			return
		}
		j.mu.Lock()
		step, steps := j.st.Step, j.st.Steps
		j.mu.Unlock()
		e.post(e.postStream, id, true, map[string]any{"output": workerapi.RunPodStream{Logs: chunk, Step: step, Steps: steps}})
	}
	stop := make(chan struct{})
	go func() {
		for {
			select {
			case <-stop:
				return
			case <-time.After(2 * time.Second):
				flush()
			}
		}
	}()
	if err := stage(j.ctx, j.dir, &j.spec); err != nil {
		w.finish(j, workerapi.JobFailed, err.Error())
	} else {
		w.submit(j)
	}
	<-j.done
	close(stop)
	flush()
	st := j.status(1 << 30) // logs went out with the stream
	if e.postStream == "" {
		st = j.status(0)
	}
	e.post(e.postOutput, id, false, map[string]any{"output": st})
	w.remove(st.ID)
}
