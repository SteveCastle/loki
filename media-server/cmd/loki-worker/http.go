package main

import (
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/stevecastle/shrike/workerapi"
)

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

func httpError(w http.ResponseWriter, code int, msg string) {
	writeJSON(w, code, map[string]string{"error": msg})
}

func (w *worker) auth(next http.HandlerFunc) http.HandlerFunc {
	return func(rw http.ResponseWriter, r *http.Request) {
		if w.cfg.token != "" {
			got := strings.TrimSpace(strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer "))
			if subtle.ConstantTimeCompare([]byte(got), []byte(w.cfg.token)) != 1 {
				httpError(rw, http.StatusUnauthorized, "missing or wrong worker token")
				return
			}
		}
		next(rw, r)
	}
}

func (w *worker) routes() *http.ServeMux {
	mux := http.NewServeMux()
	// Health for load balancers (RunPod load-balancing endpoints poll it):
	// 204 while starting or preloading models, 200 when ready.
	mux.HandleFunc("GET /ping", func(rw http.ResponseWriter, r *http.Request) {
		w.mu.Lock()
		ready := w.ready
		w.mu.Unlock()
		if !ready {
			rw.WriteHeader(http.StatusNoContent)
			return
		}
		writeJSON(rw, http.StatusOK, map[string]string{"status": "healthy"})
	})
	mux.HandleFunc("GET /{$}", func(rw http.ResponseWriter, r *http.Request) {
		rw.Header().Set("Content-Type", "text/plain; charset=utf-8")
		fmt.Fprintf(rw, "loki-worker %s (%s): engines %s on %d GPU(s). API: /v1/info, /v1/jobs, /v1/sessions (Authorization: Bearer <token>).\n", version, w.cfg.name, strings.Join(w.toolNames(), ", "), len(w.slots))
	})
	mux.HandleFunc("GET /v1/info", w.auth(func(rw http.ResponseWriter, r *http.Request) {
		writeJSON(rw, http.StatusOK, w.info(r.URL.Query().Get("help") == "1"))
	}))
	mux.HandleFunc("POST /v1/warm", w.auth(func(rw http.ResponseWriter, r *http.Request) {
		var req workerapi.WarmRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			httpError(rw, http.StatusBadRequest, err.Error())
			return
		}
		if err := w.warm(req); err != nil {
			httpError(rw, http.StatusBadRequest, err.Error())
			return
		}
		writeJSON(rw, http.StatusAccepted, map[string]bool{"ok": true})
	}))
	mux.HandleFunc("POST /v1/jobs", w.auth(w.handleSubmit))
	mux.HandleFunc("GET /v1/jobs/{id}", w.auth(func(rw http.ResponseWriter, r *http.Request) {
		j := w.job(r.PathValue("id"))
		if j == nil {
			httpError(rw, http.StatusNotFound, "no such job")
			return
		}
		from, _ := strconv.Atoi(r.URL.Query().Get("logs"))
		writeJSON(rw, http.StatusOK, j.status(from))
	}))
	mux.HandleFunc("POST /v1/jobs/{id}/cancel", w.auth(func(rw http.ResponseWriter, r *http.Request) {
		if !w.cancelJob(r.PathValue("id")) {
			httpError(rw, http.StatusNotFound, "no such job")
			return
		}
		writeJSON(rw, http.StatusOK, map[string]bool{"ok": true})
	}))
	mux.HandleFunc("DELETE /v1/jobs/{id}", w.auth(func(rw http.ResponseWriter, r *http.Request) {
		if !w.remove(r.PathValue("id")) {
			httpError(rw, http.StatusNotFound, "no such job")
			return
		}
		writeJSON(rw, http.StatusOK, map[string]bool{"ok": true})
	}))
	mux.HandleFunc("GET /v1/jobs/{id}/outputs.tar", w.auth(func(rw http.ResponseWriter, r *http.Request) {
		j := w.job(r.PathValue("id"))
		if j == nil {
			httpError(rw, http.StatusNotFound, "no such job")
			return
		}
		st := j.status(0)
		var rels []string
		for _, o := range st.Outputs {
			rels = append(rels, o.Path)
		}
		rw.Header().Set("Content-Type", "application/x-tar")
		if err := workerapi.WriteTar(rw, j.dir, rels); err != nil {
			log.Printf("job %s: sending outputs: %v", st.ID, err)
		}
	}))
	mux.HandleFunc("GET /v1/jobs/{id}/files/{path...}", w.auth(func(rw http.ResponseWriter, r *http.Request) {
		j := w.job(r.PathValue("id"))
		rel, ok := workerapi.CleanRel(r.PathValue("path"))
		if j == nil || !ok {
			httpError(rw, http.StatusNotFound, "no such file")
			return
		}
		http.ServeFile(rw, r, filepath.Join(j.dir, filepath.FromSlash(rel)))
	}))

	// sessions
	mux.HandleFunc("POST /v1/sessions", w.auth(func(rw http.ResponseWriter, r *http.Request) {
		var req struct {
			Tool string `json:"tool"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)
		if req.Tool == "" {
			req.Tool = "loki-segment"
		}
		s, err := w.sessions.open(req.Tool)
		if err != nil {
			httpError(rw, http.StatusBadRequest, err.Error())
			return
		}
		writeJSON(rw, http.StatusOK, map[string]string{"id": s.id})
	}))
	mux.HandleFunc("PUT /v1/sessions/{id}/files/{path...}", w.auth(func(rw http.ResponseWriter, r *http.Request) {
		s := w.sessions.get(r.PathValue("id"))
		rel, ok := workerapi.CleanRel(r.PathValue("path"))
		if s == nil || !ok {
			httpError(rw, http.StatusNotFound, "no such session")
			return
		}
		if _, err := workerapi.WriteFileAtomic(filepath.Join(s.dir, filepath.FromSlash(rel)), r.Body); err != nil {
			httpError(rw, http.StatusInternalServerError, err.Error())
			return
		}
		writeJSON(rw, http.StatusOK, map[string]bool{"ok": true})
	}))
	mux.HandleFunc("POST /v1/sessions/{id}/call", w.auth(func(rw http.ResponseWriter, r *http.Request) {
		s := w.sessions.get(r.PathValue("id"))
		if s == nil {
			httpError(rw, http.StatusNotFound, "no such session")
			return
		}
		var call workerapi.SessionCall
		if err := json.NewDecoder(r.Body).Decode(&call); err != nil || call.Req == nil {
			httpError(rw, http.StatusBadRequest, "a session call needs {\"req\": {...}}")
			return
		}
		rep, err := w.sessions.call(r.Context(), s, call.Req)
		if err != nil {
			httpError(rw, http.StatusBadGateway, err.Error())
			return
		}
		writeJSON(rw, http.StatusOK, rep)
	}))
	mux.HandleFunc("GET /v1/sessions/{id}/files.tar", w.auth(func(rw http.ResponseWriter, r *http.Request) {
		s := w.sessions.get(r.PathValue("id"))
		if s == nil {
			httpError(rw, http.StatusNotFound, "no such session")
			return
		}
		prefix := strings.Trim(r.URL.Query().Get("prefix"), "/")
		var rels []string
		_ = filepath.Walk(s.dir, func(p string, info os.FileInfo, err error) error {
			if err != nil || info.IsDir() {
				return err
			}
			rel, _ := filepath.Rel(s.dir, p)
			rel = filepath.ToSlash(rel)
			if prefix == "" || rel == prefix || strings.HasPrefix(rel, prefix+"/") {
				rels = append(rels, rel)
			}
			return nil
		})
		rw.Header().Set("Content-Type", "application/x-tar")
		_ = workerapi.WriteTar(rw, s.dir, rels)
	}))
	mux.HandleFunc("DELETE /v1/sessions/{id}", w.auth(func(rw http.ResponseWriter, r *http.Request) {
		if !w.sessions.close(r.PathValue("id")) {
			httpError(rw, http.StatusNotFound, "no such session")
			return
		}
		writeJSON(rw, http.StatusOK, map[string]bool{"ok": true})
	}))
	return mux
}

func (w *worker) job(id string) *job {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.jobs[id]
}

// handleSubmit takes a job: a multipart request (the "spec" part first, then
// one part per uploaded input, named by its path) or a JSON spec whose inputs
// come by URL or inline.
func (w *worker) handleSubmit(rw http.ResponseWriter, r *http.Request) {
	var spec workerapi.JobSpec
	var j *job
	fail := func(code int, err error) {
		if j != nil {
			_ = os.RemoveAll(j.dir)
		}
		httpError(rw, code, err.Error())
	}
	if strings.HasPrefix(r.Header.Get("Content-Type"), "multipart/") {
		mr, err := r.MultipartReader()
		if err != nil {
			fail(http.StatusBadRequest, err)
			return
		}
		expected := map[string]bool{}
		for {
			part, err := mr.NextPart()
			if err == io.EOF {
				break
			}
			if err != nil {
				fail(http.StatusBadRequest, fmt.Errorf("reading the upload: %w", err))
				return
			}
			name := part.FormName()
			if name == "spec" {
				if err := json.NewDecoder(part).Decode(&spec); err != nil {
					fail(http.StatusBadRequest, fmt.Errorf("spec: %w", err))
					return
				}
				if j, err = w.newJob(spec); err != nil {
					fail(http.StatusBadRequest, err)
					return
				}
				for _, in := range spec.Inputs {
					expected[in.Path] = true
				}
				continue
			}
			if j == nil {
				fail(http.StatusBadRequest, errors.New("the spec part must come first"))
				return
			}
			rel, ok := workerapi.CleanRel(name)
			if !ok || !expected[rel] {
				fail(http.StatusBadRequest, fmt.Errorf("unexpected file %q", name))
				return
			}
			if _, err := workerapi.WriteFileAtomic(filepath.Join(j.dir, filepath.FromSlash(rel)), part); err != nil {
				fail(http.StatusInternalServerError, err)
				return
			}
		}
		if j == nil {
			fail(http.StatusBadRequest, errors.New("no spec"))
			return
		}
	} else {
		if err := json.NewDecoder(r.Body).Decode(&spec); err != nil {
			fail(http.StatusBadRequest, err)
			return
		}
		var err error
		if j, err = w.newJob(spec); err != nil {
			fail(http.StatusBadRequest, err)
			return
		}
	}
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Minute)
	defer cancel()
	if err := stage(ctx, j.dir, &j.spec); err != nil {
		fail(http.StatusBadRequest, err)
		return
	}
	w.submit(j)
	writeJSON(rw, http.StatusAccepted, j.status(0))
}
