// Package handlers hosts HTTP handlers decoupled from main.go and shared by
// every platform build.
package handlers

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/stevecastle/shrike/deps/models"
	"github.com/stevecastle/shrike/deps/status"
)

// HandleDepsStatus serves GET /api/deps/status.
func HandleDepsStatus(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, status.Snapshot())
}

// HandleModelDownload serves POST /api/deps/models/{id}/download.
func HandleModelDownload(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if _, ok := models.Lookup(id); !ok {
		writeErr(w, http.StatusNotFound, fmt.Errorf("unknown model id %q", id))
		return
	}
	if cached := models.Cached(); cached[id] == models.StatusInstalled {
		if inst, ok := models.Tracker.Snapshot(id); ok {
			writeJSON(w, http.StatusOK, inst)
			return
		}
		writeJSON(w, http.StatusOK, map[string]string{"state": "installed"})
		return
	}
	inst, err := models.Tracker.StartInstall(id)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusAccepted, inst)
}

// HandleModelCancel serves POST /api/deps/models/{id}/cancel.
func HandleModelCancel(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if !models.Tracker.Cancel(id) {
		writeErr(w, http.StatusNotFound, fmt.Errorf("no active install for %q", id))
		return
	}
	w.WriteHeader(http.StatusAccepted)
}

// HandleModelDelete serves DELETE /api/deps/models/{id}.
func HandleModelDelete(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if _, ok := models.Lookup(id); !ok {
		writeErr(w, http.StatusNotFound, fmt.Errorf("unknown model id %q", id))
		return
	}
	if err := os.RemoveAll(models.ModelDir(id)); err != nil && !os.IsNotExist(err) {
		writeErr(w, http.StatusInternalServerError, err)
		return
	}
	models.RebuildState()
	w.WriteHeader(http.StatusNoContent)
}

// HandleModelVerify serves POST /api/deps/models/{id}/verify.
func HandleModelVerify(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	m, ok := models.Lookup(id)
	if !ok {
		writeErr(w, http.StatusNotFound, fmt.Errorf("unknown model id %q", id))
		return
	}
	result := struct {
		ID    string            `json:"id"`
		Files map[string]string `json:"files"`
	}{ID: id, Files: map[string]string{}}
	for _, f := range m.EffectiveFiles() {
		path := filepath.Join(models.ModelDir(id), f.RelPath)
		if f.Archive != "" {
			// The checksum covers the (discarded) archive, not the extracted
			// binary — presence is the only meaningful post-install check.
			if _, err := os.Stat(path); err != nil {
				result.Files[f.RelPath] = "missing"
			} else {
				result.Files[f.RelPath] = "ok (extracted from checksum-verified archive)"
			}
			continue
		}
		if err := models.VerifySHA256(path, f.SHA256); err != nil {
			if errors.Is(err, os.ErrNotExist) {
				result.Files[f.RelPath] = "missing"
			} else {
				result.Files[f.RelPath] = err.Error()
			}
		} else {
			result.Files[f.RelPath] = "ok"
		}
	}
	writeJSON(w, http.StatusOK, result)
}

// HandleModelProgressSSE serves GET /api/deps/models/progress.
func HandleModelProgressSSE(w http.ResponseWriter, r *http.Request) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	ch, done := models.Tracker.Subscribe()
	defer done()

	for _, inst := range models.Tracker.All() {
		writeSSE(w, flusher, inst)
	}

	ping := time.NewTicker(15 * time.Second)
	defer ping.Stop()
	for {
		select {
		case <-r.Context().Done():
			return
		case inst, ok := <-ch:
			if !ok {
				return
			}
			writeSSE(w, flusher, inst)
		case <-ping.C:
			_, _ = w.Write([]byte(": ping\n\n"))
			flusher.Flush()
		}
	}
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeErr(w http.ResponseWriter, status int, err error) {
	writeJSON(w, status, map[string]string{"error": err.Error()})
}

func writeSSE(w http.ResponseWriter, f http.Flusher, payload any) {
	b, err := json.Marshal(payload)
	if err != nil {
		return
	}
	_, _ = w.Write([]byte("data: "))
	_, _ = w.Write(b)
	_, _ = w.Write([]byte("\n\n"))
	f.Flush()
}
