package handlers

import (
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/stevecastle/shrike/platform"
)

type onboardingState struct {
	Shown       bool       `json:"shown"`
	DismissedAt *time.Time `json:"dismissed_at,omitempty"`
}

var onbMu sync.Mutex

func onboardingPath() string {
	return filepath.Join(platform.GetDataDir(), "onboarding.json")
}

// HandleOnboardingGet serves GET /api/onboarding/state.
func HandleOnboardingGet(w http.ResponseWriter, _ *http.Request) {
	onbMu.Lock()
	defer onbMu.Unlock()
	b, err := os.ReadFile(onboardingPath())
	if err != nil {
		writeJSON(w, http.StatusOK, onboardingState{Shown: false})
		return
	}
	var s onboardingState
	if err := json.Unmarshal(b, &s); err != nil {
		writeJSON(w, http.StatusOK, onboardingState{Shown: false})
		return
	}
	writeJSON(w, http.StatusOK, s)
}

// HandleOnboardingDismiss serves POST /api/onboarding/dismiss.
func HandleOnboardingDismiss(w http.ResponseWriter, _ *http.Request) {
	onbMu.Lock()
	defer onbMu.Unlock()
	now := time.Now().UTC()
	s := onboardingState{Shown: true, DismissedAt: &now}
	b, _ := json.MarshalIndent(s, "", "  ")
	if err := os.MkdirAll(filepath.Dir(onboardingPath()), 0o755); err != nil {
		writeErr(w, http.StatusInternalServerError, err)
		return
	}
	if err := os.WriteFile(onboardingPath(), b, 0o644); err != nil {
		writeErr(w, http.StatusInternalServerError, err)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// HandleOnboardingReset serves POST /api/onboarding/reset.
func HandleOnboardingReset(w http.ResponseWriter, _ *http.Request) {
	onbMu.Lock()
	defer onbMu.Unlock()
	_ = os.Remove(onboardingPath())
	w.WriteHeader(http.StatusNoContent)
}
