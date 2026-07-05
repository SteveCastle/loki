package main

import (
	"net/http"

	"github.com/stevecastle/shrike/handlers"
	"github.com/stevecastle/shrike/renderer"
)

// RegisterDepsRoutes wires every /api/deps/* handler onto the given mux.
// Called from each platform's main file after the SPA router is mounted.
// Path patterns use Go 1.22+ wildcards. CORS is applied so the Electron
// renderer (different origin) can drive point-of-use model downloads.
func RegisterDepsRoutes(mux *http.ServeMux) {
	h := func(fn http.HandlerFunc) http.HandlerFunc { return renderer.CORS(fn) }

	mux.HandleFunc("GET /api/deps/status", h(handlers.HandleDepsStatus))
	mux.HandleFunc("GET /api/deps/models/progress", h(handlers.HandleModelProgressSSE))
	mux.HandleFunc("POST /api/deps/models/{id}/download", h(handlers.HandleModelDownload))
	mux.HandleFunc("POST /api/deps/models/{id}/cancel", h(handlers.HandleModelCancel))
	mux.HandleFunc("POST /api/deps/models/{id}/verify", h(handlers.HandleModelVerify))
	mux.HandleFunc("DELETE /api/deps/models/{id}", h(handlers.HandleModelDelete))

	mux.HandleFunc("GET /api/onboarding/state", h(handlers.HandleOnboardingGet))
	mux.HandleFunc("POST /api/onboarding/dismiss", h(handlers.HandleOnboardingDismiss))
	mux.HandleFunc("POST /api/onboarding/reset", h(handlers.HandleOnboardingReset))
}
