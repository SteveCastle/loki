package main

import (
	"net/http"

	"github.com/stevecastle/shrike/deps/optional"
	"github.com/stevecastle/shrike/handlers"
	"github.com/stevecastle/shrike/renderer"
)

// RegisterDepsRoutes wires every /api/deps/* handler onto the given mux.
// Called from each platform's main file after the SPA router is mounted.
// Path patterns use Go 1.22+ wildcards. CORS is applied so the Electron
// renderer (different origin) can drive point-of-use model downloads.
// Write endpoints are open by design (the first-run wizard and Electron
// call them unauthenticated) EXCEPT while Allow Public Access is on, when
// anonymous visitors must not be able to start or delete model downloads.
func RegisterDepsRoutes(mux *http.ServeMux, deps *Dependencies) {
	h := func(fn http.HandlerFunc) http.HandlerFunc { return renderer.CORS(fn) }
	hw := func(fn http.HandlerFunc) http.HandlerFunc {
		return renderer.CORS(requireAuthWhenPublic(deps, fn))
	}

	// Populate the optional-tool detection cache before the first status
	// request; detection spawns version subprocesses and must never run in
	// the request path.
	optional.Warm()

	mux.HandleFunc("GET /api/deps/status", h(handlers.HandleDepsStatus))
	mux.HandleFunc("GET /api/deps/models/progress", h(handlers.HandleModelProgressSSE))
	mux.HandleFunc("POST /api/deps/models/{id}/download", hw(handlers.HandleModelDownload))
	mux.HandleFunc("POST /api/deps/models/{id}/cancel", hw(handlers.HandleModelCancel))
	mux.HandleFunc("POST /api/deps/models/{id}/verify", hw(handlers.HandleModelVerify))
	mux.HandleFunc("DELETE /api/deps/models/{id}", hw(handlers.HandleModelDelete))

	mux.HandleFunc("GET /api/onboarding/state", h(handlers.HandleOnboardingGet))
	mux.HandleFunc("POST /api/onboarding/dismiss", hw(handlers.HandleOnboardingDismiss))
	mux.HandleFunc("POST /api/onboarding/reset", hw(handlers.HandleOnboardingReset))
}
