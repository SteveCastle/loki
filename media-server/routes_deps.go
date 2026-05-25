package main

import (
	"net/http"

	"github.com/stevecastle/shrike/handlers"
)

// RegisterDepsRoutes wires every /api/deps/* handler onto the given mux.
// Called from each platform's main file after the SPA router is mounted.
// Path patterns use Go 1.22+ wildcards.
func RegisterDepsRoutes(mux *http.ServeMux) {
	mux.HandleFunc("GET /api/deps/status", handlers.HandleDepsStatus)
	mux.HandleFunc("GET /api/deps/models/progress", handlers.HandleModelProgressSSE)
	mux.HandleFunc("POST /api/deps/models/{id}/download", handlers.HandleModelDownload)
	mux.HandleFunc("POST /api/deps/models/{id}/cancel", handlers.HandleModelCancel)
	mux.HandleFunc("POST /api/deps/models/{id}/verify", handlers.HandleModelVerify)
	mux.HandleFunc("DELETE /api/deps/models/{id}", handlers.HandleModelDelete)
}
