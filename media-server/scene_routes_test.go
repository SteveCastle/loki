package main

import (
	"net/http"
	"testing"
)

// Overlapping route patterns (e.g. /api/scenes/x/{job} next to
// /api/scenes/{id}/render) make http.ServeMux panic at startup.
func TestSceneRoutesRegister(t *testing.T) {
	RegisterSceneRoutes(http.NewServeMux(), &Dependencies{})
}
