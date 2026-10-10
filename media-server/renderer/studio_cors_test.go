package renderer

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestIsStudioOrigin(t *testing.T) {
	for origin, want := range map[string]bool{
		"studio://app":                      true,
		"https://lowkeyviewer.com":          true,
		"http://localhost:8790":             true,
		"http://127.0.0.1:8793":             true,
		"http://localhost:1212":             false, // Electron renderer: cookie auth
		"https://evil.example":              false,
		"http://localhost.evil.com":         false,
		"https://lowkeyviewer.com.evil.com": false,
		"":                                  false,
	} {
		if got := IsStudioOrigin(origin); got != want {
			t.Errorf("IsStudioOrigin(%q) = %v; want %v", origin, got, want)
		}
	}
}

func TestStudioCorsHasNoCredentials(t *testing.T) {
	rec := httptest.NewRecorder()
	w := http.ResponseWriter(rec)
	enableCors(&w, &http.Request{Header: http.Header{
		"Origin":                                 []string{"https://lowkeyviewer.com"},
		"Access-Control-Request-Private-Network": []string{"true"},
	}})
	h := rec.Header()
	if h.Get("Access-Control-Allow-Origin") != "https://lowkeyviewer.com" {
		t.Errorf("allow-origin = %q", h.Get("Access-Control-Allow-Origin"))
	}
	if h.Get("Access-Control-Allow-Credentials") != "" {
		t.Errorf("studio origins must never get credentialed CORS")
	}
	if h.Get("Access-Control-Allow-Private-Network") != "true" {
		t.Errorf("private-network preflight not allowed")
	}
}
