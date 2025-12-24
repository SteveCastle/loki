package renderer

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// TestFormatTime tests the formatTime template function
func TestFormatTime(t *testing.T) {
	tests := []struct {
		name     string
		input    time.Time
		expected string
	}{
		{
			name:     "Standard date",
			input:    time.Date(2024, 6, 15, 14, 30, 45, 0, time.UTC),
			expected: "Jun 15, 2024 14:30:45",
		},
		{
			name:     "New Year",
			input:    time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC),
			expected: "Jan 1, 2024 00:00:00",
		},
		{
			name:     "End of year",
			input:    time.Date(2024, 12, 31, 23, 59, 59, 0, time.UTC),
			expected: "Dec 31, 2024 23:59:59",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := formatTime(tt.input)
			if result != tt.expected {
				t.Errorf("formatTime() = %q; want %q", result, tt.expected)
			}
		})
	}
}

// TestHtmlAttr tests the htmlAttr template function
func TestHtmlAttr(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "Plain text",
			input:    "hello world",
			expected: "hello world",
		},
		{
			name:     "Double quotes",
			input:    `say "hello"`,
			expected: `say &#34;hello&#34;`,
		},
		{
			name:     "Single quotes",
			input:    "it's fine",
			expected: "it&#39;s fine",
		},
		{
			name:     "Angle brackets",
			input:    "<script>alert(1)</script>",
			expected: "&lt;script&gt;alert(1)&lt;/script&gt;",
		},
		{
			name:     "Ampersand",
			input:    "fish & chips",
			expected: "fish &amp; chips",
		},
		{
			name:     "Complex path",
			input:    `C:\Users\test\file.jpg`,
			expected: `C:\Users\test\file.jpg`,
		},
		{
			name:     "Empty string",
			input:    "",
			expected: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := htmlAttr(tt.input)
			if result != tt.expected {
				t.Errorf("htmlAttr(%q) = %q; want %q", tt.input, result, tt.expected)
			}
		})
	}
}

// TestTemplates tests template initialization
func TestTemplates(t *testing.T) {
	tmpl := Templates()

	if tmpl == nil {
		t.Fatal("Templates() returned nil")
	}

	// Verify it returns the same instance (singleton)
	tmpl2 := Templates()
	if tmpl != tmpl2 {
		t.Error("Templates() should return same instance")
	}
}

// TestTemplatesHasExpectedTemplates verifies expected templates exist
func TestTemplatesHasExpectedTemplates(t *testing.T) {
	tmpl := Templates()

	// List of expected template names (based on files in templates/)
	expectedTemplates := []string{
		"home",
		"jobs",
		"detail",
		"jobRow",
		"media",
		"config",
		"stats",
		"dependencies",
		"setup",
		"editor",
		"swipe",
		"topnav",
	}

	for _, name := range expectedTemplates {
		if tmpl.Lookup(name) == nil {
			t.Errorf("Template %q not found", name)
		}
	}
}

// TestLoggerMiddleware tests the Logger middleware
func TestLoggerMiddleware(t *testing.T) {
	// Create a test handler
	innerHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("OK"))
	})

	// Wrap with Logger
	handler := Logger(innerHandler)

	// Create test request
	req := httptest.NewRequest("GET", "/test-path", nil)
	rec := httptest.NewRecorder()

	// Call handler
	handler.ServeHTTP(rec, req)

	// Verify response
	if rec.Code != http.StatusOK {
		t.Errorf("Response code = %d; want %d", rec.Code, http.StatusOK)
	}
	if rec.Body.String() != "OK" {
		t.Errorf("Response body = %q; want %q", rec.Body.String(), "OK")
	}
}

// TestCORSMiddleware tests the CORS middleware
func TestCORSMiddleware(t *testing.T) {
	// Create a test handler
	innerHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("OK"))
	})

	// Wrap with CORS
	handler := CORS(innerHandler)

	// Test regular request
	t.Run("Regular GET request", func(t *testing.T) {
		req := httptest.NewRequest("GET", "/test", nil)
		rec := httptest.NewRecorder()

		handler.ServeHTTP(rec, req)

		// Check CORS headers are set
		if rec.Header().Get("Access-Control-Allow-Origin") != "*" {
			t.Error("Access-Control-Allow-Origin header not set")
		}
		if rec.Header().Get("Access-Control-Allow-Methods") == "" {
			t.Error("Access-Control-Allow-Methods header not set")
		}
		if rec.Code != http.StatusOK {
			t.Errorf("Response code = %d; want %d", rec.Code, http.StatusOK)
		}
	})

	// Test OPTIONS (preflight) request
	t.Run("OPTIONS preflight request", func(t *testing.T) {
		req := httptest.NewRequest("OPTIONS", "/test", nil)
		rec := httptest.NewRecorder()

		handler.ServeHTTP(rec, req)

		// CORS headers should be set
		if rec.Header().Get("Access-Control-Allow-Origin") != "*" {
			t.Error("Access-Control-Allow-Origin header not set for OPTIONS")
		}
		// For OPTIONS, handler should return early (no body written by inner handler)
	})
}

// TestApplyMiddlewares tests the full middleware chain
func TestApplyMiddlewares(t *testing.T) {
	callCount := 0

	// Create a test handler
	innerHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("OK"))
	})

	// Apply all middlewares without protection
	handler := ApplyMiddlewares(innerHandler, RolePublic)

	// Create test request
	req := httptest.NewRequest("GET", "/test", nil)
	rec := httptest.NewRecorder()

	// Call handler
	handler.ServeHTTP(rec, req)

	// Verify handler was called
	if callCount != 1 {
		t.Errorf("Inner handler called %d times; want 1", callCount)
	}

	// Verify CORS headers
	if rec.Header().Get("Access-Control-Allow-Origin") != "*" {
		t.Error("CORS header not set by ApplyMiddlewares")
	}

	// Verify response
	if rec.Code != http.StatusOK {
		t.Errorf("Response code = %d; want %d", rec.Code, http.StatusOK)
	}
}

// TestApplyMiddlewaresWithAuth tests ApplyMiddlewares with auth protection
func TestApplyMiddlewaresWithAuth(t *testing.T) {
	innerCalled := false
	innerHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		innerCalled = true
		w.WriteHeader(http.StatusOK)
	})

	// Setup a mock auth middleware
	authCalled := false
	AuthMiddleware = func(next http.Handler, role AuthRole) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			authCalled = true
			if r.Header.Get("X-Auth") == "valid" {
				next.ServeHTTP(w, r)
			} else {
				w.WriteHeader(http.StatusUnauthorized)
			}
		})
	}
	defer func() { AuthMiddleware = nil }() // Clean up

	t.Run("Protected route - authorized", func(t *testing.T) {
		innerCalled = false
		authCalled = false
		handler := ApplyMiddlewares(innerHandler, RoleAdmin)
		req := httptest.NewRequest("GET", "/test", nil)
		req.Header.Set("X-Auth", "valid")
		rec := httptest.NewRecorder()

		handler.ServeHTTP(rec, req)

		if !authCalled {
			t.Error("Auth middleware was not called for protected route")
		}
		if !innerCalled {
			t.Error("Inner handler was not called for authorized request")
		}
		if rec.Code != http.StatusOK {
			t.Errorf("Code = %d; want %d", rec.Code, http.StatusOK)
		}
	})

	t.Run("Protected route - unauthorized", func(t *testing.T) {
		innerCalled = false
		authCalled = false
		handler := ApplyMiddlewares(innerHandler, RoleAdmin)
		req := httptest.NewRequest("GET", "/test", nil)
		rec := httptest.NewRecorder()

		handler.ServeHTTP(rec, req)

		if !authCalled {
			t.Error("Auth middleware was not called for protected route")
		}
		if innerCalled {
			t.Error("Inner handler was called for unauthorized request")
		}
		if rec.Code != http.StatusUnauthorized {
			t.Errorf("Code = %d; want %d", rec.Code, http.StatusUnauthorized)
		}
	})

	t.Run("Unprotected route", func(t *testing.T) {
		innerCalled = false
		authCalled = false
		handler := ApplyMiddlewares(innerHandler, RolePublic)
		req := httptest.NewRequest("GET", "/test", nil)
		rec := httptest.NewRecorder()

		handler.ServeHTTP(rec, req)

		if authCalled {
			t.Error("Auth middleware was called for unprotected route")
		}
		if !innerCalled {
			t.Error("Inner handler was not called for unprotected route")
		}
		if rec.Code != http.StatusOK {
			t.Errorf("Code = %d; want %d", rec.Code, http.StatusOK)
		}
	})
}

// TestEnableCors tests the enableCors helper function
func TestEnableCors(t *testing.T) {
	rec := httptest.NewRecorder()
	w := http.ResponseWriter(rec)

	enableCors(&w)

	expectedHeaders := map[string]string{
		"Access-Control-Allow-Origin":      "*",
		"Access-Control-Allow-Methods":     "POST, GET, OPTIONS, PUT, DELETE",
		"Access-Control-Allow-Credentials": "true",
		"Access-Control-Expose-Headers":    "Content-Length",
	}

	for header, expected := range expectedHeaders {
		actual := rec.Header().Get(header)
		if actual != expected {
			t.Errorf("Header %q = %q; want %q", header, actual, expected)
		}
	}

	// Check Authorization is in allowed headers
	allowHeaders := rec.Header().Get("Access-Control-Allow-Headers")
	if !strings.Contains(allowHeaders, "Authorization") {
		t.Error("Access-Control-Allow-Headers should contain Authorization")
	}
}

// TestMiddlewareChainOrder tests that middlewares are applied in correct order
func TestMiddlewareChainOrder(t *testing.T) {
	var order []string

	// Create handlers that record their execution
	innerHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		order = append(order, "inner")
		w.WriteHeader(http.StatusOK)
	})

	// Apply middlewares without protection
	handler := ApplyMiddlewares(innerHandler, RolePublic)

	req := httptest.NewRequest("GET", "/test", nil)
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, req)

	// Inner handler should have been called
	if len(order) == 0 || order[len(order)-1] != "inner" {
		t.Error("Inner handler was not called")
	}
}

// TestCORSPOST tests CORS with POST request
func TestCORSPOST(t *testing.T) {
	innerHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusCreated)
		w.Write([]byte(`{"id":"123"}`))
	})

	handler := CORS(innerHandler)

	req := httptest.NewRequest("POST", "/create", strings.NewReader(`{"data":"test"}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusCreated {
		t.Errorf("Response code = %d; want %d", rec.Code, http.StatusCreated)
	}

	// CORS headers should still be set
	if rec.Header().Get("Access-Control-Allow-Origin") != "*" {
		t.Error("CORS header not set for POST request")
	}
}

// TestLoggerWithDifferentMethods tests Logger with various HTTP methods
func TestLoggerWithDifferentMethods(t *testing.T) {
	methods := []string{"GET", "POST", "PUT", "DELETE", "PATCH"}

	for _, method := range methods {
		t.Run(method, func(t *testing.T) {
			innerHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusOK)
			})

			handler := Logger(innerHandler)

			req := httptest.NewRequest(method, "/test", nil)
			rec := httptest.NewRecorder()

			handler.ServeHTTP(rec, req)

			if rec.Code != http.StatusOK {
				t.Errorf("%s request failed with code %d", method, rec.Code)
			}
		})
	}
}
