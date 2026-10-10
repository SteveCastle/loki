package renderer

import (
	"embed"
	"encoding/json"
	"html"
	"html/template"
	"log"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"
)

var (
	templates *template.Template
	once      sync.Once
)

// --------------------------------------------------------------------
// Template embedding
// --------------------------------------------------------------------

//go:embed templates/*.go.html
var templatesFS embed.FS

const templateGlob = "templates/*.go.html"

// formatTime is a helper function that can be called from templates.
// Example usage in template: {{ formatTime .SomeTimeField }}
func formatTime(t time.Time) string {
	return t.Format("Jan 2, 2006 15:04:05")
}

// htmlAttr safely escapes a string for use in HTML attributes
func htmlAttr(s string) string {
	return html.EscapeString(s)
}

// jsonFunc marshals an object to JSON for use in templates
func jsonFunc(v interface{}) (template.JS, error) {
	a, err := json.Marshal(v)
	if err != nil {
		return "", err
	}
	return template.JS(a), nil
}

// progressPct converts done/total into a 0-100 integer percentage for
// rendering job progress bars.
func progressPct(done, total int) int {
	if total <= 0 {
		return 0
	}
	pct := done * 100 / total
	if pct < 0 {
		pct = 0
	}
	if pct > 100 {
		pct = 100
	}
	return pct
}

// initTemplates initializes the templates. Called only once.
func initTemplates() *template.Template {
	tmpl, err := template.New("").
		Funcs(template.FuncMap{
			"formatTime":   formatTime,
			"htmlAttr":     htmlAttr,
			"json":         jsonFunc,
			"jobInputView": jobInputView,
			"progressPct":  progressPct,
		}).
		ParseFS(templatesFS, templateGlob)
	if err != nil {
		log.Fatalf("Error parsing embedded templates: %v", err)
	}
	return tmpl
}

// Templates returns the singleton instance of the parsed templates.
func Templates() *template.Template {
	once.Do(func() { templates = initTemplates() })
	return templates
}

// --------------------------------------------------------------------
// Middleware helpers
// --------------------------------------------------------------------

func Logger(next http.Handler) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		log.Println(time.Since(start), r.Method, r.URL.Path)
	}
}

func CORS(next http.Handler) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		enableCors(&w, r)
		if r.Method == http.MethodOptions {
			return
		}
		next.ServeHTTP(w, r)
	}
}

// AuthRole defines the required access level for a route.
type AuthRole int

const (
	RolePublic AuthRole = iota
	RoleAdmin
	// RolePublicRead marks read-only routes that admins can always reach
	// and anonymous visitors can reach while the AllowPublicAccess config
	// flag is on. The flag check lives in authMiddleware (main*.go) so it
	// is evaluated per request; with the flag off these routes behave
	// exactly like RoleAdmin.
	RolePublicRead
)

// AuthMiddleware is a function that takes a handler and a required role, returning a protected handler.
// This is set from main.go to avoid circular dependencies.
var AuthMiddleware func(http.Handler, AuthRole) http.Handler

func ApplyMiddlewares(handler http.HandlerFunc, role AuthRole) http.HandlerFunc {
	var h http.Handler = handler
	if role != RolePublic && AuthMiddleware != nil {
		h = AuthMiddleware(h, role)
	}
	return Logger(CORS(h))
}

func enableCors(w *http.ResponseWriter, r *http.Request) {
	h := (*w).Header()
	// To use credentials, Access-Control-Allow-Origin cannot be "*"
	// It must be the exact origin of the requesting page.
	// Support browser extensions (chrome-extension://, moz-extension://) and Electron renderer
	origin := r.Header.Get("Origin")
	if IsStudioOrigin(origin) {
		// Lowkey Studio (docs site, local dev server, the viewer's studio://
		// window) talks to the server with an API key in a header, never a
		// cookie: its origin is echoed WITHOUT Allow-Credentials, so a
		// session cookie can never ride one of these cross-origin calls.
		h.Set("Access-Control-Allow-Origin", origin)
		h.Set("Vary", "Origin")
		h.Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
		h.Set("Access-Control-Allow-Headers", "Accept, Content-Type, Authorization, X-API-Key")
		h.Set("Access-Control-Expose-Headers", "Content-Length")
		h.Set("Access-Control-Max-Age", "600")
		// Chrome Private Network Access: the public docs site calling a
		// server on this machine must be explicitly allowed.
		if r.Header.Get("Access-Control-Request-Private-Network") == "true" {
			h.Set("Access-Control-Allow-Private-Network", "true")
		}
		return
	}
	if strings.HasPrefix(origin, "chrome-extension://") || strings.HasPrefix(origin, "moz-extension://") {
		h.Set("Access-Control-Allow-Origin", origin)
	} else {
		// Default to Electron renderer origin
		h.Set("Access-Control-Allow-Origin", "http://localhost:1212")
	}
	h.Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS, PUT, DELETE")
	h.Set("Access-Control-Allow-Headers", "Accept, Content-Type, Content-Length, Accept-Encoding, X-CSRF-Token, Authorization")
	h.Set("Access-Control-Allow-Credentials", "true")
	h.Set("Access-Control-Expose-Headers", "Content-Length")
}

// IsStudioOrigin reports whether a request comes from a Lowkey Studio page:
// the published docs site, the Electron viewer's studio:// window, or a page
// served from this machine (the studio dev server). Those origins get
// header-credential CORS only (see enableCors).
func IsStudioOrigin(origin string) bool {
	// The Electron renderer's dev server is localhost too, but it logs in
	// with the session cookie and needs the credentialed CORS below.
	if origin == "" || origin == "http://localhost:1212" {
		return false
	}
	switch origin {
	case "studio://app", "https://lowkeyviewer.com", "https://www.lowkeyviewer.com":
		return true
	}
	u, err := url.Parse(origin)
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Path != "" {
		return false
	}
	host := u.Hostname()
	return host == "localhost" || host == "127.0.0.1" || host == "::1"
}
