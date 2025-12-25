package renderer

import (
	"embed"
	"encoding/json"
	"html"
	"html/template"
	"log"
	"net/http"
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

// initTemplates initializes the templates. Called only once.
func initTemplates() *template.Template {
	tmpl, err := template.New("").
		Funcs(template.FuncMap{
			"formatTime": formatTime,
			"htmlAttr":   htmlAttr,
			"json":       jsonFunc,
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
		enableCors(&w)
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

func enableCors(w *http.ResponseWriter) {
	h := (*w).Header()
	h.Set("Access-Control-Allow-Origin", "*")
	h.Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS, PUT, DELETE")
	h.Set("Access-Control-Allow-Headers", "Accept, Content-Type, Content-Length, Accept-Encoding, X-CSRF-Token, Authorization")
	h.Set("Access-Control-Allow-Credentials", "true")
	h.Set("Access-Control-Expose-Headers", "Content-Length")
}
