package main

import (
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strings"
	"testing"
)

// sourceDir is this package's directory. Paths can't be relative to the
// working directory: the package init() chdirs to the executable's folder,
// which under go test is a temporary build directory.
func sourceDir() string {
	_, file, _, _ := runtime.Caller(0)
	return filepath.Dir(file)
}

// openAPISpec is the published HTTP API spec (GitHub Pages, docs/server/),
// rendered into docs/server/api.html by scripts/build-api-docs.js.
func openAPISpec() string {
	return filepath.Join(sourceDir(), "..", "docs", "server", "openapi.yaml")
}

var (
	routeRE    = regexp.MustCompile(`(?:HandleFunc|\.Handle)\("([^"]+)"`)
	specPathRE = regexp.MustCompile(`(?m)^  (/[^:\s]*):\s*$`)
)

// registeredRoutes collects every mux pattern registered by the server's own
// sources, across all three platform mains, with any "METHOD " prefix dropped.
func registeredRoutes(t *testing.T) map[string][]string {
	t.Helper()
	files, err := filepath.Glob(filepath.Join(sourceDir(), "*.go"))
	if err != nil {
		t.Fatal(err)
	}
	routes := map[string][]string{}
	for _, f := range files {
		if strings.HasSuffix(f, "_test.go") {
			continue
		}
		src, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		for _, m := range routeRE.FindAllStringSubmatch(string(src), -1) {
			p := m[1]
			if i := strings.Index(p, " "); i >= 0 {
				p = p[i+1:]
			}
			if !strings.HasPrefix(p, "/") {
				continue
			}
			routes[p] = append(routes[p], filepath.Base(f))
		}
	}
	return routes
}

func segments(p string) []string { return strings.Split(strings.Trim(p, "/"), "/") }

func isParam(s string) bool { return strings.HasPrefix(s, "{") && strings.HasSuffix(s, "}") }

// routeMatchesSpec: a registered pattern covers a spec path when their
// segments line up, a {param} on either side matching any segment. A
// registered subtree pattern ("/media/hls/") covers every spec path under it.
func routeMatchesSpec(route, spec string) bool {
	if route == "/" {
		return spec == "/"
	}
	if strings.HasSuffix(route, "/") {
		return strings.HasPrefix(spec+"/", route) && spec+"/" != route || spec == strings.TrimSuffix(route, "/")
	}
	rs, ss := segments(route), segments(spec)
	if len(rs) != len(ss) {
		return false
	}
	for i := range rs {
		if rs[i] != ss[i] && !isParam(rs[i]) && !isParam(ss[i]) {
			return false
		}
	}
	return true
}

// TestOpenAPICoversEveryRoute keeps docs/server/openapi.yaml (and so the
// generated api.html) complete and free of stale endpoints: every route the
// server registers must be documented, and every documented path must be
// served. Adding a route means adding it to the spec and running
// node scripts/build-api-docs.js.
func TestOpenAPICoversEveryRoute(t *testing.T) {
	spec := openAPISpec()
	raw, err := os.ReadFile(spec)
	if err != nil {
		t.Skipf("spec not available (%v)", err)
	}
	var specPaths []string
	for _, m := range specPathRE.FindAllStringSubmatch(string(raw), -1) {
		specPaths = append(specPaths, m[1])
	}
	if len(specPaths) == 0 {
		t.Fatalf("no paths found in %s", spec)
	}
	routes := registeredRoutes(t)
	if len(routes) < 100 {
		t.Fatalf("found only %d registered routes; the source scan is broken", len(routes))
	}

	var missing []string
	for r, files := range routes {
		found := false
		for _, s := range specPaths {
			if routeMatchesSpec(r, s) {
				found = true
				break
			}
		}
		if !found {
			missing = append(missing, r+" ("+strings.Join(files, ", ")+")")
		}
	}
	sort.Strings(missing)
	for _, m := range missing {
		t.Errorf("route %s is not documented in docs/server/openapi.yaml", m)
	}

	for _, s := range specPaths {
		served := false
		for r := range routes {
			if routeMatchesSpec(r, s) {
				served = true
				break
			}
		}
		if !served {
			t.Errorf("docs/server/openapi.yaml documents %s, which no route serves", s)
		}
	}
}
