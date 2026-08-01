package main

import (
	"database/sql"
	_ "embed"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// -----------------------------------------------------------------------------
// Path-coverage treemap (shared across all platform mains).
//
//   GET /viz/treemap           — WinDirStat-style squarified treemap page
//   GET /api/treemap           — per-directory row counts for one metric
//   GET /api/treemap/metrics   — available metrics with library-wide row counts
//
// Unlike a disk-usage treemap, every metric here is a row count: items in the
// media table, tag rows, and per-model rows in the embedding and face tables.
// The point is to see where the library's items live and how much of each
// directory the indexes actually cover.
// -----------------------------------------------------------------------------

//go:embed vizstatic/treemap.html
var treemapVizHTML []byte

// Metric ids are either a fixed table ("items", "tags") or a model-scoped
// prefix into the two model-keyed tables ("embed:<model>", "face:<model>").
func validTreemapMetric(metric string) bool {
	return metric == "items" || metric == "tags" ||
		strings.HasPrefix(metric, "embed:") || strings.HasPrefix(metric, "face:")
}

// pathTreeNode aggregates rows for one directory. count is total rows in the
// subtree; covered is distinct media items contributing rows (every stored
// path is visited exactly once per build, so +1 per path is exact); direct is
// rows whose file lives immediately in this directory rather than a subdir.
type pathTreeNode struct {
	name     string
	path     string // prefix exactly as stored, so drill-down keys stay valid
	count    int64
	covered  int64
	direct   int64
	children map[string]*pathTreeNode
}

// insert adds one file's rows to every directory on its path. Segments split
// on both separators so Windows, POSIX, and s3://-style paths all nest; each
// node keeps the byte-exact prefix of the original path (p[:i]) so mixed
// separators survive round-trips through the API.
func (n *pathTreeNode) insert(p string, rows int64) {
	node := n
	node.count += rows
	node.covered++
	start := 0
	for i := 0; i < len(p); i++ {
		if p[i] != '/' && p[i] != '\\' {
			continue
		}
		if seg := p[start:i]; seg != "" {
			child := node.children[seg]
			if child == nil {
				child = &pathTreeNode{name: seg, path: p[:i], children: map[string]*pathTreeNode{}}
				node.children[seg] = child
			}
			child.count += rows
			child.covered++
			node = child
		}
		start = i + 1
	}
	// The final segment is the file name — it never becomes a node.
	node.direct += rows
}

// find walks q segment-by-segment (same tokenizer as insert, but here the
// final segment is a directory too). Returns nil when q isn't in the tree.
func (n *pathTreeNode) find(q string) *pathTreeNode {
	node := n
	start := 0
	for i := 0; i <= len(q); i++ {
		if i < len(q) && q[i] != '/' && q[i] != '\\' {
			continue
		}
		if seg := q[start:i]; seg != "" {
			child := node.children[seg]
			if child == nil {
				return nil
			}
			node = child
		}
		start = i + 1
	}
	return node
}

func buildPathTree(db *sql.DB, metric string) (*pathTreeNode, error) {
	var (
		rows     *sql.Rows
		err      error
		hasCount bool
	)
	switch {
	case metric == "items":
		rows, err = db.Query(`SELECT path FROM media`)
	case metric == "tags":
		// GROUP BY rides the (media_path, ...) PK, and one row per item is
		// what keeps `covered` exact.
		hasCount = true
		rows, err = db.Query(
			`SELECT media_path, COUNT(*) FROM media_tag_by_category GROUP BY media_path`)
	case strings.HasPrefix(metric, "embed:"):
		rows, err = db.Query(
			`SELECT media_path FROM media_embedding WHERE model = ?`,
			strings.TrimPrefix(metric, "embed:"))
	case strings.HasPrefix(metric, "face:"):
		hasCount = true
		rows, err = db.Query(
			`SELECT media_path, COUNT(*) FROM face WHERE model = ? GROUP BY media_path`,
			strings.TrimPrefix(metric, "face:"))
	}
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	root := &pathTreeNode{children: map[string]*pathTreeNode{}}
	for rows.Next() {
		var p string
		n := int64(1)
		if hasCount {
			err = rows.Scan(&p, &n)
		} else {
			err = rows.Scan(&p)
		}
		if err != nil {
			return nil, err
		}
		root.insert(p, n)
	}
	return root, rows.Err()
}

const (
	treemapCacheTTL = 2 * time.Minute
	treemapCacheMax = 16 // metrics are few; this only bounds pathological input
)

type treemapCacheEntry struct {
	root    *pathTreeNode
	builtAt time.Time
}

type treemapAPI struct {
	deps  *Dependencies
	mu    sync.Mutex
	trees map[string]treemapCacheEntry
}

func newTreemapAPI(deps *Dependencies) *treemapAPI {
	return &treemapAPI{deps: deps, trees: map[string]treemapCacheEntry{}}
}

// tree returns the cached aggregate for metric, rebuilding after the TTL. The
// lock is held across the build: concurrent requests briefly serialize, which
// beats racing identical full-table scans on the shared SQLite connection.
func (a *treemapAPI) tree(metric string) (*pathTreeNode, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if e, ok := a.trees[metric]; ok && time.Since(e.builtAt) < treemapCacheTTL {
		return e.root, nil
	}
	root, err := buildPathTree(a.deps.DB, metric)
	if err != nil {
		return nil, err
	}
	if len(a.trees) >= treemapCacheMax {
		oldest, first := "", true
		var oldestAt time.Time
		for k, e := range a.trees {
			if first || e.builtAt.Before(oldestAt) {
				oldest, oldestAt, first = k, e.builtAt, false
			}
		}
		delete(a.trees, oldest)
	}
	a.trees[metric] = treemapCacheEntry{root: root, builtAt: time.Now()}
	return root, nil
}

type treemapNode struct {
	Name       string         `json:"name"`
	Path       string         `json:"path"`
	Count      int64          `json:"count"`
	Covered    int64          `json:"covered"`
	Direct     int64          `json:"direct,omitempty"`
	ItemCount  int64          `json:"itemCount,omitempty"`
	Children   []*treemapNode `json:"children,omitempty"`
	OtherCount int64          `json:"otherCount,omitempty"`
	OtherDirs  int            `json:"otherDirs,omitempty"`
}

// serializeTree flattens node down to depth levels. Children go largest-first;
// past limit they collapse into OtherCount/OtherDirs so a directory of 10k
// siblings doesn't produce a 10k-element payload. items is the matching node
// in the items tree (nil for the items metric itself, or when a directory only
// holds rows for media no longer in the library) — it feeds the coverage
// denominator on every node.
func serializeTree(node, items *pathTreeNode, depth, limit int) *treemapNode {
	out := &treemapNode{
		Name:    node.name,
		Path:    node.path,
		Count:   node.count,
		Covered: node.covered,
		Direct:  node.direct,
	}
	if items != nil {
		out.ItemCount = items.count
	}
	if depth <= 0 {
		return out
	}
	kids := make([]*pathTreeNode, 0, len(node.children))
	for _, c := range node.children {
		kids = append(kids, c)
	}
	sort.Slice(kids, func(i, j int) bool {
		if kids[i].count != kids[j].count {
			return kids[i].count > kids[j].count
		}
		return kids[i].name < kids[j].name
	})
	for i, c := range kids {
		if i >= limit {
			out.OtherCount += c.count
			out.OtherDirs++
			continue
		}
		var ic *pathTreeNode
		if items != nil {
			ic = items.children[c.name]
		}
		out.Children = append(out.Children, serializeTree(c, ic, depth-1, limit))
	}
	return out
}

func (a *treemapAPI) dataHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			httpError(w, "use GET", http.StatusMethodNotAllowed)
			return
		}
		q := r.URL.Query()
		metric := q.Get("metric")
		if metric == "" {
			metric = "items"
		}
		if !validTreemapMetric(metric) {
			httpError(w, "unknown metric", http.StatusBadRequest)
			return
		}
		depth := 3
		if v := q.Get("depth"); v != "" {
			n, err := strconv.Atoi(v)
			if err != nil || n < 1 {
				httpError(w, "invalid depth", http.StatusBadRequest)
				return
			}
			depth = min(n, 8)
		}
		limit := 100
		if v := q.Get("limit"); v != "" {
			n, err := strconv.Atoi(v)
			if err != nil || n < 1 {
				httpError(w, "invalid limit", http.StatusBadRequest)
				return
			}
			limit = min(n, 2000)
		}

		root, err := a.tree(metric)
		if err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		path := q.Get("path")
		node := root.find(path)
		if node == nil {
			httpError(w, "path not found", http.StatusNotFound)
			return
		}
		var itemsNode *pathTreeNode
		resp := map[string]any{
			"metric": metric,
			"path":   path,
			"total":  root.count,
		}
		if metric != "items" {
			itemsRoot, err := a.tree("items")
			if err != nil {
				httpError(w, err.Error(), http.StatusInternalServerError)
				return
			}
			itemsNode = itemsRoot.find(path)
			resp["totalItems"] = itemsRoot.count
		}
		resp["node"] = serializeTree(node, itemsNode, depth, limit)
		writeJSON(w, resp)
	}
}

func treemapMetricsHandler(deps *Dependencies) http.HandlerFunc {
	type metricInfo struct {
		ID    string `json:"id"`
		Label string `json:"label"`
		Count int64  `json:"count"`
	}
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			httpError(w, "use GET", http.StatusMethodNotAllowed)
			return
		}
		var out []metricInfo
		var n int64
		if err := deps.DB.QueryRow(`SELECT COUNT(*) FROM media`).Scan(&n); err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		out = append(out, metricInfo{"items", "Items", n})
		if err := deps.DB.QueryRow(`SELECT COUNT(*) FROM media_tag_by_category`).Scan(&n); err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		out = append(out, metricInfo{"tags", "Tags", n})

		perModel := func(query, prefix, label string) error {
			rows, err := deps.DB.Query(query)
			if err != nil {
				return err
			}
			defer rows.Close()
			for rows.Next() {
				var model string
				var c int64
				if err := rows.Scan(&model, &c); err != nil {
					return err
				}
				out = append(out, metricInfo{prefix + model, label + " · " + model, c})
			}
			return rows.Err()
		}
		if err := perModel(
			`SELECT model, COUNT(*) FROM media_embedding GROUP BY model ORDER BY model`,
			"embed:", "Embeddings"); err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		if err := perModel(
			`SELECT model, COUNT(*) FROM face GROUP BY model ORDER BY model`,
			"face:", "Faces"); err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		writeJSON(w, map[string]any{"metrics": out})
	}
}

func treemapPageHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			httpError(w, "use GET", http.StatusMethodNotAllowed)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Write(treemapVizHTML)
	}
}
