package main

import (
	"compress/gzip"
	"net"
	"net/http"
	"strings"
	"sync"
)

// withCompression gzips text-like responses (JSON, JS, CSS, HTML, SVG) for
// clients that accept it. Library listings are path-heavy JSON that shrink
// ~8-10x, and the SPA bundle's multi-MB chunks were sent raw on every load.
//
// Left alone: loopback clients (the local viewer gains nothing and would pay
// the CPU), SSE (must reach the client unbuffered), range/partial responses
// and media (already compressed, and ServeContent's byte ranges must match
// the file), HEAD, and anything already carrying a Content-Encoding. The
// decision is made when the handler commits its headers, from the
// Content-Type it set.
func withCompression(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodHead ||
			!strings.Contains(r.Header.Get("Accept-Encoding"), "gzip") ||
			isLoopbackRequest(r) {
			next.ServeHTTP(w, r)
			return
		}
		gw := &gzipResponseWriter{ResponseWriter: w}
		defer func() {
			if rec := recover(); rec != nil {
				// Don't finish the gzip stream of an aborted response: the
				// client must see the failure, not a well-formed body.
				panic(rec)
			}
			gw.close()
		}()
		next.ServeHTTP(gw, r)
	})
}

func isLoopbackRequest(r *http.Request) bool {
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		host = r.RemoteAddr
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

func compressibleType(ct string) bool {
	ct = strings.ToLower(ct)
	if i := strings.IndexByte(ct, ';'); i >= 0 {
		ct = ct[:i]
	}
	ct = strings.TrimSpace(ct)
	switch {
	case ct == "text/event-stream":
		return false
	case strings.HasPrefix(ct, "text/"),
		ct == "application/json",
		ct == "application/javascript",
		ct == "application/x-javascript",
		ct == "application/xml",
		ct == "application/manifest+json",
		ct == "image/svg+xml",
		ct == "application/vnd.apple.mpegurl": // HLS playlists, not segments
		return true
	}
	return false
}

var gzipWriterPool = sync.Pool{New: func() any {
	gz, _ := gzip.NewWriterLevel(nil, gzip.BestSpeed)
	return gz
}}

type gzipResponseWriter struct {
	http.ResponseWriter
	gz      *gzip.Writer
	decided bool
}

func (g *gzipResponseWriter) decide(status int) {
	if g.decided {
		return
	}
	g.decided = true
	h := g.ResponseWriter.Header()
	if status < 200 || status == http.StatusNoContent || status == http.StatusNotModified ||
		status == http.StatusPartialContent || h.Get("Content-Encoding") != "" ||
		h.Get("Content-Range") != "" || !compressibleType(h.Get("Content-Type")) {
		return
	}
	h.Del("Content-Length")
	h.Set("Content-Encoding", "gzip")
	h.Add("Vary", "Accept-Encoding")
	g.gz = gzipWriterPool.Get().(*gzip.Writer)
	g.gz.Reset(g.ResponseWriter)
}

func (g *gzipResponseWriter) WriteHeader(status int) {
	g.decide(status)
	g.ResponseWriter.WriteHeader(status)
}

func (g *gzipResponseWriter) Write(b []byte) (int, error) {
	if !g.decided {
		// Handlers that never set a Content-Type (templates) get one sniffed
		// on first write by net/http; do the same so the decision sees it.
		if h := g.ResponseWriter.Header(); h.Get("Content-Type") == "" {
			h.Set("Content-Type", http.DetectContentType(b))
		}
		g.WriteHeader(http.StatusOK)
	}
	if g.gz != nil {
		return g.gz.Write(b)
	}
	return g.ResponseWriter.Write(b)
}

func (g *gzipResponseWriter) Flush() {
	if !g.decided {
		g.WriteHeader(http.StatusOK)
	}
	if g.gz != nil {
		_ = g.gz.Flush()
	}
	if f, ok := g.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

// Unwrap lets http.ResponseController reach the underlying writer.
func (g *gzipResponseWriter) Unwrap() http.ResponseWriter { return g.ResponseWriter }

func (g *gzipResponseWriter) close() {
	if g.gz == nil {
		return
	}
	_ = g.gz.Close()
	g.gz.Reset(nil)
	gzipWriterPool.Put(g.gz)
	g.gz = nil
}
