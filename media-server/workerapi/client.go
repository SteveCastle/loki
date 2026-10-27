package workerapi

import (
	"archive/tar"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// Node is a client of one loki-worker served over HTTP (a persistent node:
// a pod, a VM, a container, or a load-balanced serverless endpoint that
// keeps a job on one replica).
type Node struct {
	Name  string
	URL   string // e.g. https://abc123-8080.proxy.runpod.net
	Token string
	HTTP  *http.Client
}

// ErrUnauthorized is a 401/403 from the worker (wrong token).
var ErrUnauthorized = errors.New("the worker rejected the token")

// HTTPError is a non-2xx answer.
type HTTPError struct {
	Status int
	Body   string
}

func (e *HTTPError) Error() string {
	return fmt.Sprintf("worker answered %d: %s", e.Status, strings.TrimSpace(e.Body))
}

func (n *Node) client() *http.Client {
	if n.HTTP != nil {
		return n.HTTP
	}
	// No overall timeout: uploads of large inputs take as long as they take;
	// every call carries its own context deadline instead.
	return http.DefaultClient
}

func (n *Node) endpoint(p string) string {
	return strings.TrimRight(n.URL, "/") + p
}

func (n *Node) newRequest(ctx context.Context, method, p string, body io.Reader) (*http.Request, error) {
	req, err := http.NewRequestWithContext(ctx, method, n.endpoint(p), body)
	if err != nil {
		return nil, err
	}
	if n.Token != "" {
		req.Header.Set("Authorization", "Bearer "+n.Token)
	}
	return req, nil
}

func (n *Node) send(req *http.Request, out any) error {
	resp, err := n.client().Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden {
		return ErrUnauthorized
	}
	if resp.StatusCode/100 != 2 {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return &HTTPError{Status: resp.StatusCode, Body: string(b)}
	}
	if out == nil {
		_, _ = io.Copy(io.Discard, resp.Body)
		return nil
	}
	return json.NewDecoder(resp.Body).Decode(out)
}

func (n *Node) doJSON(ctx context.Context, method, p string, in, out any, timeout time.Duration) error {
	if timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, timeout)
		defer cancel()
	}
	var body io.Reader
	if in != nil {
		b, err := json.Marshal(in)
		if err != nil {
			return err
		}
		body = bytes.NewReader(b)
	}
	req, err := n.newRequest(ctx, method, p, body)
	if err != nil {
		return err
	}
	if in != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	return n.send(req, out)
}

// Ping reports whether the worker is up and ready (GET /ping, no auth).
func (n *Node) Ping(ctx context.Context) error {
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	req, err := n.newRequest(ctx, http.MethodGet, "/ping", nil)
	if err != nil {
		return err
	}
	resp, err := n.client().Do(req)
	if err != nil {
		return err
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("worker not ready (%d)", resp.StatusCode)
	}
	return nil
}

// Info fetches the worker's self-description; help adds each engine's --help.
func (n *Node) Info(ctx context.Context, help bool) (*Info, error) {
	p := "/v1/info"
	if help {
		p += "?help=1"
	}
	var info Info
	if err := n.doJSON(ctx, http.MethodGet, p, nil, &info, 20*time.Second); err != nil {
		return nil, err
	}
	return &info, nil
}

// Submit queues a job, uploading files (remote path -> local file) in the
// same multipart request; large files stream, nothing is buffered.
func (n *Node) Submit(ctx context.Context, spec JobSpec, files []LocalInput) (*JobStatus, error) {
	pr, pw := io.Pipe()
	mw := multipart.NewWriter(pw)
	go func() {
		err := func() error {
			sj, err := json.Marshal(spec)
			if err != nil {
				return err
			}
			if err := mw.WriteField("spec", string(sj)); err != nil {
				return err
			}
			for _, f := range files {
				w, err := mw.CreateFormFile(f.Remote, filepath.Base(f.Remote))
				if err != nil {
					return err
				}
				fh, err := os.Open(f.Local)
				if err != nil {
					return err
				}
				_, err = io.Copy(w, fh)
				fh.Close()
				if err != nil {
					return err
				}
			}
			return mw.Close()
		}()
		pw.CloseWithError(err)
	}()
	req, err := n.newRequest(ctx, http.MethodPost, "/v1/jobs", pr)
	if err != nil {
		pr.Close()
		return nil, err
	}
	req.Header.Set("Content-Type", mw.FormDataContentType())
	var st JobStatus
	if err := n.send(req, &st); err != nil {
		pr.CloseWithError(err)
		return nil, err
	}
	return &st, nil
}

// Status polls a job; logs is the first log index wanted.
func (n *Node) Status(ctx context.Context, id string, logs int) (*JobStatus, error) {
	var st JobStatus
	err := n.doJSON(ctx, http.MethodGet, "/v1/jobs/"+url.PathEscape(id)+"?logs="+strconv.Itoa(logs), nil, &st, 30*time.Second)
	if err != nil {
		return nil, err
	}
	return &st, nil
}

// Cancel asks the worker to stop a job.
func (n *Node) Cancel(ctx context.Context, id string) error {
	return n.doJSON(ctx, http.MethodPost, "/v1/jobs/"+url.PathEscape(id)+"/cancel", nil, nil, 15*time.Second)
}

// Delete drops a finished job and its files from the worker.
func (n *Node) Delete(ctx context.Context, id string) error {
	return n.doJSON(ctx, http.MethodDelete, "/v1/jobs/"+url.PathEscape(id), nil, nil, 15*time.Second)
}

// Warm asks the worker to fetch and load engines' models in the background.
func (n *Node) Warm(ctx context.Context, w WarmRequest) error {
	return n.doJSON(ctx, http.MethodPost, "/v1/warm", w, nil, 15*time.Second)
}

// Outputs downloads a job's output files as a tar stream, handing each file
// (path relative to the job directory) to fn.
func (n *Node) Outputs(ctx context.Context, id string, fn func(rel string, r io.Reader) error) error {
	return n.tarGet(ctx, "/v1/jobs/"+url.PathEscape(id)+"/outputs.tar", fn)
}

func (n *Node) tarGet(ctx context.Context, p string, fn func(rel string, r io.Reader) error) error {
	req, err := n.newRequest(ctx, http.MethodGet, p, nil)
	if err != nil {
		return err
	}
	resp, err := n.client().Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return &HTTPError{Status: resp.StatusCode, Body: string(b)}
	}
	return ReadTar(resp.Body, fn)
}

// --- sessions

// OpenSession starts an interactive session on the worker's engine.
func (n *Node) OpenSession(ctx context.Context, tool string) (string, error) {
	var r struct {
		ID string `json:"id"`
	}
	if err := n.doJSON(ctx, http.MethodPost, "/v1/sessions", map[string]string{"tool": EngineName(tool)}, &r, 30*time.Second); err != nil {
		return "", err
	}
	return r.ID, nil
}

// PutSessionFile uploads a local file to rel inside the session directory.
func (n *Node) PutSessionFile(ctx context.Context, id, rel, local string) error {
	f, err := os.Open(local)
	if err != nil {
		return err
	}
	defer f.Close()
	req, err := n.newRequest(ctx, http.MethodPut, "/v1/sessions/"+url.PathEscape(id)+"/files/"+rel, f)
	if err != nil {
		return err
	}
	if st, err := f.Stat(); err == nil {
		req.ContentLength = st.Size()
	}
	return n.send(req, nil)
}

// CallSession sends one request to the session's engine.
func (n *Node) CallSession(ctx context.Context, id string, req map[string]any, timeout time.Duration) (*SessionReply, error) {
	var r SessionReply
	if err := n.doJSON(ctx, http.MethodPost, "/v1/sessions/"+url.PathEscape(id)+"/call", SessionCall{Req: req}, &r, timeout); err != nil {
		return nil, err
	}
	return &r, nil
}

// SessionFiles downloads the session files under prefix as a tar stream.
func (n *Node) SessionFiles(ctx context.Context, id, prefix string, fn func(rel string, r io.Reader) error) error {
	return n.tarGet(ctx, "/v1/sessions/"+url.PathEscape(id)+"/files.tar?prefix="+url.QueryEscape(prefix), fn)
}

// CloseSession ends a session and drops its files.
func (n *Node) CloseSession(ctx context.Context, id string) error {
	return n.doJSON(ctx, http.MethodDelete, "/v1/sessions/"+url.PathEscape(id), nil, nil, 15*time.Second)
}

// --- tar

// WriteTar streams files (relative path -> local file) as a tar archive.
func WriteTar(w io.Writer, root string, rels []string) error {
	tw := tar.NewWriter(w)
	for _, rel := range rels {
		p := filepath.Join(root, filepath.FromSlash(rel))
		st, err := os.Stat(p)
		if err != nil || !st.Mode().IsRegular() {
			continue
		}
		if err := tw.WriteHeader(&tar.Header{Name: filepath.ToSlash(rel), Mode: 0644, Size: st.Size(), ModTime: st.ModTime(), Typeflag: tar.TypeReg}); err != nil {
			return err
		}
		f, err := os.Open(p)
		if err != nil {
			return err
		}
		_, err = io.Copy(tw, f)
		f.Close()
		if err != nil {
			return err
		}
	}
	return tw.Close()
}

// ReadTar hands every regular file of a tar stream to fn (names are
// cleaned; anything escaping the archive root is refused).
func ReadTar(r io.Reader, fn func(rel string, r io.Reader) error) error {
	tr := tar.NewReader(r)
	for {
		h, err := tr.Next()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		if h.Typeflag != tar.TypeReg {
			continue
		}
		rel, ok := CleanRel(h.Name)
		if !ok {
			return fmt.Errorf("refusing archive entry %q", h.Name)
		}
		if err := fn(rel, tr); err != nil {
			return err
		}
	}
}

// CleanRel cleans a slash-separated relative path; false when it is absolute
// or climbs out of its root.
func CleanRel(p string) (string, bool) {
	p = filepath.ToSlash(p)
	if p == "" || strings.HasPrefix(p, "/") || (len(p) > 1 && p[1] == ':') {
		return "", false
	}
	c := filepath.ToSlash(filepath.Clean(filepath.FromSlash(p)))
	if c == "." || c == ".." || strings.HasPrefix(c, "../") {
		return "", false
	}
	return c, true
}

// WriteFileAtomic copies r to path through a temp file beside it, so a
// reader never sees half a file.
func WriteFileAtomic(path string, r io.Reader) (int64, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return 0, err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".lw-*"+filepath.Ext(path))
	if err != nil {
		return 0, err
	}
	n, err := io.Copy(tmp, r)
	if cerr := tmp.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		os.Remove(tmp.Name())
		return n, err
	}
	if err := os.Rename(tmp.Name(), path); err != nil {
		// Windows: rename onto an existing file fails
		_ = os.Remove(path)
		if err := os.Rename(tmp.Name(), path); err != nil {
			os.Remove(tmp.Name())
			return n, err
		}
	}
	return n, nil
}
