package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// APIError is any non-2xx response from the server.
type APIError struct {
	Status int
	Body   string
	Hint   string
}

func (e *APIError) Error() string {
	body := strings.TrimSpace(e.Body)
	if len(body) > 200 {
		body = body[:200] + "…"
	}
	if body == "" {
		return fmt.Sprintf("server returned HTTP %d", e.Status)
	}
	return fmt.Sprintf("server returned HTTP %d: %s", e.Status, body)
}

// Client is a thin JSON HTTP client for the media server.
type Client struct {
	Base   string
	Token  string
	HTTP   *http.Client // request/response calls, bounded by --timeout
	Stream *http.Client // SSE / long downloads, no timeout
}

func NewClient(base, token string, timeout time.Duration) *Client {
	return &Client{
		Base:   strings.TrimRight(base, "/"),
		Token:  token,
		HTTP:   &http.Client{Timeout: timeout},
		Stream: &http.Client{},
	}
}

func (c *Client) newRequest(method, path, contentType string, body io.Reader) (*http.Request, error) {
	req, err := http.NewRequest(method, c.Base+path, body)
	if err != nil {
		return nil, err
	}
	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}
	// application/json Accept makes the auth middleware answer 401 JSON
	// instead of a 302 to /login.
	req.Header.Set("Accept", "application/json")
	if c.Token != "" {
		req.Header.Set("Authorization", "Bearer "+c.Token)
	}
	return req, nil
}

func hintForStatus(status int) string {
	switch status {
	case http.StatusUnauthorized, http.StatusForbidden:
		return "run: lokictl login --password <password>"
	case http.StatusNotFound:
		return "endpoint or resource not found — the server may be older than this CLI"
	}
	return ""
}

func (c *Client) connectError(err error) error {
	return fmt.Errorf("cannot reach %s: %w — is the media server running? try: lokictl health", c.Base, err)
}

// checkResponse converts a non-2xx response into *APIError (consuming and
// closing the body). For 2xx it returns nil and leaves the body open.
func checkResponse(resp *http.Response) error {
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		return nil
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
	return &APIError{Status: resp.StatusCode, Body: string(b), Hint: hintForStatus(resp.StatusCode)}
}

// DoJSON sends body (marshalled, may be nil) and decodes the 2xx JSON
// response into out (may be nil; 204/empty bodies are skipped).
func (c *Client) DoJSON(method, path string, body, out any) error {
	var reader io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return fmt.Errorf("failed to encode request: %w", err)
		}
		reader = bytes.NewReader(b)
	}
	req, err := c.newRequest(method, path, "application/json", reader)
	if err != nil {
		return err
	}
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return c.connectError(err)
	}
	if err := checkResponse(resp); err != nil {
		return err
	}
	defer resp.Body.Close()
	if out == nil || resp.StatusCode == http.StatusNoContent {
		return nil
	}
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("failed to read response: %w", err)
	}
	if len(bytes.TrimSpace(b)) == 0 {
		return nil
	}
	if err := json.Unmarshal(b, out); err != nil {
		return fmt.Errorf("server sent invalid JSON: %w", err)
	}
	return nil
}

// DoRaw performs a request and returns the raw 2xx response (caller closes
// the body). Non-2xx becomes *APIError.
func (c *Client) DoRaw(method, path, contentType string, body io.Reader) (*http.Response, error) {
	req, err := c.newRequest(method, path, contentType, body)
	if err != nil {
		return nil, err
	}
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return nil, c.connectError(err)
	}
	if err := checkResponse(resp); err != nil {
		return nil, err
	}
	return resp, nil
}

// DoStream is DoRaw on the untimed client, for SSE and long downloads.
func (c *Client) DoStream(method, path string) (*http.Response, error) {
	req, err := c.newRequest(method, path, "", nil)
	if err != nil {
		return nil, err
	}
	resp, err := c.Stream.Do(req)
	if err != nil {
		return nil, c.connectError(err)
	}
	if err := checkResponse(resp); err != nil {
		return nil, err
	}
	return resp, nil
}
