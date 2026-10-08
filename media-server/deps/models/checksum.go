package models

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strings"
	"time"
)

var sha256SumRE = regexp.MustCompile(`^[0-9a-fA-F]{64}$`)

// resolveChecksum returns the SHA-256 to verify f against. A pinned
// File.SHA256 wins; otherwise File.SHA256URL names a sidecar (the usual
// `sha256sum` format, "<hex>  <name>") published next to the download. That
// lets a "latest release" asset be installed without the server having to be
// rebuilt around each new hash: the sidecar protects against truncated or
// corrupt downloads (it comes from the same origin, so it is not a defence
// against a compromised release).
func resolveChecksum(ctx context.Context, f File) (string, error) {
	if f.SHA256 != "" || f.SHA256URL == "" {
		return f.SHA256, nil
	}
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, f.SHA256URL, nil)
	if err != nil {
		return "", err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("fetch checksum %s: %w", f.SHA256URL, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", &httpStatusError{Status: resp.StatusCode, URL: f.SHA256URL}
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, 4096))
	if err != nil {
		return "", err
	}
	fields := strings.Fields(string(body))
	if len(fields) == 0 || !sha256SumRE.MatchString(fields[0]) {
		return "", fmt.Errorf("checksum file %s is not a sha256 sum", f.SHA256URL)
	}
	return strings.ToLower(fields[0]), nil
}
