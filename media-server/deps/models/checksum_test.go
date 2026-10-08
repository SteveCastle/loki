package models

import (
	"archive/zip"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// A "latest release" style tool: a zip with the executable in a folder, plus a
// sha256sum sidecar. Installing it extracts the executable and verifies the
// archive against the sidecar.
func TestInstallModel_ToolZipWithChecksumSidecar(t *testing.T) {
	var zbuf bytes.Buffer
	zw := zip.NewWriter(&zbuf)
	for name, body := range map[string]string{
		"demo-windows-amd64/demo.exe":  "BINARY",
		"demo-windows-amd64/README.md": "docs",
	} {
		w, _ := zw.Create(name)
		w.Write([]byte(body))
	}
	zw.Close()
	sum := sha256.Sum256(zbuf.Bytes())
	good := hex.EncodeToString(sum[:]) + "  demo-windows-amd64.zip\n"
	sidecar := good

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasSuffix(r.URL.Path, ".zip"):
			w.Write(zbuf.Bytes())
		case strings.HasSuffix(r.URL.Path, ".sha256"):
			w.Write([]byte(sidecar))
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	dir := t.TempDir()
	SetDataDirForTest(dir)
	t.Cleanup(func() { SetDataDirForTest("") })
	old := Manifest
	defer func() { Manifest = old }()
	Manifest = []Model{{ID: "demo", Category: "tool", Version: "latest", Files: []File{{
		URL: srv.URL + "/demo-windows-amd64.zip", SHA256URL: srv.URL + "/demo-windows-amd64.zip.sha256",
		RelPath: "demo.exe", Archive: "zip", ArchiveMember: "demo-windows-amd64/demo.exe", Exec: true,
	}}}}

	if err := InstallModel(context.Background(), "demo", nil); err != nil {
		t.Fatalf("install: %v", err)
	}
	got, err := os.ReadFile(filepath.Join(dir, "models", "demo", "demo.exe"))
	if err != nil || string(got) != "BINARY" {
		t.Fatalf("exe = %q, %v", got, err)
	}

	// A sidecar that disagrees with the archive must fail the install.
	os.RemoveAll(filepath.Join(dir, "models", "demo"))
	sidecar = strings.Repeat("0", 64) + "  demo-windows-amd64.zip\n"
	if err := InstallModel(context.Background(), "demo", nil); err == nil || !strings.Contains(err.Error(), "checksum") {
		t.Fatalf("want a checksum mismatch, got %v", err)
	}

	// A garbage sidecar (e.g. an HTML error page) is rejected before downloading.
	sidecar = "<html>nope</html>"
	if err := InstallModel(context.Background(), "demo", nil); err == nil || !strings.Contains(err.Error(), "not a sha256") {
		t.Fatalf("want a bad-sidecar error, got %v", err)
	}
}
