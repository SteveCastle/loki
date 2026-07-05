package tasks

import (
	"archive/zip"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

// DirectML GPU runtime sources. The ONNX Runtime DirectML build must match the
// onnxruntime_go binding's API version (ORT 1.22.0); that package depends on
// Microsoft.AI.DirectML 1.15.4 for DirectML.dll. Both NuGet packages are plain
// zips served at stable URLs.
const (
	ortDirectMLURL    = "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.DirectML/1.22.0"
	directMLURL       = "https://www.nuget.org/api/v2/package/Microsoft.AI.DirectML/1.15.4"
	ortDirectMLMember = "runtimes/win-x64/native/onnxruntime.dll"
	directMLMember    = "bin/x64-win/DirectML.dll"
)

// InstallDirectMLRuntime downloads the GPU (DirectML) ONNX Runtime + DirectML.dll
// and installs them into DirectMLRuntimeDir() so the embed task can use
// provider=directml. Windows-only. `log` receives human-readable progress lines
// (may be nil).
func InstallDirectMLRuntime(log func(string)) error {
	logf := func(format string, a ...interface{}) {
		if log != nil {
			log(fmt.Sprintf(format, a...))
		}
	}
	if runtime.GOOS != "windows" {
		return fmt.Errorf("DirectML is only available on Windows")
	}
	dir := DirectMLRuntimeDir()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}

	logf("Downloading ONNX Runtime (DirectML) …")
	if err := downloadZipMember(ortDirectMLURL, ortDirectMLMember, filepath.Join(dir, "onnxruntime.dll")); err != nil {
		return fmt.Errorf("onnxruntime (directml): %w", err)
	}
	logf("Downloading DirectML.dll …")
	if err := downloadZipMember(directMLURL, directMLMember, filepath.Join(dir, "DirectML.dll")); err != nil {
		return fmt.Errorf("directml.dll: %w", err)
	}
	logf("DirectML runtime installed to %s", dir)
	return nil
}

// downloadZipMember fetches a zip (NuGet .nupkg) from url and extracts the named
// member to dst atomically (write to dst.partial, then rename).
func downloadZipMember(url, member, dst string) error {
	client := &http.Client{Timeout: 5 * time.Minute}
	resp, err := client.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("HTTP %d fetching %s", resp.StatusCode, url)
	}

	// zip needs random access; buffer the package to a temp file.
	tmpPkg, err := os.CreateTemp("", "pkg-*.zip")
	if err != nil {
		return err
	}
	tmpPkgName := tmpPkg.Name()
	defer os.Remove(tmpPkgName)
	if _, err := io.Copy(tmpPkg, resp.Body); err != nil {
		tmpPkg.Close()
		return err
	}
	if err := tmpPkg.Close(); err != nil {
		return err
	}

	zr, err := zip.OpenReader(tmpPkgName)
	if err != nil {
		return err
	}
	defer zr.Close()

	for _, f := range zr.File {
		if !strings.EqualFold(f.Name, member) {
			continue
		}
		rc, err := f.Open()
		if err != nil {
			return err
		}
		defer rc.Close()
		partial := dst + ".partial"
		out, err := os.Create(partial)
		if err != nil {
			return err
		}
		if _, err := io.Copy(out, rc); err != nil {
			out.Close()
			os.Remove(partial)
			return err
		}
		if err := out.Close(); err != nil {
			os.Remove(partial)
			return err
		}
		return os.Rename(partial, dst)
	}
	return fmt.Errorf("member %q not found in package", member)
}
