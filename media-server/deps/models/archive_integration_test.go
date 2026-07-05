package models

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

// TestExtractSevenZipDirRealBundle extracts a real Faster-Whisper-XXL release
// archive (solid LZMA2 + BCJ2, ~4.7 GB uncompressed) through the same code
// path the installer uses. It only runs when LOKI_XXL_ARCHIVE points at a
// downloaded copy, e.g.:
//
//	LOKI_XXL_ARCHIVE=/path/to/Faster-Whisper-XXL_r245.4_windows.7z \
//	  go test -run TestExtractSevenZipDirRealBundle -timeout 60m ./deps/models
func TestExtractSevenZipDirRealBundle(t *testing.T) {
	archive := os.Getenv("LOKI_XXL_ARCHIVE")
	if archive == "" {
		t.Skip("set LOKI_XXL_ARCHIVE to a downloaded Faster-Whisper-XXL .7z to run")
	}
	dstRoot := os.Getenv("LOKI_XXL_EXTRACT_DIR")
	if dstRoot == "" {
		dstRoot = t.TempDir()
	}
	dst := filepath.Join(dstRoot, "xxl")

	var lastDone, lastTotal int64
	progress := func(_ string, done, total int64) { lastDone, lastTotal = done, total }
	if err := extractSevenZipDir(context.Background(), archive, "Faster-Whisper-XXL/", dst, true, progress); err != nil {
		t.Fatalf("extractSevenZipDir: %v", err)
	}
	if lastDone != lastTotal || lastTotal == 0 {
		t.Errorf("extraction progress ended at %d/%d", lastDone, lastTotal)
	}

	// The pieces the provider and the runtime actually need.
	for _, rel := range []string{"faster-whisper-xxl", "_xxl_data"} {
		matches, err := filepath.Glob(filepath.Join(dst, rel+"*"))
		if err != nil || len(matches) == 0 {
			t.Errorf("expected %s* in extracted bundle (err=%v)", rel, err)
		}
	}
}
