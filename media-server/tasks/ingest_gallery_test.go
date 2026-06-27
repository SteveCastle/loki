package tasks

import "testing"

// gallery-dl prints a destination path for each file. When a file already
// exists and is being skipped it prefixes the path with '#'. Those skip lines
// must NOT be treated as freshly downloaded files (and so must not be ingested).
func TestIsDownloadedFilePath(t *testing.T) {
	const dir = "/downloads/site"

	tests := []struct {
		name string
		line string
		want bool
	}{
		{"freshly written media", "/downloads/site/photo.jpg", true},
		{"skipped (already exists) with space", "# /downloads/site/photo.jpg", false},
		{"skipped (already exists) no space", "#/downloads/site/photo.jpg", false},
		{"metadata sidecar is not media", "/downloads/site/photo.jpg.json", false},
		{"blank line", "", false},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := isDownloadedFilePath(tc.line, dir); got != tc.want {
				t.Errorf("isDownloadedFilePath(%q) = %v, want %v", tc.line, got, tc.want)
			}
		})
	}
}
