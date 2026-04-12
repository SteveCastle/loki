package tasks

import (
	"os"
	"path/filepath"
	"testing"
)

func TestStripLokiTemp(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{
			filepath.Join("Z:", "gallery-dl", ".loki-temp", "abc-123", "video_grayscale.mp4"),
			filepath.Join("Z:", "gallery-dl"),
		},
		{
			filepath.Join("/tmp", ".loki-temp", "job-1", "file.mp4"),
			filepath.FromSlash("/tmp"),
		},
		{
			filepath.Join("/tmp", "no-temp", "file.mp4"),
			filepath.Join("/tmp", "no-temp"),
		},
	}

	for _, tt := range tests {
		got := stripLokiTemp(tt.input)
		if got != tt.expected {
			t.Errorf("stripLokiTemp(%q) = %q; want %q", tt.input, got, tt.expected)
		}
	}
}

func TestBuildSaveFilename(t *testing.T) {
	tests := []struct {
		name     string
		suffix   string
		ext      string
		expected string
	}{
		{"video_grayscale_blurred", "_edited", ".mp4", "video_grayscale_blurred_edited.mp4"},
		{"video", "_final", ".mp4", "video_final.mp4"},
		{"video_grayscale", "", ".mp4", "video_grayscale.mp4"},
	}

	for _, tt := range tests {
		got := buildSaveFilename(tt.name, tt.suffix, tt.ext)
		if got != tt.expected {
			t.Errorf("buildSaveFilename(%q, %q, %q) = %q; want %q", tt.name, tt.suffix, tt.ext, got, tt.expected)
		}
	}
}

func TestResolveConflictSuffix(t *testing.T) {
	dir := t.TempDir()

	existing := filepath.Join(dir, "video.mp4")
	os.WriteFile(existing, []byte("x"), 0644)

	result := resolveConflict(filepath.Join(dir, "video.mp4"), "suffix")
	expected := filepath.Join(dir, "video_1.mp4")
	if result != expected {
		t.Errorf("resolveConflict() = %q; want %q", result, expected)
	}

	os.WriteFile(expected, []byte("x"), 0644)
	result2 := resolveConflict(filepath.Join(dir, "video.mp4"), "suffix")
	expected2 := filepath.Join(dir, "video_2.mp4")
	if result2 != expected2 {
		t.Errorf("resolveConflict() = %q; want %q", result2, expected2)
	}
}

func TestResolveConflictOverwrite(t *testing.T) {
	dir := t.TempDir()
	existing := filepath.Join(dir, "video.mp4")
	os.WriteFile(existing, []byte("x"), 0644)

	result := resolveConflict(existing, "overwrite")
	if result != existing {
		t.Errorf("resolveConflict(overwrite) = %q; want %q", result, existing)
	}
}

func TestResolveConflictSkip(t *testing.T) {
	dir := t.TempDir()
	existing := filepath.Join(dir, "video.mp4")
	os.WriteFile(existing, []byte("x"), 0644)

	result := resolveConflict(existing, "skip")
	if result != "" {
		t.Errorf("resolveConflict(skip) = %q; want empty", result)
	}
}
