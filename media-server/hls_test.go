package main

import (
	"testing"
)

func TestHlsCacheDir(t *testing.T) {
	dir := hlsCacheDir("/base", "/path/to/video.mp4")
	if dir == "" {
		t.Fatal("expected non-empty cache dir")
	}
	dir2 := hlsCacheDir("/base", "/path/to/video.mp4")
	if dir != dir2 {
		t.Fatalf("expected deterministic hash, got %s vs %s", dir, dir2)
	}
	dir3 := hlsCacheDir("/base", "/path/to/other.mp4")
	if dir == dir3 {
		t.Fatal("expected different hash for different input")
	}
}

func TestValidateHlsFilename(t *testing.T) {
	valid := []string{"master.m3u8", "stream.m3u8", "segment_000.ts", "segment_123.ts"}
	for _, f := range valid {
		if !isValidHlsFilename(f) {
			t.Errorf("expected %q to be valid", f)
		}
	}
	invalid := []string{"../etc/passwd", "foo.exe", "segment_.ts", "stream.ts", "master.ts", "../../secret.m3u8"}
	for _, f := range invalid {
		if isValidHlsFilename(f) {
			t.Errorf("expected %q to be invalid", f)
		}
	}
}

func TestValidateHlsPreset(t *testing.T) {
	valid := []string{"passthrough", "480p", "720p", "1080p"}
	for _, p := range valid {
		if !isValidHlsPreset(p) {
			t.Errorf("expected %q to be valid preset", p)
		}
	}
	invalid := []string{"../hack", "4k", "", "PASSTHROUGH"}
	for _, p := range invalid {
		if isValidHlsPreset(p) {
			t.Errorf("expected %q to be invalid preset", p)
		}
	}
}
