package main

import (
	"testing"
)

func TestValidatePathWithinRoots(t *testing.T) {
	roots := []string{"/mnt/media", "/home/user/photos"}

	tests := []struct {
		name    string
		path    string
		roots   []string
		wantErr bool
	}{
		{"valid path under root", "/mnt/media/vacation", roots, false},
		{"valid path exact root", "/mnt/media", roots, false},
		{"valid second root", "/home/user/photos/2024", roots, false},
		{"path outside roots", "/etc/passwd", roots, true},
		{"traversal attack", "/mnt/media/../../../etc/passwd", roots, true},
		{"empty roots allows all", "/any/path", []string{}, false},
		{"empty path with roots", "", roots, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validatePathWithinRoots(tt.path, tt.roots)
			if (err != nil) != tt.wantErr {
				t.Errorf("validatePathWithinRoots(%q, %v) error = %v, wantErr %v",
					tt.path, tt.roots, err, tt.wantErr)
			}
		})
	}
}

func TestMediaExtensionFilter(t *testing.T) {
	tests := []struct {
		name string
		want bool
	}{
		{"photo.jpg", true},
		{"photo.JPG", true},
		{"video.mp4", true},
		{"song.mp3", true},
		{"doc.pdf", false},
		{"readme.txt", false},
		{"image.webp", true},
		{"clip.mkv", true},
		{"audio.flac", true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := isMediaFile(tt.name); got != tt.want {
				t.Errorf("isMediaFile(%q) = %v, want %v", tt.name, got, tt.want)
			}
		})
	}
}
