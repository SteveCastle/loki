package tasks

import (
	"testing"
)

func TestIsMediaFile(t *testing.T) {
	cases := []struct {
		path string
		want bool
	}{
		// Images and video
		{`C:\lib\photo.jpg`, true},
		{`C:\lib\photo.PNG`, true},
		{`C:\lib\clip.mp4`, true},
		{`C:\lib\clip.m4v`, true},
		// Audio must count as media — transcript jobs target audio libraries.
		{`C:\lib\song.mp3`, true},
		{`C:\lib\song.flac`, true},
		{`C:\lib\voice.m4a`, true},
		// Sidecars and other non-media that library scans ingest.
		{`C:\lib\photo.json`, false},
		{`C:\lib\photo.jpg.json`, false},
		{`C:\lib\notes.txt`, false},
		{`C:\lib\subs.vtt`, false},
		{`C:\lib\noextension`, false},
	}
	for _, c := range cases {
		if got := isMediaFile(c.path); got != c.want {
			t.Errorf("isMediaFile(%q) = %v, want %v", c.path, got, c.want)
		}
	}
}

// Query results are not guaranteed to be media files: scans ingest sidecar
// files (e.g. per-image .json metadata) into the media table, and batch
// tasks fed by a query used to process those too — doubling their targets.
func TestFilterMediaPaths(t *testing.T) {
	in := []string{
		`C:\lib\a.jpg`,
		`C:\lib\a.json`,
		`C:\lib\b.png`,
		`C:\lib\b.json`,
		`C:\lib\c.mp3`,
	}
	got := filterMediaPaths(in)
	want := []string{`C:\lib\a.jpg`, `C:\lib\b.png`, `C:\lib\c.mp3`}
	if len(got) != len(want) {
		t.Fatalf("filterMediaPaths returned %d paths, want %d: %v", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("filterMediaPaths[%d] = %q, want %q", i, got[i], want[i])
		}
	}
}
