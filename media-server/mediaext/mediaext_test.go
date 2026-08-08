package mediaext

import (
	"strings"
	"testing"
)

// The formats that were half-supported: accepted by one list, dropped by
// another. Each is pinned here so the next divergence fails a test instead of
// silently costing an item its embedding.
func TestFormerlyHalfSupportedFormatsAreMedia(t *testing.T) {
	for _, name := range []string{
		"photo.jfif", "photo.JFIF", "photo.pjpeg", "photo.pjp",
		"photo.avif", "photo.webp", "photo.bmp", "scan.tif", "scan.tiff",
		"photo.heic", "clip.avi", "clip.wmv", "clip.m4v", "clip.mpeg", "clip.mpg",
	} {
		if !IsMedia(name) {
			t.Errorf("IsMedia(%q) = false, want true", name)
		}
	}
}

func TestClassification(t *testing.T) {
	images := []string{"a.jpg", "a.jpeg", "a.jfif", "a.png", "a.webp", "a.avif", "a.gif", "a.bmp", "a.tif", "a.heic"}
	videos := []string{"a.mp4", "a.mov", "a.avi", "a.mkv", "a.webm", "a.wmv", "a.m4v", "a.mpeg", "a.mpg"}
	audio := []string{"a.mp3", "a.wav", "a.flac", "a.aac", "a.ogg", "a.m4a", "a.opus", "a.wma", "a.aiff", "a.ape"}

	for _, n := range images {
		if !IsImage(n) || IsVideo(n) || IsAudio(n) {
			t.Errorf("%q: want image only, got image=%v video=%v audio=%v", n, IsImage(n), IsVideo(n), IsAudio(n))
		}
	}
	for _, n := range videos {
		if !IsVideo(n) || IsImage(n) || IsAudio(n) {
			t.Errorf("%q: want video only, got image=%v video=%v audio=%v", n, IsImage(n), IsVideo(n), IsAudio(n))
		}
	}
	for _, n := range audio {
		if !IsAudio(n) || IsImage(n) || IsVideo(n) {
			t.Errorf("%q: want audio only, got image=%v video=%v audio=%v", n, IsImage(n), IsVideo(n), IsAudio(n))
		}
	}
}

func TestNonMediaIsRejected(t *testing.T) {
	for _, name := range []string{
		"notes.md", "archive.torrent", "installer.msi", "page.html", "script.ps1",
		"db.sqlite", "clip.mp4.part", "clip.mp4.ytdl", "meta.json", "subs.vtt", "subs.srt",
		"noextension",
	} {
		if IsMedia(name) {
			t.Errorf("IsMedia(%q) = true, want false", name)
		}
	}
}

// .avi must not be swept up by the .avif entry, and vice versa — the substring
// trap the viewer's own classifier had to be fixed for.
func TestSimilarExtensionsDoNotCollide(t *testing.T) {
	if !IsVideo("clip.avi") || IsImage("clip.avi") {
		t.Error(".avi must classify as video only")
	}
	if !IsImage("photo.avif") || IsVideo("photo.avif") {
		t.Error(".avif must classify as image only")
	}
	if !IsImage("scan.tif") || !IsImage("scan.tiff") {
		t.Error(".tif and .tiff must both be images")
	}
}

func TestMimeType(t *testing.T) {
	cases := map[string]string{
		"a.jfif":  "image/jpeg", // the one that reached vision models as a blob
		"a.pjpeg": "image/jpeg", "a.pjp": "image/jpeg",
		"a.jpg": "image/jpeg", "a.png": "image/png", "a.webp": "image/webp",
		"a.avif": "image/avif", "a.heic": "image/heic", "a.tiff": "image/tiff",
		"a.mp4": "video/mp4", "a.mkv": "video/x-matroska", "a.mp3": "audio/mpeg",
		"a.unknown": "application/octet-stream",
	}
	for name, want := range cases {
		if got := MimeType(name); got != want {
			t.Errorf("MimeType(%q) = %q, want %q", name, got, want)
		}
	}
}

// Every image the vision path hands over directly must have a real image MIME
// type — sending bytes with application/octet-stream is what produced
// "no image was provided" replies.
func TestDirectVisionImagesAllHaveAnImageMime(t *testing.T) {
	for _, name := range []string{"a.jpg", "a.jpeg", "a.jfif", "a.pjpeg", "a.pjp", "a.png", "a.bmp", "a.webp"} {
		if !IsDirectVisionImage(name) {
			t.Errorf("%q should be sent to the model directly", name)
		}
		if mt := MimeType(name); !strings.HasPrefix(mt, "image/") {
			t.Errorf("MimeType(%q) = %q, want an image/* type", name, mt)
		}
	}
	// Formats that need an ffmpeg pass first.
	for _, name := range []string{"a.gif", "a.tiff", "a.heic", "a.avif", "a.mp4"} {
		if IsDirectVisionImage(name) {
			t.Errorf("%q needs a frame extracted, not a direct hand-off", name)
		}
	}
}

// The exported slices back query predicates and op applicability. A caller that
// appends to one must not corrupt the next caller's view of what is media.
func TestExtSlicesAreCopies(t *testing.T) {
	first := ImageExts()
	_ = append(first[:len(first):len(first)], ".bogus")
	first[0] = ".clobbered"
	if second := ImageExts(); second[0] == ".clobbered" {
		t.Error("ImageExts returned a slice aliasing package state")
	}
	if !IsImage("a" + imageExts[0]) {
		t.Error("mutating a returned slice changed classification")
	}
}

func TestAltPattern(t *testing.T) {
	got := AltPattern([]string{".jpg", ".jfif", ".mp4"})
	if got != "jpg|jfif|mp4" {
		t.Errorf("AltPattern = %q, want %q", got, "jpg|jfif|mp4")
	}
}
