package storage

import (
	"strings"
	"testing"
)

// newTestS3Backend builds an S3Backend directly without calling AWS.
func newTestS3Backend(bucket, prefix, label, thumbnailPrefix string) *S3Backend {
	return &S3Backend{
		bucket:          bucket,
		prefix:          prefix,
		label:           label,
		thumbnailPrefix: thumbnailPrefix,
	}
}

// --- Contains ---

func TestS3BackendContains(t *testing.T) {
	b := newTestS3Backend("my-bucket", "media/", "Test", "_thumbnails")

	inside := []string{
		"s3://my-bucket/media/",
		"s3://my-bucket/media/photo.jpg",
		"s3://my-bucket/media/sub/video.mp4",
	}
	for _, p := range inside {
		if !b.Contains(p) {
			t.Errorf("Contains(%q) = false, want true", p)
		}
	}

	outside := []string{
		"s3://other-bucket/media/photo.jpg",
		"s3://my-bucket/other/photo.jpg",
		"/local/path/photo.jpg",
		"",
	}
	for _, p := range outside {
		if b.Contains(p) {
			t.Errorf("Contains(%q) = true, want false", p)
		}
	}
}

func TestS3BackendContains_NoPrefix(t *testing.T) {
	b := newTestS3Backend("my-bucket", "", "Test", "_thumbnails")

	if !b.Contains("s3://my-bucket/any/path.jpg") {
		t.Error("Contains should return true for any path in bucket when prefix is empty")
	}
	if b.Contains("s3://other-bucket/any/path.jpg") {
		t.Error("Contains should return false for a different bucket")
	}
}

// --- pathToKey ---

func TestS3PathToKey(t *testing.T) {
	b := newTestS3Backend("my-bucket", "media/", "Test", "_thumbnails")

	cases := []struct {
		path string
		want string
	}{
		{"s3://my-bucket/media/photo.jpg", "media/photo.jpg"},
		{"s3://my-bucket/media/sub/video.mp4", "media/sub/video.mp4"},
		{"s3://my-bucket/", ""},
		{"s3://my-bucket/single.jpg", "single.jpg"},
	}

	for _, tc := range cases {
		got := b.pathToKey(tc.path)
		if got != tc.want {
			t.Errorf("pathToKey(%q) = %q, want %q", tc.path, got, tc.want)
		}
	}
}

// --- keyToPath ---

func TestS3KeyToPath(t *testing.T) {
	b := newTestS3Backend("my-bucket", "media/", "Test", "_thumbnails")

	cases := []struct {
		key  string
		want string
	}{
		{"media/photo.jpg", "s3://my-bucket/media/photo.jpg"},
		{"media/sub/video.mp4", "s3://my-bucket/media/sub/video.mp4"},
		{"", "s3://my-bucket/"},
		{"single.jpg", "s3://my-bucket/single.jpg"},
	}

	for _, tc := range cases {
		got := b.keyToPath(tc.key)
		if got != tc.want {
			t.Errorf("keyToPath(%q) = %q, want %q", tc.key, got, tc.want)
		}
	}
}

// --- ThumbnailPath ---

func TestS3ThumbnailPath(t *testing.T) {
	b := newTestS3Backend("my-bucket", "media/", "Test", "_thumbnails")

	got := b.ThumbnailPath("thumb_photo.jpg")
	want := "s3://my-bucket/_thumbnails/thumb_photo.jpg"
	if got != want {
		t.Errorf("ThumbnailPath = %q, want %q", got, want)
	}
}

func TestS3ThumbnailPath_CustomPrefix(t *testing.T) {
	b := newTestS3Backend("my-bucket", "media/", "Test", "thumbs")

	got := b.ThumbnailPath("image.webp")
	if !strings.HasPrefix(got, "s3://my-bucket/thumbs/") {
		t.Errorf("ThumbnailPath = %q, want prefix \"s3://my-bucket/thumbs/\"", got)
	}
}

// --- Root ---

func TestS3Root(t *testing.T) {
	b := newTestS3Backend("my-bucket", "media/", "My S3", "_thumbnails")

	e := b.Root()

	if e.Name != "My S3" {
		t.Errorf("Root().Name = %q, want \"My S3\"", e.Name)
	}
	if e.Path != "s3://my-bucket/media/" {
		t.Errorf("Root().Path = %q, want \"s3://my-bucket/media/\"", e.Path)
	}
	if !e.IsDir {
		t.Error("Root().IsDir = false, want true")
	}
	if e.Type != "s3" {
		t.Errorf("Root().Type = %q, want \"s3\"", e.Type)
	}
}

func TestS3Root_NoPrefix(t *testing.T) {
	b := newTestS3Backend("my-bucket", "", "Bucket Root", "_thumbnails")

	e := b.Root()

	if e.Path != "s3://my-bucket/" {
		t.Errorf("Root().Path = %q, want \"s3://my-bucket/\"", e.Path)
	}
}
