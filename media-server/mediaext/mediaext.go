// Package mediaext is the single answer to "is this file media, and what kind".
//
// It exists because that question used to be answered in ten places — an
// isMediaFile switch in tasks, a regex in storage, an imageExts slice behind
// the per-item ops, a filetype: mapping in the query engine, a map in the
// thumbnailer, a MIME switch per platform main — and they disagreed. A file
// whose extension appeared in some lists but not others got a half-supported
// life: .jfif was ingested by the browser (storage's regex listed it) and then
// silently dropped by every task that ran afterwards (tasks' switch did not),
// so 227 of them sat in the library with no embedding, no tags, and no faces,
// and nothing anywhere reported an error.
//
// Adding an extension is now one edit here. Anything that classifies a path by
// extension MUST come through this package rather than write its own list.
package mediaext

import (
	"path/filepath"
	"strings"
)

// imageExts are still images. .gif is included: the library treats it as an
// image for classification, and the one place that wants it counted as video
// (thumbnail generation, which needs a frame grab) says so explicitly.
//
// .avif and .heic are listed because they are images the viewer displays and
// the library should track — but note that the Go decoders registered by
// onnxtag/onnxface cover jpeg, png, gif, bmp, tiff and webp only, so local
// ONNX work on those two formats fails at decode rather than being skipped.
// .jfif/.pjpeg/.pjp are JPEG payloads under other names and decode normally,
// since image.Decode sniffs magic bytes rather than trusting the extension.
var imageExts = []string{
	".jpg", ".jpeg", ".jfif", ".pjpeg", ".pjp",
	".png", ".webp", ".avif", ".gif", ".bmp",
	".tif", ".tiff", ".heic",
}

// videoExts are containers ffmpeg/ffprobe handle.
var videoExts = []string{
	".mp4", ".mov", ".avi", ".mkv", ".webm", ".wmv", ".m4v", ".mpeg", ".mpg",
}

// audioExts are everything faster-whisper can decode through ffmpeg, so all of
// it is transcribable. .ogg is counted as audio rather than video: it is
// overwhelmingly Vorbis audio in this library, and transcription targeting is
// the only thing that reads this set.
var audioExts = []string{
	".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".opus", ".wma", ".aiff", ".ape",
}

func toSet(lists ...[]string) map[string]struct{} {
	set := make(map[string]struct{})
	for _, list := range lists {
		for _, e := range list {
			set[e] = struct{}{}
		}
	}
	return set
}

var (
	imageSet = toSet(imageExts)
	videoSet = toSet(videoExts)
	audioSet = toSet(audioExts)
	mediaSet = toSet(imageExts, videoExts, audioExts)
)

// Ext returns a path's lower-cased extension, dot included.
func Ext(name string) string {
	return strings.ToLower(filepath.Ext(name))
}

// IsMedia reports whether a path is media of any kind. This is the gate for
// ingesting, browsing, querying and per-item work — if it returns false the
// file is invisible to the library.
func IsMedia(name string) bool { _, ok := mediaSet[Ext(name)]; return ok }

// IsImage reports whether a path is a still image (see imageExts on .gif).
func IsImage(name string) bool { _, ok := imageSet[Ext(name)]; return ok }

// IsVideo reports whether a path is a video container.
func IsVideo(name string) bool { _, ok := videoSet[Ext(name)]; return ok }

// IsAudio reports whether a path is an audio file.
func IsAudio(name string) bool { _, ok := audioSet[Ext(name)]; return ok }

// ImageExts, VideoExts and AudioExts return copies of the sets, for callers
// that need the list itself (query predicates, op applicability). Copies,
// because a caller appending to a shared slice would corrupt every other
// caller's idea of what counts as media.
func ImageExts() []string { return append([]string{}, imageExts...) }
func VideoExts() []string { return append([]string{}, videoExts...) }
func AudioExts() []string { return append([]string{}, audioExts...) }

// MediaExts returns every media extension.
func MediaExts() []string {
	out := make([]string, 0, len(imageExts)+len(videoExts)+len(audioExts))
	out = append(out, imageExts...)
	out = append(out, videoExts...)
	return append(out, audioExts...)
}

// AltPattern renders the extensions as a regex alternation without dots
// ("jpg|jpeg|…"), for the few callers that must match with a regex.
func AltPattern(exts []string) string {
	parts := make([]string, len(exts))
	for i, e := range exts {
		parts[i] = strings.TrimPrefix(e, ".")
	}
	return strings.Join(parts, "|")
}

// mimeTypes maps an extension to the Content-Type served for it and sent to
// vision models. A missing entry used to mean "application/octet-stream",
// which is how .jfif images reached the LLM as an opaque blob and came back
// described as though no image had been provided.
var mimeTypes = map[string]string{
	".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".jfif": "image/jpeg",
	".pjpeg": "image/jpeg", ".pjp": "image/jpeg",
	".png": "image/png", ".gif": "image/gif", ".webp": "image/webp",
	".avif": "image/avif", ".bmp": "image/bmp", ".heic": "image/heic",
	".tif": "image/tiff", ".tiff": "image/tiff", ".svg": "image/svg+xml",
	".ico": "image/x-icon",

	".mp4": "video/mp4", ".m4v": "video/mp4", ".webm": "video/webm",
	".mov": "video/quicktime", ".mkv": "video/x-matroska", ".avi": "video/x-msvideo",
	".wmv": "video/x-ms-wmv", ".mpeg": "video/mpeg", ".mpg": "video/mpeg",

	".mp3": "audio/mpeg", ".wav": "audio/wav", ".flac": "audio/flac",
	".aac": "audio/aac", ".ogg": "audio/ogg", ".m4a": "audio/mp4",
	".opus": "audio/opus", ".wma": "audio/x-ms-wma", ".aiff": "audio/aiff",
	".ape": "audio/x-ape",
}

// MimeType returns the Content-Type for a path, or application/octet-stream
// when the extension is unknown.
func MimeType(name string) string {
	if mt, ok := mimeTypes[Ext(name)]; ok {
		return mt
	}
	return "application/octet-stream"
}

// directVisionSet are still images whose bytes a vision model takes as-is.
// Everything else an op might be handed — animated .gif, .tiff, .heic, .avif,
// any video — goes through an ffmpeg frame extraction first, which both picks
// a frame and normalizes the encoding. .jfif belongs here: it is JPEG data,
// and routing it through frame extraction was wasted work at best.
var directVisionSet = toSet([]string{
	".jpg", ".jpeg", ".jfif", ".pjpeg", ".pjp", ".png", ".bmp", ".webp",
})

// IsDirectVisionImage reports whether a vision model can be handed this file's
// bytes without an ffmpeg pass first.
func IsDirectVisionImage(name string) bool {
	_, ok := directVisionSet[Ext(name)]
	return ok
}
