package tasks

import (
	"testing"

	"github.com/stevecastle/shrike/media"
	"github.com/stevecastle/shrike/mediaext"
	"github.com/stevecastle/shrike/storage"
)

// The bug this guards against was never in one list — it was in the gap
// between them. A .jfif passed storage's regex (so the browser ingested it)
// and failed tasks' switch (so every per-item job dropped it), and the item
// sat in the library forever with no embedding, tags, or faces, with nothing
// reporting an error. These assertions fail the moment two gates disagree.

// formats names the extensions worth pinning explicitly: the ones that were
// actually broken, plus the ordinary ones whose behavior must not regress.
var formats = []string{
	".jpg", ".jpeg", ".jfif", ".pjpeg", ".pjp", ".png", ".webp", ".avif",
	".gif", ".bmp", ".tif", ".tiff", ".heic",
	".mp4", ".mov", ".avi", ".mkv", ".webm", ".wmv", ".m4v", ".mpeg", ".mpg",
	".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".opus", ".wma", ".aiff", ".ape",
}

func TestIngestAndTaskGatesAgree(t *testing.T) {
	for _, ext := range formats {
		name := "item" + ext
		ingest := storage.IsMediaFile(name)
		task := isMediaFile(name)
		if ingest != task {
			t.Errorf("%s: storage.IsMediaFile=%v but tasks.isMediaFile=%v — a file one side accepts and the other drops is invisible work",
				ext, ingest, task)
		}
		if !task {
			t.Errorf("%s: should be media", ext)
		}
	}
}

// filterMediaPaths is what turns a job's resolved query into its item list.
// Anything IsMedia accepts has to survive it.
func TestFilterMediaPathsKeepsEveryMediaFormat(t *testing.T) {
	paths := make([]string, 0, len(formats))
	for _, ext := range formats {
		paths = append(paths, `Z:\Media\item`+ext)
	}
	kept := filterMediaPaths(paths)
	if len(kept) != len(paths) {
		got := map[string]bool{}
		for _, p := range kept {
			got[p] = true
		}
		for _, p := range paths {
			if !got[p] {
				t.Errorf("filterMediaPaths dropped %s", p)
			}
		}
	}
}

// filetype:image / :video / :audio must cover exactly what the ops classify
// the same way, or a query built to target a task's inputs silently omits
// items that task would happily have processed.
func TestFiletypeQueryMatchesOpClassification(t *testing.T) {
	inSet := func(set []string, ext string) bool {
		for _, e := range set {
			if e == ext {
				return true
			}
		}
		return false
	}
	for _, ext := range formats {
		name := "item" + ext
		switch {
		case mediaext.IsImage(name):
			if !inSet(media.ExtensionsForFiletype("image"), ext) {
				t.Errorf("%s is an image but filetype:image does not match it", ext)
			}
			if !inSet(imageExts, ext) {
				t.Errorf("%s is an image but the per-item ops' imageExts omits it", ext)
			}
		case mediaext.IsVideo(name):
			if !inSet(media.ExtensionsForFiletype("video"), ext) {
				t.Errorf("%s is a video but filetype:video does not match it", ext)
			}
		case mediaext.IsAudio(name):
			if !inSet(media.ExtensionsForFiletype("audio"), ext) {
				t.Errorf("%s is audio but filetype:audio does not match it", ext)
			}
			if !inSet(audioExts, ext) {
				t.Errorf("%s is audio but the transcribe op's audioExts omits it", ext)
			}
		default:
			t.Errorf("%s is media but classifies as neither image, video, nor audio", ext)
		}
	}
}

// Every still image must be describable — that op's applicability is the
// gate .jfif failed.
func TestDescribeAndDimensionsApplyToEveryImage(t *testing.T) {
	describe := extAppliesFn(append(append([]string{}, imageExts...), videoExts...)...)
	for _, ext := range formats {
		name := "item" + ext
		if mediaext.IsImage(name) || mediaext.IsVideo(name) {
			if !describe(name) {
				t.Errorf("describe/dimensions do not apply to %s", ext)
			}
		}
	}
}
