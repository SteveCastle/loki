package main

import (
	"encoding/binary"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	depspkg "github.com/stevecastle/shrike/deps"
)

// box builds a serialized MP4 box with a 32-bit size header.
func box(boxType string, payload []byte) []byte {
	b := make([]byte, 8+len(payload))
	binary.BigEndian.PutUint32(b[:4], uint32(8+len(payload)))
	copy(b[4:8], boxType)
	copy(b[8:], payload)
	return b
}

func writeTemp(t *testing.T, name string, data []byte) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(p, data, 0644); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestThumbnailFileValid(t *testing.T) {
	ftyp := box("ftyp", []byte("isom\x00\x00\x02\x00isomiso2"))
	moov := box("moov", make([]byte, 100))

	t.Run("missing file", func(t *testing.T) {
		if thumbnailFileValid(filepath.Join(t.TempDir(), "nope.mp4")) {
			t.Error("missing file should be invalid")
		}
	})

	t.Run("empty file", func(t *testing.T) {
		p := writeTemp(t, "empty.png", nil)
		if thumbnailFileValid(p) {
			t.Error("zero-byte file should be invalid")
		}
	})

	t.Run("non-mp4 with content", func(t *testing.T) {
		p := writeTemp(t, "thumb.png", []byte("\x89PNG fake"))
		if !thumbnailFileValid(p) {
			t.Error("non-empty png should be valid")
		}
	})

	t.Run("frameless mp4 (empty mdat)", func(t *testing.T) {
		// This is what ffmpeg leaves behind when the seek lands past the
		// only frame of a single-frame GIF: ftyp + free + empty mdat + moov.
		data := append(append(append(ftyp, box("free", nil)...), box("mdat", nil)...), moov...)
		p := writeTemp(t, "empty-mdat.mp4", data)
		if thumbnailFileValid(p) {
			t.Error("mp4 with empty mdat should be invalid")
		}
	})

	t.Run("mp4 with frames", func(t *testing.T) {
		data := append(append(append(ftyp, box("free", nil)...), box("mdat", make([]byte, 700))...), moov...)
		p := writeTemp(t, "good.mp4", data)
		if !thumbnailFileValid(p) {
			t.Error("mp4 with mdat payload should be valid")
		}
	})

	t.Run("mdat extends to EOF", func(t *testing.T) {
		// size=0 means "rest of file"
		mdat := box("mdat", make([]byte, 50))
		binary.BigEndian.PutUint32(mdat[:4], 0)
		data := append(append([]byte{}, ftyp...), mdat...)
		p := writeTemp(t, "eof-mdat.mp4", data)
		if !thumbnailFileValid(p) {
			t.Error("EOF-extending mdat with payload should be valid")
		}
	})

	t.Run("mdat with 64-bit largesize", func(t *testing.T) {
		payload := make([]byte, 50)
		b := make([]byte, 16+len(payload))
		binary.BigEndian.PutUint32(b[:4], 1)
		copy(b[4:8], "mdat")
		binary.BigEndian.PutUint64(b[8:16], uint64(16+len(payload)))
		copy(b[16:], payload)
		data := append(append([]byte{}, ftyp...), b...)
		p := writeTemp(t, "largesize-mdat.mp4", data)
		if !thumbnailFileValid(p) {
			t.Error("largesize mdat with payload should be valid")
		}
	})

	t.Run("corrupt box size", func(t *testing.T) {
		data := append(append([]byte{}, ftyp...), []byte{0, 0, 0, 3, 'j', 'u', 'n', 'k'}...)
		p := writeTemp(t, "corrupt.mp4", data)
		if thumbnailFileValid(p) {
			t.Error("corrupt mp4 should be invalid")
		}
	})
}

// TestGenerateVideoThumbnailGifs exercises the real ffmpeg flow. A
// single-frame GIF probes as ~0.04s; the old code seeked to duration/2,
// which lands after the only frame — ffmpeg exits 0 but encodes nothing.
func TestGenerateVideoThumbnailGifs(t *testing.T) {
	ffmpegPath := depspkg.BundledOrEmpty("ffmpeg")
	if ffmpegPath == "" {
		// Bundled deps resolve relative to the executable and never work
		// under `go test`; fall back to a system ffmpeg.
		ffmpegPath, _ = exec.LookPath("ffmpeg")
	}
	if ffmpegPath == "" {
		t.Skip("ffmpeg not available")
	}

	dir := t.TempDir()
	makeGif := func(name string, frames int) string {
		p := filepath.Join(dir, name)
		dur := float64(frames) / 10.0
		cmd := exec.Command(ffmpegPath, "-y",
			"-f", "lavfi", "-i", "testsrc=s=64x64:d="+formatTimeStamp(dur)+":r=10",
			"-frames:v", formatTimeStamp(float64(frames)), p)
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("failed to create %s: %v\n%s", name, err, out)
		}
		return p
	}

	t.Run("single-frame gif", func(t *testing.T) {
		gif := makeGif("single.gif", 1)
		thumb := filepath.Join(dir, "single-thumb.mp4")
		if err := generateVideoThumbnail(ffmpegPath, gif, thumb, "thumbnail_path_600", 0); err != nil {
			t.Fatalf("generateVideoThumbnail: %v", err)
		}
		if !thumbnailFileValid(thumb) {
			t.Error("single-frame gif thumbnail has no frames")
		}
	})

	t.Run("animated gif", func(t *testing.T) {
		gif := makeGif("anim.gif", 30)
		thumb := filepath.Join(dir, "anim-thumb.mp4")
		if err := generateVideoThumbnail(ffmpegPath, gif, thumb, "thumbnail_path_600", 0); err != nil {
			t.Fatalf("generateVideoThumbnail: %v", err)
		}
		if !thumbnailFileValid(thumb) {
			t.Error("animated gif thumbnail has no frames")
		}
	})

	t.Run("timestamp past end of file", func(t *testing.T) {
		gif := makeGif("single2.gif", 1)
		thumb := filepath.Join(dir, "single2-thumb.mp4")
		if err := generateVideoThumbnail(ffmpegPath, gif, thumb, "thumbnail_path_600", 5); err != nil {
			t.Fatalf("generateVideoThumbnail: %v", err)
		}
		if !thumbnailFileValid(thumb) {
			t.Error("out-of-range timestamp thumbnail has no frames")
		}
	})
}
