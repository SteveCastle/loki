package tasks

import (
	"image"
	"image/color"
	"image/png"
	"io"
	"os"
	"path/filepath"
	"testing"

	"golang.org/x/image/bmp"

	"github.com/stevecastle/shrike/appconfig"
)

// writeTestImage renders a solid w×h image and encodes it with enc into a temp
// file with the given name, returning its path.
func writeTestImage(t *testing.T, name string, w, h int, enc func(io.Writer, image.Image) error) string {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.Set(x, y, color.RGBA{R: 10, G: 120, B: 200, A: 255})
		}
	}
	path := filepath.Join(t.TempDir(), name)
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("create %s: %v", path, err)
	}
	if err := enc(f, img); err != nil {
		_ = f.Close()
		t.Fatalf("encode %s: %v", path, err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("close %s: %v", path, err)
	}
	return path
}

// decodedImage opens path and returns the registered decoder name + dimensions.
func decodedImage(t *testing.T, path string) (format string, w, h int) {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open %s: %v", path, err)
	}
	defer f.Close()
	cfg, format, err := image.DecodeConfig(f)
	if err != nil {
		t.Fatalf("decode %s: %v", path, err)
	}
	return format, cfg.Width, cfg.Height
}

// A small JPEG/PNG is already model-safe and within size, so it must pass
// through untouched (same path, no temp file).
func TestResizeImageIfNeededPassesThroughSmallPNG(t *testing.T) {
	src := writeTestImage(t, "small.png", 100, 80, png.Encode)
	got, err := resizeImageIfNeeded(src)
	if err != nil {
		t.Fatalf("resizeImageIfNeeded: %v", err)
	}
	if got != src {
		t.Fatalf("expected passthrough (same path), got %q", got)
	}
}

// The regression guard: a SMALL bmp/webp/etc. used to slip through in its
// original format. It must now be re-encoded to a PNG temp file even though it
// needs no resizing — otherwise backends that can't decode it answer blind.
func TestResizeImageIfNeededNormalizesSmallNonSafeFormat(t *testing.T) {
	src := writeTestImage(t, "small.bmp", 100, 80, bmp.Encode)
	got, err := resizeImageIfNeeded(src)
	if err != nil {
		t.Fatalf("resizeImageIfNeeded: %v", err)
	}
	if got == src {
		t.Fatal("expected a re-encoded temp file, got the original bmp path back")
	}
	t.Cleanup(func() { _ = os.Remove(got) })
	if format, _, _ := decodedImage(t, got); format != "png" {
		t.Fatalf("expected normalized output to be png, got %q", format)
	}
}

// Oversized images are still downscaled (and emitted as PNG) as before.
func TestResizeImageIfNeededResizesOversized(t *testing.T) {
	src := writeTestImage(t, "big.png", 2000, 1000, png.Encode)
	got, err := resizeImageIfNeeded(src)
	if err != nil {
		t.Fatalf("resizeImageIfNeeded: %v", err)
	}
	if got == src {
		t.Fatal("expected a resized temp file for an oversized image")
	}
	t.Cleanup(func() { _ = os.Remove(got) })
	format, w, h := decodedImage(t, got)
	if format != "png" {
		t.Fatalf("expected png output, got %q", format)
	}
	long := w
	if h > w {
		long = h
	}
	if long != 1280 {
		t.Fatalf("expected long side clamped to 1280, got %dx%d", w, h)
	}
}

func TestLooksLikeNoImageResponse(t *testing.T) {
	blind := "Since no image was provided in your prompt (only the instructions), I cannot generate specific research notes."
	if !looksLikeNoImageResponse(blind) {
		t.Fatal("expected the blind response to be detected")
	}
	legit := "A white-haired woman in a green cloak stands in a sunlit field; a small wooden sign is visible at the lower left."
	if looksLikeNoImageResponse(legit) {
		t.Fatal("false positive on a legitimate description")
	}
}

func TestResolveDescribePromptUsesCustomWhenProvided(t *testing.T) {
	got := resolveDescribePrompt("custom override")
	if got != "custom override" {
		t.Errorf("got %q, want %q", got, "custom override")
	}
}

func TestResolveDescribePromptTrimsWhitespace(t *testing.T) {
	got := resolveDescribePrompt("   custom   ")
	if got != "custom" {
		t.Errorf("got %q, want %q", got, "custom")
	}
}

func TestResolveDescribePromptFallsBackToConfigWhenEmpty(t *testing.T) {
	old := appconfig.Get()
	cfg := old
	cfg.DescribePrompt = "DEFAULT-FROM-CONFIG"
	appconfig.Set(cfg)
	t.Cleanup(func() { appconfig.Set(old) })

	got := resolveDescribePrompt("")
	if got != "DEFAULT-FROM-CONFIG" {
		t.Errorf("got %q, want %q", got, "DEFAULT-FROM-CONFIG")
	}
}

func TestResolveDescribePromptFallsBackOnAllWhitespace(t *testing.T) {
	old := appconfig.Get()
	cfg := old
	cfg.DescribePrompt = "DEFAULT-FROM-CONFIG"
	appconfig.Set(cfg)
	t.Cleanup(func() { appconfig.Set(old) })

	got := resolveDescribePrompt("   \n\t  ")
	if got != "DEFAULT-FROM-CONFIG" {
		t.Errorf("got %q, want %q", got, "DEFAULT-FROM-CONFIG")
	}
}
