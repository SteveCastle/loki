package heic

import (
	"bytes"
	"encoding/binary"
	"os"
	"strings"
	"testing"
)

func mkbox(typ string, payload ...[]byte) []byte {
	body := bytes.Join(payload, nil)
	out := binary.BigEndian.AppendUint32(nil, uint32(8+len(body)))
	return append(append(out, typ...), body...)
}

func fullHdr(version byte, flags uint32) []byte {
	return []byte{version, byte(flags >> 16), byte(flags >> 8), byte(flags)}
}

func be16(n uint16) []byte { return binary.BigEndian.AppendUint16(nil, n) }
func be32(n uint32) []byte { return binary.BigEndian.AppendUint32(nil, n) }

var testICC = []byte("fake-display-p3-profile")

// A meta box shaped like an iPhone photo: primary item 49 (the grid) with
// colr/ispe/irot, plus a depth-map item with its own ICC that must not be
// picked up.
func heicFile(rotate byte, wide bool) []byte {
	ipco := mkbox("ipco",
		mkbox("colr", []byte("prof"), testICC),
		mkbox("ispe", fullHdr(0, 0), be32(4032), be32(3024)),
		mkbox("irot", []byte{rotate}),
		mkbox("colr", []byte("prof"), []byte("depth-profile")),
	)
	idx := func(i uint16) []byte {
		if wide {
			return be16(0x8000 | i)
		}
		return []byte{0x80 | byte(i)}
	}
	var flags uint32
	if wide {
		flags = 1
	}
	ipma := mkbox("ipma", fullHdr(0, flags), be32(2),
		be16(62), []byte{1}, idx(4),
		be16(49), []byte{3}, idx(1), idx(2), idx(3),
	)
	return bytes.Join([][]byte{
		mkbox("ftyp", []byte("heic"), be32(0), []byte("mif1heic")),
		mkbox("meta", fullHdr(0, 0),
			mkbox("hdlr", fullHdr(0, 0), make([]byte, 20)),
			mkbox("pitm", fullHdr(0, 0), be16(49)),
			mkbox("iprp", ipco, ipma)),
		mkbox("mdat", make([]byte, 16)),
	}, nil)
}

// SOI, APP0 (JFIF), SOF0 (h=3024, w=4032), SOS, EOI.
func testJPEG() []byte {
	app0 := append(append([]byte{0xff, 0xe0}, be16(16)...), append([]byte("JFIF\x00"), make([]byte, 9)...)...)
	sof0 := bytes.Join([][]byte{{0xff, 0xc0}, be16(17), {8}, be16(3024), be16(4032), {3}, make([]byte, 9)}, nil)
	sos := append(append([]byte{0xff, 0xda}, be16(8)...), make([]byte, 6)...)
	return bytes.Join([][]byte{{0xff, 0xd8}, app0, sof0, sos, {0x12, 0x34, 0xff, 0xd9}}, nil)
}

func TestIsHeic(t *testing.T) {
	for name, want := range map[string]bool{
		`C:\Photos\IMG_0001.HEIC`: true, "/x/y.heif": true, "/x/heic.jpg": false,
	} {
		if got := IsHeic(name); got != want {
			t.Errorf("IsHeic(%q) = %v", name, got)
		}
	}
}

func TestParseMetaPrimaryItem(t *testing.T) {
	info, ok := ParseMeta(heicFile(0, false))
	if !ok {
		t.Fatal("ParseMeta failed")
	}
	if !bytes.Equal(info.ICC, testICC) {
		t.Errorf("ICC = %q, want the primary item's", info.ICC)
	}
	if info.Width != 4032 || info.Height != 3024 || info.Cropped {
		t.Errorf("got %dx%d cropped=%v", info.Width, info.Height, info.Cropped)
	}
}

func TestParseMetaRotationAndWideIndices(t *testing.T) {
	info, _ := ParseMeta(heicFile(1, true))
	if info.Width != 3024 || info.Height != 4032 {
		t.Errorf("quarter turn: got %dx%d, want 3024x4032", info.Width, info.Height)
	}
	if !bytes.Equal(info.ICC, testICC) {
		t.Errorf("15-bit indices: ICC = %q", info.ICC)
	}
}

func TestParseMetaTruncated(t *testing.T) {
	if _, ok := ParseMeta(mkbox("ftyp", []byte("heic"))); ok {
		t.Error("no meta box should not parse")
	}
	file := heicFile(0, false)
	for n := 0; n < len(file); n += 7 {
		ParseMeta(file[:n]) // must not panic
	}
}

func TestJPEGICC(t *testing.T) {
	src := testJPEG()
	if w, h, ok := JPEGSize(src); !ok || w != 4032 || h != 3024 {
		t.Fatalf("JPEGSize = %d %d %v", w, h, ok)
	}
	out := WithJPEGICC(src, testICC)
	// APP2 right after SOI(2) + APP0(18)
	if out[20] != 0xff || out[21] != 0xe2 || string(out[24:36]) != "ICC_PROFILE\x00" || out[36] != 1 || out[37] != 1 {
		t.Fatalf("APP2 ICC segment not after APP0: % x", out[:40])
	}
	if !bytes.Equal(out[38:38+len(testICC)], testICC) {
		t.Error("profile bytes not embedded")
	}
	if w, h, ok := JPEGSize(out); !ok || w != 4032 || h != 3024 {
		t.Error("image header damaged")
	}
	if again := WithJPEGICC(out, testICC); len(again) != len(out) {
		t.Error("a JPEG with a profile got a second one")
	}
	big := bytes.Repeat([]byte{7}, 150_000)
	if got, want := len(WithJPEGICC(src, big)), len(src)+len(big)+3*(4+12+2); got != want {
		t.Errorf("chunked profile: len %d, want %d", got, want)
	}
}

// Opt-in end to end against a real photo and an ffmpeg binary:
//
//	LOKI_HEIC_SAMPLE=<photo.heic> LOKI_HEIC_FFMPEG=<ffmpeg> go test ./heic -run Render
//
// With ffmpeg older than 8.0 this must fail with the tile-grid error rather
// than return the wrong image.
func TestRenderRealHEIC(t *testing.T) {
	sample, ffmpeg := os.Getenv("LOKI_HEIC_SAMPLE"), os.Getenv("LOKI_HEIC_FFMPEG")
	if sample == "" || ffmpeg == "" {
		t.Skip("set LOKI_HEIC_SAMPLE and LOKI_HEIC_FFMPEG")
	}
	info, ok := ReadInfo(sample)
	if !ok || info.Width == 0 {
		t.Fatalf("ReadInfo: %+v %v", info, ok)
	}
	jpeg, err := Render(ffmpeg, sample)
	if err != nil {
		if strings.Contains(err.Error(), "tile grids") {
			t.Skipf("ffmpeg too old, refused as intended: %v", err)
		}
		t.Fatal(err)
	}
	w, h, _ := JPEGSize(jpeg)
	if w != info.Width || h != info.Height {
		t.Errorf("rendered %dx%d, want %dx%d", w, h, info.Width, info.Height)
	}
	if !jpegHasICC(jpeg) {
		t.Error("rendered JPEG has no ICC profile")
	}
	if out := os.Getenv("LOKI_HEIC_OUT"); out != "" {
		_ = os.WriteFile(out, jpeg, 0o644)
	}
}
