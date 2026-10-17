package main

import (
	"bytes"
	"compress/zlib"
	"encoding/binary"
	"hash/crc32"
	"os"
	"path/filepath"
	"testing"
	"unicode/utf16"
)

func pngChunk(typ string, data []byte) []byte {
	var b bytes.Buffer
	binary.Write(&b, binary.BigEndian, uint32(len(data)))
	body := append([]byte(typ), data...)
	b.Write(body)
	binary.Write(&b, binary.BigEndian, crc32.ChecksumIEEE(body))
	return b.Bytes()
}

func testPng(chunks ...[]byte) []byte {
	ihdr := make([]byte, 13)
	binary.BigEndian.PutUint32(ihdr, 1)
	binary.BigEndian.PutUint32(ihdr[4:], 1)
	ihdr[8], ihdr[9] = 8, 6
	var idat bytes.Buffer
	zw := zlib.NewWriter(&idat)
	zw.Write(make([]byte, 5))
	zw.Close()
	out := append([]byte{}, pngSig...)
	out = append(out, pngChunk("IHDR", ihdr)...)
	for _, c := range chunks {
		out = append(out, c...)
	}
	out = append(out, pngChunk("IDAT", idat.Bytes())...)
	return append(out, pngChunk("IEND", nil)...)
}

func deflate(b []byte) []byte {
	var out bytes.Buffer
	zw := zlib.NewWriter(&out)
	zw.Write(b)
	zw.Close()
	return out.Bytes()
}

// testTiff builds a little-endian TIFF: IFD0 {Make, Model, ExifIFD} and an
// Exif sub-IFD holding UserComment.
func testTiff(makeTag, model string, userComment []byte) []byte {
	type entry struct {
		tag, typ uint16
		val      []byte
	}
	var ifd0 []entry
	if makeTag != "" {
		ifd0 = append(ifd0, entry{0x010f, 2, append([]byte(makeTag), 0)})
	}
	if model != "" {
		ifd0 = append(ifd0, entry{0x0110, 2, append([]byte(model), 0)})
	}
	var exif []entry
	if userComment != nil {
		exif = append(exif, entry{0x9286, 7, userComment})
	}
	n0 := len(ifd0)
	if len(exif) > 0 {
		n0++
	}
	ifd0Pos := 8
	exifPos := ifd0Pos + 2 + 12*n0 + 4
	exifLen := 0
	if len(exif) > 0 {
		exifLen = 2 + 12*len(exif) + 4
	}
	dataPos := uint32(exifPos + exifLen)
	var data []byte
	writeEntry := func(b *bytes.Buffer, e entry) {
		binary.Write(b, binary.LittleEndian, e.tag)
		binary.Write(b, binary.LittleEndian, e.typ)
		binary.Write(b, binary.LittleEndian, uint32(len(e.val)))
		if len(e.val) <= 4 {
			v := make([]byte, 4)
			copy(v, e.val)
			b.Write(v)
		} else {
			binary.Write(b, binary.LittleEndian, dataPos)
			data = append(data, e.val...)
			dataPos += uint32(len(e.val))
		}
	}
	var b bytes.Buffer
	b.Write([]byte{'I', 'I', 0x2a, 0, 8, 0, 0, 0})
	binary.Write(&b, binary.LittleEndian, uint16(n0))
	for _, e := range ifd0 {
		writeEntry(&b, e)
	}
	if len(exif) > 0 {
		binary.Write(&b, binary.LittleEndian, uint16(0x8769))
		binary.Write(&b, binary.LittleEndian, uint16(4))
		binary.Write(&b, binary.LittleEndian, uint32(1))
		binary.Write(&b, binary.LittleEndian, uint32(exifPos))
	}
	b.Write(make4())
	if len(exif) > 0 {
		binary.Write(&b, binary.LittleEndian, uint16(len(exif)))
		for _, e := range exif {
			writeEntry(&b, e)
		}
		b.Write(make4())
	}
	b.Write(data)
	return b.Bytes()
}

func make4() []byte { return make([]byte, 4) }

func testWebp(exif []byte) []byte {
	chunk := func(fourcc string, data []byte) []byte {
		var b bytes.Buffer
		b.WriteString(fourcc)
		binary.Write(&b, binary.LittleEndian, uint32(len(data)))
		b.Write(data)
		if len(data)%2 == 1 {
			b.WriteByte(0)
		}
		return b.Bytes()
	}
	body := append([]byte("WEBP"), chunk("VP8X", make([]byte, 10))...)
	body = append(body, chunk("EXIF", exif)...)
	body = append(body, chunk("VP8 ", make([]byte, 3))...)
	var b bytes.Buffer
	b.WriteString("RIFF")
	binary.Write(&b, binary.LittleEndian, uint32(len(body)))
	b.Write(body)
	return b.Bytes()
}

func testJpeg(exif []byte, comment string) []byte {
	seg := func(marker byte, data []byte) []byte {
		var b bytes.Buffer
		b.Write([]byte{0xff, marker})
		binary.Write(&b, binary.BigEndian, uint16(len(data)+2))
		b.Write(data)
		return b.Bytes()
	}
	out := []byte{0xff, 0xd8}
	if exif != nil {
		out = append(out, seg(0xe1, append([]byte("Exif\x00\x00"), exif...))...)
	}
	if comment != "" {
		out = append(out, seg(0xfe, []byte(comment))...)
	}
	out = append(out, seg(0xdb, make([]byte, 65))...)
	out = append(out, 0xff, 0xda, 0, 2)
	out = append(out, make([]byte, 20)...)
	return append(out, 0xff, 0xd9)
}

func utf16be(s string) []byte {
	var out []byte
	for _, u := range utf16.Encode([]rune(s)) {
		out = append(out, byte(u>>8), byte(u))
	}
	return out
}

func writeGenFile(t *testing.T, name string, data []byte) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(p, data, 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func expectChunks(t *testing.T, got, want map[string]string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("got %d records %v, want %d %v", len(got), got, len(want), want)
	}
	for k, v := range want {
		if got[k] != v {
			t.Errorf("%s = %q, want %q", k, got[k], v)
		}
	}
}

func TestGenerationTextPNG(t *testing.T) {
	prompt := `{"3":{"class_type":"KSampler","inputs":{"seed":1}}}`
	file := writeGenFile(t, "comfy.png", testPng(
		pngChunk("tEXt", []byte("prompt\x00"+prompt)),
		pngChunk("zTXt", append([]byte("workflow\x00\x00"), deflate([]byte(`{"nodes":[]}`))...)),
		pngChunk("iTXt", []byte("parameters\x00\x00\x00en\x00\x00über — Steps: 1")),
	))
	expectChunks(t, readGenerationText(file), map[string]string{
		"prompt":     prompt,
		"workflow":   `{"nodes":[]}`,
		"parameters": "über — Steps: 1",
	})
	// Lowkey's own tools (loki-retouch) stamp one JSON record in an iTXt
	// chunk keyed `loki`; it comes back verbatim for the SPA to interpret.
	loki := `{"loki":1,"tool":"loki-retouch","version":"0.1.0","model":"Qwen Image 2.1","prompt":"über — make it night","seed":3,"steps":25,"inputs":[{"tag":"<image1>","path":"photo.jpg","width":1024,"height":768}]}`
	expectChunks(t, readGenerationText(writeGenFile(t, "retouch.png", testPng(
		pngChunk("iTXt", []byte("loki\x00\x00\x00\x00\x00"+loki)),
	))), map[string]string{"loki": loki})
	if got := readGenerationText(writeGenFile(t, "plain.png", testPng())); got != nil {
		t.Errorf("plain png: got %v, want nil", got)
	}
	if got := readGenerationText(writeGenFile(t, "junk.bin", []byte("not an image"))); got != nil {
		t.Errorf("junk: got %v, want nil", got)
	}
	if got := readGenerationText(filepath.Join(t.TempDir(), "missing.png")); got != nil {
		t.Errorf("missing: got %v, want nil", got)
	}
}

func TestGenerationTextWebPAndJPEG(t *testing.T) {
	uc := append([]byte("UNICODE\x00"), utf16be("a cat\nSteps: 20, Seed: 5")...)
	webp := writeGenFile(t, "comfy.webp", testWebp(testTiff(`workflow:{"nodes":[]}`, `prompt:{"1":{}}`, uc)))
	expectChunks(t, readGenerationText(webp), map[string]string{
		"Make":        `workflow:{"nodes":[]}`,
		"Model":       `prompt:{"1":{}}`,
		"UserComment": "a cat\nSteps: 20, Seed: 5",
	})

	ascii := append([]byte("ASCII\x00\x00\x00"), []byte("prompt text\nSteps: 9")...)
	jpg := writeGenFile(t, "webui.jpg", testJpeg(testTiff("", "", ascii), "made with love"))
	expectChunks(t, readGenerationText(jpg), map[string]string{
		"UserComment": "prompt text\nSteps: 9",
		"Comment":     "made with love",
	})

	// A truncated directory is just empty, never a panic.
	tiff := testTiff("workflow:{}", "prompt:{}", nil)
	if got := readTiffText(tiff[:20]); len(got) != 0 {
		t.Errorf("truncated tiff: got %v", got)
	}
	if got := readTiffText([]byte("nope")); len(got) != 0 {
		t.Errorf("garbage tiff: got %v", got)
	}
}
