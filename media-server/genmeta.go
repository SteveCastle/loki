package main

// Embedded generation text — the metadata AI image tools leave inside the
// files they write. This only EXTRACTS it: every text record a file carries
// comes back as keyword → text and the SPA decides what it means (ComfyUI
// `prompt` / `workflow` JSON, a WebUI `parameters` block…). It mirrors
// src/main/generation-text.ts in the Electron app so both backends feed
// the renderer the same shape.
//
//	PNG   tEXt / zTXt / iTXt chunks
//	WebP  the RIFF EXIF chunk → TIFF directory (ComfyUI: Model=prompt:…,
//	      Make=workflow:…; WebUI: UserComment)
//	JPEG  APP1 Exif (same TIFF layout) and COM comment segments
//	video container tags via ffprobe (VideoHelperSuite puts
//	      {"prompt":…,"workflow":…} in `comment`)

import (
	"bytes"
	"compress/zlib"
	"encoding/binary"
	"encoding/json"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"unicode/utf16"

	depspkg "github.com/stevecastle/shrike/deps"
	"github.com/stevecastle/shrike/platform"
)

const genTextMax = 64 << 20

var pngSig = []byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'}

var videoExt = map[string]bool{".mp4": true, ".m4v": true, ".mov": true, ".webm": true, ".mkv": true, ".avi": true}

// readGenerationText returns every embedded text record in the file, or nil
// when it has none. It never fails loudly: a file we can't open or don't
// understand just has no generation text.
func readGenerationText(path string) map[string]string {
	if videoExt[strings.ToLower(filepath.Ext(path))] {
		return nonEmpty(readVideoText(path))
	}
	f, err := os.Open(path)
	if err != nil {
		return nil
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return nil
	}
	head := make([]byte, 12)
	if _, err := io.ReadFull(f, head); err != nil {
		return nil
	}
	switch {
	case bytes.Equal(head[:8], pngSig):
		return nonEmpty(readPngText(f, st.Size()))
	case string(head[:4]) == "RIFF" && string(head[8:12]) == "WEBP":
		return nonEmpty(readWebpText(f, st.Size()))
	case head[0] == 0xff && head[1] == 0xd8:
		return nonEmpty(readJpegText(f, st.Size()))
	}
	return nil
}

func nonEmpty(m map[string]string) map[string]string {
	if len(m) == 0 {
		return nil
	}
	return m
}

func readAt(r io.ReaderAt, pos int64, n int64) []byte {
	if n <= 0 || n > genTextMax {
		return nil
	}
	buf := make([]byte, n)
	got, _ := r.ReadAt(buf, pos)
	return buf[:got]
}

// ---- PNG ----------------------------------------------------------------

func readPngText(r io.ReaderAt, size int64) map[string]string {
	out := map[string]string{}
	pos := int64(len(pngSig))
	for pos+8 <= size {
		head := readAt(r, pos, 8)
		if len(head) < 8 {
			break
		}
		n := int64(binary.BigEndian.Uint32(head[:4]))
		typ := string(head[4:8])
		dataPos := pos + 8
		if typ == "IEND" {
			break
		}
		if (typ == "tEXt" || typ == "zTXt" || typ == "iTXt") && n <= genTextMax {
			if k, v, ok := decodePngTextChunk(typ, readAt(r, dataPos, n)); ok && v != "" {
				out[k] = v
			}
		}
		pos = dataPos + n + 4 // data + CRC; IDAT is skipped, not read
	}
	return out
}

func decodePngTextChunk(typ string, data []byte) (string, string, bool) {
	nul := bytes.IndexByte(data, 0)
	if nul < 0 {
		return "", "", false
	}
	keyword := string(data[:nul])
	switch typ {
	case "tEXt":
		return keyword, latin1(data[nul+1:]), true
	case "zTXt":
		if nul+2 > len(data) {
			return "", "", false
		}
		text, ok := inflate(data[nul+2:])
		return keyword, latin1(text), ok
	default: // iTXt: keyword \0 flag(1) method(1) language \0 translated \0 text
		if nul+3 > len(data) {
			return "", "", false
		}
		compressed := data[nul+1] == 1
		p := nul + 3
		for i := 0; i < 2; i++ {
			j := bytes.IndexByte(data[p:], 0)
			if j < 0 {
				return "", "", false
			}
			p += j + 1
		}
		body := data[p:]
		if compressed {
			text, ok := inflate(body)
			return keyword, string(text), ok
		}
		return keyword, string(body), true
	}
}

func inflate(b []byte) ([]byte, bool) {
	zr, err := zlib.NewReader(bytes.NewReader(b))
	if err != nil {
		return nil, false
	}
	defer zr.Close()
	out, err := io.ReadAll(io.LimitReader(zr, genTextMax))
	if err != nil && len(out) == 0 {
		return nil, false
	}
	return out, true
}

// latin1 maps each byte to the rune of the same value (PNG tEXt is Latin-1).
func latin1(b []byte) string {
	if !hasHighBytes(b) {
		return string(b)
	}
	rs := make([]rune, len(b))
	for i, c := range b {
		rs[i] = rune(c)
	}
	return string(rs)
}

func hasHighBytes(b []byte) bool {
	for _, c := range b {
		if c >= 0x80 {
			return true
		}
	}
	return false
}

// ---- TIFF / Exif ----------------------------------------------------------

const (
	tagImageDescription = 0x010e
	tagMake             = 0x010f
	tagModel            = 0x0110
	tagExifIFD          = 0x8769
	tagUserComment      = 0x9286
)

var tiffTypeSize = map[uint16]uint32{1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 7: 1, 9: 4, 10: 8}

// readTiffText returns the text-bearing tags of a TIFF blob (IFD0 plus the
// Exif sub-IFD). Tolerates truncated or malformed directories.
func readTiffText(tiff []byte) map[string]string {
	out := map[string]string{}
	if len(tiff) < 8 {
		return out
	}
	var bo binary.ByteOrder
	switch {
	case tiff[0] == 'I' && tiff[1] == 'I':
		bo = binary.LittleEndian
	case tiff[0] == 'M' && tiff[1] == 'M':
		bo = binary.BigEndian
	default:
		return out
	}
	visited := map[uint32]bool{}
	var walk func(ifd uint32)
	walk = func(ifd uint32) {
		if ifd < 8 || int(ifd)+2 > len(tiff) || visited[ifd] {
			return
		}
		visited[ifd] = true
		n := int(bo.Uint16(tiff[ifd:]))
		for i := 0; i < n; i++ {
			e := int(ifd) + 2 + i*12
			if e+12 > len(tiff) {
				return
			}
			tag := bo.Uint16(tiff[e:])
			typ := bo.Uint16(tiff[e+2:])
			count := bo.Uint32(tiff[e+4:])
			unit, ok := tiffTypeSize[typ]
			if !ok {
				unit = 1
			}
			byteLen := uint64(count) * uint64(unit)
			if tag == tagExifIFD {
				walk(bo.Uint32(tiff[e+8:]))
				continue
			}
			valuePos := uint64(e + 8)
			if byteLen > 4 {
				valuePos = uint64(bo.Uint32(tiff[e+8:]))
			}
			if valuePos+byteLen > uint64(len(tiff)) {
				continue
			}
			raw := tiff[valuePos : valuePos+byteLen]
			switch tag {
			case tagUserComment:
				if t := decodeUserComment(raw, bo == binary.LittleEndian); t != "" {
					out["UserComment"] = t
				}
			case tagImageDescription, tagMake, tagModel:
				t := strings.TrimRight(string(raw), "\x00")
				if t == "" {
					continue
				}
				name := "Model"
				if tag == tagImageDescription {
					name = "ImageDescription"
				} else if tag == tagMake {
					name = "Make"
				}
				out[name] = t
			}
		}
	}
	walk(bo.Uint32(tiff[4:]))
	return out
}

// decodeUserComment: the value opens with an 8-byte character code.
func decodeUserComment(raw []byte, le bool) string {
	if len(raw) < 8 {
		return strings.TrimRight(string(raw), "\x00")
	}
	code := string(raw[:8])
	body := raw[8:]
	var text string
	switch {
	case strings.HasPrefix(code, "UNICODE"):
		// Pillow (WebUI) writes big-endian whatever the TIFF order says;
		// sniff: real text has a zero high byte first in BE.
		be := len(body) >= 2 && body[0] == 0 && body[1] != 0
		text = decodeUTF16(body, be || !le)
	case strings.HasPrefix(code, "ASCII"), strings.HasPrefix(code, "JIS"), strings.Trim(code, "\x00") == "":
		text = string(body)
	default:
		text = string(raw)
	}
	return strings.TrimRight(text, "\x00")
}

func decodeUTF16(b []byte, bigEndian bool) string {
	u := make([]uint16, len(b)/2)
	for i := range u {
		if bigEndian {
			u[i] = binary.BigEndian.Uint16(b[2*i:])
		} else {
			u[i] = binary.LittleEndian.Uint16(b[2*i:])
		}
	}
	return string(utf16.Decode(u))
}

// ---- WebP -----------------------------------------------------------------

var xmpDescription = regexp.MustCompile(`(?s)<dc:description>.*?<rdf:li[^>]*>(.*?)</rdf:li>`)

func readWebpText(r io.ReaderAt, size int64) map[string]string {
	out := map[string]string{}
	pos := int64(12)
	for pos+8 <= size {
		head := readAt(r, pos, 8)
		if len(head) < 8 {
			break
		}
		fourcc := string(head[:4])
		n := int64(binary.LittleEndian.Uint32(head[4:]))
		switch {
		case fourcc == "EXIF" && n <= genTextMax:
			data := readAt(r, pos+8, n)
			data = bytes.TrimPrefix(data, []byte("Exif\x00\x00"))
			for k, v := range readTiffText(data) {
				out[k] = v
			}
		case fourcc == "XMP " && n <= genTextMax:
			if m := xmpDescription.FindSubmatch(readAt(r, pos+8, n)); m != nil {
				out["XMPDescription"] = string(m[1])
			}
		}
		pos += 8 + n + (n & 1)
	}
	return out
}

// ---- JPEG -----------------------------------------------------------------

func readJpegText(r io.ReaderAt, size int64) map[string]string {
	out := map[string]string{}
	var comments []string
	pos := int64(2)
	for pos+4 <= size {
		head := readAt(r, pos, 4)
		if len(head) < 4 || head[0] != 0xff {
			break
		}
		marker := head[1]
		if marker == 0xd9 || marker == 0xda { // EOI / SOS: no more headers
			break
		}
		if marker == 0xff {
			pos++
			continue
		}
		n := int64(binary.BigEndian.Uint16(head[2:]))
		if n < 2 {
			break
		}
		body := n - 2
		switch {
		case marker == 0xe1 && body <= genTextMax:
			data := readAt(r, pos+4, body)
			if bytes.HasPrefix(data, []byte("Exif\x00\x00")) {
				for k, v := range readTiffText(data[6:]) {
					out[k] = v
				}
			}
		case marker == 0xfe && body <= genTextMax:
			if t := strings.TrimRight(string(readAt(r, pos+4, body)), "\x00"); t != "" {
				comments = append(comments, t)
			}
		}
		pos += 2 + n
	}
	if len(comments) > 0 {
		out["Comment"] = strings.Join(comments, "\n")
	}
	return out
}

// ---- video containers -----------------------------------------------------

func readVideoText(path string) map[string]string {
	ffprobe := depspkg.BundledOrEmpty("ffprobe")
	if ffprobe == "" {
		if p, err := exec.LookPath("ffprobe"); err == nil {
			ffprobe = p
		} else {
			return nil
		}
	}
	cmd := exec.Command(ffprobe, "-v", "error", "-show_entries", "format_tags", "-of", "json", path)
	platform.HideSubprocessWindow(cmd)
	stdout, err := cmd.Output()
	if err != nil {
		return nil
	}
	var probe struct {
		Format struct {
			Tags map[string]string `json:"tags"`
		} `json:"format"`
	}
	if json.Unmarshal(stdout, &probe) != nil {
		return nil
	}
	out := map[string]string{}
	for k, v := range probe.Format.Tags {
		if v != "" {
			out[k] = v
		}
	}
	return out
}
