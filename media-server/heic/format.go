// Package heic renders HEIC/HEIF photos as JPEG for clients that can't decode
// them (every browser but Safari). ffmpeg 8.0+ does the decoding — including
// the 512×512 tile grids iPhone photos are stored as — and this package puts
// back the ICC profile that grid assembly drops, caches the result, and
// refuses output from an older ffmpeg that decoded the wrong image.
//
// It mirrors src/main/heic-format.ts and src/main/heic.ts in the desktop app.
package heic

import (
	"bytes"
	"encoding/binary"
	"path/filepath"
	"strings"
)

// IsHeic reports whether name has a .heic or .heif extension.
func IsHeic(name string) bool {
	ext := strings.ToLower(filepath.Ext(name))
	return ext == ".heic" || ext == ".heif"
}

// Info is what the primary item's properties declare.
type Info struct {
	// ICC is the primary image's ICC profile, nil when it declares none.
	ICC []byte
	// Width and Height are the display size (after irot), 0 when undeclared.
	Width, Height int
	// Cropped is set when a clean-aperture crop applies, so the decoded size
	// won't match ispe.
	Cropped bool
}

type box struct {
	typ       string
	body, end int
}

// boxes returns the ISOBMFF boxes laid end to end in b[start:end], stopping
// at the first malformed header so a truncated read just yields fewer.
func boxes(b []byte, start, end int) []box {
	var out []box
	p := start
	for p+8 <= end {
		size := int(binary.BigEndian.Uint32(b[p:]))
		typ := string(b[p+4 : p+8])
		body := p + 8
		switch size {
		case 1:
			if p+16 > end {
				return out
			}
			large := binary.BigEndian.Uint64(b[p+8:])
			if large > uint64(end-p) {
				return out
			}
			size = int(large)
			body = p + 16
		case 0:
			size = end - p
		}
		if size < body-p || p+size > end {
			return out
		}
		out = append(out, box{typ, body, p + size})
		p += size
	}
	return out
}

func child(b []byte, parent box, typ string, skip int) (box, bool) {
	if parent.body+skip > parent.end {
		return box{}, false
	}
	for _, c := range boxes(b, parent.body+skip, parent.end) {
		if c.typ == typ {
			return c, true
		}
	}
	return box{}, false
}

// ParseMeta reads the primary item's ICC profile and display size from a
// buffer holding the top-level meta box (with or without the boxes around
// it). ok is false when there is no usable meta box.
func ParseMeta(b []byte) (info Info, ok bool) {
	// Malformed property payloads index past their box; treat as unparseable.
	defer func() {
		if recover() != nil {
			info, ok = Info{}, false
		}
	}()

	var meta box
	found := false
	for _, c := range boxes(b, 0, len(b)) {
		if c.typ == "meta" {
			meta, found = c, true
		}
	}
	if !found {
		return Info{}, false
	}
	// meta, pitm and ipma are FullBoxes: 1 byte version + 3 bytes flags.
	pitm, ok1 := child(b, meta, "pitm", 4)
	iprp, ok2 := child(b, meta, "iprp", 4)
	if !ok1 || !ok2 {
		return Info{}, false
	}
	var primary uint32
	if b[pitm.body] == 0 {
		primary = uint32(binary.BigEndian.Uint16(b[pitm.body+4:]))
	} else {
		primary = binary.BigEndian.Uint32(b[pitm.body+4:])
	}
	ipco, ok1 := child(b, iprp, "ipco", 0)
	ipma, ok2 := child(b, iprp, "ipma", 0)
	if !ok1 || !ok2 {
		return Info{}, false
	}
	props := boxes(b, ipco.body, ipco.end)

	// ipma: entry_count, then per item its id and a list of 1-based ipco
	// indices (7 or 15 bits; the top bit flags the property as essential).
	version := b[ipma.body]
	wide := b[ipma.body+3]&1 == 1
	p := ipma.body + 4
	entries := binary.BigEndian.Uint32(b[p:])
	p += 4
	var indices []int
	for i := uint32(0); i < entries && p < ipma.end; i++ {
		var item uint32
		if version < 1 {
			item = uint32(binary.BigEndian.Uint16(b[p:]))
			p += 2
		} else {
			item = binary.BigEndian.Uint32(b[p:])
			p += 4
		}
		n := int(b[p])
		p++
		list := make([]int, 0, n)
		for j := 0; j < n; j++ {
			if wide {
				list = append(list, int(binary.BigEndian.Uint16(b[p:])&0x7fff))
				p += 2
			} else {
				list = append(list, int(b[p]&0x7f))
				p++
			}
		}
		if item == primary {
			indices = list
		}
	}

	quarterTurns := 0
	for _, idx := range indices {
		if idx < 1 || idx > len(props) {
			continue
		}
		prop := props[idx-1]
		switch prop.typ {
		case "colr":
			kind := string(b[prop.body : prop.body+4])
			if kind == "prof" || kind == "rICC" {
				info.ICC = append([]byte(nil), b[prop.body+4:prop.end]...)
			}
		case "ispe":
			info.Width = int(binary.BigEndian.Uint32(b[prop.body+4:]))
			info.Height = int(binary.BigEndian.Uint32(b[prop.body+8:]))
		case "irot":
			quarterTurns = int(b[prop.body] & 3)
		case "clap":
			info.Cropped = true
		}
	}
	if quarterTurns%2 == 1 {
		info.Width, info.Height = info.Height, info.Width
	}
	return info, true
}

const (
	markerSOI  = 0xd8
	markerSOS  = 0xda
	markerAPP0 = 0xe0
	markerAPP2 = 0xe2
)

var iccTag = []byte("ICC_PROFILE\x00")

// Segment length is a u16 that counts itself, the tag and the seq/count bytes.
var iccChunk = 0xffff - 2 - len(iccTag) - 2

type segment struct {
	marker           byte
	start, body, end int
}

// jpegSegments returns the marker segments up to and including SOS.
func jpegSegments(b []byte) []segment {
	if len(b) < 4 || b[0] != 0xff || b[1] != markerSOI {
		return nil
	}
	var out []segment
	p := 2
	for p+4 <= len(b) && b[p] == 0xff {
		m := b[p+1]
		n := int(binary.BigEndian.Uint16(b[p+2:]))
		out = append(out, segment{m, p, p + 4, p + 2 + n})
		if m == markerSOS {
			break
		}
		p += 2 + n
	}
	return out
}

// JPEGSize returns the pixel size from the JPEG's SOF header.
func JPEGSize(b []byte) (width, height int, ok bool) {
	for _, s := range jpegSegments(b) {
		m := s.marker
		// SOF0–SOF15, minus DHT (C4), JPG (C8) and DAC (CC).
		if m >= 0xc0 && m <= 0xcf && m != 0xc4 && m != 0xc8 && m != 0xcc && s.body+5 <= len(b) {
			return int(binary.BigEndian.Uint16(b[s.body+3:])), int(binary.BigEndian.Uint16(b[s.body+1:])), true
		}
	}
	return 0, 0, false
}

func jpegHasICC(b []byte) bool {
	for _, s := range jpegSegments(b) {
		if s.marker == markerAPP2 && s.body+len(iccTag) <= len(b) && bytes.Equal(b[s.body:s.body+len(iccTag)], iccTag) {
			return true
		}
	}
	return false
}

// WithJPEGICC returns jpeg with icc embedded as APP2 ICC_PROFILE segment(s)
// after SOI/APP0. A JPEG that already carries a profile is returned as-is.
func WithJPEGICC(jpeg, icc []byte) []byte {
	if len(icc) == 0 || jpegHasICC(jpeg) {
		return jpeg
	}
	count := (len(icc) + iccChunk - 1) / iccChunk
	if count > 255 {
		return jpeg
	}
	insertAt := 2
	if segs := jpegSegments(jpeg); len(segs) > 0 && segs[0].marker == markerAPP0 {
		insertAt = segs[0].end
	}
	out := make([]byte, 0, len(jpeg)+len(icc)+count*(4+len(iccTag)+2))
	out = append(out, jpeg[:insertAt]...)
	for i := 0; i < count; i++ {
		data := icc[i*iccChunk : min((i+1)*iccChunk, len(icc))]
		n := 2 + len(iccTag) + 2 + len(data)
		out = append(out, 0xff, markerAPP2, byte(n>>8), byte(n))
		out = append(out, iccTag...)
		out = append(out, byte(i+1), byte(count))
		out = append(out, data...)
	}
	return append(out, jpeg[insertAt:]...)
}
