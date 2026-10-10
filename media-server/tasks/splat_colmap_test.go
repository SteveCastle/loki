package tasks

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeColmapModel writes a minimal binary model: one PINHOLE camera and
// the given images (no 2D points).
func writeColmapModel(t *testing.T, dir string, fy float64, h uint64, imgs []colmapImage) {
	t.Helper()
	var cams bytes.Buffer
	le := binary.LittleEndian
	_ = binary.Write(&cams, le, uint64(1))
	_ = binary.Write(&cams, le, uint32(1))
	_ = binary.Write(&cams, le, int32(1)) // PINHOLE
	_ = binary.Write(&cams, le, uint64(960))
	_ = binary.Write(&cams, le, h)
	_ = binary.Write(&cams, le, []float64{fy, fy, 480, float64(h) / 2})
	var ims bytes.Buffer
	_ = binary.Write(&ims, le, uint64(len(imgs)))
	for i, im := range imgs {
		_ = binary.Write(&ims, le, uint32(i+1))
		_ = binary.Write(&ims, le, im.Q)
		_ = binary.Write(&ims, le, im.T)
		_ = binary.Write(&ims, le, uint32(1))
		ims.WriteString(im.Name)
		ims.WriteByte(0)
		_ = binary.Write(&ims, le, uint64(2))
		ims.Write(make([]byte, 48)) // two (x, y, id) records
	}
	if err := os.WriteFile(filepath.Join(dir, "cameras.bin"), cams.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "images.bin"), ims.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
}

func near(a, b [3]float64) bool {
	for i := range a {
		if math.Abs(a[i]-b[i]) > 1e-6 {
			return false
		}
	}
	return true
}

func TestColmapCaptureView(t *testing.T) {
	dir := t.TempDir()
	// Identity rotation: camera looks down +Z with y down; t = -C.
	// A second frame, rotated 180° about Z, must not change the first-frame
	// view (names sort b > a) but does cancel half of the mean up.
	writeColmapModel(t, dir, 467.6, 540, []colmapImage{
		{Name: "b.jpg", Q: [4]float64{0, 0, 0, 1}, T: [3]float64{0, 0, 0}},
		{Name: "a.jpg", Q: [4]float64{1, 0, 0, 0}, T: [3]float64{-1, -2, -3}},
		{Name: "c.jpg", Q: [4]float64{1, 0, 0, 0}, T: [3]float64{0, 0, 0}},
	})
	cv, err := colmapCaptureView(dir)
	if err != nil {
		t.Fatal(err)
	}
	if !near(cv.Pos, [3]float64{1, 2, 3}) || !near(cv.Forward, [3]float64{0, 0, 1}) || !near(cv.Up, [3]float64{0, -1, 0}) {
		t.Errorf("first view = %+v", cv)
	}
	if math.Abs(cv.FovY-60) > 0.1 {
		t.Errorf("fov = %v; want ~60", cv.FovY)
	}
	if !near(cv.WorldUp, [3]float64{0, -1, 0}) {
		t.Errorf("world up = %v", cv.WorldUp)
	}
}

func TestAddPlyComments(t *testing.T) {
	p := filepath.Join(t.TempDir(), "s.ply")
	body := []byte{1, 2, 3, 10, 13, 0, 255}
	head := "ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty uchar x\nend_header\n"
	if err := os.WriteFile(p, append([]byte(head), body...), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := addPlyComments(p, []string{"comment lowkey_capture_up 0 1 0"}); err != nil {
		t.Fatal(err)
	}
	got, _ := os.ReadFile(p)
	want := "ply\nformat binary_little_endian 1.0\ncomment lowkey_capture_up 0 1 0\nelement vertex 1\n"
	if !strings.HasPrefix(string(got), want) || !bytes.HasSuffix(got, body) {
		t.Errorf("rewritten = %q", got)
	}
}
