package tasks

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Reading the COLMAP model the pipeline produced, to tell Studio where the
// capture camera was. The trained .ply is in the same world space as the
// COLMAP model (Brush trains on its poses as-is), so a pose from here can be
// replayed exactly as the splat layer's opening camera.

type colmapImage struct {
	Name string
	Q    [4]float64 // cam_from_world rotation, (w, x, y, z)
	T    [3]float64 // cam_from_world translation
	Cam  uint32
}

type colmapCamera struct {
	Model         int32
	Width, Height uint64
	Params        []float64
}

// Parameter counts per COLMAP camera model id.
var colmapModelParams = map[int32]int{0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12}

func readColmapImages(modelDir string) ([]colmapImage, error) {
	f, err := os.Open(filepath.Join(modelDir, "images.bin"))
	if err != nil {
		return nil, err
	}
	defer f.Close()
	r := bufio.NewReader(f)
	var n uint64
	if err := binary.Read(r, binary.LittleEndian, &n); err != nil {
		return nil, err
	}
	out := make([]colmapImage, 0, n)
	for i := uint64(0); i < n; i++ {
		var head struct {
			ID  uint32
			Q   [4]float64
			T   [3]float64
			Cam uint32
		}
		if err := binary.Read(r, binary.LittleEndian, &head); err != nil {
			return nil, err
		}
		name, err := r.ReadString(0)
		if err != nil {
			return nil, err
		}
		var pts uint64
		if err := binary.Read(r, binary.LittleEndian, &pts); err != nil {
			return nil, err
		}
		if _, err := r.Discard(int(pts * 24)); err != nil { // (x, y double, point3D id uint64)
			return nil, err
		}
		out = append(out, colmapImage{Name: strings.TrimSuffix(name, "\x00"), Q: head.Q, T: head.T, Cam: head.Cam})
	}
	return out, nil
}

func readColmapCameras(modelDir string) (map[uint32]colmapCamera, error) {
	f, err := os.Open(filepath.Join(modelDir, "cameras.bin"))
	if err != nil {
		return nil, err
	}
	defer f.Close()
	r := bufio.NewReader(f)
	var n uint64
	if err := binary.Read(r, binary.LittleEndian, &n); err != nil {
		return nil, err
	}
	out := map[uint32]colmapCamera{}
	for i := uint64(0); i < n; i++ {
		var head struct {
			ID     uint32
			Model  int32
			Width  uint64
			Height uint64
		}
		if err := binary.Read(r, binary.LittleEndian, &head); err != nil {
			return nil, err
		}
		np, ok := colmapModelParams[head.Model]
		if !ok {
			return nil, fmt.Errorf("unknown COLMAP camera model %d", head.Model)
		}
		params := make([]float64, np)
		if err := binary.Read(r, binary.LittleEndian, params); err != nil {
			return nil, err
		}
		out[head.ID] = colmapCamera{Model: head.Model, Width: head.Width, Height: head.Height, Params: params}
	}
	return out, nil
}

// rotT returns Rᵀ·v for the rotation of unit quaternion q = (w, x, y, z).
func rotT(q [4]float64, v [3]float64) [3]float64 {
	w, x, y, z := q[0], q[1], q[2], q[3]
	n := math.Sqrt(w*w + x*x + y*y + z*z)
	w, x, y, z = w/n, x/n, y/n, z/n
	r := [3][3]float64{
		{1 - 2*(y*y+z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)},
		{2 * (x*y + w*z), 1 - 2*(x*x+z*z), 2 * (y*z - w*x)},
		{2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2*(x*x+y*y)},
	}
	var o [3]float64
	for i := 0; i < 3; i++ {
		o[i] = r[0][i]*v[0] + r[1][i]*v[1] + r[2][i]*v[2]
	}
	return o
}

// captureView summarises a capture for Studio: where its first frame was
// taken and which way "up" points on average over all frames.
type captureView struct {
	Pos, Forward, Up [3]float64 // first frame, world space (Up = camera's up)
	FovY             float64    // degrees
	WorldUp          [3]float64 // mean camera up over the capture
}

func colmapCaptureView(modelDir string) (captureView, error) {
	var cv captureView
	imgs, err := readColmapImages(modelDir)
	if err != nil || len(imgs) == 0 {
		return cv, fmt.Errorf("reading poses: %v", err)
	}
	cams, err := readColmapCameras(modelDir)
	if err != nil {
		return cv, fmt.Errorf("reading cameras: %v", err)
	}
	sort.Slice(imgs, func(a, b int) bool { return imgs[a].Name < imgs[b].Name })
	var up [3]float64
	for _, im := range imgs {
		u := rotT(im.Q, [3]float64{0, -1, 0}) // camera y points down
		for k := range up {
			up[k] += u[k]
		}
	}
	norm := math.Sqrt(up[0]*up[0] + up[1]*up[1] + up[2]*up[2])
	if norm < 1e-9 {
		return cv, fmt.Errorf("degenerate camera set")
	}
	for k := range up {
		cv.WorldUp[k] = up[k] / norm
	}
	first := imgs[0]
	c := rotT(first.Q, first.T)
	cv.Pos = [3]float64{-c[0], -c[1], -c[2]}
	cv.Forward = rotT(first.Q, [3]float64{0, 0, 1})
	cv.Up = rotT(first.Q, [3]float64{0, -1, 0})
	cv.FovY = 50
	if cam, ok := cams[first.Cam]; ok && cam.Height > 0 {
		fy := cam.Params[0]                                     // SIMPLE_* models share one focal
		if cam.Model == 1 || cam.Model == 4 || cam.Model == 6 { // PINHOLE / OPENCV / FULL_OPENCV: fx, fy
			fy = cam.Params[1]
		}
		if fy > 0 {
			cv.FovY = 2 * math.Atan(float64(cam.Height)/(2*fy)) * 180 / math.Pi
		}
	}
	return cv, nil
}

// plyCaptureComments renders the PLY header comments Studio reads
// (studio/splat.js parsePly): object-space vectors, so they survive any
// up-axis choice made later.
func plyCaptureComments(cv captureView) []string {
	v := func(a [3]float64) string { return fmt.Sprintf("%.6g %.6g %.6g", a[0], a[1], a[2]) }
	return []string{
		"comment lowkey_capture_camera " + v(cv.Pos) + " " + v(cv.Forward) + " " + v(cv.Up) + fmt.Sprintf(" %.4g", cv.FovY),
		"comment lowkey_capture_up " + v(cv.WorldUp),
	}
}

// addPlyComments rewrites path with extra comment lines after its
// "format" line (streamed: trained scenes are tens of MB).
func addPlyComments(path string, comments []string) error {
	in, err := os.Open(path)
	if err != nil {
		return err
	}
	defer in.Close()
	r := bufio.NewReaderSize(in, 1<<20)
	var header bytes.Buffer
	for {
		line, err := r.ReadString('\n')
		if err != nil {
			return fmt.Errorf("not a PLY header: %v", err)
		}
		header.WriteString(line)
		if strings.HasPrefix(line, "format ") {
			for _, c := range comments {
				header.WriteString(c + "\n")
			}
		}
		if strings.TrimSpace(line) == "end_header" {
			break
		}
		if header.Len() > 1<<16 {
			return fmt.Errorf("PLY header too long")
		}
	}
	tmp := path + ".partial"
	out, err := os.Create(tmp)
	if err != nil {
		return err
	}
	if _, err := out.Write(header.Bytes()); err == nil {
		_, err = io.Copy(out, r)
	}
	if cerr := out.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		os.Remove(tmp)
		return err
	}
	in.Close()
	return os.Rename(tmp, path)
}
