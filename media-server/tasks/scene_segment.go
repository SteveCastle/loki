package tasks

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha1"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/png"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// --- Scene editor selection: loki-segment (SAM 3.1) behind the editor ---
//
// The editor picks objects (a click, a box, "person", every object) on the
// picture it shows, and on frames of a motion layer's clip. Each scene gets
// one `loki-segment --serve` process: models load once (~1.5 s), a picture
// is encoded once (~100 ms) and each pick then costs ~5-25 ms. Sessions close
// after segSessionIdle without use.
//
// VRAM: a session holds ~2.2 GB (stills; ~5 GB for video). The diffusion
// engines take ~20 GB, so while one runs (gpuHeavy > 0) every session is
// closed and each pick runs in a short-lived session that closes right after:
// slower (the models load each time) but it never holds memory against a
// generation.

const segSessionIdle = 3 * time.Minute

// segBinary finds loki-segment: $LOKI_SEGMENT_BIN, else PATH. A variable so
// tests can stand in a fake engine.
var segBinary = func() (string, error) {
	if p := strings.TrimSpace(os.Getenv("LOKI_SEGMENT_BIN")); p != "" {
		return p, nil
	}
	if p, err := exec.LookPath("loki-segment"); err == nil {
		return p, nil
	}
	return "", errors.New("loki-segment is not installed (put it on PATH: tools/loki-segment)")
}

// gpuHeavy counts diffusion / tracking runs holding the GPU right now.
var gpuHeavy atomic.Int32

// beginGPUHeavy marks a GPU-heavy run (retouch, reshoot, splat, tracking):
// selection sessions let go of their VRAM until it ends. Call the returned
// func when it ends.
func beginGPUHeavy() func() {
	gpuHeavy.Add(1)
	closeSegSessions()
	return func() { gpuHeavy.Add(-1) }
}

type segSession struct {
	mu     sync.Mutex
	cmd    *exec.Cmd
	in     io.WriteCloser
	out    *bufio.Reader
	opened string // the image or video open in it
	used   time.Time
}

var segSessions = struct {
	sync.Mutex
	m map[string]*segSession
}{m: map[string]*segSession{}}

var segReaper sync.Once

func closeSegSessions() {
	segSessions.Lock()
	all := segSessions.m
	segSessions.m = map[string]*segSession{}
	segSessions.Unlock()
	for _, s := range all {
		s.mu.Lock()
		s.close()
		s.mu.Unlock()
	}
}

// close ends the process (the caller holds s.mu).
func (s *segSession) close() {
	if s.cmd == nil {
		return
	}
	_ = s.in.Close()
	done := make(chan struct{})
	go func() { _ = s.cmd.Wait(); close(done) }()
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		_ = s.cmd.Process.Kill()
	}
	s.cmd = nil
	s.opened = ""
}

func (s *segSession) start() error {
	bin, err := segBinary()
	if err != nil {
		return err
	}
	cmd := exec.Command(bin, "--serve", "-q")
	cmd.Env = append(os.Environ(), aiToolEnv(aiToolSpec{Bin: "loki-segment"})...)
	in, err := cmd.StdinPipe()
	if err != nil {
		return err
	}
	out, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("starting loki-segment: %w", err)
	}
	s.cmd, s.in, s.out, s.opened = cmd, in, bufio.NewReaderSize(out, 1<<16), ""
	return nil
}

// ask sends one request and reads its one-line answer (the caller holds s.mu).
// A request that doesn't answer in time takes the process down with it.
func (s *segSession) ask(req map[string]any, timeout time.Duration) (map[string]any, error) {
	b, _ := json.Marshal(req)
	if _, err := s.in.Write(append(b, '\n')); err != nil {
		s.close()
		return nil, fmt.Errorf("loki-segment stopped: %w", err)
	}
	type answer struct {
		line string
		err  error
	}
	ch := make(chan answer, 1)
	go func() {
		line, err := s.out.ReadString('\n')
		ch <- answer{line, err}
	}()
	select {
	case a := <-ch:
		if a.err != nil {
			s.close()
			return nil, fmt.Errorf("loki-segment stopped: %w", a.err)
		}
		var resp map[string]any
		if err := json.Unmarshal([]byte(a.line), &resp); err != nil {
			return nil, fmt.Errorf("loki-segment answered %q", strings.TrimSpace(a.line))
		}
		if e, ok := resp["error"].(string); ok {
			return nil, errors.New(e)
		}
		return resp, nil
	case <-time.After(timeout):
		_ = s.cmd.Process.Kill()
		s.close()
		return nil, errors.New("loki-segment did not answer in time")
	}
}

// segRequest runs one segment request on path in scene id's session.
func segRequest(id, path string, req map[string]any) (map[string]any, error) {
	segReaper.Do(func() {
		go func() {
			for range time.Tick(30 * time.Second) {
				segSessions.Lock()
				for k, s := range segSessions.m {
					if s.mu.TryLock() {
						if time.Since(s.used) > segSessionIdle {
							s.close()
							delete(segSessions.m, k)
						}
						s.mu.Unlock()
					}
				}
				segSessions.Unlock()
			}
		}()
	})
	segSessions.Lock()
	s := segSessions.m[id]
	if s == nil {
		s = &segSession{}
		segSessions.m[id] = s
	}
	segSessions.Unlock()

	s.mu.Lock()
	defer s.mu.Unlock()
	s.used = time.Now()
	// While a generation holds the GPU, nothing stays loaded between picks.
	if gpuHeavy.Load() > 0 {
		defer func() {
			s.close()
			segSessions.Lock()
			delete(segSessions.m, id)
			segSessions.Unlock()
		}()
	}
	if s.cmd == nil {
		if err := s.start(); err != nil {
			return nil, err
		}
	}
	if s.opened != path {
		if _, err := s.ask(map[string]any{"cmd": "open", "path": path}, 2*time.Minute); err != nil {
			return nil, err
		}
		s.opened = path
	}
	req["cmd"] = "segment"
	return s.ask(req, time.Minute)
}

// SegmentRequest is one pick in the editor. Points are [x, y, 1|0] (1 = this,
// 0 = not this) and Box is [x, y, w, h], in the pixels of what is segmented:
// the picture (frame size), or with Layer + Version a motion version's clip
// frame Frame (clip pixels).
type SegmentRequest struct {
	Text       string       `json:"text,omitempty"`
	Points     [][3]float64 `json:"points,omitempty"`
	Box        []float64    `json:"box,omitempty"`
	Everything bool         `json:"everything,omitempty"`
	Multimask  bool         `json:"multimask,omitempty"`
	Layer      string       `json:"layer,omitempty"`
	Version    *int         `json:"version,omitempty"`
	Frame      int          `json:"frame,omitempty"`
}

// SegmentObject is one object found: its mask (white, alpha = coverage) as a
// PNG data URL covering Rect, its box, and for clicks/boxes the other
// candidate readings (part / object / whole) when asked for.
type SegmentObject struct {
	ID         int             `json:"id"`
	Label      string          `json:"label"`
	Score      float64         `json:"score"`
	BBox       SceneRect       `json:"bbox"`
	Mask       string          `json:"mask"`
	Rect       SceneRect       `json:"rect"`
	Candidates []SegmentObject `json:"candidates,omitempty"`
}

type SegmentResult struct {
	Width   int             `json:"width"`
	Height  int             `json:"height"`
	Objects []SegmentObject `json:"objects"`
	Ms      int64           `json:"ms"`
}

// pictureSigs remembers what each scene's selection picture was flattened
// from, so it is only rebuilt when the layers change.
var pictureSigs sync.Map // scene id -> signature

// scenePicture is what the editor shows (every visible still and paint
// layer over the base) flattened to <dir>/seg-picture.png.
func scenePicture(ctx context.Context, s *Scene, dir string) (string, error) {
	layers, err := sceneGraphLayers(dir, s.Width, s.Height, s.Layers, true)
	if err != nil {
		return "", err
	}
	h := sha1.New()
	fmt.Fprintf(h, "%s|%dx%d", s.Base, s.Width, s.Height)
	for _, g := range layers {
		fmt.Fprintf(h, "|%s|%v", g.Media, g.Region)
		if b, err := os.ReadFile(g.Mask); err == nil {
			h.Write(b)
		}
	}
	sig := hex.EncodeToString(h.Sum(nil))
	out := filepath.Join(dir, "seg-picture.png")
	if prev, ok := pictureSigs.Load(s.ID); ok && prev == sig && fileExists(out) {
		return out, nil
	}
	tmp := filepath.Join(dir, "seg-picture.tmp.png")
	if err := runSceneFFmpeg(ctx, append([]string{"-y"}, append(sceneGraphArgs(s.Base, s.Width, s.Height, layers, false, 0, SceneRect{}), tmp)...)); err != nil {
		return "", fmt.Errorf("flattening the picture: %w", err)
	}
	if err := os.Rename(tmp, out); err != nil {
		return "", err
	}
	pictureSigs.Store(s.ID, sig)
	return out, nil
}

// SegmentScene answers one pick on scene id.
func SegmentScene(ctx context.Context, id string, in SegmentRequest) (*SegmentResult, error) {
	sceneMu.Lock()
	s, err := loadSceneLocked(id)
	sceneMu.Unlock()
	if err != nil {
		return nil, err
	}
	dir, _ := sceneDir(id)
	var path string
	frame := -1
	if in.Layer != "" && in.Version != nil {
		for _, l := range s.Layers {
			if l.ID == in.Layer && l.Kind == SceneKindMotion && *in.Version >= 0 && *in.Version < len(l.Versions) {
				path = l.Versions[*in.Version].Path
			}
		}
		if path == "" {
			return nil, ErrLayerNotFound
		}
		frame = max(0, in.Frame)
	} else if path, err = scenePicture(ctx, s, dir); err != nil {
		return nil, err
	}

	out := filepath.Join(dir, "seg-out")
	_ = os.RemoveAll(out)
	req := map[string]any{"out": out}
	if t := strings.TrimSpace(in.Text); t != "" {
		req["text"] = t
	}
	if len(in.Points) > 0 {
		req["points"] = in.Points
	}
	if len(in.Box) == 4 {
		req["box"] = in.Box
	}
	if in.Everything {
		req["everything"] = true
	}
	if in.Multimask {
		req["multimask"] = true
	}
	if frame >= 0 {
		req["frame"] = frame
	}
	t0 := time.Now()
	resp, err := segRequest(id, path, req)
	if err != nil {
		return nil, err
	}
	res := &SegmentResult{Ms: time.Since(t0).Milliseconds()}
	b, _ := json.Marshal(resp)
	var raw struct {
		Width   int         `json:"width"`
		Height  int         `json:"height"`
		Objects []segRawObj `json:"objects"`
	}
	if err := json.Unmarshal(b, &raw); err != nil {
		return nil, err
	}
	res.Width, res.Height = raw.Width, raw.Height
	for _, o := range raw.Objects {
		obj, err := o.load(max(0, frame))
		if err != nil {
			return nil, err
		}
		if obj != nil {
			res.Objects = append(res.Objects, *obj)
		}
	}
	return res, nil
}

type segRawObj struct {
	ID    int       `json:"id"`
	Label string    `json:"label"`
	Score float64   `json:"score"`
	BBox  []float64 `json:"bbox"`
	Mask  string    `json:"mask"`
	// Frames (video): the object's folder, the mask of frame N being <N:05>.png.
	Frames     string      `json:"frames"`
	Candidates []segRawObj `json:"candidates"`
}

// load reads an object's mask file(s) into data URLs (frame: which mask of a
// video object's folder); nil when it covers nothing.
func (o segRawObj) load(frame int) (*SegmentObject, error) {
	mask := o.Mask
	if mask == "" && o.Frames != "" {
		mask = filepath.Join(o.Frames, fmt.Sprintf("%05d.png", frame))
	}
	url, rect, err := maskFileDataURL(mask)
	if err != nil || url == "" {
		return nil, err
	}
	obj := &SegmentObject{ID: o.ID, Label: o.Label, Score: o.Score, Mask: url, Rect: rect}
	if len(o.BBox) == 4 {
		obj.BBox = SceneRect{X: int(o.BBox[0]), Y: int(o.BBox[1]), W: int(o.BBox[2]), H: int(o.BBox[3])}
	}
	for _, c := range o.Candidates {
		if co, err := c.load(frame); err == nil && co != nil {
			obj.Candidates = append(obj.Candidates, *co)
		}
	}
	return obj, nil
}

// maskFileDataURL turns one of loki-segment's masks (8-bit gray, soft) into a
// white PNG whose alpha is the mask, cropped to what it covers (plus a little
// for the soft edge), and the rectangle that crop covers.
func maskFileDataURL(path string) (string, SceneRect, error) {
	if path == "" {
		return "", SceneRect{}, nil
	}
	f, err := os.Open(path)
	if err != nil {
		return "", SceneRect{}, err
	}
	img, err := png.Decode(f)
	f.Close()
	if err != nil {
		return "", SceneRect{}, err
	}
	b := img.Bounds()
	gray := func(x, y int) uint8 {
		r, _, _, _ := img.At(x, y).RGBA()
		return uint8(r >> 8)
	}
	x0, y0, x1, y1 := b.Max.X, b.Max.Y, b.Min.X-1, b.Min.Y-1
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			if gray(x, y) >= 8 {
				x0, y0, x1, y1 = min(x0, x), min(y0, y), max(x1, x), max(y1, y)
			}
		}
	}
	if x1 < x0 {
		return "", SceneRect{}, nil
	}
	const pad = 2
	x0, y0 = max(b.Min.X, x0-pad), max(b.Min.Y, y0-pad)
	x1, y1 = min(b.Max.X-1, x1+pad), min(b.Max.Y-1, y1+pad)
	out := image.NewNRGBA(image.Rect(0, 0, x1-x0+1, y1-y0+1))
	for y := y0; y <= y1; y++ {
		for x := x0; x <= x1; x++ {
			i := out.PixOffset(x-x0, y-y0)
			out.Pix[i], out.Pix[i+1], out.Pix[i+2], out.Pix[i+3] = 255, 255, 255, gray(x, y)
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, out); err != nil {
		return "", SceneRect{}, err
	}
	rect := SceneRect{X: x0 - b.Min.X, Y: y0 - b.Min.Y, W: x1 - x0 + 1, H: y1 - y0 + 1}
	return "data:image/png;base64," + base64.StdEncoding.EncodeToString(buf.Bytes()), rect, nil
}
