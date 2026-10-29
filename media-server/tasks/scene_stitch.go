package tasks

import (
	"context"
	"crypto/sha1"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"image"
	"image/draw"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	xdraw "golang.org/x/image/draw"

	"github.com/stevecastle/shrike/features"
	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/stitch"
)

// --- stitching (seamless edges for still layers) ---
//
// A still layer's generation is composited through its feathered mask; where
// the engine drifted (lines a few pixels off, a slightly different tone, no
// grain) the edge shows. Stitching (package stitch) blends each still layer's
// active version into the picture under it: G aligned, tone- and
// grain-matched to O outside the mask, cut along the path where the two agree
// and blended across it. The result is a region-sized RGBA patch whose alpha
// carries the blend, composited over the stills below in place of the
// version through its mask: by the stage, the export and every crop an
// engine is given.
//
// A stitch depends on the version, the layer's mask and everything below it
// (the stills below are stitched too, bottom up), so each is a file named by
// a key over all of those: stitch-<layer>-<key>.png (+ .json: its record,
// seam-…png: the seam drawn for the editor, heal-…png: where it still shows).
// Any change makes new keys; the editor polls while Scene.Stitching says a
// pass is running (kickStitch), and exports and generations make the ones
// they need first (ensureStitches). A layer opts out with options.stitch =
// false.

// sceneStitchVersion is part of every key: bump it when the algorithm
// changes and every stitch is made again.
const sceneStitchVersion = "2"

// sceneStitchMaxEdge caps the long edge of a stitched patch.
const sceneStitchMaxEdge = 2048

// SceneStitch is a version's stitched result (server-owned, derived).
type SceneStitch struct {
	Path   string        `json:"path"`
	Seam   string        `json:"seam,omitempty"`
	Heal   string        `json:"heal,omitempty"`
	Key    string        `json:"key"`
	Report stitch.Report `json:"report"`
}

// stitchable reports whether l's active version is stitched.
func stitchable(l *SceneLayer) bool {
	if l.Kind != SceneKindStill || l.Hidden || l.Mask == "" || l.Options["stitch"] == false {
		return false
	}
	v := l.active()
	return v != nil && fileExists(v.Path)
}

// sceneStitchKeys is the key of every stitchable layer's stitch: a hash over
// its version, mask and feather and, chained bottom up, everything in the
// picture below it (paint and stills, stitched or not).
func sceneStitchKeys(layers []*SceneLayer) map[string]string {
	keys := map[string]string{}
	chain := sha1hex(sceneStitchVersion)
	for _, l := range layers {
		if paintInPicture(l) {
			chain = sha1hex(chain, "paint", l.ID, sha1hex(l.Mask))
			continue
		}
		v := l.active()
		if l.Kind != SceneKindStill || l.Hidden || v == nil || l.Mask == "" {
			continue
		}
		st, err := os.Stat(v.Path)
		if err != nil {
			continue
		}
		own := sha1hex(chain, l.ID, v.Path, fmt.Sprint(st.Size(), st.ModTime().UnixNano(), v.Region, l.Feather), sha1hex(l.Mask))
		if stitchable(l) {
			keys[l.ID] = own
			chain = sha1hex(chain, "stitched", own)
		} else {
			chain = sha1hex(chain, "plain", own)
		}
	}
	return keys
}

func sha1hex(parts ...string) string {
	h := sha1.New()
	for _, p := range parts {
		h.Write([]byte(p))
		h.Write([]byte{0})
	}
	return hex.EncodeToString(h.Sum(nil))
}

func stitchFiles(dir, layerID, key string) (img, seam, heal, record string) {
	stem := filepath.Join(dir, fmt.Sprintf("stitch-%s-%s", safeLayerID(layerID), key[:16]))
	return stem + ".png", stem + "-seam.png", stem + "-heal.png", stem + ".json"
}

// readStitch is the stitch made for key, or nil when there is none yet.
func readStitch(dir, layerID, key string) *SceneStitch {
	img, _, _, record := stitchFiles(dir, layerID, key)
	b, err := os.ReadFile(record)
	if err != nil || !fileExists(img) {
		return nil
	}
	var st SceneStitch
	if json.Unmarshal(b, &st) != nil || st.Key != key {
		return nil
	}
	return &st
}

// DecorateStitches is decorateStitches for the API's answers.
func DecorateStitches(s *Scene) {
	if s != nil {
		decorateStitches(s)
	}
}

// decorateStitches sets each active still version's Stitch (nil while it is
// still to be made) and Scene.Stitching, and starts a pass when one is due.
func decorateStitches(s *Scene) {
	dir, err := sceneDir(s.ID)
	if err != nil {
		return
	}
	keys := sceneStitchKeys(s.Layers)
	s.Stitching = false
	for _, l := range s.Layers {
		for i := range l.Versions {
			l.Versions[i].Stitch = nil
		}
		key, ok := keys[l.ID]
		if !ok {
			continue
		}
		if st := readStitch(dir, l.ID, key); st != nil {
			l.active().Stitch = st
		} else {
			s.Stitching = true
		}
	}
	if s.Stitching {
		kickStitch(s.ID)
	}
}

// --- the background pass ---

var stitchRuns = struct {
	sync.Mutex
	running map[string]bool
	again   map[string]bool
}{running: map[string]bool{}, again: map[string]bool{}}

// kickStitch makes the scene's missing stitches in the background (one pass
// per scene at a time; a kick during a pass runs another after it).
func kickStitch(id string) {
	stitchRuns.Lock()
	defer stitchRuns.Unlock()
	if stitchRuns.running[id] {
		stitchRuns.again[id] = true
		return
	}
	stitchRuns.running[id] = true
	go func() {
		for {
			// Mask edits arrive in bursts (autosave): let them settle.
			time.Sleep(400 * time.Millisecond)
			sceneMu.Lock()
			s, err := loadSceneLocked(id)
			sceneMu.Unlock()
			if err == nil {
				ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
				if err := ensureStitches(ctx, s, len(s.Layers)); err != nil {
					fmt.Fprintf(os.Stderr, "scene %s: stitching: %v\n", id, err)
				}
				cancel()
				pruneStitches(s)
			}
			stitchRuns.Lock()
			if !stitchRuns.again[id] {
				delete(stitchRuns.running, id)
				stitchRuns.Unlock()
				return
			}
			delete(stitchRuns.again, id)
			stitchRuns.Unlock()
		}
	}()
}

// pruneStitches removes stitch files no current key names (older than a few
// minutes: a pass for a newer key may have just written one).
func pruneStitches(s *Scene) {
	dir, err := sceneDir(s.ID)
	if err != nil {
		return
	}
	keep := map[string]bool{}
	for lid, key := range sceneStitchKeys(s.Layers) {
		a, b, c, d := stitchFiles(dir, lid, key)
		keep[a], keep[b], keep[c], keep[d] = true, true, true, true
	}
	files, _ := filepath.Glob(filepath.Join(dir, "stitch-*"))
	for _, f := range files {
		if keep[f] {
			continue
		}
		if st, err := os.Stat(f); err == nil && time.Since(st.ModTime()) > 3*time.Minute {
			os.Remove(f)
		}
	}
}

// ensureStitches makes every missing stitch of layers[:upto], bottom up: the
// picture is built as it goes (base, paint, stills as composited), so each
// layer is stitched into exactly what is under it.
func ensureStitches(ctx context.Context, s *Scene, upto int) error {
	dir, err := sceneDir(s.ID)
	if err != nil {
		return err
	}
	layers := s.Layers[:min(upto, len(s.Layers))]
	keys := sceneStitchKeys(layers)
	missing := false
	for lid, key := range keys {
		if readStitch(dir, lid, key) == nil {
			missing = true
		}
	}
	if !missing {
		return nil
	}
	_, err = buildPicture(ctx, s, upto)
	return err
}

// buildPicture is the picture made of the base and layers[:upto] as the
// compositor draws them (paint, stills; stitched where stitched), making
// any missing stitch on the way.
func buildPicture(ctx context.Context, s *Scene, upto int) (*image.RGBA, error) {
	dir, err := sceneDir(s.ID)
	if err != nil {
		return nil, err
	}
	layers := s.Layers[:min(upto, len(s.Layers))]
	keys := sceneStitchKeys(layers)
	w, h := s.Width, s.Height
	base, err := loadScaled(s.Base, w, h)
	if err != nil {
		return nil, fmt.Errorf("base: %w", err)
	}
	// Premultiplied: draw's fast paths composite NRGBA patches onto it.
	frame := image.NewRGBA(base.Bounds())
	draw.Draw(frame, frame.Bounds(), base, image.Point{}, draw.Src)
	for _, l := range layers {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if paintInPicture(l) {
			paint, err := decodeMaskImage(l.Mask, w, h)
			if err != nil {
				return nil, err
			}
			draw.Draw(frame, frame.Bounds(), paint, image.Point{}, draw.Over)
			continue
		}
		v := l.active()
		if l.Kind != SceneKindStill || l.Hidden || v == nil || l.Mask == "" || !fileExists(v.Path) {
			continue
		}
		r := image.Rect(v.Region.X, v.Region.Y, v.Region.X+v.Region.W, v.Region.Y+v.Region.H).Intersect(frame.Bounds())
		if r.Empty() {
			continue
		}
		key, stitched := keys[l.ID]
		var patch *image.NRGBA
		if stitched {
			if st := readStitch(dir, l.ID, key); st != nil {
				if patch, err = loadScaled(st.Path, r.Dx(), r.Dy()); err != nil {
					return nil, err
				}
			} else {
				if patch, err = stitchLayer(dir, l, key, frame, r); err != nil {
					return nil, fmt.Errorf("layer %q: %w", l.Name, err)
				}
			}
		} else {
			if patch, err = plainPatch(l, r, w, h); err != nil {
				return nil, err
			}
		}
		// a stitch is at the version's resolution: into the picture at the region's
		if b := patch.Bounds(); b.Dx() != r.Dx() || b.Dy() != r.Dy() {
			patch = scaleTo(patch, r.Dx(), r.Dy(), xdraw.CatmullRom)
		}
		draw.Draw(frame, r, patch, image.Point{}, draw.Over)
	}
	return frame, nil
}

// stitchLayer stitches l's active version into frame (the picture under it)
// over r, writes its files and returns the patch.
func stitchLayer(dir string, l *SceneLayer, key string, frame *image.RGBA, r image.Rectangle) (*image.NRGBA, error) {
	v := l.active()
	g, err := loadScaled(v.Path, r.Dx(), r.Dy())
	if err != nil {
		return nil, err
	}
	o := image.NewNRGBA(image.Rect(0, 0, r.Dx(), r.Dy()))
	draw.Draw(o, o.Bounds(), frame, r.Min, draw.Src)
	mask, err := maskRegion(l.Mask, frame.Bounds().Dx(), frame.Bounds().Dy(), r)
	if err != nil {
		return nil, err
	}
	var seed int64
	for _, c := range key[:15] {
		seed = seed*16 + int64(strings.IndexRune("0123456789abcdef", c))
	}
	opt := stitch.Options{Feather: l.Feather, Seed: seed}
	// The patch is made at the version's own resolution (capped), so the
	// stage keeps its detail when zoomed; the compositor scales it into place.
	if full, err := loadScaled(v.Path, 0, 0); err == nil {
		fw, fh := full.Bounds().Dx(), full.Bounds().Dy()
		if k := float64(sceneStitchMaxEdge) / float64(max(fw, fh)); k < 1 {
			full = scaleTo(full, int(float64(fw)*k+0.5), int(float64(fh)*k+0.5), xdraw.CatmullRom)
			fw, fh = full.Bounds().Dx(), full.Bounds().Dy()
		}
		if fw > r.Dx()+r.Dx()/10 || fh > r.Dy()+r.Dy()/10 {
			opt.Detail = full
		}
	}
	res, err := stitch.Stitch(o, g, mask, opt)
	if err != nil {
		return nil, err
	}
	imgPath, seamPath, healPath, record := stitchFiles(dir, l.ID, key)
	st := SceneStitch{Path: imgPath, Key: key, Report: res.Report}
	if err := writeStitchPNG(imgPath, res.Image); err != nil {
		return nil, err
	}
	if res.Seam != nil {
		if err := writeStitchPNG(seamPath, res.Seam); err == nil {
			st.Seam = seamPath
		}
	}
	if res.Heal != nil {
		if err := writeStitchPNG(healPath, res.Heal); err == nil {
			st.Heal = healPath
		}
	}
	// The record last, atomically: it is what marks the stitch made.
	b, _ := json.Marshal(st)
	tmp := fmt.Sprintf("%s.%d.tmp", record, time.Now().UnixNano())
	if err := os.WriteFile(tmp, b, 0644); err != nil {
		return nil, err
	}
	if err := os.Rename(tmp, record); err != nil {
		os.Remove(tmp)
		return nil, err
	}
	return res.Image, nil
}

// plainPatch is l's active version through its mask over r: what the
// compositor draws for a layer that isn't stitched.
func plainPatch(l *SceneLayer, r image.Rectangle, w, h int) (*image.NRGBA, error) {
	g, err := loadScaled(l.active().Path, r.Dx(), r.Dy())
	if err != nil {
		return nil, err
	}
	mask, err := maskRegion(l.Mask, w, h, r)
	if err != nil {
		return nil, err
	}
	for i, a := range mask {
		g.Pix[i*4+3] = uint8(float32(g.Pix[i*4+3])*a + 0.5)
	}
	return g, nil
}

// maskRegion is a layer mask's coverage (alpha, 0-1) over r of the w×h frame.
func maskRegion(dataURL string, w, h int, r image.Rectangle) ([]float32, error) {
	m, err := decodeMaskImage(dataURL, w, h)
	if err != nil {
		return nil, err
	}
	out := make([]float32, r.Dx()*r.Dy())
	for y := 0; y < r.Dy(); y++ {
		for x := 0; x < r.Dx(); x++ {
			out[y*r.Dx()+x] = float32(m.Pix[(r.Min.Y+y)*m.Stride+(r.Min.X+x)*4+3]) / 255
		}
	}
	return out, nil
}

// decodeMaskImage decodes a mask (or painting) data URL scaled to w×h.
func decodeMaskImage(dataURL string, w, h int) (*image.NRGBA, error) {
	const prefix = "data:image/png;base64,"
	if !strings.HasPrefix(dataURL, prefix) {
		return nil, fmt.Errorf("not a PNG data URL")
	}
	b, err := base64.StdEncoding.DecodeString(dataURL[len(prefix):])
	if err != nil {
		return nil, err
	}
	img, err := png.Decode(strings.NewReader(string(b)))
	if err != nil {
		return nil, err
	}
	return scaleTo(img, w, h, xdraw.BiLinear), nil
}

// loadScaled decodes an image at w×h (0×0: its own size).
func loadScaled(path string, w, h int) (*image.NRGBA, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", filepath.Base(path), err)
	}
	if w == 0 || h == 0 {
		w, h = img.Bounds().Dx(), img.Bounds().Dy()
	}
	return scaleTo(img, w, h, xdraw.CatmullRom), nil
}

func loadNRGBA(path string) (*image.NRGBA, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, err := png.Decode(f)
	if err != nil {
		return nil, err
	}
	b := img.Bounds()
	return scaleTo(img, b.Dx(), b.Dy(), xdraw.NearestNeighbor), nil
}

// scaleTo is img at w×h as straight-alpha NRGBA.
func scaleTo(img image.Image, w, h int, k xdraw.Interpolator) *image.NRGBA {
	b := img.Bounds()
	if n, ok := img.(*image.NRGBA); ok && b.Dx() == w && b.Dy() == h && b.Min == (image.Point{}) {
		return n
	}
	// Scale in premultiplied space (RGBA), then un-premultiply: scaling
	// straight alpha bleeds the colour of transparent pixels into the edge.
	tmp := image.NewRGBA(image.Rect(0, 0, w, h))
	if b.Dx() == w && b.Dy() == h {
		draw.Draw(tmp, tmp.Bounds(), img, b.Min, draw.Src)
	} else {
		k.Scale(tmp, tmp.Bounds(), img, b, draw.Src, nil)
	}
	out := image.NewNRGBA(tmp.Bounds())
	draw.Draw(out, out.Bounds(), tmp, image.Point{}, draw.Src)
	return out
}

func writeStitchPNG(path string, img image.Image) error {
	// Unique: a background pass and an export can make the same stitch at once.
	tmp := fmt.Sprintf("%s.%d.tmp", path, time.Now().UnixNano())
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	enc := png.Encoder{CompressionLevel: png.BestSpeed}
	if err := enc.Encode(f, img); err != nil {
		f.Close()
		os.Remove(tmp)
		return err
	}
	if err := f.Close(); err != nil {
		os.Remove(tmp)
		return err
	}
	return os.Rename(tmp, path)
}

// --- healing ---

// sceneHealStrength is how hard the seam is redrawn: enough for Kromoss to
// rework a line that doesn't continue, little enough to keep what is there.
const sceneHealStrength = 0.5

// HealVersion queues a generation that redraws where version n's seam still
// shows (its stitch's heal map). loki-dream inpaints that band of the picture
// as the stage shows it (layer stitched in), at the stitch's own resolution;
// scene-heal-merge then pastes only the band back, so everything else stays
// exactly as it was. The result is a new version ("healed vN"), stitched like
// any other, so the previous one is a click away.
func HealVersion(ctx context.Context, q *jobqueue.Queue, id, lid string, n int, strength float64) (*Scene, error) {
	if !features.Dream {
		return nil, fmt.Errorf("healing seams needs loki-dream, which this build doesn't have")
	}
	sceneMu.Lock()
	defer sceneMu.Unlock()
	s, err := loadSceneLocked(id)
	if err != nil {
		return nil, err
	}
	reconcileScene(q, s)
	idx := -1
	for i, l := range s.Layers {
		if l.ID == lid {
			idx = i
		}
	}
	if idx < 0 {
		return nil, ErrLayerNotFound
	}
	l := s.Layers[idx]
	if l.Kind != SceneKindStill {
		return nil, fmt.Errorf("only still layers are stitched")
	}
	if n != l.Active || l.active() == nil {
		return nil, fmt.Errorf("show the version first: a seam is healed in the picture as shown")
	}
	if len(l.jobs()) >= sceneMaxGenerations {
		return nil, ErrSceneBusy
	}
	if !stitchable(l) {
		return nil, fmt.Errorf("this layer isn't stitched (Blend edges is off)")
	}
	if err := ensureStitches(ctx, s, idx+1); err != nil {
		return nil, err
	}
	dir, _ := sceneDir(id)
	st := readStitch(dir, l.ID, sceneStitchKeys(s.Layers[:idx+1])[l.ID])
	if st == nil {
		return nil, fmt.Errorf("the stitch isn't ready yet")
	}
	if st.Heal == "" {
		return nil, fmt.Errorf("the seam already blends in: nothing to heal")
	}
	v := l.active()
	region := v.Region

	// The picture under the layer, at the stitch's resolution, with the stitch over it.
	below, err := buildPicture(ctx, s, idx)
	if err != nil {
		return nil, err
	}
	patch, err := loadNRGBA(st.Path)
	if err != nil {
		return nil, err
	}
	pw, ph := patch.Bounds().Dx(), patch.Bounds().Dy()
	r := image.Rect(region.X, region.Y, region.X+region.W, region.Y+region.H).Intersect(below.Bounds())
	crop := image.NewRGBA(image.Rect(0, 0, r.Dx(), r.Dy()))
	draw.Draw(crop, crop.Bounds(), below, r.Min, draw.Src)
	init := scaleTo(crop, pw, ph, xdraw.CatmullRom)
	draw.Draw(init, init.Bounds(), patch, image.Point{}, draw.Over)
	heal, err := loadNRGBA(st.Heal)
	if err != nil {
		return nil, err
	}
	healHi := scaleTo(heal, pw, ph, xdraw.BiLinear)
	mask := image.NewGray(healHi.Bounds())
	for i := range mask.Pix {
		mask.Pix[i] = healHi.Pix[i*4+3]
	}

	stem := fmt.Sprintf("still-%s-%d", safeLayerID(l.ID), len(l.Versions)+1)
	input := filepath.Join(dir, stem+".png")
	for i := 2; fileExists(input); i++ {
		input = filepath.Join(dir, fmt.Sprintf("%s.%d.png", stem, i))
	}
	base := strings.TrimSuffix(input, filepath.Ext(input))
	maskPath := base + "-mask.png"
	if err := writeStitchPNG(input, init); err != nil {
		return nil, err
	}
	if err := writeStitchPNG(maskPath, mask); err != nil {
		os.Remove(input)
		return nil, err
	}
	if strength <= 0 {
		strength = sceneHealStrength
	}
	what := strings.TrimRight(strings.TrimSpace(l.Prompt), ". ")
	if what == "" {
		what = "the scene"
	}
	w, h := sceneDreamSize(region)
	dream := jobqueue.WorkflowTask{ID: uuid.NewString(), Command: "dream", Input: input, Arguments: []string{
		"--prompt=" + what + ", one seamless picture, continuous with its surroundings, in the same style, light and perspective",
		"--mask=" + maskPath,
		"--strength=" + formatFloatArg(clampFloat(strength, 0.1, 1)),
		fmt.Sprintf("--size=%dx%d", w, h),
		"--nolibrary=true",
	}}
	merge := jobqueue.WorkflowTask{ID: uuid.NewString(), Command: "scene-heal-merge", Dependencies: []string{dream.ID},
		Arguments: []string{"--init=" + input, "--mask=" + maskPath, "--out=" + base + "_healed.png"}}
	ids, err := q.AddWorkflow(jobqueue.Workflow{Tasks: []jobqueue.WorkflowTask{dream, merge}})
	if err != nil || len(ids) == 0 {
		os.Remove(input)
		os.Remove(maskPath)
		if err == nil {
			err = fmt.Errorf("the job queue returned no job")
		}
		return nil, err
	}
	l.Error = ""
	l.startJob(&SceneJob{ID: ids[len(ids)-1], Chain: ids, State: "pending", Region: region, Prompt: l.Prompt, Input: input,
		Note: fmt.Sprintf("healed v%d", n+1), Below: sceneBelow(s.Layers[:idx]),
		Mask: &SceneMask{Shapes: l.Shapes, MaskOpts: l.MaskOpts, Feather: l.Feather}})
	if err := saveSceneLocked(s); err != nil {
		return nil, err
	}
	decorateStitches(s)
	return s, nil
}

var sceneHealMergeOptions = []TaskOption{
	{Name: "init", Label: "Picture", Type: "string", Default: "", Description: "The picture the heal started from"},
	{Name: "mask", Label: "Heal Mask", Type: "string", Default: "", Description: "Where it was redrawn (white), the picture's size"},
	{Name: "out", Label: "Output", Type: "string", Default: "", Description: "The healed picture (beside the init)"},
}

// sceneHealMergeTask pastes a heal's redrawn band (the input, any size) into
// the picture it started from through the soft heal mask: the band is new,
// every other pixel is the picture's own (an inpainting engine resamples and
// re-encodes the whole image; that must not reach the rest).
func sceneHealMergeTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	const tag = "scene-heal-merge"
	fail := func(msg string, err error) error {
		q.PushJobStdout(j.ID, tag+": "+msg)
		q.ErrorJob(j.ID)
		if err == nil {
			err = fmt.Errorf("%s: %s", tag, msg)
		}
		return err
	}
	opts := ParseOptions(j, sceneHealMergeOptions)
	initPath, maskPath, out := optString(opts, "init"), optString(opts, "mask"), optString(opts, "out")
	if initPath == "" || maskPath == "" || out == "" || filepath.Dir(absPath(out)) != filepath.Dir(absPath(initPath)) {
		return fail("needs --init, --mask and --out beside the init", nil)
	}
	healed := ""
	for _, line := range strings.Split(j.Input, "\n") {
		if p := strings.TrimSpace(line); p != "" && isImageExt(filepath.Ext(p)) && fileExists(p) {
			healed = p
		}
	}
	if healed == "" {
		return fail("no healed image in the input", nil)
	}
	init, err := loadNRGBA(initPath)
	if err != nil {
		return fail(err.Error(), err)
	}
	b := init.Bounds()
	hi, err := loadScaled(healed, b.Dx(), b.Dy())
	if err != nil {
		return fail(err.Error(), err)
	}
	mask, err := loadScaled(maskPath, b.Dx(), b.Dy())
	if err != nil {
		return fail(err.Error(), err)
	}
	for i := 0; i < b.Dx()*b.Dy(); i++ {
		// the mask is gray: its red channel is the coverage
		a := float32(mask.Pix[i*4]) / 255
		for k := 0; k < 3; k++ {
			init.Pix[i*4+k] = uint8(float32(init.Pix[i*4+k])*(1-a) + float32(hi.Pix[i*4+k])*a + 0.5)
		}
		init.Pix[i*4+3] = 255
	}
	if err := writeStitchPNG(out, init); err != nil {
		return fail(err.Error(), err)
	}
	q.PushJobStdout(j.ID, fmt.Sprintf("%s: pasted the healed seam into %s -> %s", tag, filepath.Base(initPath), filepath.Base(out)))
	q.RegisterOutputFile(j.ID, out, healed)
	q.CompleteJob(j.ID)
	return nil
}
