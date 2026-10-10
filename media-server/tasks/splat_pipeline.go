package tasks

import (
	"context"
	"encoding/binary"
	"fmt"
	"image"
	_ "image/jpeg"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/stevecastle/shrike/deps"
	"github.com/stevecastle/shrike/platform"
)

// The splat pipeline: a capture (one video, or photos of one scene) becomes
// a trained Gaussian-splat .ply in four stages, all in a scratch workspace:
//
//	frames   video → evenly spaced candidates (ffmpeg), the sharpest of
//	         each run of three kept; photos → resized copies
//	poses    COLMAP: SIFT features → sequential (video) or exhaustive
//	         (photos) matching → focal calibration → global mapper (the
//	         GLOMAP solver, built into COLMAP 4) → undistortion
//	train    Brush, headless, on the undistorted COLMAP model
//	export   Brush's final .ply moved to the output path
//
// COLMAP and Brush are managed dependencies (deps/models manifest ids
// "colmap" and "brush"), installed on first use unless already on PATH.

// splatStage maps a stage's own 0..1 progress into the job's 0..1000.
type splatStage struct{ from, to int }

var (
	stageFrames   = splatStage{0, 40}
	stageFeatures = splatStage{40, 80}
	stageMatch    = splatStage{80, 160}
	stageCalib    = splatStage{160, 180}
	stageMapper   = splatStage{180, 330}
	stageUndist   = splatStage{330, 350}
	stageTrain    = splatStage{350, 990}
)

// Training steps per quality preset.
var splatQualitySteps = map[string]int{"draft": 5000, "standard": 15000, "high": 30000}

type splatTools struct {
	FFmpeg, FFprobe, Colmap, Brush string
}

// installSplatDep is a variable so tests never touch the network.
var installSplatDep = installDep

// resolveSplatTools finds COLMAP and Brush — on PATH first, then the
// managed copies, installing them on first use (logged to the job).
func resolveSplatTools(ctx context.Context, logf func(string, ...any)) (splatTools, error) {
	t := splatTools{FFmpeg: deps.BundledOrEmpty("ffmpeg"), FFprobe: deps.BundledOrEmpty("ffprobe")}
	if t.FFmpeg == "" {
		if p, err := exec.LookPath("ffmpeg"); err == nil {
			t.FFmpeg = p
		}
	}
	if t.FFprobe == "" {
		if p, err := exec.LookPath("ffprobe"); err == nil {
			t.FFprobe = p
		}
	}
	if t.FFmpeg == "" || t.FFprobe == "" {
		return t, fmt.Errorf("ffmpeg/ffprobe not found (install them from the Dependencies page)")
	}
	ext := platform.BinaryExtension()
	// rel is the executable inside the dependency's folder.
	managed := func(id, rel, name string) (string, error) {
		return deps.ModelPath(id, rel)
	}
	get := func(id, rel, name, human string) (string, error) {
		if p, err := exec.LookPath(name); err == nil {
			return p, nil
		}
		if p, err := managed(id, rel, name); err == nil {
			return p, nil
		}
		if runtime.GOOS != "windows" && id == "colmap" {
			return "", fmt.Errorf("COLMAP is not on PATH — no prebuilt Linux/macOS release exists; install colmap 4.x (package manager or source build) and restart the server")
		}
		logf("installing %s (one-time download)", human)
		if err := installSplatDep(ctx, logf, id); err != nil {
			return "", fmt.Errorf("could not install %s: %w (it can also be put on PATH)", human, err)
		}
		return managed(id, rel, name)
	}
	var err error
	if t.Colmap, err = get("colmap", filepath.Join("colmap", "colmap"+ext), "colmap", "COLMAP"); err != nil {
		return t, err
	}
	if t.Brush, err = get("brush", "brush_app"+ext, "brush_app", "Brush"); err != nil {
		return t, err
	}
	return t, nil
}

// splatRun runs one tool to completion. Every output line goes to the
// workspace log; onLine sees each one (to report progress or pick the few
// worth showing in the job log).
func splatRun(ctx context.Context, logFile io.Writer, env []string, bin string, args []string, onLine func(string)) error {
	cmd := exec.CommandContext(ctx, bin, args...)
	cmd.Env = append(os.Environ(), env...)
	platform.HideSubprocessWindow(cmd)
	cmd.WaitDelay = 10 * time.Second
	var tail []string
	w := &lineWriter{fn: func(s string) {
		fmt.Fprintln(logFile, s)
		tail = append(tail, s)
		if len(tail) > 8 {
			tail = tail[1:]
		}
		if onLine != nil {
			onLine(s)
		}
	}}
	cmd.Stdout = w
	cmd.Stderr = w
	err := cmd.Run()
	w.Flush()
	if err != nil && ctx.Err() == nil {
		return fmt.Errorf("%s %s failed: %v — last output: %s", filepath.Base(bin), firstArg(args), err, strings.Join(tail, " | "))
	}
	return err
}

func firstArg(a []string) string {
	if len(a) == 0 {
		return ""
	}
	return a[0]
}

// scaleFilter shrinks so the longest side is at most max (never enlarges).
func scaleFilter(max int) string {
	m := strconv.Itoa(max)
	return "scale='if(gt(iw,ih),min(" + m + ",iw),-2)':'if(gt(iw,ih),-2,min(" + m + ",ih))'"
}

// probeVideo returns a video's duration (s) and frame rate (0 if unknown).
func probeVideo(ctx context.Context, ffprobe, path string) (dur, fps float64, err error) {
	cmd := exec.CommandContext(ctx, ffprobe, "-v", "error", "-select_streams", "v:0",
		"-show_entries", "stream=avg_frame_rate:format=duration", "-of", "default=nw=1", path)
	platform.HideSubprocessWindow(cmd)
	out, err := cmd.Output()
	if err != nil {
		return 0, 0, err
	}
	for _, line := range strings.Split(string(out), "\n") {
		k, v, ok := strings.Cut(strings.TrimSpace(line), "=")
		if !ok {
			continue
		}
		switch k {
		case "duration":
			dur, _ = strconv.ParseFloat(v, 64)
		case "avg_frame_rate":
			if n, d, ok := strings.Cut(v, "/"); ok {
				nf, _ := strconv.ParseFloat(n, 64)
				df, _ := strconv.ParseFloat(d, 64)
				if df > 0 {
					fps = nf / df
				}
			}
		}
	}
	if dur <= 0 {
		return 0, 0, fmt.Errorf("no duration")
	}
	return dur, fps, nil
}

// sharpness is the variance of a 4-neighbour Laplacian over a downsampled
// grey copy — the standard "how blurry is this frame" score.
func sharpness(path string) float64 {
	f, err := os.Open(path)
	if err != nil {
		return 0
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		return 0
	}
	b := img.Bounds()
	step := b.Dx() / 480
	if step < 1 {
		step = 1
	}
	w, h := b.Dx()/step, b.Dy()/step
	if w < 3 || h < 3 {
		return 0
	}
	grey := make([]float64, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, bl, _ := img.At(b.Min.X+x*step, b.Min.Y+y*step).RGBA()
			grey[y*w+x] = 0.299*float64(r) + 0.587*float64(g) + 0.114*float64(bl)
		}
	}
	var sum, sum2 float64
	n := 0
	for y := 1; y < h-1; y++ {
		for x := 1; x < w-1; x++ {
			i := y*w + x
			l := grey[i-1] + grey[i+1] + grey[i-w] + grey[i+w] - 4*grey[i]
			sum += l
			sum2 += l * l
			n++
		}
	}
	mean := sum / float64(n)
	return sum2/float64(n) - mean*mean
}

// splatFramesFromVideo samples ~3×want candidates evenly and keeps the
// sharpest of each consecutive run of three (motion blur is the main thing
// that sinks feature matching on handheld video).
func splatFramesFromVideo(ctx context.Context, t splatTools, logf func(string, ...any), logFile io.Writer, video, imagesDir string, want, maxRes int, report func(float64)) (int, error) {
	dur, srcFPS, err := probeVideo(ctx, t.FFprobe, video)
	if err != nil {
		return 0, fmt.Errorf("could not read the video's duration: %w", err)
	}
	cand := filepath.Join(filepath.Dir(imagesDir), "candidates")
	if err := os.MkdirAll(cand, 0o755); err != nil {
		return 0, err
	}
	rate := float64(want*3) / dur
	if srcFPS > 0 && rate > srcFPS {
		rate = srcFPS // a short clip: every frame is a candidate
	}
	logf("sampling %.1f s of video at %.2f fps", dur, rate)
	args := []string{"-hide_banner", "-loglevel", "error", "-i", video,
		"-vf", fmt.Sprintf("fps=%.4f,%s", rate, scaleFilter(maxRes)), "-q:v", "2",
		filepath.Join(cand, "c%05d.jpg")}
	if err := splatRun(ctx, logFile, nil, t.FFmpeg, args, nil); err != nil {
		return 0, err
	}
	files, _ := filepath.Glob(filepath.Join(cand, "c*.jpg"))
	sort.Strings(files)
	if len(files) == 0 {
		return 0, fmt.Errorf("ffmpeg produced no frames")
	}
	// Keep the sharpest of each run of `group` candidates — three normally,
	// fewer when a short clip couldn't supply 3x the frames we want.
	group := int(math.Round(float64(len(files)) / float64(want)))
	if group < 1 {
		group = 1
	} else if group > 3 {
		group = 3
	}
	kept := 0
	for i := 0; i < len(files); i += group {
		if ctx.Err() != nil {
			return kept, ctx.Err()
		}
		best, bestScore := "", -1.0
		for k := i; k < i+group && k < len(files); k++ {
			if s := sharpness(files[k]); s > bestScore {
				best, bestScore = files[k], s
			}
		}
		kept++
		if err := os.Rename(best, filepath.Join(imagesDir, fmt.Sprintf("frame%05d.jpg", kept))); err != nil {
			return kept, err
		}
		report(float64(i+group) / float64(len(files)))
	}
	_ = os.RemoveAll(cand)
	return kept, nil
}

// splatFramesFromPhotos re-encodes each photo (resized) into the workspace.
func splatFramesFromPhotos(ctx context.Context, t splatTools, logFile io.Writer, photos []string, imagesDir string, maxRes int, report func(float64)) (int, error) {
	for i, p := range photos {
		dst := filepath.Join(imagesDir, fmt.Sprintf("photo%05d.jpg", i+1))
		args := []string{"-hide_banner", "-loglevel", "error", "-i", p, "-vf", scaleFilter(maxRes), "-frames:v", "1", "-q:v", "2", dst}
		if err := splatRun(ctx, logFile, nil, t.FFmpeg, args, nil); err != nil {
			return i, err
		}
		report(float64(i+1) / float64(len(photos)))
	}
	return len(photos), nil
}

// registeredImages reads the image count from a COLMAP images.bin.
func registeredImages(modelDir string) int {
	f, err := os.Open(filepath.Join(modelDir, "images.bin"))
	if err != nil {
		return 0
	}
	defer f.Close()
	var n uint64
	if binary.Read(f, binary.LittleEndian, &n) != nil {
		return 0
	}
	return int(n)
}

// bestSparseModel picks the reconstruction that registered the most images
// (the global mapper writes sparse/0, sparse/1, … when the capture splits
// into pieces it can't link) and reports how many pieces there were.
func bestSparseModel(sparse string) (best string, bestN, pieces int) {
	dirs, _ := os.ReadDir(sparse)
	for _, d := range dirs {
		if !d.IsDir() {
			continue
		}
		p := filepath.Join(sparse, d.Name())
		n := registeredImages(p)
		if n > 0 {
			pieces++
		}
		if n > bestN {
			best, bestN = p, n
		}
	}
	return best, bestN, pieces
}

var (
	colmapProcessedRe = regexp.MustCompile(`Processed file \[(\d+)/(\d+)\]`)
	colmapMatchRe     = regexp.MustCompile(`Processing image \[(\d+)/(\d+)\]|Matching block \[(\d+)/(\d+)`)
	brushIterRe       = regexp.MustCompile(`Refine iter (\d+), (\d+) splats`)
)

// splatPipelineSpec is everything runSplatPipeline needs.
type splatPipelineSpec struct {
	Params  splatParams
	Video   string   // one video, or
	Photos  []string // three or more photos
	Output  string   // .ply to write
	WorkDir string   // empty scratch directory (removed by the caller)
}

func runSplatPipeline(ctx context.Context, t splatTools, s splatPipelineSpec, logf func(string, ...any), progress func(done int)) error {
	p := s.Params
	at := func(st splatStage, f float64) {
		f = math.Max(0, math.Min(1, f))
		progress(st.from + int(f*float64(st.to-st.from)))
	}
	logFile, err := os.Create(filepath.Join(s.WorkDir, "pipeline.log"))
	if err != nil {
		return err
	}
	defer logFile.Close()

	images := filepath.Join(s.WorkDir, "images")
	if err := os.MkdirAll(images, 0o755); err != nil {
		return err
	}

	// 1. frames
	var n int
	if s.Video != "" {
		n, err = splatFramesFromVideo(ctx, t, logf, logFile, s.Video, images, p.Frames, p.MaxRes, func(f float64) { at(stageFrames, f) })
	} else {
		n, err = splatFramesFromPhotos(ctx, t, logFile, s.Photos, images, p.MaxRes, func(f float64) { at(stageFrames, f) })
	}
	if err != nil {
		return fmt.Errorf("preparing frames: %w", err)
	}
	logf("%d frames ready", n)

	// 2. poses
	db := filepath.Join(s.WorkDir, "colmap.db")
	colmap := func(st splatStage, re *regexp.Regexp, args ...string) error {
		return splatRun(ctx, logFile, nil, t.Colmap, args, func(line string) {
			if re == nil {
				return
			}
			if m := re.FindStringSubmatch(line); m != nil {
				a, b := m[1], m[2]
				if a == "" && len(m) > 4 {
					a, b = m[3], m[4]
				}
				i, _ := strconv.Atoi(a)
				total, _ := strconv.Atoi(b)
				if total > 0 {
					at(st, float64(i)/float64(total))
				}
			}
		})
	}
	logf("finding features")
	singleCam := "0"
	if s.Video != "" {
		singleCam = "1" // one lens, one zoom: share the intrinsics
	}
	if err := colmap(stageFeatures, colmapProcessedRe, "feature_extractor",
		"--database_path", db, "--image_path", images,
		"--ImageReader.single_camera", singleCam, "--ImageReader.camera_model", "OPENCV"); err != nil {
		return err
	}
	at(stageFeatures, 1)
	if s.Video != "" {
		logf("matching neighbouring frames")
		err = colmap(stageMatch, colmapMatchRe, "sequential_matcher", "--database_path", db,
			"--SequentialMatching.overlap", "10")
	} else {
		logf("matching every pair of photos")
		err = colmap(stageMatch, colmapMatchRe, "exhaustive_matcher", "--database_path", db)
	}
	if err != nil {
		return err
	}
	at(stageMatch, 1)
	logf("calibrating the lens")
	if err := colmap(stageCalib, nil, "view_graph_calibrator", "--database_path", db); err != nil {
		// Calibration only sharpens the focal prior; the mapper copes without.
		logf("lens calibration skipped: %v", err)
	}
	at(stageCalib, 1)
	logf("solving the camera path (global mapper)")
	sparse := filepath.Join(s.WorkDir, "sparse")
	if err := os.MkdirAll(sparse, 0o755); err != nil {
		return err
	}
	started := time.Now()
	stopTick := make(chan struct{})
	go func() { // no fine-grained progress from the mapper: creep toward its end
		tk := time.NewTicker(2 * time.Second)
		defer tk.Stop()
		for {
			select {
			case <-stopTick:
				return
			case <-tk.C:
				at(stageMapper, 1-math.Exp(-time.Since(started).Seconds()/90))
			}
		}
	}()
	err = colmap(stageMapper, nil, "global_mapper", "--database_path", db, "--image_path", images, "--output_path", sparse)
	close(stopTick)
	if err != nil {
		return err
	}
	model, reg, pieces := bestSparseModel(sparse)
	if reg < 3 {
		return fmt.Errorf("could not work out the camera path (%d of %d frames placed) — the capture needs more overlap: move slower, keep the subject in view, avoid blank walls and moving things", reg, n)
	}
	logf("camera path solved: %d of %d frames placed (%.0f s)", reg, n, time.Since(started).Seconds())
	switch {
	case pieces > 1:
		logf("warning: the capture broke into %d pieces that couldn't be joined; training the largest (%d of %d frames) — a slower, steadier pass with more overlap avoids this", pieces, reg, n)
	case reg < n*2/3:
		logf("warning: only %d of %d frames could be placed — the scene may have gaps", reg, n)
	}
	at(stageMapper, 1)
	undist := filepath.Join(s.WorkDir, "undistorted")
	if err := colmap(stageUndist, nil, "image_undistorter", "--image_path", images, "--input_path", model,
		"--output_path", undist, "--output_type", "COLMAP"); err != nil {
		return err
	}
	at(stageUndist, 1)
	_ = os.RemoveAll(images) // the undistorted copies are what training reads
	// Where the capture started and which way is up: written into the .ply
	// so Studio opens the layer on the first frame's view, levelled.
	capture, capErr := colmapCaptureView(filepath.Join(undist, "sparse"))
	if capErr != nil {
		logf("note: no capture camera for Studio (%v)", capErr)
	}

	// 3. train
	steps := p.Steps
	if steps <= 0 {
		steps = splatQualitySteps[p.Quality]
	}
	logf("training the splat: %d steps", steps)
	out := filepath.Join(s.WorkDir, "out")
	if err := os.MkdirAll(out, 0o755); err != nil {
		return err
	}
	lastLog := time.Time{}
	trainStart := time.Now()
	err = splatRun(ctx, logFile, []string{"RUST_LOG=brush_cli=info", "NO_COLOR=1"}, t.Brush, []string{
		undist,
		"--total-steps", strconv.Itoa(steps),
		"--export-every", strconv.Itoa(steps),
		"--export-path", out,
		"--export-name", "splat.ply",
		"--max-resolution", strconv.Itoa(p.MaxRes),
		"--sh-degree", strconv.Itoa(p.SHDegree),
	}, func(line string) {
		m := brushIterRe.FindStringSubmatch(line)
		if m == nil {
			return
		}
		it, _ := strconv.Atoi(m[1])
		at(stageTrain, float64(it)/float64(steps))
		// `step i/n` is the shared progress-line shape (see runAICLI).
		if time.Since(lastLog) > 15*time.Second {
			lastLog = time.Now()
			logf("step %d/%d (%s splats, %.0f s)", it, steps, m[2], time.Since(trainStart).Seconds())
		}
	})
	if err != nil {
		return err
	}
	at(stageTrain, 1)

	// 4. export
	ply := filepath.Join(out, "splat.ply")
	if _, err := os.Stat(ply); err != nil {
		// Older/newer Brush builds may still template the name.
		if found, _ := filepath.Glob(filepath.Join(out, "*.ply")); len(found) > 0 {
			sort.Strings(found)
			ply = found[len(found)-1]
		} else {
			return fmt.Errorf("training finished but wrote no .ply")
		}
	}
	if capErr == nil {
		if err := addPlyComments(ply, plyCaptureComments(capture)); err != nil {
			logf("note: could not record the capture camera (%v)", err)
		}
	}
	if err := moveFile(ply, s.Output); err != nil {
		return fmt.Errorf("saving the splat: %w", err)
	}
	logf("trained in %.0f s", time.Since(trainStart).Seconds())
	return nil
}
