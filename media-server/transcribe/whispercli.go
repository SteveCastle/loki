package transcribe

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/stevecastle/shrike/appconfig"
	"github.com/stevecastle/shrike/deps"
	"github.com/stevecastle/shrike/platform"
)

func init() { Register(&whisperCLI{}) }

// whisperCLI runs the Purfview Faster-Whisper standalone binary — the
// Faster-Whisper-XXL bundle on Windows/Linux (CPU + CUDA, supports the fast
// large-v3-turbo model), the legacy CPU-only build on macOS where no XXL
// release exists. The binary downloads its speech model on first use into
// its own _models directory.
type whisperCLI struct{}

func (w *whisperCLI) ID() string          { return "whisper-cli" }
func (w *whisperCLI) DisplayName() string { return "Faster-Whisper (local)" }

// xxlPlatform reports whether this OS runs the XXL bundle. The macOS legacy
// build predates the turbo model, so model choices differ per platform.
func xxlPlatform() bool { return runtime.GOOS == "windows" || runtime.GOOS == "linux" }

func (w *whisperCLI) DefaultModel() string {
	if xxlPlatform() {
		// Turbo trims the decoder from 32 layers to 4: near large-v2
		// quality at a fraction of the (per-token) decode cost, which is
		// where transcription spends its time.
		return "large-v3-turbo"
	}
	return "large-v2"
}

func (w *whisperCLI) Models() []ModelChoice {
	// faster-whisper itself warns that large-v3 can produce worse results
	// than large-v2 on general content (more hallucinations, some accents
	// regressed) — hence v2, not v3, as the quality pick.
	common := []ModelChoice{
		{ID: "large-v3", DisplayName: "Large v3 (~3 GB; can hallucinate more than v2)"},
		{ID: "medium", DisplayName: "Medium (~1.5 GB)"},
		{ID: "small", DisplayName: "Small (~460 MB)"},
		{ID: "base", DisplayName: "Base (~140 MB)"},
		{ID: "tiny", DisplayName: "Tiny — fastest, lowest quality (~75 MB)"},
	}
	if xxlPlatform() {
		return append([]ModelChoice{
			{ID: "large-v3-turbo", DisplayName: "Large v3 Turbo — near-large quality, many times faster, recommended (~1.6 GB download)"},
			{ID: "large-v2", DisplayName: "Large v2 — best quality, slow (~3 GB download)"},
		}, common...)
	}
	return append([]ModelChoice{
		{ID: "large-v2", DisplayName: "Large v2 — best quality, recommended (~3 GB download)"},
	}, common...)
}

// depBinaryRelPath is where the assisted download places the executable,
// relative to the faster-whisper model dir.
func depBinaryRelPath() string {
	if xxlPlatform() {
		return filepath.Join("xxl", "faster-whisper-xxl"+platform.BinaryExtension())
	}
	return "whisper-faster" + platform.BinaryExtension()
}

// binaryPath resolves the executable: explicit config path wins, then the
// binary installed via the Dependencies downloader, then anything on PATH.
func (w *whisperCLI) binaryPath() string {
	if p := strings.TrimSpace(appconfig.Get().FasterWhisperPath); p != "" {
		return p
	}
	if p, err := deps.ModelPath("faster-whisper", depBinaryRelPath()); err == nil {
		return p
	}
	for _, name := range []string{"faster-whisper-xxl", "whisper-faster"} {
		if p, err := exec.LookPath(name); err == nil {
			return p
		}
	}
	return ""
}

func (w *whisperCLI) Available() error {
	if w.binaryPath() == "" {
		return fmt.Errorf("faster-whisper not found: download it from the Dependencies page (Transcription), set fasterWhisperPath in settings, or put faster-whisper-xxl / whisper-faster on PATH")
	}
	return nil
}

// buildArgs translates a Request into the Purfview CLI's flags.
func (w *whisperCLI) buildArgs(req Request) []string {
	model := req.Model
	if model == "" {
		model = w.DefaultModel()
	}
	args := []string{
		"--beep_off",
		"--output_format=vtt",
		"--output_dir=source",
		"--model", model,
	}
	// --vad_filter trims non-speech, which dramatically reduces
	// hallucinations during silent stretches in long clips.
	if req.VADFilter {
		args = append(args, "--vad_filter", "true")
	}
	// A language hint skips the (often-wrong on silent openings)
	// auto-detect; empty means let whisper detect.
	if req.Language != "" {
		args = append(args, "--language", req.Language)
	}
	return append(args, req.MediaPath)
}

func (w *whisperCLI) Transcribe(ctx context.Context, req Request) (Result, error) {
	exePath := w.binaryPath()
	if exePath == "" {
		return Result{}, w.Available()
	}
	logf := req.Log
	if logf == nil {
		logf = func(string) {}
	}
	logf("using " + exePath)

	args := w.buildArgs(req)
	cmd := exec.CommandContext(ctx, exePath, args...)
	platform.HideSubprocessWindow(cmd)
	logf(fmt.Sprintf("running: %s %s", exePath, strings.Join(args, " ")))

	// Pipe both stdout and stderr line-by-line into the log so failures
	// surface in the job's output (rather than disappearing into a dropped
	// process buffer). The binary writes progress and errors to stderr.
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return Result{}, fmt.Errorf("whisper-cli: stdout pipe: %w", err)
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return Result{}, fmt.Errorf("whisper-cli: stderr pipe: %w", err)
	}
	if err := cmd.Start(); err != nil {
		return Result{}, fmt.Errorf("whisper-cli: start: %w", err)
	}

	scanReader := func(r io.Reader) {
		scanner := bufio.NewScanner(r)
		// Whisper progress lines can be long; bump the buffer so we don't drop them.
		scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
		for scanner.Scan() {
			logf(scanner.Text())
		}
	}
	go scanReader(stdout)
	go scanReader(stderr)

	waitErr := cmd.Wait()

	vttPath := req.MediaPath[:len(req.MediaPath)-len(filepath.Ext(req.MediaPath))] + ".vtt"

	// Trust the artifact, not the exit code. The standalone binary is a
	// PyInstaller bundle that on Windows sometimes returns 0xc0000409
	// (STATUS_STACK_BUFFER_OVERRUN) AFTER all transcription work is complete
	// and the .vtt is written — a known teardown crash in the bundled
	// CRT/CUDA runtime, not a transcription failure. If the expected output
	// is on disk, treat the run as success regardless of the exit code.
	if stat, statErr := os.Stat(vttPath); statErr == nil && stat.Size() > 0 {
		if waitErr != nil {
			var exitErr *exec.ExitError
			if errors.As(waitErr, &exitErr) {
				logf(fmt.Sprintf("exited with code %d but VTT is present (%d bytes); treating as success", exitErr.ExitCode(), stat.Size()))
			} else {
				logf(fmt.Sprintf("wait error %v but VTT is present (%d bytes); treating as success", waitErr, stat.Size()))
			}
		} else {
			logf("transcription complete; reading " + vttPath)
		}
		text, err := os.ReadFile(vttPath)
		if err != nil {
			return Result{}, err
		}
		return Result{Text: string(text), TranscriptPath: vttPath}, nil
	}

	// No VTT produced — this is a real failure.
	if waitErr != nil {
		var exitErr *exec.ExitError
		if errors.As(waitErr, &exitErr) {
			logf(fmt.Sprintf("exited with code %d, no VTT produced", exitErr.ExitCode()))
			return Result{}, fmt.Errorf("whisper-cli exited with code %d (no VTT produced): %w", exitErr.ExitCode(), waitErr)
		}
		return Result{}, fmt.Errorf("whisper-cli failed (no VTT produced): %w", waitErr)
	}
	return Result{}, fmt.Errorf("whisper-cli exited cleanly but no VTT was produced at %s", vttPath)
}
