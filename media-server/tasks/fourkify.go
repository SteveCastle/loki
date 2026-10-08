package tasks

import (
	"context"
	"encoding/base64"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/stevecastle/shrike/jobqueue"
)

// --- 4kify (deprecated alias of retouch) ---
//
// The original task wrapped a standalone `4kify` binary. That engine now ships
// as `loki-retouch --preset 4kify|4kify-phone`, so this task keeps its id and
// option list (saved workflows and older frontends still call it) and maps
// them onto the shared retouch runner (retouch.go):
//
//	phone    -> preset 4kify-phone (else 4kify)
//	prompt64 -> prompt (base64 UTF-8)
//	steps    -> steps
//	time     -> time (video frame sampling)

var fourKifyOptions = []TaskOption{
	{Name: "phone", Label: "Phone Wallpaper", Type: "bool", Default: false, Description: "Make a vertical 1296x2800 phone wallpaper instead of a 3840x2160 desktop one"},
	{Name: "time", Label: "Video Time", Type: "number", Default: 0.0, Description: "Seconds into a video to sample the frame from (videos only; images ignore it)"},
	{Name: "prompt64", Label: "Prompt Override (base64)", Type: "string", Default: "", Description: "Base64-encoded UTF-8 prompt that replaces the built-in 4kify prompt (must mention <image1>). Empty = default behavior"},
	{Name: "steps", Label: "Steps", Type: "number", Default: 25.0, Description: "Sampling steps (more is slower)"},
}

// fourKifyParams maps the legacy 4kify options onto retouch parameters.
func fourKifyParams(opts map[string]any) (retouchParams, error) {
	ro := map[string]any{
		"preset": "4kify",
		"steps":  optFloat(opts, "steps"),
		"time":   optFloat(opts, "time"),
		"seed":   -1.0,
	}
	if optBool(opts, "phone") {
		ro["preset"] = "4kify-phone"
	}
	if enc := strings.TrimSpace(optString(opts, "prompt64")); enc != "" {
		dec, err := base64.StdEncoding.DecodeString(enc)
		if err != nil {
			return retouchParams{}, err
		}
		ro["prompt"] = strings.TrimSpace(string(dec))
	}
	return retouchParamsFromOptions(ro), nil
}

func fourKifyTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	p, err := fourKifyParams(ParseOptions(j, fourKifyOptions))
	if err != nil {
		q.PushJobStdout(j.ID, "4kify: invalid prompt64 option: "+err.Error())
		q.ErrorJob(j.ID)
		return err
	}
	return runRetouchJob(j, q, "4kify", p)
}

// sampleVideoFrame renders the frame at `at` seconds to a lossless PNG in a
// fresh temp dir (the caller removes the directory).
func sampleVideoFrame(ctx context.Context, src string, at float64) (string, error) {
	dir, err := os.MkdirTemp("", "retouch-frame-")
	if err != nil {
		return "", err
	}
	frame := filepath.Join(dir, "frame.png")
	if err := runFFmpegSingleFrame(ctx, src, frame, at); err != nil {
		os.RemoveAll(dir)
		return "", err
	}
	return frame, nil
}
