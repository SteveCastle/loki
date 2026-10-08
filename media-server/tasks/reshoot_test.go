package tasks

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stevecastle/shrike/jobqueue"
)

func TestReshootClassifyInputs(t *testing.T) {
	in := []string{
		`C:\m\a.png`, `C:\m\b.JPG`, `C:\m\c.gif`, `C:\m\d.avif`, `C:\m\e.webp`,
		`C:\m\clip.mp4`, `C:\m\clip2.MOV`,
		`C:\m\song.mp3`, `C:\m\v.wav`, `C:\m\x.flac`, `C:\m\y.ogg`, `C:\m\z.m4a`, `C:\m\w.aac`, `C:\m\o.opus`,
		`C:\m\notes.txt`, `C:\m\data.json`,
		`C:\m\a.png`, // duplicate
	}
	imgs, vids, auds, skipped := classifyReshootInputs(in)
	if len(imgs) != 5 || imgs[0] != `C:\m\a.png` || imgs[2] != `C:\m\c.gif` {
		t.Errorf("images = %q", imgs)
	}
	if len(vids) != 2 {
		t.Errorf("videos = %q", vids)
	}
	if len(auds) != 7 {
		t.Errorf("audios = %q", auds)
	}
	if strings.Join(skipped, "|") != `C:\m\notes.txt|C:\m\data.json` {
		t.Errorf("skipped = %q", skipped)
	}
}

func TestReshootCapInputs(t *testing.T) {
	mk := func(prefix string, n int) []string {
		var out []string
		for i := 0; i < n; i++ {
			out = append(out, fmt.Sprintf("%s%d", prefix, i))
		}
		return out
	}
	imgs, vids, auds, dropped := capReshootInputs(mk("i", 11), mk("v", 4), mk("a", 3))
	if len(imgs) != 9 || len(vids) != 3 || len(auds) != 3 {
		t.Errorf("caps = %d/%d/%d", len(imgs), len(vids), len(auds))
	}
	if strings.Join(dropped, ",") != "i9,i10,v3" {
		t.Errorf("dropped = %q", dropped)
	}
	_, _, _, dropped = capReshootInputs(mk("i", 2), nil, nil)
	if len(dropped) != 0 {
		t.Errorf("dropped = %q", dropped)
	}
}

func TestReshootParamsFromOptions(t *testing.T) {
	p := reshootParamsFromOptions(ParseOptions(&jobqueue.Job{}, reshootOptions))
	if p.Duration != 5 || p.Steps != 20 || p.Seed != -1 || !p.Native || p.Megapixels != 0.4 ||
		p.Shake != "subtle" || p.RefSize != "match" || p.Animate || p.NoAudio || p.NoVideoAudio {
		t.Errorf("defaults = %+v", p)
	}
	j := &jobqueue.Job{Arguments: []string{
		"--duration", "30", "--steps", "100", "--native", "false", "--shake", "wild",
		"--refsize", "max", "--seed", "77", "--noaudio", "--novideoaudio=true", "--animate",
	}}
	p = reshootParamsFromOptions(ParseOptions(j, reshootOptions))
	if p.Duration != 15.08 || p.Steps != 60 || p.Native || p.Shake != "subtle" || p.RefSize != "max" ||
		p.Seed != 77 || !p.NoAudio || !p.NoVideoAudio || !p.Animate {
		t.Errorf("parsed = %+v", p)
	}
	p = reshootParamsFromOptions(map[string]any{"duration": 0.1, "steps": 1.0})
	if p.Duration != 0.25 || p.Steps != 4 {
		t.Errorf("low clamp: duration %v steps %d", p.Duration, p.Steps)
	}
}

func TestBuildReshootArgs(t *testing.T) {
	base := reshootParams{Duration: 5, Native: true, Megapixels: 0.4, Steps: 20, Seed: 3, RefSize: "match", Shake: "subtle"}

	// Full-control mode: prompt, native canvas, all reference kinds.
	p := base
	p.Prompt = "<Picture 1> walks"
	args := buildReshootArgs(p, []string{"a.png", "b.png"}, []string{"c.mp4"}, []string{"d.mp3"}, "out.mp4", "")
	want := []string{
		"--prompt=<Picture 1> walks", "-d", "5", "--native", "--steps", "20", "--seed", "3",
		"--ref-image-size", "match", "-i", "a.png", "-i", "b.png", "-v", "c.mp4", "-a", "d.mp3", "-o", "out.mp4",
	}
	if strings.Join(args, "|") != strings.Join(want, "|") {
		t.Errorf("args =\n%q\nwant\n%q", args, want)
	}

	// Living photo: first image -> --animate, describe + shake, extra direction.
	p = base
	p.Animate, p.Describe, p.Shake, p.Prompt = true, "the dog on the sofa", "handheld", "it yawns"
	p.Duration, p.NoAudio, p.NoVideoAudio, p.RefSize = 7.5, true, true, "max"
	args = buildReshootArgs(p, []string{"dog.jpg", "sofa.jpg"}, nil, nil, "o.mp4", "")
	want = []string{
		"--animate", "dog.jpg", "--describe=the dog on the sofa", "--shake", "handheld", "--prompt=it yawns",
		"-d", "7.5", "--native", "--steps", "20", "--seed", "3", "--ref-image-size", "max",
		"--no-audio", "--no-video-audio", "-i", "sofa.jpg", "-o", "o.mp4",
	}
	if strings.Join(args, "|") != strings.Join(want, "|") {
		t.Errorf("animate args =\n%q\nwant\n%q", args, want)
	}

	// Explicit size beats native; native off -> megapixels; prompt file.
	p = base
	p.Size = "1344x768"
	args = buildReshootArgs(p, []string{"a.png"}, nil, nil, "o.mp4", "p.txt")
	if v, _ := argValue(args, "--size"); v != "1344x768" || hasArg(args, "--native") {
		t.Errorf("size args = %q", args)
	}
	if v, _ := argValue(args, "--prompt-file"); v != "p.txt" {
		t.Errorf("prompt file args = %q", args)
	}
	p = base
	p.Native, p.Megapixels = false, 0.9
	args = buildReshootArgs(p, []string{"a.png"}, nil, nil, "o.mp4", "")
	if v, _ := argValue(args, "--megapixels"); v != "0.9" || hasArg(args, "--native") {
		t.Errorf("megapixels args = %q", args)
	}
}

func TestReshootTaskFake(t *testing.T) {
	argsLog := installFakeAI(t)
	db := setupItemOpsDB(t)
	dir := t.TempDir()
	img := writeFileForTest(t, filepath.Join(dir, "hero.png"), "x")
	vid := writeFileForTest(t, filepath.Join(dir, "move.mp4"), "x")
	txt := writeFileForTest(t, filepath.Join(dir, "notes.txt"), "x")
	song := writeFileForTest(t, filepath.Join(t.TempDir(), "song.mp3"), "x")
	writeFileForTest(t, filepath.Join(dir, "hero_reshoot.mp4"), "old")

	q, j := newItemOpsJob(t, db, "reshoot", []string{
		"--prompt", "<Subject 1> is the man in <Picture 1>.\nHe dances like <Video 1> to <Audio 2>.",
		"--refs", song, "--steps", "8", "--duration", "6", "--seed", "11",
	}, img+"\n"+txt+"\n"+vid)
	if err := reshootTask(j, q, &sync.Mutex{}); err != nil {
		t.Fatalf("reshoot: %v\n%s", err, jobLog(q, j.ID))
	}
	want := filepath.Join(dir, "hero_reshoot_1.mp4")
	if _, err := os.Stat(want); err != nil {
		t.Fatalf("output missing: %v\n%s", err, jobLog(q, j.ID))
	}
	got := q.GetJob(j.ID)
	if got.State != jobqueue.StateCompleted {
		t.Errorf("state = %v", got.State)
	}
	if got.ProgressTotal != 10 || got.ProgressDone != 10 {
		t.Errorf("progress = %d/%d; want 10/10", got.ProgressDone, got.ProgressTotal)
	}
	if len(got.OutputFiles) != 1 || got.OutputFiles[0] != want || got.SourceFiles[0] != img {
		t.Errorf("outputs = %v sources = %v", got.OutputFiles, got.SourceFiles)
	}
	if !mediaRowExists(t, db, want) {
		t.Error("terminal mp4 not inserted into media table")
	}
	log := jobLog(q, j.ID)
	for _, s := range []string{"reshoot: seed 11", "skipping unsupported file notes.txt", "reshoot: step 1/8", "reshoot: step 5/8", "reshoot: step 8/8", "reshoot: fake info"} {
		if !strings.Contains(log, s) {
			t.Errorf("log missing %q:\n%s", s, log)
		}
	}
	a := readFakeArgs(t, argsLog)[0]
	if v := argValues(a, "-i"); len(v) != 1 || v[0] != img {
		t.Errorf("-i = %q", v)
	}
	if v := argValues(a, "-v"); len(v) != 1 || v[0] != vid {
		t.Errorf("-v = %q", v)
	}
	if v := argValues(a, "-a"); len(v) != 1 || v[0] != song {
		t.Errorf("-a = %q", v)
	}
	if v, _ := argValue(a, "--prompt"); !strings.Contains(v, "\nHe dances") {
		t.Errorf("--prompt = %q (multiline must survive)", v)
	}
	if v, _ := argValue(a, "-d"); v != "6" {
		t.Errorf("-d = %q", v)
	}
	if !hasArg(a, "--native") {
		t.Errorf("--native missing: %q", a)
	}
}

func TestReshootTaskAnimateChainedFake(t *testing.T) {
	argsLog := installFakeAI(t)
	db := setupItemOpsDB(t)
	dir := t.TempDir()
	img := writeFileForTest(t, filepath.Join(dir, "cat.jpg"), "x")

	q := jobqueue.NewQueueWithDB(db)
	if _, err := q.AddWorkflow(jobqueue.Workflow{Tasks: []jobqueue.WorkflowTask{
		{ID: "rs", Command: "reshoot", Arguments: []string{"--animate", "--describe", "the grey cat on a windowsill", "--shake", "none", "--steps", "4"}, Input: img},
		{ID: "after", Command: "wait", Dependencies: []string{"rs"}},
	}}); err != nil {
		t.Fatal(err)
	}
	j, _ := q.ClaimJob()
	if j == nil || j.ID != "rs" {
		t.Fatalf("claimed %v", j)
	}
	if err := reshootTask(j, q, &sync.Mutex{}); err != nil {
		t.Fatalf("reshoot: %v\n%s", err, jobLog(q, j.ID))
	}
	want := filepath.Join(dir, ".loki-temp", "rs", "cat_reshoot.mp4")
	if _, err := os.Stat(want); err != nil {
		t.Fatalf("chained output missing: %v", err)
	}
	if mediaRowExists(t, db, want) {
		t.Error("intermediate output must not be in the library")
	}
	a := readFakeArgs(t, argsLog)[0]
	if v, _ := argValue(a, "--animate"); v != img {
		t.Errorf("--animate = %q", v)
	}
	if v, _ := argValue(a, "--describe"); v != "the grey cat on a windowsill" {
		t.Errorf("--describe = %q", v)
	}
	if v, _ := argValue(a, "--shake"); v != "none" {
		t.Errorf("--shake = %q", v)
	}
	if hasArg(a, "-i") {
		t.Errorf("animate image also passed as -i: %q", a)
	}
}

func TestReshootTaskValidation(t *testing.T) {
	installFakeAI(t)
	db := setupItemOpsDB(t)
	dir := t.TempDir()
	img := writeFileForTest(t, filepath.Join(dir, "a.png"), "x")
	txt := writeFileForTest(t, filepath.Join(dir, "a.txt"), "x")
	song := writeFileForTest(t, filepath.Join(dir, "a.mp3"), "x")

	cases := []struct {
		name  string
		args  []string
		input string
		want  string
	}{
		{"no prompt", nil, img, "reshoot: needs a prompt"},
		{"no references", []string{"--prompt", "x"}, txt, "reshoot: needs at least one reference"},
		{"animate without image", []string{"--animate"}, song, "needs an image input"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			q, j := newItemOpsJob(t, db, "reshoot", c.args, c.input)
			if err := reshootTask(j, q, &sync.Mutex{}); err == nil {
				t.Fatal("expected error")
			}
			if !strings.Contains(jobLog(q, j.ID), c.want) {
				t.Errorf("log missing %q:\n%s", c.want, jobLog(q, j.ID))
			}
			if q.GetJob(j.ID).State != jobqueue.StateError {
				t.Errorf("state = %v", q.GetJob(j.ID).State)
			}
		})
	}
}

func TestReshootTaskMissingBinary(t *testing.T) {
	t.Setenv("PATH", t.TempDir())
	db := setupItemOpsDB(t)
	img := writeFileForTest(t, filepath.Join(t.TempDir(), "a.png"), "x")
	q, j := newItemOpsJob(t, db, "reshoot", []string{"--prompt", "x"}, img)
	if err := reshootTask(j, q, &sync.Mutex{}); err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(jobLog(q, j.ID), "reshoot: loki-reshoot not found on PATH") {
		t.Errorf("log:\n%s", jobLog(q, j.ID))
	}
}

// Real smoke run (LOKI_AI_E2E=1): the actual loki-reshoot on the GPU, a tiny clip.
func TestReshootRealE2E(t *testing.T) {
	if os.Getenv("LOKI_AI_E2E") != "1" {
		t.Skip("set LOKI_AI_E2E=1 to run the real loki-reshoot (needs a GPU with ~22 GB free; takes a minute or two)")
	}
	db := setupItemOpsDB(t)
	dir := t.TempDir()
	in := filepath.Join(dir, "gradient.png")
	img := image.NewRGBA(image.Rect(0, 0, 384, 480))
	for y := 0; y < 480; y++ {
		for x := 0; x < 384; x++ {
			img.Set(x, y, color.RGBA{uint8(x * 255 / 384), uint8(y * 255 / 480), uint8(255 - x*255/384), 255})
		}
	}
	f, err := os.Create(in)
	if err != nil {
		t.Fatal(err)
	}
	if err := png.Encode(f, img); err != nil {
		t.Fatal(err)
	}
	f.Close()

	// The shortest valid clip (22 frames) at the native canvas, few steps: a pipeline check, not a quality check.
	q, j := newItemOpsJob(t, db, "reshoot", []string{
		"--animate", "--describe", "a soft colour gradient", "--duration", "0.25", "--steps", "8", "--seed", "1", "--noaudio",
	}, in)
	start := time.Now()
	err = reshootTask(j, q, &sync.Mutex{})
	t.Logf("job log:\n%s", jobLog(q, j.ID))
	if err != nil {
		t.Fatalf("reshoot: %v", err)
	}
	out := filepath.Join(dir, "gradient_reshoot.mp4")
	st, err := os.Stat(out)
	if err != nil {
		t.Fatalf("output missing: %v", err)
	}
	if st.Size() < 1000 {
		t.Errorf("output suspiciously small: %d bytes", st.Size())
	}
	t.Logf("output %s: %d bytes in %v", out, st.Size(), time.Since(start))
	got := q.GetJob(j.ID)
	if got.ProgressDone != got.ProgressTotal {
		t.Errorf("progress = %d/%d", got.ProgressDone, got.ProgressTotal)
	}
}
