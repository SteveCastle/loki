package tasks

import (
	"sort"
	"strings"
	"testing"

	"github.com/stevecastle/shrike/appconfig"
	"github.com/stevecastle/shrike/jobqueue"
)

// The Transform flow (src/renderer/components/transform) submits jobs as
// POST /create {input, fields}; the server turns every non-empty field into
// "--key value" arguments (createjob_args.go: appendFieldArgs). These tests
// feed the task option parsing the exact field maps the UI produces (see
// src/__tests__/transform-intents.test.ts) and check the resulting CLI
// command lines, so a rename on either side fails here.

func uiFieldsToArgs(fields map[string]string) []string {
	keys := make([]string, 0, len(fields))
	for k := range fields {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var args []string
	for _, k := range keys {
		if fields[k] == "" {
			continue
		}
		args = append(args, "--"+k, fields[k])
	}
	return args
}

func uiRetouchArgs(fields map[string]string, input string, refs []string) []string {
	j := &jobqueue.Job{Arguments: uiFieldsToArgs(fields)}
	p := retouchParamsFromOptions(ParseOptions(j, retouchOptions))
	if p.Seed < 0 {
		p.Seed = 5
	}
	return buildRetouchArgs(p, input, refs, "out.png", "")
}

func TestUIContractRetouch(t *testing.T) {
	has := func(t *testing.T, args []string, want ...string) {
		t.Helper()
		joined := "\x00" + strings.Join(args, "\x00") + "\x00"
		if !strings.Contains(joined, "\x00"+strings.Join(want, "\x00")+"\x00") {
			t.Errorf("args %q do not contain %q", args, want)
		}
	}
	lacks := func(t *testing.T, args []string, flag string) {
		t.Helper()
		for _, a := range args {
			if a == flag || strings.HasPrefix(a, flag+"=") {
				t.Errorf("args %q should not contain %s", args, flag)
			}
		}
	}

	t.Run("restore / upscale / wallpaper presets", func(t *testing.T) {
		a := uiRetouchArgs(map[string]string{"preset": "restore", "steps": "25", "seed": "123"}, "in.png", nil)
		has(t, a, "--preset", "restore")
		has(t, a, "--steps", "25")
		has(t, a, "--seed", "123")
		lacks(t, a, "--prompt")
		lacks(t, a, "--scale")

		a = uiRetouchArgs(map[string]string{"preset": "upscale", "scale": "3", "steps": "12"}, "in.png", nil)
		has(t, a, "--preset", "upscale")
		has(t, a, "--scale", "3")
		has(t, a, "--steps", "12")

		a = uiRetouchArgs(map[string]string{"preset": "4kify-phone", "steps": "40", "append": "keep the film grain"}, "in.png", nil)
		has(t, a, "--preset", "4kify-phone")
		has(t, a, "--append=keep the film grain")
	})

	t.Run("free-form edit keeps a multi-line quoted prompt verbatim, with a custom size", func(t *testing.T) {
		prompt := "make it \"night\"\nwith rain"
		a := uiRetouchArgs(map[string]string{"prompt": prompt, "size": "1920x1080", "steps": "25", "seed": "7"}, "in.png", nil)
		has(t, a, "--prompt="+prompt)
		has(t, a, "--size", "1920x1080")
		lacks(t, a, "--preset")
	})

	t.Run("combine and frame time are parsed from the UI's string flags", func(t *testing.T) {
		j := &jobqueue.Job{Arguments: uiFieldsToArgs(map[string]string{"prompt": "put <image2> into <image1>", "combine": "1", "time": "12.346", "steps": "25"})}
		p := retouchParamsFromOptions(ParseOptions(j, retouchOptions))
		if !p.Combine {
			t.Errorf("combine=1 was not read as true: %+v", p)
		}
		if p.Time < 12.345 || p.Time > 12.347 {
			t.Errorf("time = %v; want 12.346", p.Time)
		}
		if p.Prompt != "put <image2> into <image1>" {
			t.Errorf("prompt = %q", p.Prompt)
		}
	})
}

func TestUIContractReshoot(t *testing.T) {
	parse := func(fields map[string]string) reshootParams {
		j := &jobqueue.Job{Arguments: uiFieldsToArgs(fields)}
		p := reshootParamsFromOptions(ParseOptions(j, reshootOptions))
		if p.Seed < 0 {
			p.Seed = 5
		}
		return p
	}

	t.Run("bring to life", func(t *testing.T) {
		p := parse(map[string]string{"animate": "1", "describe": "a woman in a sunlit room", "shake": "handheld", "duration": "5", "steps": "20", "seed": "9"})
		if !p.Animate || p.Describe != "a woman in a sunlit room" || p.Shake != "handheld" || p.Duration != 5 || p.Steps != 20 || p.Seed != 9 {
			t.Fatalf("params = %+v", p)
		}
		args := buildReshootArgs(p, []string{"a.png"}, nil, nil, "o.mp4", "")
		got := strings.Join(args, "|")
		for _, want := range []string{"--animate|a.png", "--describe=a woman in a sunlit room", "--shake|handheld", "-d|5", "--steps|20", "--seed|9"} {
			if !strings.Contains(got, want) {
				t.Errorf("args %q missing %q", args, want)
			}
		}
	})

	t.Run("direct a scene with every reference kind", func(t *testing.T) {
		p := parse(map[string]string{"prompt": "She dances to <Audio 1>", "duration": "7.5", "steps": "32", "seed": "1", "refsize": "max", "novideoaudio": "1", "noaudio": "1"})
		if p.Animate || p.Prompt != "She dances to <Audio 1>" || p.Duration != 7.5 || p.RefSize != "max" || !p.NoVideoAudio || !p.NoAudio {
			t.Fatalf("params = %+v", p)
		}
		args := buildReshootArgs(p, []string{"a.png"}, []string{"c.mp4"}, []string{"d.mp3"}, "o.mp4", "")
		got := strings.Join(args, "|")
		for _, want := range []string{"--prompt=She dances to <Audio 1>", "--ref-image-size|max", "--no-audio", "--no-video-audio", "-i|a.png", "-v|c.mp4", "-a|d.mp3"} {
			if !strings.Contains(got, want) {
				t.Errorf("args %q missing %q", args, want)
			}
		}
	})

	t.Run("fast model: model=fast and steps=8 become --model fast --steps 8", func(t *testing.T) {
		p := parse(map[string]string{"prompt": "x", "model": "fast", "steps": "8", "duration": "5", "seed": "2"})
		if p.Model != "fast" || p.Steps != 8 {
			t.Fatalf("params = %+v", p)
		}
		args := buildReshootArgs(p, []string{"a.png"}, nil, nil, "o.mp4", "")
		got := strings.Join(args, "|")
		for _, want := range []string{"--model|fast", "--steps|8"} {
			if !strings.Contains(got, want) {
				t.Errorf("args %q missing %q", args, want)
			}
		}
		// The UI omits `model` for H3: no --model reaches the engine.
		p = parse(map[string]string{"prompt": "x", "steps": "20"})
		if args := buildReshootArgs(p, []string{"a.png"}, nil, nil, "o.mp4", ""); hasArg(args, "--model") || p.Model != "h3" {
			t.Errorf("h3 args = %q (model %q)", args, p.Model)
		}
	})

	t.Run("an extra model: dit=FILE becomes --dit FILE with the model's recipe", func(t *testing.T) {
		p := parse(map[string]string{"prompt": "x", "model": "fast", "dit": "turbo.safetensors", "steps": "8", "duration": "5"})
		c := appconfig.Config{ReshootModels: []appconfig.ReshootModel{{File: "turbo.safetensors", Recipe: "fast"}}}
		if err := applyReshootDit(c, &p); err != nil {
			t.Fatal(err)
		}
		got := strings.Join(buildReshootArgs(p, []string{"a.png"}, nil, nil, "o.mp4", ""), "|")
		for _, want := range []string{"--model|fast", "--dit|turbo.safetensors", "--steps|8"} {
			if !strings.Contains(got, want) {
				t.Errorf("args %q missing %q", got, want)
			}
		}
	})

	t.Run("the UI's maximum duration is accepted unchanged", func(t *testing.T) {
		if p := parse(map[string]string{"duration": "15", "animate": "1"}); p.Duration != 15 {
			t.Errorf("duration = %v; want 15", p.Duration)
		}
	})
}

func TestUIContractDream(t *testing.T) {
	parse := func(fields map[string]string) dreamParams {
		j := &jobqueue.Job{Arguments: uiFieldsToArgs(fields)}
		return dreamParamsFromOptions(ParseOptions(j, dreamOptions))
	}
	joined := func(args []string) string { return "\x00" + strings.Join(args, "\x00") + "\x00" }
	has := func(t *testing.T, args []string, want ...string) {
		t.Helper()
		if !strings.Contains(joined(args), "\x00"+strings.Join(want, "\x00")+"\x00") {
			t.Errorf("args %q do not contain %q", args, want)
		}
	}

	t.Run("generate: prompt, negative, size, steps, seed and count (no input)", func(t *testing.T) {
		p := parse(map[string]string{"prompt": "a red fox\nin \"snow\"", "negative": "blurry", "size": "1344x768", "steps": "30", "seed": "42", "count": "3"})
		if p.Prompt != "a red fox\nin \"snow\"" || p.Negative != "blurry" || p.Size != "1344x768" || p.Steps != 30 || p.Seed != 42 || p.Count != 3 || p.Turbo {
			t.Fatalf("params = %+v", p)
		}
		args := buildDreamArgs(p, "", "outdir", "")
		has(t, args, "--negative", "blurry")
		has(t, args, "--size", "1344x768")
		has(t, args, "--steps", "30")
		has(t, args, "--seed", "42")
		has(t, args, "--count", "3")
		if hasArg(args, "--init") || hasArg(args, "--turbo") {
			t.Errorf("text-to-image args %q", args)
		}
	})

	t.Run("generate turbo: turbo=1 and no steps or negative from the UI", func(t *testing.T) {
		p := parse(map[string]string{"prompt": "x", "turbo": "1", "size": "1024x1024", "seed": "1"})
		if !p.Turbo || p.Steps != 0 {
			t.Fatalf("params = %+v", p)
		}
		args := buildDreamArgs(p, "", "out.png", "")
		has(t, args, "--turbo")
		if hasArg(args, "--steps") || hasArg(args, "--negative") || hasArg(args, "--count") {
			t.Errorf("turbo args %q", args)
		}
	})

	t.Run("reimagine: strength with the input image as init", func(t *testing.T) {
		p := parse(map[string]string{"prompt": "as an oil painting", "strength": "0.35", "steps": "50", "seed": "7"})
		if p.Strength != 0.35 {
			t.Fatalf("strength = %v; want 0.35", p.Strength)
		}
		args := buildDreamArgs(p, "in.png", "in_dream.png", "")
		has(t, args, "--init", "in.png", "--strength", "0.35")
		has(t, args, "--steps", "50")
	})

	t.Run("the recipe rides along as --meta", func(t *testing.T) {
		meta := `{"kind":"loki-transform","v":1,"intent":"generate","paths":[]}`
		p := parse(map[string]string{"prompt": "x", "meta": meta})
		has(t, buildDreamArgs(p, "", "out.png", ""), "--meta="+meta)
	})
}
