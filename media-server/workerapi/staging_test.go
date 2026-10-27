package workerapi

import (
	"encoding/json"
	"path/filepath"
	"reflect"
	"testing"
)

func fakeFS(t *testing.T, files map[string]int64, dirs map[string]bool) {
	t.Helper()
	oldF, oldD := statFile, statDir
	statFile = func(p string) (int64, bool) { s, ok := files[p]; return s, ok }
	statDir = func(p string) bool { return dirs[p] }
	t.Cleanup(func() { statFile, statDir = oldF, oldD })
}

func TestPlanArgs(t *testing.T) {
	root := filepath.Join(string(filepath.Separator)+"lib", "media")
	if filepath.VolumeName(t.TempDir()) != "" {
		root = `C:\lib\media`
	}
	photo := filepath.Join(root, "My Photo.jpg")
	ref := filepath.Join(root, "ref.png")
	prompt := filepath.Join(root, "prompt.txt")
	clip := filepath.Join(root, "clip.mp4")
	out := filepath.Join(root, "out", "result.png")
	gen := filepath.Join(root, "Generated")
	fakeFS(t, map[string]int64{photo: 10, ref: 20, prompt: 3, clip: 40}, map[string]bool{gen: true})

	p := PlanArgs([]string{"--prompt-file", prompt, "-p", "make it night", "--ref", ref, "--meta=" + prompt, "-v", clip + "@1.5,2", "-o", out, photo, "relative.png"})
	want := []string{"--prompt-file", "$JOB/in/0/prompt.txt", "-p", "make it night", "--ref", "$JOB/in/1/ref.png", "--meta=$JOB/in/0/prompt.txt", "-v", "$JOB/in/2/clip.mp4@1.5,2", "-o", "$JOB/out/0/result.png", "$JOB/in/3/My_Photo.jpg", "relative.png"}
	if !reflect.DeepEqual(p.Args, want) {
		t.Fatalf("args\n got %q\nwant %q", p.Args, want)
	}
	if len(p.Inputs) != 4 || p.Inputs[3].Local != photo || p.Inputs[3].Remote != "in/3/My_Photo.jpg" || p.Inputs[2].Size != 40 {
		t.Fatalf("inputs %+v", p.Inputs)
	}
	for rel, local := range map[string]string{
		"out/0/result.png":    out,
		"in/3/My_Photo.jpg":   photo,                                 // an input itself (log lines)
		"in/3/My_Photo_x.png": filepath.Join(root, "My_Photo_x.png"), // a new file beside an input
	} {
		if got, ok := p.LocalPath(rel); !ok || got != local {
			t.Errorf("LocalPath(%q) = %q, %v; want %q", rel, got, ok, local)
		}
	}
	if _, ok := p.LocalPath("in/3/sub/x.png"); ok {
		t.Error("a file in a subfolder of an input folder must not map")
	}
	if _, ok := p.LocalPath("elsewhere/x"); ok {
		t.Error("an unknown path must not map")
	}

	// an existing output folder: created on the worker, written into by name
	p = PlanArgs([]string{"-o", gen, "-p", "a cat"})
	if p.Args[1] != "$JOB/out/0/Generated/" || !reflect.DeepEqual(p.Dirs, []string{"out/0/Generated"}) {
		t.Fatalf("folder output: %q %q", p.Args, p.Dirs)
	}
	if got, _ := p.LocalPath("out/0/Generated/dream_1.png"); got != filepath.Join(gen, "dream_1.png") {
		t.Fatalf("folder output maps to %q", got)
	}
}

func TestRewriteLine(t *testing.T) {
	root := t.TempDir()
	in := filepath.Join(root, "a.png")
	out := filepath.Join(root, "o.png")
	fakeFS(t, map[string]int64{in: 1}, nil)
	p := PlanArgs([]string{"-o", out, in})
	if got := p.RewriteLine("$JOB/out/0/o.png"); got != out {
		t.Fatalf("plain: %q", got)
	}
	if got := p.RewriteLine("encoding $JOB/in/0/a.png, skipping $JOB/unknown"); got != "encoding "+in+", skipping $JOB/unknown" {
		t.Fatalf("mixed: %q", got)
	}
	line, _ := json.Marshal(map[string]string{"output": "$JOB/out/0/o.png"})
	var back map[string]string
	if err := json.Unmarshal([]byte(p.RewriteLine(string(line))), &back); err != nil || back["output"] != out {
		t.Fatalf("json: %v %q", err, back)
	}
}

func TestSafeNameAndCleanRel(t *testing.T) {
	for in, want := range map[string]string{"My Photo (1).jpg": "My_Photo__1_.jpg", "日本.png": "__.png", "..": "file", ".hidden": "hidden"} {
		if got := SafeName(in); got != want {
			t.Errorf("SafeName(%q) = %q, want %q", in, got, want)
		}
	}
	for in, ok := range map[string]bool{"in/0/a.png": true, "../x": false, "/etc/passwd": false, `C:\x`: false, "a/../../b": false, "a/./b": true} {
		if _, got := CleanRel(in); got != ok {
			t.Errorf("CleanRel(%q) ok = %v, want %v", in, got, ok)
		}
	}
}

func TestEngineName(t *testing.T) {
	for in, want := range map[string]string{"retouch": "loki-retouch", "loki-segment.exe": "loki-segment", " Loki-Dream ": "loki-dream", "": ""} {
		if got := EngineName(in); got != want {
			t.Errorf("EngineName(%q) = %q, want %q", in, got, want)
		}
	}
}
