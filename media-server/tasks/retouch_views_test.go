package tasks

import (
	"strings"
	"testing"

	"github.com/stevecastle/shrike/jobqueue"
)

func TestRetouchViewsCoverFullOrbit(t *testing.T) {
	// 8 orbit positions x 3 heights, minus the original (front, eye level).
	if len(retouchViews) != 23 {
		t.Fatalf("got %d views; want 23", len(retouchViews))
	}
	seen := map[string]bool{}
	for _, v := range retouchViews {
		if seen[v.ID] {
			t.Errorf("duplicate view id %q", v.ID)
		}
		seen[v.ID] = true
		p := v.Prompt()
		for _, want := range []string{"Edit <image1>", "Only the camera has moved", "The pose is frozen", "identity, pose"} {
			if !strings.Contains(p, want) {
				t.Errorf("%s prompt lacks %q:\n%s", v.ID, want, p)
			}
		}
	}
	for _, id := range []string{"reverse", "right-90", "left-45", "front-high", "front-low", "reverse-high", "left-135-low"} {
		if !seen[id] {
			t.Errorf("missing view %q", id)
		}
	}
	if seen["front"] {
		t.Error("front at eye level is the original photo, not a preset")
	}
}

func TestRetouchViewPrompts(t *testing.T) {
	rev, _ := retouchViewByID("reverse")
	p := rev.Prompt()
	for _, want := range []string{
		"photographed from directly behind.",
		"their back faces the camera and their face is hidden.",
		"Match the original camera height, subject distance, lens, crop, headroom, and placement within the frame.",
		"Reveal the rear surfaces of the subject and the background visible from this opposite viewpoint.",
	} {
		if !strings.Contains(p, want) {
			t.Errorf("reverse prompt lacks %q:\n%s", want, p)
		}
	}
	r90, _ := retouchViewByID("right-90")
	if !strings.Contains(r90.Prompt(), "the side that was on the right of the original frame facing the camera") {
		t.Errorf("right-90:\n%s", r90.Prompt())
	}
	l45, _ := retouchViewByID("left-45-high")
	if p := l45.Prompt(); !strings.Contains(p, "45 degrees to the left") || !strings.Contains(p, "looking down") || strings.Contains(p, "Match the original camera height") {
		t.Errorf("left-45-high:\n%s", p)
	}
	fl, _ := retouchViewByID("front-low")
	if p := fl.Prompt(); !strings.Contains(p, "from a low angle.") || !strings.Contains(p, "looking up") || !strings.Contains(p, "foreshortened") {
		t.Errorf("front-low:\n%s", p)
	}
}

func TestRetouchViewPresetRunsAsPlainEdit(t *testing.T) {
	j := &jobqueue.Job{Arguments: []string{"--preset", "reverse-low", "--append", "keep the red scarf", "--steps", "25", "--seed", "1"}}
	p := retouchParamsFromOptions(ParseOptions(j, retouchOptions))
	v, _ := retouchViewByID("reverse-low")
	if p.Prompt != v.Prompt() {
		t.Fatalf("prompt not expanded from the view: %q", p.Prompt)
	}
	args := strings.Join(buildRetouchArgs(p, "in.png", nil, "out.png", ""), "\x00")
	if strings.Contains(args, "--preset") {
		t.Errorf("a view preset must not reach loki-retouch as --preset: %q", args)
	}
	if !strings.Contains(args, "--prompt="+v.Prompt()) || !strings.Contains(args, "--append=keep the red scarf") {
		t.Errorf("args %q", args)
	}
	if got := retouchSuffix("reverse-low", false, 0); got != "_reverse-low" {
		t.Errorf("suffix = %q", got)
	}

	// An explicit prompt replaces the view's.
	j = &jobqueue.Job{Arguments: []string{"--preset", "right-90", "--prompt", "my own words"}}
	if p := retouchParamsFromOptions(ParseOptions(j, retouchOptions)); p.Prompt != "my own words" {
		t.Errorf("manual prompt = %q", p.Prompt)
	}
}
