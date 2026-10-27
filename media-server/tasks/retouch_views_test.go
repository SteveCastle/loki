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
	// Facing unknown: the move is said relatively, true whichever way the subject faces.
	rev, _ := retouchViewByID("reverse")
	p := rev.Prompt()
	for _, want := range []string{
		"Edit <image1> to show the same subject photographed from the opposite side of the subject from the original camera.",
		"Everything that faced away from the original camera now faces the camera",
		"Match the original camera height, subject distance, lens, crop, headroom, and placement within the frame.",
		"Reveal the surfaces of the subject that faced away from the original camera and the background visible from this opposite viewpoint.",
	} {
		if !strings.Contains(p, want) {
			t.Errorf("reverse prompt lacks %q:\n%s", want, p)
		}
	}
	// The old wording assumed a subject facing the camera ("their back faces the
	// camera"): a subject already facing away came back unchanged.
	if strings.Contains(p, "their back faces the camera") || strings.Contains(p, "seen in full profile") {
		t.Errorf("unknown facing must not say what the new view shows:\n%s", p)
	}
	r90, _ := retouchViewByID("right-90")
	if !strings.Contains(r90.Prompt(), "The side of the subject that was on the right of the original picture now faces the camera") {
		t.Errorf("right-90:\n%s", r90.Prompt())
	}
	l45, _ := retouchViewByID("left-45-high")
	if p := l45.Prompt(); !strings.Contains(p, "45 degrees around the subject toward the left") || !strings.Contains(p, "looking down") || strings.Contains(p, "Match the original camera height") {
		t.Errorf("left-45-high:\n%s", p)
	}
	fl, _ := retouchViewByID("front-low")
	if p := fl.Prompt(); !strings.Contains(p, "from a low angle.") || !strings.Contains(p, "looking up") || !strings.Contains(p, "foreshortened") {
		t.Errorf("front-low:\n%s", p)
	}

	// Facing known: the prompt leads with the view before and after.
	if p := rev.PromptFacing(180, true); !strings.HasPrefix(p, "Edit <image1> so the same subject is now seen from the front, facing the camera, instead of from directly behind, their back to the camera and their face hidden as in <image1>. This is a new camera position, not a new pose: the same subject photographed from the opposite side") {
		t.Errorf("reverse of a subject facing away:\n%s", p)
	}
	l90, _ := retouchViewByID("left-90")
	if p := l90.PromptFacing(270, true); !strings.HasPrefix(p, "Edit <image1> so the same subject is now seen from the front, facing the camera, instead of in full profile, facing the left of the picture as in <image1>.") {
		t.Errorf("left-90 of a subject facing left:\n%s", p)
	}
	if p := r90.PromptFacing(270, true); !strings.Contains(p, "now seen from directly behind") {
		t.Errorf("right-90 of a subject facing left:\n%s", p)
	}
	if p := rev.PromptFacing(270, true); !strings.Contains(p, "now seen in full profile, facing the right of the picture") {
		t.Errorf("reverse of a subject facing left:\n%s", p)
	}
	// The front view at a new height doesn't name views.
	if p := fl.PromptFacing(180, true); !strings.HasPrefix(p, "Edit <image1> to show the same subject photographed from a low angle.") {
		t.Errorf("front-low with facing:\n%s", p)
	}
}

// Keypoints RTMPose read (whole picture as the box) on the user's scene photos
// and on back views generated from them.
func TestSubjectFacing(t *testing.T) {
	pose := func(kp map[int][3]float64) *ScenePose {
		p := &ScenePose{Joints: make([]ScenePoseJoint, 18)}
		for i := range p.Joints {
			p.Joints[i] = ScenePoseJoint{Score: 0.8}
		}
		for i, v := range kp {
			p.Joints[i] = ScenePoseJoint{X: v[0], Y: v[1], Score: v[2]}
		}
		return p
	}
	cases := []struct {
		name string
		kp   map[int][3]float64
		want int
	}{
		// nose, neck, r shoulder, l shoulder, r hip, l hip, r ear, l ear
		{"front (pool)", map[int][3]float64{0: {697, 619, .73}, 1: {715, 677, .82}, 2: {636, 675, .82}, 5: {795, 680, .83}, 8: {701, 984, .88}, 11: {800, 984, .85}, 16: {664, 595, .83}, 17: {743, 591, .81}}, 0},
		{"front (suit)", map[int][3]float64{0: {758, 399, .83}, 1: {658, 712, .71}, 2: {398, 749, .82}, 5: {918, 674, .71}, 8: {482, 1629, .71}, 11: {792, 1614, .74}, 16: {542, 399, .78}, 17: {838, 329, .72}}, 0},
		{"profile facing left (window)", map[int][3]float64{0: {431, 362, .83}, 1: {533, 524, .66}, 2: {522, 517, .66}, 5: {544, 531, .75}, 8: {529, 988, .66}, 11: {529, 981, .73}, 16: {582, 369, .51}, 17: {533, 362, .80}}, 270},
		{"back (pool)", map[int][3]float64{0: {234, 196, .65}, 1: {256, 201, .83}, 2: {286, 205, .83}, 5: {226, 205, .86}, 8: {280, 326, .8}, 11: {236, 326, .8}, 16: {274, 190, .88}, 17: {236, 190, .86}}, 180},
		{"back (suit)", map[int][3]float64{0: {138, 200, .46}, 1: {206, 221, .82}, 2: {298, 225, .83}, 5: {115, 225, .82}, 8: {250, 493, .8}, 11: {160, 493, .8}, 16: {249, 180, .76}, 17: {151, 180, .78}}, 180},
	}
	for _, c := range cases {
		got, ok := subjectFacing(pose(c.kp))
		if !ok || got != c.want {
			t.Errorf("%s: got %d (ok %v), want %d", c.name, got, ok, c.want)
		}
	}
	if _, ok := subjectFacing(pose(map[int][3]float64{2: {10, 10, .1}})); ok {
		t.Error("an unsure shoulder must not give a facing")
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
