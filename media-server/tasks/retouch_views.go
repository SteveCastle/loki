package tasks

import (
	"fmt"
	"strings"
)

// --- retouch camera-angle presets ---
//
// "View" presets re-photograph the subject from another camera position: a
// full orbit in 45° steps (front, right-45 … reverse … left-45), each at eye
// level, from a high angle and from a low angle. Unlike the engine presets
// (4kify, upscale, …), whose prompts live in loki-retouch, these are written
// here: the server expands the preset into the edit prompt (an explicit
// prompt still replaces it, --append still extends it) and runs loki-retouch
// as a plain edit, so they need no engine rebuild.
//
// Directions are frame-relative: "right-90" means the camera orbits 90°
// toward the right of the original frame, so the subject's side that was on
// the right of the picture turns toward the camera.

type retouchView struct {
	ID    string
	Label string
	// Azimuth is the orbit in degrees toward the right of the original
	// frame: 0 = original direction, 180 = reverse, 270 = left-90.
	Azimuth int
	// Elevation is "" (eye level, the original height), "high" or "low".
	Elevation string
}

var retouchViewAzimuths = []struct {
	az    int
	id    string
	label string
}{
	{0, "front", "Front"},
	{45, "right-45", "45° right"},
	{90, "right-90", "90° right"},
	{135, "right-135", "135° right"},
	{180, "reverse", "Reverse"},
	{225, "left-135", "135° left"},
	{270, "left-90", "90° left"},
	{315, "left-45", "45° left"},
}

// retouchViews lists every camera-angle preset in orbit order: eye level,
// then high, then low. The front view at eye level is the original photo, so
// it is not a preset.
var retouchViews = func() []retouchView {
	var out []retouchView
	for _, elev := range []string{"", "high", "low"} {
		for _, a := range retouchViewAzimuths {
			if a.az == 0 && elev == "" {
				continue
			}
			v := retouchView{ID: a.id, Label: a.label, Azimuth: a.az, Elevation: elev}
			if elev != "" {
				v.ID += "-" + elev
				v.Label += ", " + elev + " angle"
			}
			out = append(out, v)
		}
	}
	return out
}()

func retouchViewIDs() []string {
	ids := make([]string, len(retouchViews))
	for i, v := range retouchViews {
		ids[i] = v.ID
	}
	return ids
}

// retouchViewByID returns the camera-angle preset with this id.
func retouchViewByID(id string) (retouchView, bool) {
	for _, v := range retouchViews {
		if v.ID == id {
			return v, true
		}
	}
	return retouchView{}, false
}

// isRetouchView reports whether preset is a camera-angle preset (expanded
// here rather than by loki-retouch).
func isRetouchView(preset string) bool {
	_, ok := retouchViewByID(preset)
	return ok
}

// Prompt is the edit instruction for this camera position.
func (v retouchView) Prompt() string {
	side := "right"
	if v.Azimuth > 180 {
		side = "left"
	}
	deg := v.Azimuth
	if deg > 180 {
		deg = 360 - deg
	}

	var where, camera, facing, surfaces string
	switch deg {
	case 0:
		camera = "The camera keeps its original direction to the subject"
		facing = "The subject remains facing the same direction in the scene."
	case 45:
		where = fmt.Sprintf("a three-quarter angle, 45 degrees to the %s of the original camera position", side)
		camera = fmt.Sprintf("The camera has orbited 45 degrees around the subject toward the %s of the original frame, still aimed at the subject", side)
		facing = fmt.Sprintf("The subject remains facing the same direction in the scene, so they are now seen in three-quarter view, with the side that was on the %s of the original frame turned toward the camera.", side)
		surfaces = "the surfaces of the subject that this angle turns toward the camera"
	case 90:
		where = fmt.Sprintf("directly from the side, 90 degrees to the %s of the original camera position", side)
		camera = fmt.Sprintf("The camera has orbited a quarter turn around the subject toward the %s of the original frame, still aimed at the subject", side)
		facing = fmt.Sprintf("The subject remains facing the same direction in the scene, so they are now seen in full profile, with the side that was on the %s of the original frame facing the camera.", side)
		surfaces = "the side of the subject facing the camera"
	case 135:
		where = fmt.Sprintf("behind at a three-quarter angle, 135 degrees to the %s of the original camera position", side)
		camera = fmt.Sprintf("The camera has orbited 135 degrees around the subject toward the %s of the original frame, past their side and around toward their back, still aimed at the subject", side)
		facing = fmt.Sprintf("The subject remains facing the same direction in the scene, so they are now seen mostly from behind, with their back and the side that was on the %s of the original frame toward the camera and their face mostly hidden.", side)
		surfaces = "the rear and side surfaces of the subject"
	default: // 180
		where = "directly behind"
		camera = "The camera is now positioned on the opposite side of the subject from its original position"
		facing = "The subject remains facing the same direction in the scene, so their back faces the camera and their face is hidden."
		surfaces = "the rear surfaces of the subject"
	}

	var b strings.Builder
	b.WriteString("Edit <image1> to show the same subject photographed from ")
	switch {
	case where != "" && v.Elevation == "high":
		b.WriteString(where + ", from a high angle")
	case where != "" && v.Elevation == "low":
		b.WriteString(where + ", from a low angle")
	case where != "":
		b.WriteString(where)
	case v.Elevation == "high":
		b.WriteString("a high angle")
	default:
		b.WriteString("a low angle")
	}
	b.WriteString(". ")
	b.WriteString(camera)
	switch v.Elevation {
	case "high":
		b.WriteString(", and it is raised well above the subject, looking down at them at about 45 degrees")
	case "low":
		b.WriteString(", and it is lowered close to the ground, looking up at them at about 30 degrees")
	}
	b.WriteString(". ")
	b.WriteString(facing)
	// Anchor: the model's failure mode is turning (or mirroring) the subject
	// toward the new camera, or redrawing someone else, instead of moving
	// the camera around an unchanged subject.
	b.WriteString(" Only the camera has moved: the subject stays on exactly the same spot in the scene, in exactly the same pose, and is not turned, rotated, mirrored or replaced.")
	b.WriteString("\n\n")
	b.WriteString("The pose is frozen. Keep every part of the body exactly where it is in the scene: the same head tilt and turn, the same gaze direction, the same arm, hand and finger positions, the same leg and foot placement, weight distribution, and body lean. Do not re-pose the subject for the new camera; show the identical pose as it looks from this angle")
	if deg == 0 {
		b.WriteString(", foreshortened by the new camera height.")
	} else {
		b.WriteString(", with limbs that were toward the original camera now seen from the side or behind accordingly.")
	}
	b.WriteString("\n\n")

	switch v.Elevation {
	case "":
		b.WriteString("Match the original camera height, subject distance, lens, crop, headroom, and placement within the frame.")
	default:
		b.WriteString("Match the original subject distance, lens, and crop, keeping the subject at the same size and placement within the frame.")
	}
	b.WriteString(" Keep the subject the focus of the shot and unmistakably the same subject: preserve their identity, pose, clothing, hair, proportions, colours, and appearance exactly. Reveal ")
	var reveal []string
	if surfaces != "" {
		reveal = append(reveal, surfaces)
	}
	switch v.Elevation {
	case "high":
		reveal = append(reveal, "the tops of the subject's head, shoulders and other upward-facing surfaces", "the ground around them")
	case "low":
		reveal = append(reveal, "the undersides of the subject's chin, arms and other downward-facing surfaces", "the sky or ceiling above them")
	}
	switch len(reveal) {
	case 0:
	case 1:
		b.WriteString(reveal[0] + " and ")
	default:
		b.WriteString(strings.Join(reveal[:len(reveal)-1], ", ") + ", " + reveal[len(reveal)-1] + ", and ")
	}
	if deg == 180 && v.Elevation == "" {
		b.WriteString("the background visible from this opposite viewpoint.")
	} else {
		b.WriteString("the background visible from this new viewpoint.")
	}
	return b.String()
}
