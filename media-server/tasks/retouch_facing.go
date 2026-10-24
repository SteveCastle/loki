package tasks

import (
	"context"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"math"
	"os"
	"time"
)

// --- which way the subject faces, for the camera-angle presets ---
//
// A camera-angle prompt that says what the new view shows ("now seen in
// profile", "their back faces the camera") is only right for a subject who
// faced the original camera: a subject already in profile or facing away
// came back unchanged, because the prompt described the picture as it was.
// The facing is read from the person's pose (RTMPose, as the Scene editor's
// Detect pose): shoulder order and width against the torso say front, side or
// back (RTMPose labels left and right correctly from behind too), and the
// head's offset from the shoulders says which way a side view faces.

// subjectFacing is the direction the person in pose faces, in degrees with
// the camera-angle presets' convention: 0 = toward the camera, 90 = toward the
// right of the picture, 180 = away from the camera, 270 = toward the left;
// rounded to 45. ok is false when the pose is too unsure to say.
func subjectFacing(p *ScenePose) (deg int, ok bool) {
	if p == nil || len(p.Joints) != 18 {
		return 0, false
	}
	j := p.Joints
	nose, neck, rsho, lsho, rhip, lhip, rear, lear := j[0], j[1], j[2], j[5], j[8], j[11], j[16], j[17]
	for _, k := range []ScenePoseJoint{neck, rsho, lsho} {
		if k.Score < 0.4 {
			return 0, false
		}
	}
	hipY := (rhip.Y + lhip.Y) / 2
	torso := math.Hypot(neck.X-(rhip.X+lhip.X)/2, neck.Y-hipY)
	if rhip.Score < 0.3 || lhip.Score < 0.3 {
		// Hips cropped off: a shoulder span is about 0.7 of a torso.
		torso = math.Hypot(lsho.X-rsho.X, lsho.Y-rsho.Y) / 0.7
	}
	if torso < 8 {
		return 0, false
	}
	clamp := func(v float64) float64 { return math.Max(-1, math.Min(1, v)) }
	// Facing the camera, the person's right shoulder is on the picture's left.
	front := clamp((lsho.X - rsho.X) / (0.7 * torso))
	midX := (lsho.X + rsho.X) / 2
	// The head leads a side view. From behind the nose is a guess through the
	// hair, so the ears place the head there.
	headX := nose.X
	if front < -0.3 {
		headX = (rear.X + lear.X) / 2
	}
	side := clamp((headX - midX) / (0.4 * torso))
	a := math.Atan2(side, front) * 180 / math.Pi
	return ((int(math.Round(a/45))*45)%360 + 360) % 360, true
}

// detectSubjectFacing reads which way the main person in the image file faces,
// with the whole picture as their box. ok is false when there is no person to
// read, the model isn't installed yet (its install is started for next
// time), or detection fails: the camera prompt then says the move only
// relatively.
func detectSubjectFacing(ctx context.Context, pic string) (deg int, ok bool) {
	if missing := missingModels(scenePoseModel); len(missing) > 0 {
		_ = startModelInstalls("pose detection model", scenePoseModel)
		return 0, false
	}
	f, err := os.Open(pic)
	if err != nil {
		return 0, false
	}
	cfg, _, err := image.DecodeConfig(f)
	f.Close()
	if err != nil || cfg.Width < 16 || cfg.Height < 16 {
		return 0, false
	}
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	p, err := detectPoseInImage(ctx, pic, SceneRect{W: cfg.Width, H: cfg.Height})
	if err != nil {
		return 0, false
	}
	return subjectFacing(p)
}
