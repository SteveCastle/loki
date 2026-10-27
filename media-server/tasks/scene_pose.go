package tasks

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/stevecastle/shrike/deps"
	"github.com/stevecastle/shrike/platform"
)

// --- pose detection for the Scene editor's mannequin ---
//
// "Detect pose" starts a paint layer's mannequin from a character in the
// picture: the editor sends the character's box (the Select tool's pick, or
// a box dragged with the Mannequin tool), the picture under the paint layer
// is flattened, and RTMPose (onnxpose, run by the bundled embed binary's
// --pose mode on the CPU) reads the person in that box. The answer is
// OpenPose's 18 joints in frame pixels with their confidence; the editor
// keeps the confident ones and fills the rest from a standing figure.

// scenePoseModel is the deps/models id of the RTMPose body model.
const scenePoseModel = "rtmpose-body"

// ScenePoseJoint is one detected joint, frame pixels.
type ScenePoseJoint struct {
	X     float64 `json:"x"`
	Y     float64 `json:"y"`
	Score float64 `json:"score"`
}

// ScenePose is DetectScenePose's answer: OpenPose's 18 joints in its order
// (nose, neck, right shoulder, ... left ear).
type ScenePose struct {
	Joints []ScenePoseJoint `json:"joints"`
}

// DetectScenePose reads the body pose of the person in box (frame pixels) in
// the picture as the layer lid sees it (every layer under it, flattened).
func DetectScenePose(ctx context.Context, id, lid string, box SceneRect) (*ScenePose, error) {
	if box.W < 8 || box.H < 8 {
		return nil, fmt.Errorf("select the character first (a click with Select object, or a box)")
	}
	if err := startModelInstalls("pose detection model", scenePoseModel); err != nil {
		return nil, err
	}

	sceneMu.Lock()
	s, err := loadSceneLocked(id)
	if err != nil {
		sceneMu.Unlock()
		return nil, err
	}
	idx := len(s.Layers)
	for i, l := range s.Layers {
		if l.ID == lid {
			idx = i
		}
	}
	dir, _ := sceneDir(id)
	below, err := sceneGraphLayers(dir, s.Width, s.Height, s.Layers[:idx], true)
	base, w, h := s.Base, s.Width, s.Height
	sceneMu.Unlock()
	if err != nil {
		return nil, err
	}
	box = box.clamp(w, h)

	tmp, err := os.CreateTemp(dir, "pose-*.png")
	if err != nil {
		return nil, err
	}
	pic := tmp.Name()
	tmp.Close()
	defer os.Remove(pic)
	args := sceneGraphArgs(base, w, h, below, false, 0, SceneRect{W: w, H: h})
	if err := runSceneFFmpeg(ctx, append([]string{"-y"}, append(args, pic)...)); err != nil {
		return nil, fmt.Errorf("flattening the picture: %w", err)
	}
	return detectPoseInImage(ctx, pic, box)
}

// detectPoseInImage runs RTMPose on the person in box (image pixels) of the
// image file pic. The model must already be installed (startModelInstalls).
func detectPoseInImage(ctx context.Context, pic string, box SceneRect) (*ScenePose, error) {
	model, err := deps.ModelPath(scenePoseModel, "model.onnx")
	if err != nil || model == "" {
		return nil, fmt.Errorf("the pose detection model is not installed")
	}
	embedBin := deps.BundledOrEmpty("embed")
	if embedBin == "" {
		return nil, bundledMissing("embed")
	}
	cmdArgs := []string{"--pose", "--model=" + model, "--image=" + pic, fmt.Sprintf("--box=%d,%d,%d,%d", box.X, box.Y, box.W, box.H)}
	if ortLib := deps.BundledOrEmpty("onnxruntime"); ortLib != "" {
		cmdArgs = append(cmdArgs, "--ort="+ortLib)
	}
	cmd := exec.CommandContext(ctx, embedBin, cmdArgs...)
	platform.HideSubprocessWindow(cmd)
	cmd.Dir = filepath.Dir(embedBin)
	var stderr strings.Builder
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("pose detection: %v: %s", err, strings.TrimSpace(stderr.String()))
	}
	var p ScenePose
	if err := json.Unmarshal(out, &p); err != nil {
		return nil, fmt.Errorf("pose detection answered %q: %w", strings.TrimSpace(string(out)), err)
	}
	if len(p.Joints) != 18 {
		return nil, fmt.Errorf("pose detection answered %d joints, want 18", len(p.Joints))
	}
	return &p, nil
}
