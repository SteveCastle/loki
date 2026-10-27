package main

import (
	"encoding/json"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"os"
	"strconv"
	"strings"

	"github.com/stevecastle/shrike/onnxpose"
)

// poseJSON is --pose's answer: one person's body as OpenPose's 18 joints
// (nose, neck, right shoulder, ... left ear), x/y in image pixels, with the
// model's confidence (below ~0.3 a joint is a guess).
type poseJSON struct {
	Joints []poseJoint `json:"joints"`
}

type poseJoint struct {
	X     float32 `json:"x"`
	Y     float32 `json:"y"`
	Score float32 `json:"score"`
}

// runPose estimates the pose of the person in box ("x,y,w,h", image pixels)
// of imagePath with an RTMPose model, printing poseJSON.
func runPose(modelPath, ortLib, imagePath, box string) error {
	parts := strings.Split(box, ",")
	if len(parts) != 4 {
		return fmt.Errorf("--box must be x,y,w,h (got %q)", box)
	}
	var b [4]float64
	for i, p := range parts {
		v, err := strconv.ParseFloat(strings.TrimSpace(p), 64)
		if err != nil {
			return fmt.Errorf("--box: %w", err)
		}
		b[i] = v
	}
	if b[2] <= 1 || b[3] <= 1 {
		return fmt.Errorf("--box is empty")
	}
	f, err := os.Open(imagePath)
	if err != nil {
		return err
	}
	img, _, err := image.Decode(f)
	f.Close()
	if err != nil {
		return fmt.Errorf("decoding %s: %w", imagePath, err)
	}
	est, err := onnxpose.NewEstimator(modelPath, ortLib)
	if err != nil {
		return err
	}
	defer est.Destroy()
	coco, err := est.Estimate(img, b[0], b[1], b[2], b[3])
	if err != nil {
		return err
	}
	out := poseJSON{}
	for _, k := range onnxpose.ToOpenPose(coco) {
		out.Joints = append(out.Joints, poseJoint{X: k.X, Y: k.Y, Score: k.Score})
	}
	return json.NewEncoder(os.Stdout).Encode(out)
}
