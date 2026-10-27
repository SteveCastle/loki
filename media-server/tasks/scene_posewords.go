package tasks

import (
	"encoding/json"
	"math"
	"sort"
	"strings"
)

// --- a mannequin's pose in words, for Repose ---
//
// The engine reads the OpenPose figure better with the pose also said in
// words ("both arms folded across the chest; the leg on the left of the
// picture lifted, ..."): measured in tools/repose-lab, where this is
// posewords.py's describe. Sides are the picture's (the figure's own left and
// right flip with the way it faces), and a limb whose root joint is outside
// the crop is left out.

// posePt is a joint in crop pixels.
type posePt struct{ X, Y float64 }

func poseDist(a, b posePt) float64 { return math.Hypot(a.X-b.X, a.Y-b.Y) }

// poseTorso is the neck to mid-hip length: the figure's scale.
func poseTorso(kp []posePt) float64 {
	return poseDist(kp[1], posePt{(kp[8].X + kp[11].X) / 2, (kp[8].Y + kp[11].Y) / 2})
}

// poseArm says where the arm from shoulder s (elbow e, wrist w, same-side hip) is.
func poseArm(kp []posePt, s, e, w, hip int, t float64) string {
	S, E, W := kp[s], kp[e], kp[w]
	up, fore := poseDist(S, E), poseDist(E, W)
	straight := poseDist(S, W) > 0.9*(up+fore)
	switch {
	case straight && W.Y > S.Y+0.8*(up+fore)*0.9:
		return "hanging relaxed at the side"
	case W.Y < kp[0].Y-0.15*t:
		return "raised above the head"
	case poseDist(W, kp[0]) < 0.4*t:
		return "up at the face"
	case poseDist(W, kp[hip]) < 0.5*t && E.Y < W.Y+0.2*t:
		return "resting on the hip, the elbow pointing out to the side"
	case math.Abs(W.Y-S.Y) < 0.3*t && math.Abs(W.X-S.X) > 0.6*t:
		return "stretched straight out to the side at shoulder height"
	case W.Y < S.Y+0.1*t:
		return "raised with the elbow out to the side and the hand up"
	case W.Y > kp[hip].Y-0.1*t && math.Abs(W.X-S.X) < 0.35*t:
		return "hanging relaxed at the side"
	case (W.X-kp[1].X)*(S.X-kp[1].X) < 0:
		return "across the front of the body"
	}
	return "held down and away from the body"
}

// poseLeg says where the leg from hip h (knee k, ankle a) is.
func poseLeg(kp []posePt, h, k, a int) string {
	H, K, A := kp[h], kp[k], kp[a]
	thigh := poseDist(H, K)
	switch {
	case K.Y < H.Y+0.45*thigh:
		out := ""
		if math.Abs(K.X-H.X) > 0.5*thigh {
			out = " toward the right of the picture"
			if K.X < H.X {
				out = " toward the left of the picture"
			}
		}
		return "lifted, the knee raised high" + out + " and the foot off the ground"
	case A.Y < K.Y+0.2*thigh:
		return "bent at the knee with the foot lifted up behind or beside it"
	case math.Abs(A.X-H.X) > 0.5*thigh:
		return "stepping out to the side"
	}
	return "standing straight"
}

// describePose says an OpenPose-18 figure's pose (crop pixels, crop w×h).
func describePose(kp []posePt, w, h float64) string {
	if len(kp) != 18 {
		return ""
	}
	t := poseTorso(kp)
	if t <= 0 {
		return ""
	}
	seen := func(js ...int) bool {
		for _, j := range js {
			if kp[j].X < 0 || kp[j].Y < 0 || kp[j].X >= w || kp[j].Y >= h {
				return false
			}
		}
		return true
	}
	// sides names two limbs by where their far joints are against the body's middle.
	sides := func(limb string, a, b float64) (string, string) {
		xa, xb := a-kp[1].X, b-kp[1].X
		if xa*xb < 0 || math.Abs(xa-xb) > 0.3*t {
			l, r := "the "+limb+" on the left of the picture", "the "+limb+" on the right of the picture"
			if xa < xb {
				return l, r
			}
			return r, l
		}
		return "one " + limb, "the other " + limb
	}
	type part struct {
		x    float64
		text string
	}
	order := func(ps []part) {
		sort.SliceStable(ps, func(i, j int) bool {
			oi, oj := strings.HasPrefix(ps[i].text, "the other"), strings.HasPrefix(ps[j].text, "the other")
			if oi != oj {
				return !oi
			}
			return ps[i].x < ps[j].x
		})
	}
	var arms, legs []part
	folded := poseDist(kp[4], kp[7]) < 0.35*t
	if folded {
		mid := (kp[4].Y + kp[7].Y) / 2
		folded = math.Min(kp[2].Y, kp[5].Y)-0.1*t < mid && mid < (kp[8].Y+kp[11].Y)/2
	}
	if folded && seen(2, 5) {
		arms = append(arms, part{0, "both arms folded across the chest"})
	}
	na, nb := sides("arm", (kp[3].X+kp[4].X)/2, (kp[6].X+kp[7].X)/2)
	for _, a := range []struct {
		s, e, w, hip int
		name         string
	}{{2, 3, 4, 8, na}, {5, 6, 7, 11, nb}} {
		if seen(a.s) && !folded {
			arms = append(arms, part{kp[a.w].X, a.name + " " + poseArm(kp, a.s, a.e, a.w, a.hip, t)})
		}
	}
	ma, mb := sides("leg", (kp[9].X+kp[10].X)/2, (kp[12].X+kp[13].X)/2)
	for _, l := range []struct {
		h, k, a int
		name    string
	}{{8, 9, 10, ma}, {11, 12, 13, mb}} {
		if seen(l.h, l.k) {
			legs = append(legs, part{kp[l.k].X, l.name + " " + poseLeg(kp, l.h, l.k, l.a)})
		}
	}
	order(arms)
	order(legs)
	var out []string
	for _, p := range append(arms, legs...) {
		out = append(out, p.text)
	}
	return strings.Join(out, "; ")
}

// paintPoseWords is the pose of a paint layer's one live mannequin in words,
// with its joints taken into region (the generation's crop); "" when the
// layer has no mannequin or more than one.
func paintPoseWords(l *SceneLayer, region SceneRect) string {
	if l == nil {
		return ""
	}
	var ops []struct {
		T   string    `json:"t"`
		Off bool      `json:"off"`
		KP  []float64 `json:"kp"`
	}
	if json.Unmarshal(l.Shapes, &ops) != nil {
		return ""
	}
	var kp []posePt
	for _, op := range ops {
		if op.Off || op.T != "pose" {
			continue
		}
		if kp != nil || len(op.KP) != 36 {
			return ""
		}
		for i := 0; i < 18; i++ {
			kp = append(kp, posePt{op.KP[2*i] - float64(region.X), op.KP[2*i+1] - float64(region.Y)})
		}
	}
	if kp == nil {
		return ""
	}
	return describePose(kp, float64(region.W), float64(region.H))
}
