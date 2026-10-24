package tasks

import (
	"fmt"
	"regexp"
	"strings"
)

// Finish layers: an Edit layer made by "Finish with AI" on a paint layer, in
// one of three modes: make a sketch real (add what it stands for), recolour
// from rough colour, or repose a character to a drawn pose. Its prompt is
// only the user's words ("a stone cottage", "warm light", "waving"); the
// grounding around it tells the engine how to read the paint, and is composed
// in here so the prompt box stays the user's own words. The user may replace
// the grounding (options.grounding). Placeholders: {what} = the prompt,
// {guide} = the token of the paint layer when it is a reference, {where} =
// where the sketch is in the crop (" at the bottom left"), {target} = the
// token of the sketch rendered alone on black (sceneFinishTarget), {pose} =
// the mannequin's pose in words (describePose), {self} = the token of the
// picture given again as an identity reference (Repose). A sentence naming
// {target}, {pose} or {self} is dropped when it is empty (Kromoss takes no
// images; a hand-drawn figure has no words).
//
// Retouch takes no mask, so the words and the target image are all that say
// which part of the crop to change: given a whole frame and "replace the
// sketch with a man" it replaced the biggest figure in it, not the sketch.
//
// Both groundings are style-neutral ("this picture", never "photograph"): the
// scene may be a photo or an artist's painting, and the result must match it.
// Mirrors FINISH_GROUNDING / finishPrompt in the editor's scene-api.ts.

const (
	// FinishSketch: paint in the picture is a rough sketch to make real.
	FinishSketch = "sketch"
	// FinishColour: a guide paint layer colours the picture below.
	FinishColour = "colour"
	// FinishRepose: a guide paint layer is a rough drawing of a new pose for
	// the character below.
	FinishRepose = "repose"
)

// finishUsesGuide reports whether a mode reads the paint as a guide (a
// reference over the clean picture) rather than as part of the picture.
func finishUsesGuide(mode string) bool {
	return mode == FinishColour || mode == FinishRepose
}

var sceneFinishGrounding = map[string]string{
	FinishSketch: "In <image1>, the rough painted sketch{where} stands in for {what}: replace that sketch, and only it, with {what} as it would really appear in this picture. " +
		"{target} shows the sketch alone on black, in exactly the same place: it marks the only part of <image1> to change. " +
		"The sketch is a crude placeholder that only marks where it goes, roughly how big it is and a hint of its colours, " +
		"so do not copy its flat paint or hard outline: correct its shape and proportions, and give it the real look, light, texture and focus " +
		"it would have here, in the picture's own style. Leave everything that was not sketched exactly as it is: " +
		"do not change, move or replace any other figure or object, even one that looks like {what}.",
	FinishColour: "Colour <image1> using the rough colours painted on {guide} as a guide: {what}. " +
		"The paint is loose and only marks which colour goes where: give each whole object or area the colour painted mostly over it, " +
		"following the shapes and edges of <image1> rather than the edges of the brush strokes. " +
		"Keep every line and detail of <image1> as it is, and finish with clean, natural shading.",
	// Evolved in tools/repose-lab (FAVORITE.md, strategy x1idref): limbs follow
	// the figure's directions, not its lengths (a mannequin's shin read from
	// an ankle off the frame is short); "keep the person exactly as they are"
	// kept the old arms too (third arms); the words and the picture again as
	// <image3> hold the pose and the person's face and body type.
	FinishRepose: "New pose for the person in <image1>: {pose}. " +
		"Make the person in <image1> do the exact same pose as shown in {guide}: {what}. " +
		"{guide} only shows the pose: where the head, shoulders, elbows, hands, hips, knees and feet go. " +
		"Follow the direction each part of the body points in {guide}, but keep this person's own body: the same proportions, limb lengths, build and size in the frame as in <image1>, so it is clearly the same body, only moved. " +
		"Only the arms and legs that are placed differently in {guide} move; the rest of the body stays exactly where it is. " +
		"They have exactly two arms, two hands and two legs, each joined naturally to the body; an arm or leg that moves leaves its old place, which then shows the background or their clothing instead. " +
		"{self} is the same picture again, for their face and body. " +
		"Keep their identity exactly: the same face, hair, skin and figure, and the same clothing, which drapes and folds naturally with the new pose. " +
		"Keep the background, the lighting, the camera angle and the framing of <image1> unchanged.",
}

// sceneFinishPrompt is what the engine is given for a still layer: its prompt
// as it is, or for a finish layer the grounding around it. refs are the
// layer's references (the guide's and the target's tokens are looked up among
// them); where is sketchWhere's phrase, or ""; pose is the mannequin's pose
// in words (paintPoseWords), or "".
func sceneFinishPrompt(opts map[string]any, prompt string, refs []sceneRef, where, pose string) string {
	mode := optString(opts, "finish")
	grounding := strings.TrimSpace(optString(opts, "grounding"))
	if grounding == "" {
		grounding = sceneFinishGrounding[mode]
	}
	if grounding == "" {
		return prompt
	}
	guide, target, self := "<image2>", "", ""
	paint := optString(opts, "paint")
	for i, t := range sceneRefTokens(SceneKindStill, refs, false) {
		if refs[i].Layer != "" && refs[i].Layer == paint && t != "" {
			guide = t
		}
		if refs[i].Target && t != "" {
			target = t
		}
		if refs[i].Identity && t != "" {
			self = t
		}
	}
	return composeFinish(fillOrDrop(fillOrDrop(grounding, "{pose}", pose), "{self}", self), prompt, guide, where, target)
}

// fillOrDrop puts val in for key, or drops the sentences naming key when val
// is empty. The sentence ends at a full stop, so val must hold none.
func fillOrDrop(s, key, val string) string {
	if val == "" {
		return strings.TrimSpace(regexp.MustCompile(`[^.]*`+regexp.QuoteMeta(key)+`[^.]*\.`).ReplaceAllString(s, ""))
	}
	return strings.ReplaceAll(s, key, val)
}

// composeFinish fills a grounding's placeholders. With no prompt a
// "…: {what}." clause (or ", even one that looks like {what}.") folds away
// instead of dangling, and any other {what} reads "the real thing"; with no
// target its sentence goes.
func composeFinish(grounding, what, guide, where, target string) string {
	what = strings.TrimRight(strings.TrimSpace(what), ".")
	out := grounding
	if what == "" {
		out = strings.ReplaceAll(out, ": {what}.", ".")
		out = strings.ReplaceAll(out, ", even one that looks like {what}.", ".")
		what = "the real thing"
	}
	out = fillOrDrop(out, "{target}", target)
	out = strings.ReplaceAll(out, "{what}", what)
	out = strings.ReplaceAll(out, "{where}", where)
	return strings.ReplaceAll(out, "{guide}", guide)
}

// sketchWhere says where paint (its bounds) sits in a crop: " at the bottom
// left", " on the right", " in the middle"; "" when it fills most of it or
// misses it. Mirrors sketchWhere in the editor's scene-api.ts.
func sketchWhere(paint, region SceneRect) string {
	if paint.empty() || region.empty() {
		return ""
	}
	if float64(paint.W*paint.H) > 0.5*float64(region.W*region.H) {
		return ""
	}
	third := func(c, start, size int) int {
		f := float64(c-start) / float64(size)
		switch {
		case f < 1.0/3:
			return 0
		case f > 2.0/3:
			return 2
		}
		return 1
	}
	cx, cy := paint.X+paint.W/2, paint.Y+paint.H/2
	if cx < region.X || cy < region.Y || cx > region.X+region.W || cy > region.Y+region.H {
		return ""
	}
	v := []string{"top", "", "bottom"}[third(cy, region.Y, region.H)]
	h := []string{"left", "", "right"}[third(cx, region.X, region.W)]
	switch {
	case v == "" && h == "":
		return " in the middle"
	case v == "":
		return " on the " + h
	case h == "":
		return fmt.Sprintf(" at the %s", v)
	}
	return fmt.Sprintf(" at the %s %s", v, h)
}
