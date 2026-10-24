package tasks

import "strings"

// Finish layers: an Edit layer made by "Finish with AI" on a paint layer, in
// one of three modes: make a sketch real (add what it stands for), recolour
// from rough colour, or repose a character to a drawn pose. Its prompt is
// only the user's words ("a stone cottage", "warm light", "waving"); the
// grounding around it tells the engine how to read the paint, and is composed
// in here so the prompt box stays the user's own words. The user may replace
// the grounding (options.grounding). Placeholders: {what} = the prompt,
// {guide} = the token of the paint layer when it is a reference.
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
	FinishSketch: "In <image1>, a rough painted sketch stands in for {what}: replace it with {what} as it would really appear in this picture. " +
		"The sketch is a crude placeholder that only marks where it goes, roughly how big it is and a hint of its colours, " +
		"so do not copy its flat paint or hard outline: correct its shape and proportions, and give it the real look, light, texture and focus " +
		"it would have here, in the picture's own style. Leave everything that was not sketched as it is.",
	FinishColour: "Colour <image1> using the rough colours painted on {guide} as a guide: {what}. " +
		"The paint is loose and only marks which colour goes where: give each whole object or area the colour painted mostly over it, " +
		"following the shapes and edges of <image1> rather than the edges of the brush strokes. " +
		"Keep every line and detail of <image1> as it is, and finish with clean, natural shading.",
	FinishRepose: "Repose the character in <image1> so their body takes the pose drawn over the picture on {guide}: {what}. " +
		"The drawing is a rough gesture sketch that only shows the new pose, where the head, body, arms, hands and legs go, " +
		"so do not draw its lines or colours. Keep the character exactly as they are in <image1>: the same identity, face, hair, " +
		"clothing, colours, proportions and style; only their pose changes. Where the new pose uncovers what was behind the character, " +
		"fill it in to match its surroundings. Leave everything else as it is.",
}

// sceneFinishPrompt is what the engine is given for a still layer: its prompt
// as it is, or for a finish layer the grounding around it. refs are the
// layer's references (the guide's token is looked up among them).
func sceneFinishPrompt(opts map[string]any, prompt string, refs []sceneRef) string {
	mode := optString(opts, "finish")
	grounding := strings.TrimSpace(optString(opts, "grounding"))
	if grounding == "" {
		grounding = sceneFinishGrounding[mode]
	}
	if grounding == "" {
		return prompt
	}
	guide := "<image2>"
	paint := optString(opts, "paint")
	for i, t := range sceneRefTokens(SceneKindStill, refs, false) {
		if refs[i].Layer != "" && refs[i].Layer == paint && t != "" {
			guide = t
		}
	}
	return composeFinish(grounding, prompt, guide)
}

// composeFinish fills a grounding's placeholders. With no prompt a
// "…: {what}." clause folds away instead of leaving a dangling colon, and any
// other {what} reads "the real thing".
func composeFinish(grounding, what, guide string) string {
	what = strings.TrimRight(strings.TrimSpace(what), ".")
	out := grounding
	if what == "" {
		out = strings.ReplaceAll(out, ": {what}.", ".")
		what = "the real thing"
	}
	out = strings.ReplaceAll(out, "{what}", what)
	return strings.ReplaceAll(out, "{guide}", guide)
}
