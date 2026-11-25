package tasks

import (
	"strings"
	"testing"
)

// The shape of the enhanced long-video prompts that replayed their story in
// every chunk: the shared part kept the whole video's summary and description
// (dialogue included) and no {chunk} line, and the segment parts were bare
// prose, appended after non_diegetic_music.
const wholeStoryShared = `subject_definitions:
<Subject 1> is the woman in <Picture 1>, with short grey hair and a green raincoat.
<Picture 1> is the first frame of [Shot 1].

summary:
[keyframe completion + audio reference] The video begins from <Picture 1> and shows <Subject 1> crossing the harbour and calling out to a boat.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - her face, hair and raincoat are retained.
<Picture 1> ([Shot 1] first frame): fully_preserved - the composition is retained at the opening.

detailed_description:
The target video is a live-action, naturalistic film in soft overcast light.
[Shot 1] The shot begins from <Picture 1>. <Subject 1> (S1) walks along the quay and says, <d>[English] Wait for me!</d> She waves.

overall_soundscape:
Gulls, lapping water and her footsteps on wet stone.

non_diegetic_music:
N/A`

func TestComposeChunkPromptReplacesTheWholeStory(t *testing.T) {
	parts := []string{
		`[Shot 1] The shot begins from <Picture 1>. <Subject 1> (S1) walks along the quay. She says, <d>[English] Wait for me!</d>`,
		`<Subject 1> reaches the end of the quay, still waving, and the boat slows.`,
	}
	first := ComposeChunkPrompt(wholeStoryShared, parts[0], 0)
	later := ComposeChunkPrompt(wholeStoryShared, parts[1], 1)
	for i, p := range []string{first, later} {
		// the shared story is gone: its dialogue only where the chunk's part has it
		if strings.Contains(p, "crossing the harbour") || strings.Count(p, "Wait for me!") != 1-i {
			t.Errorf("chunk %d keeps the shared story:\n%s", i+1, p)
		}
		// the part is the chunk's description, not text after the music section
		if !strings.HasSuffix(strings.TrimSpace(p), "N/A") || strings.Count(p, "detailed_description:") != 1 || strings.Count(p, "summary:") != 1 {
			t.Errorf("chunk %d sections:\n%s", i+1, p)
		}
		// the description keeps the shared style sentence
		if !strings.Contains(p, "detailed_description:\nThe target video is a live-action, naturalistic film in soft overcast light.\n") {
			t.Errorf("chunk %d lost the style lead:\n%s", i+1, p)
		}
		if !strings.Contains(p, "short grey hair") || !strings.Contains(p, "Gulls, lapping water") {
			t.Errorf("chunk %d lost shared sections:\n%s", i+1, p)
		}
		var order []string
		for _, s := range promptSections(p) {
			order = append(order, s.name)
		}
		if strings.Join(order, ",") != strings.Join(h3Sections, ",") {
			t.Errorf("chunk %d section order %v", i+1, order)
		}
	}
	// chunk 1 may begin from the picture; a later chunk continues the one before
	if !strings.Contains(first, "summary:\n[keyframe completion + audio reference] The shot begins from <Picture 1>.") || !strings.Contains(first, "is the first frame of [Shot 1]") {
		t.Errorf("chunk 1:\n%s", first)
	}
	if strings.Contains(later, "keyframe completion") || strings.Contains(later, "first frame") || strings.Contains(later, "opening") {
		t.Errorf("chunk 2 still starts over:\n%s", later)
	}
	if !strings.Contains(later, "summary:\n[audio reference] <Subject 1> reaches the end of the quay, still waving, and the boat slows.") {
		t.Errorf("chunk 2 summary:\n%s", later)
	}
	if !strings.Contains(later, "<Subject 1> (appears in [Shot 1]): fully_preserved") {
		t.Errorf("chunk 2 lost the subject's retention line:\n%s", later)
	}
	// an empty part keeps the shared part as it is (the scene carries on), less the start for a later chunk
	if got := ComposeChunkPrompt(wholeStoryShared, "", 0); got != wholeStoryShared {
		t.Errorf("empty part, chunk 1:\n%s", got)
	}
}

func TestStructureChunksSlotsAnEnhancedRewrite(t *testing.T) {
	shared, segs := structureChunks(wholeStoryShared, []string{"She walks.", "", "She waves."})
	if !hasSlot(shared) || hasSection(shared, "summary") || hasSection(shared, "detailed_description") {
		t.Fatalf("shared:\n%s", shared)
	}
	// the slot takes the summary's place, between the definitions and retention_analysis
	if !strings.Contains(shared, "first frame of [Shot 1].\n\n{chunk}\n\nretention_analysis:") {
		t.Errorf("slot position:\n%s", shared)
	}
	if segs[1] != "" || !strings.HasPrefix(segs[0], "summary:\n[keyframe completion + audio reference] She walks.\n\ndetailed_description:\nThe target video is") {
		t.Errorf("segments: %q", segs)
	}
	// composing the normalized form gives the same prompts as composing the raw one
	for i, raw := range []string{"She walks.", "She waves."} {
		k := i * 2
		if a, b := ComposeChunkPrompt(shared, segs[k], k), ComposeChunkPrompt(wholeStoryShared, raw, k); a != b {
			t.Errorf("chunk %d differs:\n%s\n----\n%s", k+1, a, b)
		}
	}
	// a compliant rewrite is kept
	good := "subject_definitions:\nS1.\n\n{chunk}\n\noverall_soundscape:\nRain."
	seg := "summary:\n[reference generation] She walks.\n\ndetailed_description:\nShe walks."
	if s, g := structureChunks(good, []string{seg}); s != good || g[0] != seg {
		t.Errorf("compliant rewrite changed: %q %q", s, g)
	}
	// a bare part in a structured slot becomes the description
	if _, g := structureChunks(good, []string{"She walks."}); g[0] != "detailed_description:\nShe walks." {
		t.Errorf("bare part in slot: %q", g)
	}
	// plain prose is left alone
	if s, g := structureChunks("a red tram", []string{"it stops"}); s != "a red tram" || g[0] != "it stops" {
		t.Errorf("plain: %q %q", s, g)
	}
}

func TestEnhanceAsksForHeadedSegmentsWithoutSharedStory(t *testing.T) {
	sys := enhanceSystemPrompt(EnhanceRequest{Task: "video", Model: "h3-video", Chunks: []int{209, 209, 192, 192}}, "GUIDE")
	for _, want := range []string{"each starting with its name and a colon", "never tells the story", "no dialogue lines", "never names keyframe completion"} {
		if !strings.Contains(sys, want) {
			t.Errorf("system prompt lacks %q", want)
		}
	}
	if strings.Contains(sys, "No headings") {
		t.Error("system prompt still forbids the section headings")
	}
}
