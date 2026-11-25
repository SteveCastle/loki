package tasks

import (
	"regexp"
	"sort"
	"strings"
)

// A long video's chunked prompt: a shared part and one part per chunk. Each
// chunk runs with ComposeChunkPrompt(shared, its part, its index); the
// renderer's composeChunkPrompt (components/transform/intents.ts) is the same
// rule, and tasks/prompt_chunks_test.go and transform-intents.test.ts pin it.
//
// For the H3 video format (prompt_guides/h3-video.md: six sections, each a
// name and a colon) the time-dependent sections, summary and
// detailed_description, belong to the chunk: a shared part that still has
// them (no {chunk} line) would tell every chunk the whole story, so each one
// replays it, dialogue included. A chunk's part takes their place instead.
// Later chunks continue the one before, so they lose the shared part's
// first-frame anchoring and the keyframe completion task type.

// ChunkSlot is the line of a chunked prompt's shared part where each chunk's
// own part goes; without it the chunk's part follows the shared one.
const ChunkSlot = "{chunk}"

// h3Header is a section header line of the H3 video format.
var h3Header = regexp.MustCompile(`^\s*(subject_definitions|summary|retention_analysis|detailed_description|overall_soundscape|non_diegetic_music)\s*:`)

// promptSection is one section of a structured prompt with its raw lines
// (the header line first); name "" is the text before the first header.
type promptSection struct {
	name  string
	lines []string
}

func promptSections(s string) []promptSection {
	var out []promptSection
	for _, l := range strings.Split(strings.ReplaceAll(s, "\r\n", "\n"), "\n") {
		if m := h3Header.FindStringSubmatch(l); m != nil {
			out = append(out, promptSection{name: m[1], lines: []string{l}})
			continue
		}
		if len(out) == 0 {
			out = append(out, promptSection{})
		}
		out[len(out)-1].lines = append(out[len(out)-1].lines, l)
	}
	return out
}

// body is a section's text without its header.
func (p promptSection) body() string {
	lines := append([]string{}, p.lines...)
	if p.name != "" && len(lines) > 0 {
		lines[0] = lines[0][strings.Index(lines[0], ":")+1:]
	}
	return strings.TrimSpace(strings.Join(lines, "\n"))
}

func isTimeSection(name string) bool { return name == "summary" || name == "detailed_description" }

func hasSection(s, name string) bool {
	for _, p := range promptSections(s) {
		if p.name == name {
			return true
		}
	}
	return false
}

func isStructured(s string) bool {
	for _, l := range strings.Split(s, "\n") {
		if h3Header.MatchString(l) {
			return true
		}
	}
	return false
}

func hasSlot(s string) bool {
	for _, l := range strings.Split(s, "\n") {
		if strings.TrimSpace(l) == ChunkSlot {
			return true
		}
	}
	return false
}

var (
	summaryTag = regexp.MustCompile(`^\s*\[([^\]]+)\]`)
	// a summary line's task type, also on the header line itself
	summaryTagLine = regexp.MustCompile(`^(?:\s*summary\s*:)?\s*\[([^\]]+)\]`)
	shotMarker     = regexp.MustCompile(`\[Shot \d+\]\s*`)
	dialogueTag    = regexp.MustCompile(`(?s)<d>.*?</d>`)
	sentenceEnd    = regexp.MustCompile(`[.!?](\s|$)`)
	// a shared-part line that pins a reference picture to the video's start
	firstFrameLine = regexp.MustCompile(`(?i)^\s*[-*]?\s*<Picture \d+>.*\b(first frame|opening|begins|starts)\b`)
)

// leadSentence is the gist of a chunk's part for its summary: its first
// sentence without dialogue or shot markers.
func leadSentence(part string) string {
	s := strings.Join(strings.Fields(shotMarker.ReplaceAllString(dialogueTag.ReplaceAllString(part, ""), "")), " ")
	if loc := sentenceEnd.FindStringIndex(s); loc != nil {
		s = s[:loc[0]+1]
	}
	if len(s) > 240 {
		cut := strings.LastIndex(s[:240], " ")
		if cut < 0 {
			cut = 240
		}
		s = s[:cut] + "…"
	}
	return s
}

// chunkTimeSections writes a chunk's part as the time-dependent sections of
// the structured shared part `secs`: the shared summary's task type before
// the part's first sentence, and the shared description's opening style
// sentence(s) before the part. A part with section headers of its own is kept.
func chunkTimeSections(secs []promptSection, part string) string {
	if isStructured(part) {
		return part
	}
	var tag, lead string
	hadSummary := false
	for _, p := range secs {
		switch p.name {
		case "summary":
			hadSummary = true
			if m := summaryTag.FindStringSubmatch(p.body()); m != nil {
				tag = "[" + strings.TrimSpace(m[1]) + "] "
			}
		case "detailed_description":
			lead = p.body()
			if i := strings.Index(lead, "[Shot"); i >= 0 {
				lead = lead[:i]
			}
			lead = strings.TrimSpace(lead)
		}
	}
	var b strings.Builder
	if hadSummary {
		b.WriteString("summary:\n" + tag + leadSentence(part) + "\n\n")
	}
	b.WriteString("detailed_description:\n")
	if lead != "" && !strings.HasPrefix(part, lead) {
		b.WriteString(lead + "\n")
	}
	b.WriteString(part)
	return b.String()
}

// slotChunk is the shared part with a {chunk} line and the chunk's part to
// put there: a structured shared part without the line gives up its
// time-dependent sections to the part (the slot takes the first one's place);
// a bare part in a structured shared part's slot becomes its description.
func slotChunk(shared, part string) (string, string) {
	part = strings.TrimSpace(part)
	if part == "" || !isStructured(shared) {
		return shared, part
	}
	if hasSlot(shared) {
		if !isStructured(part) && !hasSection(shared, "detailed_description") {
			part = "detailed_description:\n" + part
		}
		return shared, part
	}
	secs := promptSections(shared)
	if !hasSection(shared, "detailed_description") && !hasSection(shared, "summary") {
		return shared, part
	}
	var lines []string
	slotted := false
	for _, p := range secs {
		if !isTimeSection(p.name) {
			lines = append(lines, p.lines...)
			continue
		}
		if !slotted {
			lines = append(lines, ChunkSlot, "")
			slotted = true
		}
	}
	return strings.TrimSpace(strings.Join(lines, "\n")), chunkTimeSections(secs, part)
}

// continuationOnly drops what makes a later chunk start over: lines pinning a
// reference picture to the first frame or the opening, and the keyframe
// completion task type (a later chunk begins from the one before).
func continuationOnly(s string) string {
	if !isStructured(s) {
		return s
	}
	var out []string
	section := ""
	for _, l := range strings.Split(s, "\n") {
		if m := h3Header.FindStringSubmatch(l); m != nil {
			section = m[1]
		}
		switch section {
		case "subject_definitions", "retention_analysis":
			if firstFrameLine.MatchString(l) {
				continue
			}
		case "summary":
			if m := summaryTagLine.FindStringSubmatchIndex(l); m != nil {
				var keep []string
				for _, t := range strings.Split(l[m[2]:m[3]], "+") {
					if t = strings.TrimSpace(t); t != "" && !strings.EqualFold(t, "keyframe completion") {
						keep = append(keep, t)
					}
				}
				if len(keep) == 0 {
					keep = []string{"reference generation"}
				}
				l = l[:m[2]] + strings.Join(keep, " + ") + l[m[3]:]
			}
		}
		out = append(out, l)
	}
	return strings.Join(out, "\n")
}

// ComposeChunkPrompt is the prompt chunk `chunk` (0-based) of a long video
// runs with: the shared part with the chunk's own part in its {chunk} line
// (or in place of its time-dependent sections, or after it).
func ComposeChunkPrompt(shared, segment string, chunk int) string {
	shared, segment = slotChunk(shared, segment)
	var out string
	lines := strings.Split(strings.ReplaceAll(shared, "\r\n", "\n"), "\n")
	slot := -1
	for i, l := range lines {
		if strings.TrimSpace(l) == ChunkSlot {
			slot = i
			break
		}
	}
	if slot >= 0 {
		rest := append([]string{}, lines[:slot]...)
		if segment != "" {
			rest = append(rest, segment)
		}
		out = strings.TrimSpace(strings.Join(append(rest, lines[slot+1:]...), "\n"))
	} else {
		var parts []string
		for _, p := range []string{strings.TrimSpace(shared), segment} {
			if p != "" {
				parts = append(parts, p)
			}
		}
		out = strings.Join(parts, "\n\n")
	}
	if chunk > 0 {
		out = continuationOnly(out)
	}
	return h3Order(out)
}

// h3Sections is the H3 video format's section order.
var h3Sections = []string{"subject_definitions", "summary", "retention_analysis", "detailed_description", "overall_soundscape", "non_diegetic_music"}

// h3Order puts a structured prompt's sections in the guide's order (one
// {chunk} slot holds both time-dependent sections, which are not adjacent).
func h3Order(s string) string {
	if !isStructured(s) {
		return s
	}
	rank := func(name string) int {
		for i, n := range h3Sections {
			if n == name {
				return i + 1
			}
		}
		return 0
	}
	secs := promptSections(s)
	sort.SliceStable(secs, func(a, b int) bool { return rank(secs[a].name) < rank(secs[b].name) })
	var blocks []string
	for _, p := range secs {
		if t := strings.TrimSpace(strings.Join(p.lines, "\n")); t != "" {
			blocks = append(blocks, t)
		}
	}
	return strings.Join(blocks, "\n\n")
}

// structureChunks puts an enhanced long-video prompt in the chunked form: a
// structured shared part that kept its time-dependent sections gets the
// {chunk} line, and each segment becomes those sections (slotChunk).
func structureChunks(shared string, segments []string) (string, []string) {
	out := make([]string, len(segments))
	slotted := shared
	for i, sg := range segments {
		var s string
		s, out[i] = slotChunk(shared, sg)
		if out[i] != "" {
			slotted = s
		}
	}
	return slotted, out
}
