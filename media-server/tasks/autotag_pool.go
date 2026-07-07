package tasks

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/stevecastle/shrike/deps"
)

// classifyViaWorker runs one image through an onnxtag --serve worker using the
// length-framed protocol: the worker replies "OK <n>" followed by n tag lines,
// or "ERR <msg>".
func classifyViaWorker(w *serveWorker, imagePath string) ([]string, error) {
	if err := w.writeLine(imagePath); err != nil {
		return nil, err
	}
	head, ok := w.readLine()
	if !ok {
		return nil, fmt.Errorf("worker died: %s", w.stderrString())
	}
	if msg, found := strings.CutPrefix(head, "ERR "); found {
		return nil, fmt.Errorf("%s", msg)
	}
	countStr, found := strings.CutPrefix(head, "OK ")
	if !found {
		return nil, fmt.Errorf("unexpected response %q", head)
	}
	n, err := strconv.Atoi(strings.TrimSpace(countStr))
	if err != nil {
		return nil, fmt.Errorf("bad tag count %q", countStr)
	}
	tags := make([]string, 0, n)
	for i := 0; i < n; i++ {
		line, ok := w.readLine()
		if !ok {
			return nil, fmt.Errorf("worker died mid-response: %s", w.stderrString())
		}
		tags = append(tags, line)
	}
	return tags, nil
}

// tagsToTagInfos parses the worker's "name:score" lines into TagInfo, dropping
// the score suffix and assigning the "Suggested" category (matching the old
// per-image autotag behavior).
func tagsToTagInfos(tags []string) []TagInfo {
	var out []TagInfo
	for _, t := range tags {
		name := t
		if pos := strings.LastIndex(t, ":"); pos > 0 {
			name = t[:pos]
		}
		name = strings.TrimSpace(name)
		if name == "" {
			continue
		}
		out = append(out, TagInfo{Label: name, Category: "Suggested"})
	}
	return out
}

// depModelPathOrEmpty returns the installed model file path or "" if missing.
func depModelPathOrEmpty(modelID, rel string) string {
	p, _ := deps.ModelPath(modelID, rel)
	return p
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if strings.TrimSpace(v) != "" {
			return v
		}
	}
	return ""
}
