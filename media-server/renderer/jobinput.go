package renderer

import (
	"encoding/base64"
	"regexp"
	"strings"
	"unicode/utf8"
)

// JobInputView is a display-only, human-readable interpretation of a job's raw
// input string. It does NOT change the stored input — it only drives how the
// templates render it (see the "jobInput" partial). Kind is one of:
//
//	"query" — input carried a base64 --query64=… ; Query is the decoded text and
//	          Prefix is the surrounding command/flags (for context).
//	"paths" — input is a path or a list of paths; Paths holds them.
//	"raw"   — anything else; Raw is the input verbatim.
type JobInputView struct {
	Kind   string
	Query  string
	Prefix string
	Paths  []string
	Raw    string
}

var (
	query64Re = regexp.MustCompile(`--query64(?:=|\s+)(\S+)`)
	driveRe   = regexp.MustCompile(`^[A-Za-z]:[\\/]`)
	mediaExt  = regexp.MustCompile(`(?i)\.(mp4|mkv|avi|mov|m4v|webm|wmv|mp3|wav|flac|aac|ogg|m4a|opus|jpg|jpeg|jfif|webp|avif|png|gif|bmp|tiff?|vtt|srt|ass)$`)
)

// jobInputView is the template function that produces the human-readable view.
func jobInputView(input string) JobInputView {
	s := strings.TrimSpace(input)
	if s == "" {
		return JobInputView{Kind: "raw", Raw: ""}
	}
	if q, prefix, ok := decodeQuery64(s); ok {
		return JobInputView{Kind: "query", Query: q, Prefix: prefix}
	}
	if paths, ok := asPathList(s); ok {
		return JobInputView{Kind: "paths", Paths: paths}
	}
	return JobInputView{Kind: "raw", Raw: s}
}

// decodeQuery64 finds a --query64=<base64> (or "--query64 <base64>") token,
// decodes it, and returns the decoded query plus the input with that token
// removed (whitespace collapsed) as Prefix. ok is false when there's no valid
// query64 token.
func decodeQuery64(input string) (query, prefix string, ok bool) {
	m := query64Re.FindStringSubmatchIndex(input)
	if m == nil {
		return "", "", false
	}
	token := input[m[2]:m[3]]
	dec, err := base64.StdEncoding.DecodeString(token)
	if err != nil || !utf8.Valid(dec) || len(dec) == 0 {
		return "", "", false
	}
	pre := input[:m[0]] + input[m[1]:]
	pre = strings.Join(strings.Fields(pre), " ") // collapse whitespace
	return string(dec), strings.TrimSpace(pre), true
}

// asPathList returns the input split into paths when it looks like a path or a
// (newline/comma-separated) list of paths and carries no CLI flags.
func asPathList(input string) ([]string, bool) {
	if strings.Contains(input, "--") {
		return nil, false // has flags → not a plain path list
	}
	parts := strings.FieldsFunc(input, func(r rune) bool {
		return r == '\n' || r == '\r' || r == ','
	})
	var paths []string
	for _, p := range parts {
		p = strings.TrimSpace(p)
		p = strings.Trim(p, `"'`)
		if p == "" {
			continue
		}
		if !looksLikePath(p) {
			return nil, false
		}
		paths = append(paths, p)
	}
	if len(paths) == 0 {
		return nil, false
	}
	return paths, true
}

func looksLikePath(s string) bool {
	switch {
	case driveRe.MatchString(s): // C:\ or C:/
		return true
	case strings.HasPrefix(s, "/"), strings.HasPrefix(s, "\\"),
		strings.HasPrefix(s, "./"), strings.HasPrefix(s, "../"),
		strings.HasPrefix(s, "~/"):
		return true
	case strings.ContainsAny(s, "/\\") && !strings.Contains(s, " "):
		return true
	case mediaExt.MatchString(s): // bare filename with a media extension
		return true
	default:
		return false
	}
}
