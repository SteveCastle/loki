package workerapi

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// OutputFlags are the engines' output options (CONVENTIONS.md: -o/--out):
// their values are outputs to map back, never inputs to send.
var OutputFlags = []string{"-o", "--out"}

// LocalInput is one local file a plan sends, and where it lands on the worker.
type LocalInput struct {
	Local  string
	Remote string // relative to the job directory
	Size   int64
}

type outMap struct {
	rel   string // "out/0/name"
	local string
}

type inMap struct {
	dir   string // "in/0"
	local string // the local input's folder
}

// Plan is a local command line rewritten for a worker: every existing local
// file it names becomes an input under in/<k>/, every -o/--out an output
// under out/<k>/, and the worker's paths map back to local ones afterwards.
type Plan struct {
	Args   []string
	Inputs []LocalInput
	// Dirs are output folders to create on the worker (an -o naming an
	// existing local folder: engines write into it by name).
	Dirs []string
	outs []outMap
	ins  []inMap
	sent map[string]string // remote path of each input -> its local file
}

// statFile reports an existing regular file's size; a variable for tests.
var statFile = func(p string) (int64, bool) {
	st, err := os.Stat(p)
	if err != nil || !st.Mode().IsRegular() {
		return 0, false
	}
	return st.Size(), true
}

var statDir = func(p string) bool {
	st, err := os.Stat(p)
	return err == nil && st.IsDir()
}

// PlanArgs maps args (the command line after the executable) onto a
// worker. A value is an input when it is an absolute path to an existing
// local file, as a bare argument or after "--flag="; loki-reshoot's
// FILE@START,DURATION media specs are recognized too.
func PlanArgs(args []string) *Plan {
	p := &Plan{sent: map[string]string{}}
	byLocal := map[string]string{}
	isOut := func(a string) bool {
		for _, f := range OutputFlags {
			if a == f {
				return true
			}
		}
		return false
	}
	for i := 0; i < len(args); i++ {
		a := args[i]
		if isOut(a) && i+1 < len(args) {
			p.Args = append(p.Args, a, p.output(args[i+1]))
			i++
			continue
		}
		if strings.HasPrefix(a, "--out=") {
			p.Args = append(p.Args, "--out="+p.output(strings.TrimPrefix(a, "--out=")))
			continue
		}
		if strings.HasPrefix(a, "--") {
			if k := strings.Index(a, "="); k > 0 {
				if r, ok := p.input(a[k+1:], byLocal); ok {
					p.Args = append(p.Args, a[:k+1]+r)
					continue
				}
				p.Args = append(p.Args, a)
				continue
			}
		}
		if r, ok := p.input(a, byLocal); ok {
			p.Args = append(p.Args, r)
			continue
		}
		p.Args = append(p.Args, a)
	}
	return p
}

func (p *Plan) output(v string) string {
	if v == "-" || v == "" {
		return v
	}
	rel := fmt.Sprintf("out/%d/%s", len(p.outs), SafeName(filepath.Base(v)))
	p.outs = append(p.outs, outMap{rel: rel, local: v})
	if statDir(v) {
		// an existing folder: the engine writes into it, so it must exist there too
		p.Dirs = append(p.Dirs, rel)
		return JobDir + "/" + rel + "/"
	}
	return JobDir + "/" + rel
}

func (p *Plan) input(v string, byLocal map[string]string) (string, bool) {
	if v == "" || !filepath.IsAbs(v) {
		return "", false
	}
	if r, ok := byLocal[v]; ok {
		return JobDir + "/" + r, true
	}
	if size, ok := statFile(v); ok {
		rel := fmt.Sprintf("in/%d/%s", len(p.ins), SafeName(filepath.Base(v)))
		p.ins = append(p.ins, inMap{dir: fmt.Sprintf("in/%d", len(p.ins)), local: filepath.Dir(v)})
		p.Inputs = append(p.Inputs, LocalInput{Local: v, Remote: rel, Size: size})
		p.sent[rel] = v
		byLocal[v] = rel
		return JobDir + "/" + rel, true
	}
	// FILE@START[,DURATION]
	if k := strings.LastIndex(v, "@"); k > 0 {
		if r, ok := p.input(v[:k], byLocal); ok {
			return r + v[k:], true
		}
	}
	return "", false
}

// LocalPath maps a path relative to the job directory to the local path it
// stands for: an input, an output (or a file inside an output folder), or a
// new file the engine wrote beside one of the inputs.
func (p *Plan) LocalPath(rel string) (string, bool) {
	rel = strings.TrimPrefix(filepath.ToSlash(rel), "/")
	for _, o := range p.outs {
		if rel == o.rel {
			return o.local, true
		}
		if rest, ok := strings.CutPrefix(rel, o.rel+"/"); ok && rest != "" {
			return filepath.Join(o.local, filepath.FromSlash(rest)), true
		}
	}
	if local, ok := p.sent[rel]; ok {
		return local, true
	}
	for _, in := range p.ins {
		if rest, ok := strings.CutPrefix(rel, in.dir+"/"); ok && rest != "" && !strings.Contains(rest, "/") {
			return filepath.Join(in.local, rest), true
		}
	}
	return "", false
}

var jobPathRe = regexp.MustCompile(`\$(?:JOB|SESSION)(/[^\s"',;]*)?`)

// RewriteLine replaces the worker paths ($JOB/...) in a line of the engine's
// output with the local paths they map to; inside a JSON line the
// replacement is JSON-escaped.
func (p *Plan) RewriteLine(line string) string {
	return RewritePaths(line, p.LocalPath)
}

// RewritePaths replaces every $JOB/... or $SESSION/... path in line through
// mapFn (rel -> local); unmapped paths are left as they are.
func RewritePaths(line string, mapFn func(rel string) (string, bool)) string {
	if !strings.Contains(line, "$") {
		return line
	}
	jsonish := strings.HasPrefix(strings.TrimSpace(line), "{") || strings.HasPrefix(strings.TrimSpace(line), "[")
	return jobPathRe.ReplaceAllStringFunc(line, func(m string) string {
		k := strings.Index(m, "/")
		if k < 0 {
			return m
		}
		local, ok := mapFn(m[k+1:])
		if !ok {
			return m
		}
		if jsonish {
			b, _ := json.Marshal(local)
			return string(b[1 : len(b)-1])
		}
		return local
	})
}

// SafeName keeps a file name to letters, digits, dot, dash and underscore,
// so a worker path never needs quoting (and never contains a separator).
func SafeName(name string) string {
	var b strings.Builder
	for _, r := range name {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '.', r == '-', r == '_':
			b.WriteRune(r)
		default:
			b.WriteByte('_')
		}
	}
	s := strings.Trim(b.String(), ".")
	if s == "" {
		return "file"
	}
	return s
}
