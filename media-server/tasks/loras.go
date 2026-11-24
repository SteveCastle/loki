package tasks

// LoRA discovery for the Generate / Reimagine LoRA picker (GET /api/loras).
//
// LoRAs are found in two places:
//   - the training runs folder (<data>/loras, where the train-lora task writes): every run's final LoRA
//     (<run>/<run>.safetensors) and its checkpoints (<run>/checkpoints/step-N/*.safetensors), plus any LoRA file
//     dropped directly in the folder;
//   - a `loras` folder next to the engines' models (e.g. ~/bin/models/loras), for LoRAs from elsewhere.
//
// Only safetensors headers are read. A LoRA's architecture comes from its tensor names (MOSS-Image block linears:
// `layers.N.proj.{q,k,v,qkv,o,ffn_up_gate,ffn_down}` in kohya, ComfyUI, flattened kohya or PEFT naming, as
// loki-dream's loader accepts), so the picker offers it only for models of that architecture. loki-tune's `loki`
// metadata record adds the trigger word, base model and training step.

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/stevecastle/shrike/platform"
)

// LoraInfo is one LoRA file as the picker shows it.
type LoraInfo struct {
	Path string `json:"path"`
	File string `json:"file"`
	// Name: the run name for loki-tune LoRAs, else the file name without extension.
	Name string `json:"name"`
	// Arch: the architecture it applies to ("moss" = MOSS-Image), "" when not recognised.
	Arch      string `json:"arch"`
	Trigger   string `json:"trigger,omitempty"`
	BaseModel string `json:"base_model,omitempty"`
	Rank      int    `json:"rank,omitempty"`
	// Step: the training step (loki-tune checkpoints and finals).
	Step  int `json:"step,omitempty"`
	Steps int `json:"steps,omitempty"`
	// Checkpoint: an intermediate checkpoint of a training run (not the run's final LoRA).
	Checkpoint bool   `json:"checkpoint,omitempty"`
	Run        string `json:"run,omitempty"`
	SizeMB     int64  `json:"size_mb"`
	Modified   string `json:"modified"`
}

// LoraList is the GET /api/loras answer.
type LoraList struct {
	Loras []LoraInfo `json:"loras"`
	Dirs  []string   `json:"dirs"`
}

// LoraTrainingDir is where training runs (and their final LoRAs) live.
func LoraTrainingDir() string {
	return loraTrainingRoot()
}

// loraTrainingRoot: <data>/loras (a variable so tests can point it at a temp folder).
var loraTrainingRoot = func() string {
	return filepath.Join(platform.GetDataDir(), "loras")
}

// loraSearchDirs: the training folder, then `loras` next to the engines' models.
var loraSearchDirs = func() []string {
	dirs := []string{LoraTrainingDir()}
	for _, bin := range []string{"loki-dream", "loki-tune"} {
		if p, err := exec.LookPath(bin); err == nil {
			d := filepath.Join(filepath.Dir(p), "models", "loras")
			if !containsPath(dirs, d) {
				dirs = append(dirs, d)
			}
		}
	}
	for _, d := range filepath.SplitList(os.Getenv("LOKI_MODELS")) {
		if d = strings.TrimSpace(d); d != "" && !containsPath(dirs, filepath.Join(d, "loras")) {
			dirs = append(dirs, filepath.Join(d, "loras"))
		}
	}
	return dirs
}

func containsPath(xs []string, p string) bool {
	for _, x := range xs {
		if strings.EqualFold(filepath.Clean(x), filepath.Clean(p)) {
			return true
		}
	}
	return false
}

// readSafetensorsHeader returns the tensor names and the string metadata of a safetensors file.
func readSafetensorsHeader(path string) ([]string, map[string]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()
	var n uint64
	if err := binary.Read(f, binary.LittleEndian, &n); err != nil {
		return nil, nil, err
	}
	if n == 0 || n > 64<<20 {
		return nil, nil, fmt.Errorf("%s: not a safetensors file", path)
	}
	buf := make([]byte, n)
	if _, err := io.ReadFull(f, buf); err != nil {
		return nil, nil, err
	}
	var h map[string]json.RawMessage
	if err := json.Unmarshal(buf, &h); err != nil {
		return nil, nil, fmt.Errorf("%s: bad safetensors header: %w", path, err)
	}
	meta := map[string]string{}
	names := make([]string, 0, len(h))
	for k, v := range h {
		if k == "__metadata__" {
			var m map[string]any
			if json.Unmarshal(v, &m) == nil {
				for mk, mv := range m {
					if s, ok := mv.(string); ok {
						meta[mk] = s
					}
				}
			}
			continue
		}
		names = append(names, k)
	}
	return names, meta, nil
}

var mossLoraKey = regexp.MustCompile(`layers[._](\d+)[._]proj[._](q|k|v|qkv|o|ffn_up_gate|ffn_down)\.(lora_down|lora_A)\.weight$`)

// loraArch: the architecture a LoRA's tensors target ("" when none loki-dream can apply).
func loraArch(names []string) string {
	for _, n := range names {
		if mossLoraKey.MatchString(n) {
			return "moss"
		}
	}
	return ""
}

// describeLora reads one file's header into a LoraInfo (nil when it is not a LoRA).
func describeLora(path string, info os.FileInfo) *LoraInfo {
	names, meta, err := readSafetensorsHeader(path)
	if err != nil {
		return nil
	}
	isLora := false
	for _, n := range names {
		if strings.Contains(n, "lora_down") || strings.Contains(n, "lora_A") || strings.HasSuffix(n, ".alpha") {
			isLora = true
			break
		}
	}
	if !isLora {
		return nil
	}
	l := &LoraInfo{
		Path:     path,
		File:     filepath.Base(path),
		Name:     strings.TrimSuffix(filepath.Base(path), filepath.Ext(path)),
		Arch:     loraArch(names),
		SizeMB:   info.Size() >> 20,
		Modified: info.ModTime().UTC().Format(time.RFC3339),
	}
	if raw := meta["loki"]; raw != "" {
		var rec map[string]any
		if json.Unmarshal([]byte(raw), &rec) == nil && rec["tool"] == "loki-tune" {
			str := func(k string) string { s, _ := rec[k].(string); return s }
			num := func(k string) int { f, _ := rec[k].(float64); return int(f) }
			if s := str("name"); s != "" {
				l.Name = s
			}
			l.Trigger = str("trigger")
			l.BaseModel = str("base_model")
			l.Rank, l.Step, l.Steps = num("rank"), num("step"), num("steps")
		}
	}
	if l.Rank == 0 {
		l.Rank, _ = strconv.Atoi(meta["ss_network_dim"])
	}
	return l
}

// ListLoras scans the LoRA folders (headers only), newest first; a run's final LoRA comes before its checkpoints.
func ListLoras() LoraList {
	out := LoraList{Loras: []LoraInfo{}}
	seen := map[string]bool{}
	add := func(path string, checkpoint bool, run string) {
		key := strings.ToLower(filepath.Clean(path))
		if seen[key] {
			return
		}
		info, err := os.Stat(path)
		if err != nil || info.IsDir() {
			return
		}
		if l := describeLora(path, info); l != nil {
			seen[key] = true
			l.Checkpoint, l.Run = checkpoint, run
			out.Loras = append(out.Loras, *l)
		}
	}
	isST := func(n string) bool { return strings.EqualFold(filepath.Ext(n), ".safetensors") }
	for _, dir := range loraSearchDirs() {
		entries, err := os.ReadDir(dir)
		if err != nil {
			continue
		}
		out.Dirs = append(out.Dirs, dir)
		for _, e := range entries {
			p := filepath.Join(dir, e.Name())
			if !e.IsDir() {
				if isST(e.Name()) {
					add(p, false, "")
				}
				continue
			}
			// a training run: its final LoRA, then its checkpoints; any other folder: its LoRA files
			if _, err := os.Stat(filepath.Join(p, "run.json")); err == nil {
				add(filepath.Join(p, e.Name()+".safetensors"), false, e.Name())
				ckpts, _ := filepath.Glob(filepath.Join(p, "checkpoints", "step-*", "*.safetensors"))
				for _, c := range ckpts {
					if filepath.Base(c) != "state.safetensors" {
						add(c, true, e.Name())
					}
				}
				continue
			}
			files, _ := filepath.Glob(filepath.Join(p, "*.safetensors"))
			for _, f := range files {
				add(f, false, "")
			}
		}
	}
	sort.SliceStable(out.Loras, func(i, j int) bool {
		a, b := out.Loras[i], out.Loras[j]
		if a.Run != "" && a.Run == b.Run {
			if a.Checkpoint != b.Checkpoint {
				return !a.Checkpoint
			}
			return a.Step > b.Step
		}
		return a.Modified > b.Modified
	})
	return out
}

// LoraRun is one loki-tune training run, for the Continue a run dialog (GET /api/lora-runs).
type LoraRun struct {
	Name       string  `json:"name"`
	Dir        string  `json:"dir"`
	Kind       string  `json:"kind,omitempty"`
	Trigger    string  `json:"trigger,omitempty"`
	Class      string  `json:"class,omitempty"`
	Model      string  `json:"model,omitempty"`
	Rank       int     `json:"rank,omitempty"`
	Resolution int     `json:"resolution,omitempty"`
	LR         float64 `json:"lr,omitempty"`
	// Steps: the run's planned total.
	Steps  int `json:"steps"`
	Images int `json:"images"`
	// Step: the last step logged; Loss: its smoothed loss.
	Step int     `json:"step"`
	Loss float64 `json:"loss,omitempty"`
	// Finished: the final LoRA exists (the run reached its steps).
	Finished    bool                `json:"finished"`
	Checkpoints []LoraRunCheckpoint `json:"checkpoints"`
	Modified    string              `json:"modified"`
}

// LoraRunCheckpoint: a resumable checkpoint (it has optimizer state) and its preview images.
type LoraRunCheckpoint struct {
	Step    int      `json:"step"`
	Dir     string   `json:"dir"`
	Lora    string   `json:"lora,omitempty"`
	Samples []string `json:"samples"`
}

var stepDirRe = regexp.MustCompile(`^step-(\d+)$`)

// ListLoraRuns reads every run folder in the training folder (run.json, checkpoints, samples, the log's last line),
// newest first.
func ListLoraRuns() []LoraRun {
	out := []LoraRun{}
	entries, err := os.ReadDir(LoraTrainingDir())
	if err != nil {
		return out
	}
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		dir := filepath.Join(LoraTrainingDir(), e.Name())
		b, err := os.ReadFile(filepath.Join(dir, "run.json"))
		if err != nil {
			continue
		}
		var c struct {
			Name       string            `json:"name"`
			Kind       string            `json:"kind"`
			Trigger    string            `json:"trigger"`
			Class      string            `json:"class"`
			Model      string            `json:"model"`
			Rank       int               `json:"rank"`
			Resolution int               `json:"resolution"`
			LR         float64           `json:"lr"`
			Steps      int               `json:"steps"`
			Items      []json.RawMessage `json:"items"`
		}
		if json.Unmarshal(b, &c) != nil {
			continue
		}
		r := LoraRun{Name: c.Name, Dir: dir, Kind: c.Kind, Trigger: c.Trigger, Class: c.Class, Model: filepath.Base(c.Model),
			Rank: c.Rank, Resolution: c.Resolution, LR: c.LR, Steps: c.Steps, Images: len(c.Items), Checkpoints: []LoraRunCheckpoint{}}
		if r.Name == "" {
			r.Name = e.Name()
		}
		if c.Model == "" {
			r.Model = ""
		}
		if info, err := os.Stat(filepath.Join(dir, "run.json")); err == nil {
			r.Modified = info.ModTime().UTC().Format(time.RFC3339)
		}
		_, ferr := os.Stat(filepath.Join(dir, r.Name+".safetensors"))
		r.Finished = ferr == nil
		ck, _ := os.ReadDir(filepath.Join(dir, "checkpoints"))
		for _, ce := range ck {
			m := stepDirRe.FindStringSubmatch(ce.Name())
			if m == nil || !ce.IsDir() {
				continue
			}
			cdir := filepath.Join(dir, "checkpoints", ce.Name())
			if _, err := os.Stat(filepath.Join(cdir, "state.safetensors")); err != nil {
				continue
			}
			step, _ := strconv.Atoi(m[1])
			cp := LoraRunCheckpoint{Step: step, Dir: cdir, Samples: []string{}}
			if loras, _ := filepath.Glob(filepath.Join(cdir, "*.safetensors")); len(loras) > 0 {
				for _, l := range loras {
					if filepath.Base(l) != "state.safetensors" {
						cp.Lora = l
					}
				}
			}
			samples, _ := filepath.Glob(filepath.Join(dir, "samples", ce.Name()+"_*.png"))
			sort.Strings(samples)
			cp.Samples = append(cp.Samples, samples...)
			r.Checkpoints = append(r.Checkpoints, cp)
			if info, err := os.Stat(cdir); err == nil && info.ModTime().UTC().Format(time.RFC3339) > r.Modified {
				r.Modified = info.ModTime().UTC().Format(time.RFC3339)
			}
		}
		sort.Slice(r.Checkpoints, func(i, j int) bool { return r.Checkpoints[i].Step < r.Checkpoints[j].Step })
		r.Step, r.Loss = lastLogged(filepath.Join(dir, "log.jsonl"))
		out = append(out, r)
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Modified > out[j].Modified })
	return out
}

// lastLogged: the last step and smoothed loss in a run's log.jsonl (reads only its tail).
func lastLogged(path string) (int, float64) {
	f, err := os.Open(path)
	if err != nil {
		return 0, 0
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return 0, 0
	}
	off := info.Size() - 4096
	if off < 0 {
		off = 0
	}
	buf := make([]byte, info.Size()-off)
	if _, err := f.ReadAt(buf, off); err != nil && err != io.EOF {
		return 0, 0
	}
	lines := strings.Split(strings.TrimSpace(string(buf)), "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		var l struct {
			Step int     `json:"step"`
			Avg  float64 `json:"avg"`
		}
		if json.Unmarshal([]byte(lines[i]), &l) == nil && l.Step > 0 {
			return l.Step, l.Avg
		}
	}
	return 0, 0
}
