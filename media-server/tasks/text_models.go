package tasks

import (
	"encoding/binary"
	"errors"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/stevecastle/shrike/deps/models"
)

// TextModel is a GGUF language model found on this machine, for the Config
// page's model picker. Spec is what goes into Config.TextModel: the LM Studio
// / Hugging Face style publisher/repo/file.gguf when the file sits in that
// layout, else its path.
type TextModel struct {
	Spec      string `json:"spec"`
	Path      string `json:"path"`
	Name      string `json:"name"`
	SizeBytes int64  `json:"sizeBytes"`
	Arch      string `json:"arch"`
	// Supported: loki-scribe runs this architecture (qwen35).
	Supported bool `json:"supported"`
	// Vision: a vision projector (mmproj*.gguf) sits beside it.
	Vision bool `json:"vision"`
	// Managed: the dependency-managed default model.
	Managed bool `json:"managed"`
}

// splitShard matches the second and later files of a split GGUF
// (model-00002-of-00003.gguf): the first file stands for the model.
var splitShard = regexp.MustCompile(`-0*([2-9]|[1-9]\d+)-of-\d+\.gguf$`)

// textModelDirs are the folders loki-scribe searches, as far as the server
// can tell: the managed default model's folder, $LOKI_MODELS and LM Studio's.
func textModelDirs() []string {
	var dirs []string
	dirs = append(dirs, models.ModelDir(scribeDefaultModel))
	if v := os.Getenv("LOKI_MODELS"); v != "" {
		dirs = append(dirs, filepath.SplitList(v)...)
	}
	if h, err := os.UserHomeDir(); err == nil {
		dirs = append(dirs, filepath.Join(h, ".lmstudio", "models"), filepath.Join(h, ".cache", "lm-studio", "models"))
	}
	return dirs
}

// LocalTextModels lists the GGUF models in the text-model folders (up to
// three levels deep, which covers LM Studio's publisher/repo/file layout).
func LocalTextModels() []TextModel {
	var out []TextModel
	seen := map[string]bool{}
	managedDir := filepath.Clean(models.ModelDir(scribeDefaultModel))
	for _, root := range textModelDirs() {
		root = filepath.Clean(root)
		_ = filepath.WalkDir(root, func(p string, d os.DirEntry, err error) error {
			if err != nil {
				return nil
			}
			rel, _ := filepath.Rel(root, p)
			if d.IsDir() {
				if rel != "." && strings.Count(filepath.ToSlash(rel), "/") >= 2 {
					return filepath.SkipDir
				}
				return nil
			}
			name := strings.ToLower(d.Name())
			if !strings.HasSuffix(name, ".gguf") || strings.Contains(name, "mmproj") || seen[p] || splitShard.MatchString(name) {
				return nil
			}
			seen[p] = true
			st, err := d.Info()
			if err != nil {
				return nil
			}
			arch, _ := ggufArch(p)
			m := TextModel{Path: p, Name: d.Name(), SizeBytes: st.Size(), Arch: arch, Supported: arch == "qwen35", Managed: filepath.Dir(p) == managedDir}
			parts := strings.Split(filepath.ToSlash(rel), "/")
			switch {
			case m.Managed:
				m.Spec = ""
			case len(parts) == 3:
				m.Spec = strings.Join(parts, "/")
			default:
				m.Spec = p
			}
			if mm, _ := filepath.Glob(filepath.Join(filepath.Dir(p), "*mmproj*.gguf")); len(mm) > 0 {
				m.Vision = true
			}
			out = append(out, m)
			return nil
		})
	}
	sort.SliceStable(out, func(a, b int) bool {
		if out[a].Supported != out[b].Supported {
			return out[a].Supported
		}
		return strings.ToLower(out[a].Spec+out[a].Path) < strings.ToLower(out[b].Spec+out[b].Path)
	})
	return out
}

// ggufArch reads general.architecture from a GGUF header (it is the first
// key in files written by llama.cpp's tools; arrays before it are skipped).
func ggufArch(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	r := &ggufReader{r: f}
	var magic [4]byte
	if _, err := io.ReadFull(f, magic[:]); err != nil || string(magic[:]) != "GGUF" {
		return "", errors.New("not a GGUF file")
	}
	r.u32()
	r.u64()
	nkv := r.u64()
	for i := uint64(0); i < nkv && i < 64 && r.err == nil; i++ {
		key := r.str()
		typ := r.u32()
		if key == "general.architecture" && typ == 8 {
			return r.str(), r.err
		}
		r.skip(typ)
	}
	if r.err != nil {
		return "", r.err
	}
	return "", errors.New("no general.architecture")
}

type ggufReader struct {
	r   io.Reader
	err error
}

func (g *ggufReader) read(n int) []byte {
	if g.err != nil || n < 0 || n > 1<<20 {
		if g.err == nil {
			g.err = errors.New("bad GGUF header")
		}
		return nil
	}
	b := make([]byte, n)
	_, g.err = io.ReadFull(g.r, b)
	return b
}
func (g *ggufReader) u32() uint32 {
	b := g.read(4)
	if b == nil {
		return 0
	}
	return binary.LittleEndian.Uint32(b)
}
func (g *ggufReader) u64() uint64 {
	b := g.read(8)
	if b == nil {
		return 0
	}
	return binary.LittleEndian.Uint64(b)
}
func (g *ggufReader) str() string { return string(g.read(int(g.u64()))) }

// skip passes over one value of GGUF type typ.
func (g *ggufReader) skip(typ uint32) {
	sizes := map[uint32]int{0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
	switch typ {
	case 8:
		g.read(int(g.u64()))
	case 9:
		at := g.u32()
		n := g.u64()
		for i := uint64(0); i < n && g.err == nil; i++ {
			g.skip(at)
		}
	default:
		if n, ok := sizes[typ]; ok {
			g.read(n)
		} else if g.err == nil {
			g.err = errors.New("bad GGUF value type")
		}
	}
}
