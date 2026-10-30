package tasks

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"

	"github.com/stevecastle/shrike/deps/status"
	"github.com/stevecastle/shrike/features"
	"github.com/stevecastle/shrike/platform"
)

// LocalEngine is one loki-* engine as installed on this machine: the
// executable (a PATH build or the managed download), its version, and the
// model dependencies it runs with. What the GPU Engines config tab lists
// under "This machine".
type LocalEngine struct {
	Engine  string        `json:"engine"`
	Tool    status.Item   `json:"tool"`
	Version string        `json:"version,omitempty"`
	Models  []status.Item `json:"models"`
}

// localEngineSpecs are the engines and their models, in display order.
func localEngineSpecs() []aiToolSpec {
	specs := []aiToolSpec{retouchTool, reshootSpec(reshootModelFast, "")}
	if features.Dream {
		specs = append(specs, dreamTool)
	}
	return append(specs, segSpec)
}

// LocalEngines reports every engine's local install from the dependency
// snapshot (never installs anything).
func LocalEngines() []LocalEngine {
	byID := map[string]status.Item{}
	for _, it := range status.Snapshot() {
		byID[it.ID] = it
	}
	var out []LocalEngine
	for _, spec := range localEngineSpecs() {
		tool, ok := byID[spec.Bin]
		if !ok {
			tool = status.Item{ID: spec.Bin, Name: spec.Bin, State: "missing"}
		}
		e := LocalEngine{Engine: spec.Bin, Tool: tool}
		if tool.State == "installed" && tool.Path != "" {
			e.Version = engineVersion(tool.Path)
		}
		for _, id := range spec.modelIDs() {
			if m, ok := byID[id]; ok {
				e.Models = append(e.Models, m)
			}
		}
		out = append(out, e)
	}
	return out
}

// engineVersion is the binary's --version line without the name, cached per
// file (a rebuild changes its size or time).
var engineVersions sync.Map // path|size|mtime -> string

func engineVersion(path string) string {
	st, err := os.Stat(path)
	if err != nil || st.IsDir() {
		return ""
	}
	key := fmt.Sprint(path, "|", st.Size(), "|", st.ModTime().UnixNano())
	if v, ok := engineVersions.Load(key); ok {
		return v.(string)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, path, "--version")
	platform.HideSubprocessWindow(cmd)
	b, err := cmd.Output()
	v := ""
	if err == nil {
		f := strings.Fields(strings.TrimSpace(string(b)))
		if len(f) > 0 {
			v = f[len(f)-1]
		}
	}
	engineVersions.Store(key, v)
	return v
}
