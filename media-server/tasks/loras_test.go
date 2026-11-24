package tasks

import (
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// writeST writes a header-only safetensors file (zero-length tensors are enough for the scan).
func writeST(t *testing.T, path string, names []string, meta map[string]string) {
	t.Helper()
	h := map[string]any{}
	for _, n := range names {
		h[n] = map[string]any{"dtype": "BF16", "shape": []int{0}, "data_offsets": []int{0, 0}}
	}
	if meta != nil {
		h["__metadata__"] = meta
	}
	b, _ := json.Marshal(h)
	_ = os.MkdirAll(filepath.Dir(path), 0o755)
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	_ = binary.Write(f, binary.LittleEndian, uint64(len(b)))
	_, _ = f.Write(b)
}

func TestListLoras(t *testing.T) {
	root := t.TempDir()
	runs := filepath.Join(root, "runs")
	extra := filepath.Join(root, "models", "loras")
	old := loraSearchDirs
	loraSearchDirs = func() []string { return []string{runs, extra, filepath.Join(root, "missing")} }
	defer func() { loraSearchDirs = old }()

	moss := []string{"diffusion_model.layers.0.proj.q.lora_down.weight", "diffusion_model.layers.0.proj.q.lora_up.weight", "diffusion_model.layers.0.proj.q.alpha"}
	rec, _ := json.Marshal(map[string]any{"loki": 1, "tool": "loki-tune", "name": "zkx", "trigger": "zkx", "base_model": "kromoss_bf16.safetensors", "rank": 16, "step": 500, "steps": 500})
	_ = os.MkdirAll(filepath.Join(runs, "zkx"), 0o755)
	_ = os.WriteFile(filepath.Join(runs, "zkx", "run.json"), []byte("{}"), 0o644)
	writeST(t, filepath.Join(runs, "zkx", "zkx.safetensors"), moss, map[string]string{"loki": string(rec)})
	writeST(t, filepath.Join(runs, "zkx", "checkpoints", "step-000250", "zkx-step000250.safetensors"), moss, nil)
	writeST(t, filepath.Join(runs, "zkx", "checkpoints", "step-000250", "state.safetensors"), []string{"params", "adam_m"}, nil)
	// PEFT naming, ComfyUI prefix: still MOSS-Image
	writeST(t, filepath.Join(extra, "peft_style.safetensors"), []string{"transformer.layers.3.proj.ffn_down.lora_A.weight", "transformer.layers.3.proj.ffn_down.lora_B.weight"}, map[string]string{"ss_network_dim": "32"})
	// an SDXL LoRA: listed, no architecture loki-dream applies
	writeST(t, filepath.Join(extra, "sdxl.safetensors"), []string{"lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight"}, nil)
	// not a LoRA
	writeST(t, filepath.Join(extra, "model.safetensors"), []string{"layers.0.proj.q.weight"}, nil)

	got := ListLoras()
	if len(got.Dirs) != 2 {
		t.Errorf("dirs: %v", got.Dirs)
	}
	byFile := map[string]LoraInfo{}
	for _, l := range got.Loras {
		byFile[l.File] = l
	}
	if len(got.Loras) != 4 {
		t.Fatalf("loras: %+v", got.Loras)
	}
	z := byFile["zkx.safetensors"]
	if z.Arch != "moss" || z.Trigger != "zkx" || z.BaseModel != "kromoss_bf16.safetensors" || z.Step != 500 || z.Checkpoint || z.Run != "zkx" || z.Rank != 16 {
		t.Errorf("final: %+v", z)
	}
	c := byFile["zkx-step000250.safetensors"]
	if !c.Checkpoint || c.Run != "zkx" || c.Arch != "moss" {
		t.Errorf("checkpoint: %+v", c)
	}
	if p := byFile["peft_style.safetensors"]; p.Arch != "moss" || p.Rank != 32 || p.Name != "peft_style" {
		t.Errorf("peft: %+v", p)
	}
	if s := byFile["sdxl.safetensors"]; s.Arch != "" {
		t.Errorf("sdxl: %+v", s)
	}
	// a run's final LoRA comes right before its checkpoints
	for i, l := range got.Loras {
		if l.File == "zkx.safetensors" && (i+1 >= len(got.Loras) || got.Loras[i+1].File != "zkx-step000250.safetensors") {
			t.Errorf("order: %+v", got.Loras)
		}
	}
}

func TestListLoraRuns(t *testing.T) {
	data := t.TempDir()
	old := loraTrainingRoot
	loraTrainingRoot = func() string { return filepath.Join(data, "loras") }
	defer func() { loraTrainingRoot = old }()
	run := filepath.Join(data, "loras", "inkwash")
	_ = os.MkdirAll(run, 0o755)
	_ = os.WriteFile(filepath.Join(run, "run.json"), []byte(`{"name":"inkwash","kind":"style","trigger":"inkwash","model":"C:/m/kromoss_bf16.safetensors","rank":16,"resolution":1024,"lr":5e-05,"steps":1000,"items":[{},{},{}]}`), 0o644)
	for _, s := range []string{"000250", "000500"} {
		d := filepath.Join(run, "checkpoints", "step-"+s)
		writeST(t, filepath.Join(d, "state.safetensors"), []string{"params"}, nil)
		writeST(t, filepath.Join(d, "inkwash-step"+s+".safetensors"), []string{"diffusion_model.layers.0.proj.q.lora_down.weight"}, nil)
		_ = os.MkdirAll(filepath.Join(run, "samples"), 0o755)
		_ = os.WriteFile(filepath.Join(run, "samples", "step-"+s+"_1.png"), nil, 0o644)
		_ = os.WriteFile(filepath.Join(run, "samples", "step-"+s+"_0.png"), nil, 0o644)
	}
	// a checkpoint without optimizer state cannot be resumed
	_ = os.MkdirAll(filepath.Join(run, "checkpoints", "step-000600"), 0o755)
	_ = os.WriteFile(filepath.Join(run, "log.jsonl"), []byte("{\"step\":499,\"avg\":0.41}\n{\"step\":500,\"avg\":0.40}\n"), 0o644)
	runs := ListLoraRuns()
	if len(runs) != 1 {
		t.Fatalf("runs: %+v", runs)
	}
	r := runs[0]
	if r.Name != "inkwash" || r.Kind != "style" || r.Model != "kromoss_bf16.safetensors" || r.Steps != 1000 || r.Images != 3 || r.Step != 500 || r.Loss != 0.40 || r.Finished {
		t.Errorf("run: %+v", r)
	}
	if len(r.Checkpoints) != 2 || r.Checkpoints[1].Step != 500 || len(r.Checkpoints[1].Samples) != 2 || filepath.Base(r.Checkpoints[1].Lora) != "inkwash-step000500.safetensors" {
		t.Errorf("checkpoints: %+v", r.Checkpoints)
	}
}
