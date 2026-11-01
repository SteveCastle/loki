package tasks

import (
	"context"
	"testing"
	"time"
)

func stubDreamModels(t *testing.T, body string, err error) {
	t.Helper()
	old := listDreamModels
	listDreamModels = func(context.Context) ([]byte, string, error) { return []byte(body), "this machine", err }
	dreamModelsMu.Lock()
	dreamModelsCache = nil
	dreamModelsMu.Unlock()
	t.Cleanup(func() {
		listDreamModels = old
		dreamModelsMu.Lock()
		dreamModelsCache = nil
		dreamModelsMu.Unlock()
	})
}

const dreamListJSON = `{"models":[
 {"file":"kromoss_int8_convrot.safetensors","label":"kromoss_int8_convrot","architecture":"moss","architecture_name":"MOSS-Image","distilled":false,"steps":50,"cfg":3,"turbo":true,"loras":true,"local":true,"format":"int8","size_gb":5.4,"text_encoder":{"file":"qwen3.5_9b_bf16.safetensors","present":true},"vae":{"file":"flux2-vae_small.safetensors","present":true}},
 {"file":"krea2_turbo_fp8_scaled.safetensors","label":"krea2_turbo_fp8_scaled","architecture":"krea2","architecture_name":"Krea 2","distilled":true,"steps":8,"cfg":1,"turbo":false,"loras":false,"local":true,"text_encoder":{"file":"qwen3vl_4b_bf16.safetensors","present":false,"download_gb":8.9,"from":"Comfy-Org/Krea-2"}},
 {"file":"","label":"broken"}
],"default":"kromoss_int8_convrot.safetensors"}`

func TestDreamModelsParsesAndCaches(t *testing.T) {
	stubDreamModels(t, dreamListJSON, nil)
	l := DreamModels(context.Background(), false)
	if len(l.Models) != 2 || l.Default != "kromoss_int8_convrot.safetensors" || l.Note != "" {
		t.Fatalf("list: %+v", l)
	}
	k := l.Models[1]
	if k.Architecture != "krea2" || !k.Distilled || k.Steps != 8 || k.TextEncoder == nil || k.TextEncoder.Present || k.TextEncoder.DownloadGB != 8.9 {
		t.Errorf("krea2 entry: %+v", k)
	}
	// cached: a second call does not ask the engine
	listDreamModels = func(context.Context) ([]byte, string, error) { t.Fatal("asked again"); return nil, "", nil }
	if l2 := DreamModels(context.Background(), false); len(l2.Models) != 2 {
		t.Errorf("cached list: %+v", l2)
	}
	if dreamModelArch("KREA2_TURBO_FP8_SCALED") != "krea2" || dreamModelArch("krea2_turbo_fp8_scaled.safetensors") != "krea2" || dreamModelArch("other") != "" {
		t.Error("dreamModelArch")
	}
}

func TestDreamModelsWithoutEngine(t *testing.T) {
	stubDreamModels(t, "", errDreamNotInstalled)
	l := DreamModels(context.Background(), false)
	if len(l.Models) != 0 || l.Note == "" || l.Models == nil {
		t.Fatalf("list: %+v", l)
	}
	// a failure is not cached
	dreamModelsMu.Lock()
	cached := dreamModelsCache
	dreamModelsMu.Unlock()
	if cached != nil {
		t.Error("failed answer cached")
	}
	if time.Since(l.Fetched) > time.Minute {
		t.Error("fetched time")
	}
}

func TestDreamSpecFollowsTheArchitecture(t *testing.T) {
	stubDreamModels(t, dreamListJSON, nil)
	DreamModels(context.Background(), false)
	if s := dreamSpecFor("krea2_turbo_fp8_scaled.safetensors"); s.Model != "" || len(s.modelIDs()) != 0 {
		t.Errorf("krea2 spec %+v: no Kromoss download", s)
	}
	for _, m := range []string{"", "kromoss_int8_convrot.safetensors", "unknown.safetensors"} {
		if s := dreamSpecFor(m); s.Model != dreamTool.Model {
			t.Errorf("%q: spec %+v", m, s)
		}
	}
}

func TestValidDreamModelName(t *testing.T) {
	for _, ok := range []string{"krea2_turbo_fp8_scaled.safetensors", "anima-turbo-v1.1", "flux-2-klein-4b"} {
		if !validDreamModelName(ok) {
			t.Errorf("%q rejected", ok)
		}
	}
	for _, bad := range []string{"", "../x", `C:\models\x.safetensors`, "/etc/x", "a/b", "..", ".hidden"} {
		if validDreamModelName(bad) {
			t.Errorf("%q accepted", bad)
		}
	}
}

func TestDreamParamsModel(t *testing.T) {
	p := dreamParamsFromOptions(map[string]any{"prompt": "x", "model": " krea2_turbo_fp8_scaled "})
	if p.Model != "krea2_turbo_fp8_scaled" {
		t.Errorf("model %q", p.Model)
	}
}
