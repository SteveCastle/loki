package transcribe

import (
	"slices"
	"testing"

	"github.com/stevecastle/shrike/appconfig"
)

func withConfig(t *testing.T, c appconfig.Config) {
	t.Helper()
	orig := appconfig.Get()
	appconfig.Set(c)
	t.Cleanup(func() { appconfig.Set(orig) })
}

func TestWhisperCLIRegistered(t *testing.T) {
	p, ok := Lookup("whisper-cli")
	if !ok {
		t.Fatal("whisper-cli not registered")
	}
	if p.DefaultModel() != "large-v2" {
		t.Errorf("DefaultModel = %q; want large-v2", p.DefaultModel())
	}
	if len(p.Models()) == 0 {
		t.Error("Models() empty — the config UI dropdown would be blank")
	}
	if p.Models()[0].ID != p.DefaultModel() {
		t.Errorf("first model choice %q should be the default %q", p.Models()[0].ID, p.DefaultModel())
	}
}

func TestProvidersDefaultFirst(t *testing.T) {
	list := Providers()
	if len(list) == 0 {
		t.Fatal("no providers registered")
	}
	if list[0].ID() != DefaultProviderID {
		t.Errorf("Providers()[0] = %q; want default %q first", list[0].ID(), DefaultProviderID)
	}
}

func TestActiveFallsBackToDefault(t *testing.T) {
	withConfig(t, appconfig.Config{TranscriptionProvider: "no-such-provider"})
	p, err := Active()
	if err != nil {
		t.Fatalf("Active: %v", err)
	}
	if p.ID() != DefaultProviderID {
		t.Errorf("Active = %q; want fallback to %q", p.ID(), DefaultProviderID)
	}
}

func TestFromConfigBuildsRequest(t *testing.T) {
	withConfig(t, appconfig.Config{
		TranscriptionProvider:  "whisper-cli",
		TranscriptionModel:     "small",
		TranscriptionLanguage:  "ja",
		TranscriptionVADFilter: true,
	})
	p, req, err := FromConfig("/media/clip.mp4", nil)
	if err != nil {
		t.Fatalf("FromConfig: %v", err)
	}
	if p.ID() != "whisper-cli" {
		t.Errorf("provider = %q", p.ID())
	}
	if req.Model != "small" || req.Language != "ja" || !req.VADFilter || req.MediaPath != "/media/clip.mp4" {
		t.Errorf("request = %+v", req)
	}
}

func TestFromConfigEmptyModelUsesProviderDefault(t *testing.T) {
	withConfig(t, appconfig.Config{TranscriptionProvider: "whisper-cli"})
	p, req, err := FromConfig("x.mp4", nil)
	if err != nil {
		t.Fatalf("FromConfig: %v", err)
	}
	if req.Model != p.DefaultModel() {
		t.Errorf("Model = %q; want provider default %q", req.Model, p.DefaultModel())
	}
}

func TestWhisperBuildArgs(t *testing.T) {
	w := &whisperCLI{}

	args := w.buildArgs(Request{MediaPath: "a.mp4", Model: "medium", Language: "en", VADFilter: true})
	want := []string{"--beep_off", "--output_format=vtt", "--output_dir=source", "--model", "medium", "--vad_filter", "true", "--language", "en", "a.mp4"}
	if !slices.Equal(args, want) {
		t.Errorf("args = %v\nwant  %v", args, want)
	}

	// No VAD, auto-detect language, default model.
	args = w.buildArgs(Request{MediaPath: "b.mkv"})
	want = []string{"--beep_off", "--output_format=vtt", "--output_dir=source", "--model", "large-v2", "b.mkv"}
	if !slices.Equal(args, want) {
		t.Errorf("args = %v\nwant  %v", args, want)
	}
}
