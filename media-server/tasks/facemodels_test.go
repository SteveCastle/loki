package tasks

import (
	"testing"

	"github.com/stevecastle/shrike/appconfig"
)

func setFaceConfig(t *testing.T, mutate func(*appconfig.Config)) {
	t.Helper()
	prev := appconfig.Get()
	t.Cleanup(func() { appconfig.Set(prev) })
	cfg := prev
	mutate(&cfg)
	appconfig.Set(cfg)
}

func TestActiveFaceModelDefaultsToSFace(t *testing.T) {
	setFaceConfig(t, func(c *appconfig.Config) { c.FaceModel = "" })
	m := ActiveFaceModel()
	if m.ID != "sface" || m.Dim != 128 || m.BYO {
		t.Fatalf("default model = %+v, want built-in sface/128", m)
	}
	if m.InputName != "data" || m.OutputName != "fc1" || m.ColorOrder != "BGR" {
		t.Fatalf("sface I/O defaults wrong: %+v", m)
	}
}

func TestActiveFaceModelUnknownFallsBack(t *testing.T) {
	setFaceConfig(t, func(c *appconfig.Config) { c.FaceModel = "no-such-model" })
	if m := ActiveFaceModel(); m.ID != "sface" {
		t.Fatalf("unknown model should fall back to sface, got %q", m.ID)
	}
}

func TestByoFaceModelResolution(t *testing.T) {
	setFaceConfig(t, func(c *appconfig.Config) {
		c.FaceModel = "arcface-glint360k"
		c.ByoFaceModels = []appconfig.ByoFaceModel{
			{
				ID:             "arcface-glint360k",
				ModelPath:      `C:\models\glintr100.onnx`,
				Dim:            512,
				InputName:      "input.1",
				OutputName:     "683",
				Mean:           []float64{127.5, 127.5, 127.5},
				Std:            []float64{127.5, 127.5, 127.5},
				ColorOrder:     "RGB",
				MatchThreshold: 0.42,
			},
		}
	})
	m := ActiveFaceModel()
	if !m.BYO || m.ID != "arcface-glint360k" || m.Dim != 512 {
		t.Fatalf("BYO model not resolved: %+v", m)
	}
	if m.Mean[0] != 127.5 || m.Std[2] != 127.5 || m.ColorOrder != "RGB" {
		t.Fatalf("BYO preprocessing not carried: %+v", m)
	}
	if m.InputName != "input.1" || m.OutputName != "683" {
		t.Fatalf("BYO tensor names not carried: %+v", m)
	}
	if m.MatchThreshold != 0.42 {
		t.Fatalf("BYO threshold = %v, want 0.42", m.MatchThreshold)
	}
}

func TestByoFaceModelDefaults(t *testing.T) {
	setFaceConfig(t, func(c *appconfig.Config) {
		c.ByoFaceModels = []appconfig.ByoFaceModel{
			{ID: "minimal", ModelPath: `C:\models\m.onnx`, Dim: 512},
		}
	})
	m, ok := FaceModelByID("minimal")
	if !ok {
		t.Fatal("minimal BYO entry not found")
	}
	if m.InputName != "data" || m.OutputName != "fc1" || m.ColorOrder != "BGR" {
		t.Fatalf("BYO defaults not filled: %+v", m)
	}
	if m.Mean != ([3]float32{0, 0, 0}) || m.Std != ([3]float32{1, 1, 1}) {
		t.Fatalf("BYO mean/std defaults wrong: %+v", m)
	}
	if m.MatchThreshold != defaultByoMatchThreshold {
		t.Fatalf("BYO threshold default = %v, want %v", m.MatchThreshold, defaultByoMatchThreshold)
	}
}

func TestByoFaceModelInvalidEntriesSkipped(t *testing.T) {
	setFaceConfig(t, func(c *appconfig.Config) {
		c.ByoFaceModels = []appconfig.ByoFaceModel{
			{ID: "", ModelPath: `C:\m.onnx`, Dim: 512},   // no ID
			{ID: "no-path", Dim: 512},                    // no path
			{ID: "no-dim", ModelPath: `C:\m.onnx`},       // no dim
			{ID: "sface", ModelPath: `C:\m.onnx`, Dim: 4}, // shadows built-in
		}
	})
	for _, id := range []string{"", "no-path", "no-dim"} {
		if _, ok := FaceModelByID(id); ok {
			t.Fatalf("invalid BYO entry %q resolved", id)
		}
	}
	// The built-in must win over a shadowing BYO entry.
	if m, ok := FaceModelByID("sface"); !ok || m.BYO || m.Dim != 128 {
		t.Fatalf("built-in sface shadowed: %+v", m)
	}
	// FaceModelList: only the built-in survives (all BYO entries invalid/shadowed).
	list := FaceModelList()
	if len(list) != 1 || list[0].ID != "sface" {
		t.Fatalf("list = %+v, want [sface]", list)
	}
}

func TestFacesServeArgs(t *testing.T) {
	m, _ := FaceModelByID("sface")
	args := buildFacesServeArgs(`C:\d\yunet.onnx`, `C:\d\sface.onnx`, `C:\ort.dll`, m, "cpu", 2)
	want := map[string]bool{
		"--faces": true, "--serve": true,
		`--detect-model=C:\d\yunet.onnx`: true,
		`--model=C:\d\sface.onnx`:        true,
		"--dim=128":                      true,
		"--face-input=data":              true,
		"--face-output=fc1":              true,
		"--face-mean=0,0,0":              true,
		"--face-std=1,1,1":               true,
		"--face-color=BGR":               true,
		"--provider=cpu":                 true,
		"--threads=2":                    true,
		`--ort=C:\ort.dll`:               true,
	}
	if len(args) != len(want) {
		t.Fatalf("args = %v (len %d), want %d entries", args, len(args), len(want))
	}
	for _, a := range args {
		if !want[a] {
			t.Fatalf("unexpected arg %q in %v", a, args)
		}
	}
}

func TestParseFacesLine(t *testing.T) {
	// vec = base64(embedvec.Encode([1,0])) — little-endian float32.
	line := `{"image_w":100,"image_h":50,"faces":[{"x":0.1,"y":0.2,"w":0.3,"h":0.4,"score":0.9,"landmarks":[[0,0],[0,0],[0,0],[0,0],[0,0]],"vec":"AACAPwAAAAA="}]}`
	faces, err := parseFacesLine(line)
	if err != nil {
		t.Fatal(err)
	}
	if len(faces) != 1 {
		t.Fatalf("got %d faces, want 1", len(faces))
	}
	f := faces[0]
	if f.X != 0.1 || f.H != 0.4 || f.Score != 0.9 {
		t.Fatalf("face fields: %+v", f)
	}
	if len(f.Vec) != 2 || f.Vec[0] != 1 || f.Vec[1] != 0 {
		t.Fatalf("vec = %v, want [1 0]", f.Vec)
	}

	if _, err := parseFacesLine(`{"faces":[{"vec":"!!!"}]}`); err == nil {
		t.Fatal("expected error for bad base64")
	}
	if _, err := parseFacesLine(`not json`); err == nil {
		t.Fatal("expected error for bad JSON")
	}

	// No faces is a valid, storable result (records the scan).
	faces, err = parseFacesLine(`{"image_w":10,"image_h":10,"faces":[]}`)
	if err != nil || len(faces) != 0 {
		t.Fatalf("empty faces: %v, %v", faces, err)
	}
}
