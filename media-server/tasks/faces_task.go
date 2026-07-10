package tasks

import (
	"encoding/base64"
	"encoding/json"
	"fmt"

	"github.com/stevecastle/shrike/appconfig"
	"github.com/stevecastle/shrike/embedvec"
	"github.com/stevecastle/shrike/media"
)

// FaceProviderFromConfig returns the configured face-task execution provider.
func FaceProviderFromConfig() string {
	return normalizeProvider(appconfig.Get().FaceProvider)
}

// ResolveFaceResources mirrors ResolveEmbedResources for the face task,
// reading the Face* config fields.
func ResolveFaceResources() (workers, threads int) {
	cfg := appconfig.Get()
	return resolveResources(cfg.FacePerformance, cfg.FaceWorkers, cfg.FaceThreadsPerWorker, FaceProviderFromConfig())
}

// faceLineJSON mirrors cmd/embed's --faces output: per-image dimensions plus
// relative-coordinate faces with base64 vectors.
type faceLineJSON struct {
	ImageW int `json:"image_w"`
	ImageH int `json:"image_h"`
	Faces  []struct {
		X         float64       `json:"x"`
		Y         float64       `json:"y"`
		W         float64       `json:"w"`
		H         float64       `json:"h"`
		Score     float64       `json:"score"`
		Landmarks [5][2]float64 `json:"landmarks"`
		Vec       string        `json:"vec"`
	} `json:"faces"`
}

// parseFacesLine decodes one worker output line into storable faces.
func parseFacesLine(line string) ([]media.NewFace, error) {
	var parsed faceLineJSON
	if err := json.Unmarshal([]byte(line), &parsed); err != nil {
		return nil, fmt.Errorf("decode faces JSON: %w", err)
	}
	out := make([]media.NewFace, 0, len(parsed.Faces))
	for _, f := range parsed.Faces {
		raw, err := base64.StdEncoding.DecodeString(f.Vec)
		if err != nil {
			return nil, fmt.Errorf("decode face vector: %w", err)
		}
		vec, err := embedvec.Decode(raw)
		if err != nil {
			return nil, err
		}
		out = append(out, media.NewFace{
			X: f.X, Y: f.Y, W: f.W, H: f.H,
			Score: f.Score,
			Vec:   vec,
		})
	}
	return out, nil
}

// faceModelArgs assembles the model-driven `embed --faces` flags shared by
// serve and one-shot invocations. When m is a fused model (Secondary set),
// the primary dim is m.Dim minus the secondary's — m.Dim is the concat total.
func faceModelArgs(m FaceModel, detectorPath, recognizerPath, secondaryPath string) []string {
	primaryDim := m.Dim
	if m.Secondary != nil {
		primaryDim = m.Dim - m.Secondary.Dim
	}
	args := []string{
		"--faces",
		"--detect-model=" + detectorPath,
		"--model=" + recognizerPath,
		fmt.Sprintf("--dim=%d", primaryDim),
		"--face-input=" + m.InputName,
		"--face-output=" + m.OutputName,
		fmt.Sprintf("--face-mean=%g,%g,%g", m.Mean[0], m.Mean[1], m.Mean[2]),
		fmt.Sprintf("--face-std=%g,%g,%g", m.Std[0], m.Std[1], m.Std[2]),
		"--face-color=" + m.ColorOrder,
	}
	if m.InputSize > 0 {
		args = append(args, fmt.Sprintf("--face-size=%d", m.InputSize))
	}
	if m.DetectorKind != "" {
		args = append(args, "--detect-kind="+m.DetectorKind)
	}
	if m.Align != "" {
		args = append(args, "--align="+m.Align)
	}
	if m.CropExpand > 0 {
		args = append(args, fmt.Sprintf("--crop-expand=%g", m.CropExpand))
	}
	if m.Weight > 0 {
		args = append(args, fmt.Sprintf("--face-weight=%g", m.Weight))
	}
	if m.Secondary != nil && secondaryPath != "" {
		s := m.Secondary
		args = append(args,
			"--face2-model="+secondaryPath,
			fmt.Sprintf("--face2-dim=%d", s.Dim),
			"--face2-input="+s.InputName,
			"--face2-output="+s.OutputName,
			fmt.Sprintf("--face2-size=%d", s.InputSize),
			fmt.Sprintf("--face2-mean=%g,%g,%g", s.Mean[0], s.Mean[1], s.Mean[2]),
			fmt.Sprintf("--face2-std=%g,%g,%g", s.Std[0], s.Std[1], s.Std[2]),
			"--face2-color="+s.ColorOrder,
		)
		if m.SecondaryWeight > 0 {
			args = append(args, fmt.Sprintf("--face2-weight=%g", m.SecondaryWeight))
		}
	}
	return args
}

// buildFacesServeArgs assembles the `embed.exe --faces --serve` arguments for
// a recognizer + provider + thread config.
func buildFacesServeArgs(detectorPath, recognizerPath, secondaryPath, ortLib string, m FaceModel, provider string, threads int) []string {
	args := append(faceModelArgs(m, detectorPath, recognizerPath, secondaryPath),
		"--serve",
		"--provider="+provider,
	)
	if threads > 0 {
		args = append(args, fmt.Sprintf("--threads=%d", threads))
	}
	if ortLib != "" {
		args = append(args, "--ort="+ortLib)
	}
	return args
}
