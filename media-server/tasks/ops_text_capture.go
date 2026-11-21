package tasks

import (
	"context"
	"fmt"
	"strings"

	"github.com/stevecastle/shrike/appconfig"
)

// textCaptureOp is the Text capture item op: an exact transcription of the
// text in an image (comics as a script in narrative order), optionally
// translated, stored in the media transcript column — for images that column
// holds this plain text, for video/audio it holds the speech transcript.
const textCaptureOp = "capture-text"

// textCaptureMaxTokens: a busy comic page runs to a few hundred lines.
const textCaptureMaxTokens = 3072

// textCaptureLanguage normalizes a language choice: "" means as written.
func textCaptureLanguage(v string) string {
	v = strings.TrimSpace(v)
	switch strings.ToLower(v) {
	case "", "original", "none", "as written", "as-written", "no", "off":
		return ""
	}
	return v
}

// textCapturePrompt is the instruction for one run: the custom prompt or the
// configured one, plus the translation instruction when a language is set.
func textCapturePrompt(custom, language string) string {
	p := strings.TrimSpace(custom)
	if p == "" {
		p = strings.TrimSpace(appconfig.Get().TextCapturePrompt)
	}
	if p == "" {
		p = appconfig.DefaultTextCapturePrompt
	}
	if lang := textCaptureLanguage(language); lang != "" {
		p += fmt.Sprintf("\n\nWrite your answer in %[1]s: translate every piece of text into %[1]s, whatever language it is written in, keeping its meaning and tone and the structure described above (speaker names, panel markers and sound effects too). Text already in %[1]s stays exactly as written. Give only the %[1]s version.", lang)
	}
	return p
}

func registerTextCaptureItemOp() {
	RegisterItemOp(ItemOp{
		ID:   textCaptureOp,
		Name: "Text Capture (transcribe and translate)",
		Options: []TaskOption{
			{Name: "language", Label: "Translate To", Type: "enum", Choices: appconfig.TextCaptureLanguages, Description: "Write the text in this language whatever it is written in (unset: the configured default; Original: as written)"},
			{Name: "prompt", Label: "Custom Prompt", Type: "string", Description: "Override the configured text capture prompt for this run"},
			{Name: "model", Label: "Text Model", Type: "string", Description: "Override the configured text model"},
		},
		Concurrency: scribeConcurrency,
		Applies:     extAppliesFn(imageExts...),
		Prepare:     prepareTextCaptureOp,
	})
}

func prepareTextCaptureOp(run *ItemRun) (*ItemProcessor, error) {
	q, jobID := run.Queue, run.Job.ID
	db := q.Db
	model, _ := run.Opts["model"].(string)
	custom, _ := run.Opts["prompt"].(string)
	language, ok := run.Opts["language"].(string)
	if !ok || strings.TrimSpace(language) == "" {
		language = appconfig.Get().TextCaptureLanguage
	}
	if lang := textCaptureLanguage(language); lang != "" {
		q.PushJobStdout(jobID, "Text capture: translating into "+lang)
	}
	prompt := textCapturePrompt(custom, language)
	return &ItemProcessor{
		SkipExisting: func(path string) (bool, error) { return hasExistingMetadata(db, path, "transcript") },
		Process: func(ctx context.Context, path, localPath string) (*ItemCommit, error) {
			text, err := imageText(ctx, q, jobID, localPath, textRequest{Prompt: prompt, Model: model, MaxTokens: textCaptureMaxTokens, Greedy: true})
			if err != nil {
				return nil, err
			}
			text = strings.TrimSpace(text)
			detail := "text captured"
			if text == "" {
				detail = "no text in the image"
			}
			return &ItemCommit{
				Commit: func() error {
					if err := updateMediaMetadata(db, path, "transcript", text); err != nil {
						return err
					}
					notifyProgress(ProgressTranscript, 1)
					return nil
				},
				Detail: detail,
			}, nil
		},
	}, nil
}
