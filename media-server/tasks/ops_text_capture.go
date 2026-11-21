package tasks

import (
	"context"
	"fmt"
	"path/filepath"
	"strings"

	"github.com/stevecastle/shrike/appconfig"
)

// textCaptureOp is the Text capture item op: an exact transcription of the
// text in an image (comics as a script in narrative order), optionally
// translated, stored in the media transcript column — for images that column
// holds this plain text, for video/audio it holds the speech transcript.
const textCaptureOp = "capture-text"

// Transcription sampling: temperature 0.2 keeps the reading nearly
// deterministic without greedy decoding (which loops on Qwen3: a manga page
// repeated one line to the token limit), top-k 20 / top-p 0.8 are the
// engine's non-thinking defaults, and the DRY loop guard (NoRepeat) stops
// runaway repeats. Presence and repetition penalties stay OFF on purpose:
// they hit every token already written, and a script repeats PANEL, speaker
// names and punctuation on every line.
const textCaptureTemperature = 0.2

// textCaptureMaxTokensLimit bounds a per-job --max-tokens override.
const textCaptureMaxTokensLimit = 16384

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
			{Name: "max-tokens", Label: "Max Length", Type: "number", Description: "Most tokens to write for one image (0: the configured default); the stop for a model that never finishes"},
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
	maxTokens := int(optFloat(run.Opts, "max-tokens"))
	if maxTokens <= 0 {
		maxTokens = appconfig.Get().TextCaptureMaxTokens
	}
	if maxTokens <= 0 {
		maxTokens = appconfig.DefaultTextCaptureMaxTokens
	}
	maxTokens = min(maxTokens, textCaptureMaxTokensLimit)
	return &ItemProcessor{
		SkipExisting: func(path string) (bool, error) { return hasExistingMetadata(db, path, "transcript") },
		Process: func(ctx context.Context, path, localPath string) (*ItemCommit, error) {
			var finish string
			text, err := imageText(ctx, q, jobID, localPath, textRequest{
				Prompt: prompt, Model: model, MaxTokens: maxTokens,
				Temperature: textCaptureTemperature, NoRepeat: true,
				OnFinish: func(f string) { finish = f },
			})
			if err != nil {
				return nil, err
			}
			text = strings.TrimSpace(text)
			// Only an answer that ran into the limit is a loop; a finished
			// one keeps every repeated balloon it has.
			if finish == "length" {
				var looped bool
				if text, looped = trimRepeatedTail(text); looped {
					q.PushJobStdout(jobID, "  "+filepath.Base(path)+": the model got stuck repeating itself; kept one copy of the repeated part")
				}
			}
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

// trimRepeatedTail cuts a generation loop off the end of a text: when the
// text ends with one block (a run of lines, or of characters inside a line)
// repeated at least loopRepeats times back to back, only the first copy is
// kept. Legitimate repetition (a balloon of "HA HA HA", the same SFX twice)
// stays well under that.
func trimRepeatedTail(text string) (string, bool) {
	const loopRepeats = 4
	// whole lines: the block is the last k lines
	lines := strings.Split(text, "\n")
	for k := 1; k <= 12 && k*loopRepeats <= len(lines); k++ {
		block := strings.Join(lines[len(lines)-k:], "\n")
		if strings.TrimSpace(block) == "" {
			continue
		}
		n := 1
		for end := len(lines) - k; end-k >= 0 && strings.Join(lines[end-k:end], "\n") == block; end -= k {
			n++
		}
		if n >= loopRepeats {
			keep := len(lines) - (n-1)*k
			return strings.TrimSpace(strings.Join(lines[:keep], "\n")), true
		}
	}
	// characters: a phrase repeating inside one line ("ah ah ah ..." for pages)
	r := []rune(text)
	for p := 2; p <= 120 && p*loopRepeats*2 <= len(r); p++ {
		n := 1
		for end := len(r) - p; end-p >= 0 && string(r[end-p:end]) == string(r[end:end+p]); end -= p {
			n++
		}
		// a long run only: 8 copies and at least 80 characters of repetition
		if n >= loopRepeats*2 && n*p >= 80 {
			return strings.TrimSpace(string(r[:len(r)-(n-1)*p])), true
		}
	}
	return text, false
}
