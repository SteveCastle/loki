// Package transcribe hides transcription implementation details behind a
// provider interface. The rest of the server talks to this facade only;
// whether speech-to-text runs through a local CLI (Faster-Whisper today), an
// HTTP service, or some future engine is a provider concern. New providers
// register themselves in an init() the same way tasks do.
package transcribe

import (
	"context"
	"fmt"
	"sort"
	"strings"

	"github.com/stevecastle/shrike/appconfig"
)

// Request describes one transcription job.
type Request struct {
	// MediaPath is the audio/video file to transcribe.
	MediaPath string
	// Model is a provider-specific model id ("" = provider default).
	Model string
	// Language is an ISO language hint ("" = auto-detect).
	Language string
	// VADFilter trims non-speech before transcribing, which reduces
	// hallucinations during silent stretches.
	VADFilter bool
	// Log receives human-readable progress lines. May be nil.
	Log func(string)
}

// Result is a finished transcription.
type Result struct {
	// Text is the transcript content (VTT).
	Text string
	// TranscriptPath is the artifact written next to the media file, if the
	// provider produces one ("" otherwise).
	TranscriptPath string
}

// ModelChoice is one selectable model for a provider (config UI dropdown).
type ModelChoice struct {
	ID          string
	DisplayName string
}

// Provider is one transcription implementation.
type Provider interface {
	ID() string
	DisplayName() string
	// Models lists the selectable models, best-default first.
	Models() []ModelChoice
	// DefaultModel is used when the config names no model.
	DefaultModel() string
	// Available returns nil when the provider can run right now, or an
	// actionable error (e.g. "binary not installed — download it from the
	// Dependencies page").
	Available() error
	Transcribe(ctx context.Context, req Request) (Result, error)
}

// DefaultProviderID is used when the config names no provider.
const DefaultProviderID = "whisper-cli"

var registry = map[string]Provider{}

// Register adds a provider. Called from provider init() functions.
func Register(p Provider) { registry[p.ID()] = p }

// Providers returns all registered providers, default first, rest sorted by id.
func Providers() []Provider {
	out := make([]Provider, 0, len(registry))
	for _, p := range registry {
		out = append(out, p)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].ID() == DefaultProviderID {
			return true
		}
		if out[j].ID() == DefaultProviderID {
			return false
		}
		return out[i].ID() < out[j].ID()
	})
	return out
}

// Lookup returns the provider with the given id.
func Lookup(id string) (Provider, bool) {
	p, ok := registry[id]
	return p, ok
}

// Active resolves the configured provider, falling back to the default when
// the config is empty or names an unknown provider.
func Active() (Provider, error) {
	id := strings.TrimSpace(appconfig.Get().TranscriptionProvider)
	if id == "" {
		id = DefaultProviderID
	}
	p, ok := registry[id]
	if !ok {
		if p, ok = registry[DefaultProviderID]; !ok {
			return nil, fmt.Errorf("transcribe: no provider registered for %q and no default available", id)
		}
	}
	return p, nil
}

// FromConfig builds a Request for mediaPath from the persisted transcription
// settings and resolves the active provider.
func FromConfig(mediaPath string, logFn func(string)) (Provider, Request, error) {
	p, err := Active()
	if err != nil {
		return nil, Request{}, err
	}
	cfg := appconfig.Get()
	model := strings.TrimSpace(cfg.TranscriptionModel)
	if model == "" {
		model = p.DefaultModel()
	}
	return p, Request{
		MediaPath: mediaPath,
		Model:     model,
		Language:  strings.TrimSpace(cfg.TranscriptionLanguage),
		VADFilter: cfg.TranscriptionVADFilter,
		Log:       logFn,
	}, nil
}
