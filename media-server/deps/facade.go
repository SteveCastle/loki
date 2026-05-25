// Package deps re-exports thin helpers over the new bundled/optional/models
// packages. The legacy types (Dependency, registry, MetadataStore) will be
// removed in a separate task; for now this file coexists with them.
package deps

import (
	"fmt"

	"github.com/stevecastle/shrike/deps/bundled"
	"github.com/stevecastle/shrike/deps/models"
	"github.com/stevecastle/shrike/deps/optional"
)

// MustBundled returns the absolute path to the named bundled binary. Panics
// if missing — after bundled.VerifyAll runs at boot, callers can assume
// every entry is present.
func MustBundled(id string) string {
	p, err := bundled.Resolve(id)
	if err != nil {
		panic(fmt.Sprintf("deps: bundled %q unresolvable: %v", id, err))
	}
	return p
}

// BundledOrEmpty returns the path or "" if the binary is missing. Useful for
// callers that gracefully degrade (e.g. ffplay-on-macOS).
func BundledOrEmpty(id string) string {
	p, err := bundled.Resolve(id)
	if err != nil {
		return ""
	}
	return p
}

// ModelPath returns the absolute path to relPath inside the named model, or
// IsModelNotInstalled(err) → true if the model isn't present.
func ModelPath(id, relPath string) (string, error) {
	p, err := models.Path(id, relPath)
	if err != nil {
		return "", err
	}
	return p, nil
}

// IsModelNotInstalled forwards to models.IsNotInstalled.
func IsModelNotInstalled(err error) bool { return models.IsNotInstalled(err) }

// DetectOptional forwards to optional.Detect.
func DetectOptional(id string) (optional.Status, error) { return optional.Detect(id) }
