package main

import (
	"encoding/json"
	"net/http"

	"github.com/stevecastle/shrike/appconfig"
)

// describePromptHandler returns the currently configured describe prompt as
// JSON. The React metadata UI calls this to render the default text as a
// placeholder inside the optional custom-prompt panel. Read-only and tiny;
// safe to call on every component mount.
func describePromptHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Use GET", http.StatusMethodNotAllowed)
		return
	}
	resp := struct {
		Prompt string `json:"prompt"`
	}{
		Prompt: appconfig.Get().DescribePrompt,
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}
