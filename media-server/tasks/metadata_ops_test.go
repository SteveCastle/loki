package tasks

import (
	"testing"

	"github.com/stevecastle/shrike/appconfig"
)

func TestResolveDescribePromptUsesCustomWhenProvided(t *testing.T) {
	got := resolveDescribePrompt("custom override")
	if got != "custom override" {
		t.Errorf("got %q, want %q", got, "custom override")
	}
}

func TestResolveDescribePromptTrimsWhitespace(t *testing.T) {
	got := resolveDescribePrompt("   custom   ")
	if got != "custom" {
		t.Errorf("got %q, want %q", got, "custom")
	}
}

func TestResolveDescribePromptFallsBackToConfigWhenEmpty(t *testing.T) {
	old := appconfig.Get()
	cfg := old
	cfg.DescribePrompt = "DEFAULT-FROM-CONFIG"
	appconfig.Set(cfg)
	t.Cleanup(func() { appconfig.Set(old) })

	got := resolveDescribePrompt("")
	if got != "DEFAULT-FROM-CONFIG" {
		t.Errorf("got %q, want %q", got, "DEFAULT-FROM-CONFIG")
	}
}

func TestResolveDescribePromptFallsBackOnAllWhitespace(t *testing.T) {
	old := appconfig.Get()
	cfg := old
	cfg.DescribePrompt = "DEFAULT-FROM-CONFIG"
	appconfig.Set(cfg)
	t.Cleanup(func() { appconfig.Set(old) })

	got := resolveDescribePrompt("   \n\t  ")
	if got != "DEFAULT-FROM-CONFIG" {
		t.Errorf("got %q, want %q", got, "DEFAULT-FROM-CONFIG")
	}
}
