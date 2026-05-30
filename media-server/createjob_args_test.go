package main

import (
	"reflect"
	"strings"
	"testing"
)

func TestAppendFieldArgs_NilFields(t *testing.T) {
	args := []string{"--type", "description"}
	got := appendFieldArgs(args, nil)
	want := []string{"--type", "description"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestAppendFieldArgs_EmptyFields(t *testing.T) {
	args := []string{"--type", "description"}
	got := appendFieldArgs(args, map[string]string{})
	want := []string{"--type", "description"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestAppendFieldArgs_SingleField(t *testing.T) {
	args := []string{"--type", "description"}
	got := appendFieldArgs(args, map[string]string{"prompt": "describe this"})
	want := []string{"--type", "description", "--prompt", "describe this"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestAppendFieldArgs_SkipsEmptyValue(t *testing.T) {
	args := []string{"--type", "description"}
	got := appendFieldArgs(args, map[string]string{"prompt": ""})
	want := []string{"--type", "description"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestAppendFieldArgs_SkipsEmptyKey(t *testing.T) {
	args := []string{"--type", "description"}
	got := appendFieldArgs(args, map[string]string{"": "ignored"})
	want := []string{"--type", "description"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestAppendFieldArgs_PreservesArbitraryText(t *testing.T) {
	// Quotes, newlines, equals signs must pass through unchanged because
	// the helper appends them as separate slice elements (not a shell string).
	args := []string{"--type", "description"}
	value := `weird "quoted" text with
newlines and = signs`
	got := appendFieldArgs(args, map[string]string{"prompt": value})
	want := []string{"--type", "description", "--prompt", value}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestAppendFieldArgs_MultipleFields(t *testing.T) {
	args := []string{"--type", "description"}
	got := appendFieldArgs(args, map[string]string{
		"prompt": "describe this",
		"model":  "llava",
	})
	if len(got) != 6 {
		t.Fatalf("len = %d, want 6; got %v", len(got), got)
	}
	// First two elements come from the input slice — verify they're intact.
	if got[0] != "--type" || got[1] != "description" {
		t.Errorf("input prefix mutated: %v", got[:2])
	}
	// Remaining four elements are two appended pairs, in undefined order.
	// Read them as a map and check by key.
	pairs := map[string]string{}
	for i := 2; i < len(got); i += 2 {
		if !strings.HasPrefix(got[i], "--") {
			t.Errorf("got[%d] = %q, want a --flag", i, got[i])
		}
		pairs[strings.TrimPrefix(got[i], "--")] = got[i+1]
	}
	if pairs["prompt"] != "describe this" {
		t.Errorf("prompt = %q, want %q", pairs["prompt"], "describe this")
	}
	if pairs["model"] != "llava" {
		t.Errorf("model = %q, want %q", pairs["model"], "llava")
	}
}
