package main

import (
	"reflect"
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
