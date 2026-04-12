package tasks

import (
	"testing"

	"github.com/stevecastle/shrike/jobqueue"
)

func TestParseOptionsDefaults(t *testing.T) {
	options := []TaskOption{
		{Name: "name", Type: "string", Default: "hello"},
		{Name: "verbose", Type: "bool"},
		{Name: "count", Type: "number", Default: 5.0},
		{Name: "empty", Type: "string"},
	}
	j := &jobqueue.Job{Arguments: []string{}}
	result := ParseOptions(j, options)

	if result["name"] != "hello" {
		t.Errorf("name: got %v, want %q", result["name"], "hello")
	}
	if result["verbose"] != false {
		t.Errorf("verbose: got %v, want false", result["verbose"])
	}
	if result["count"] != 5.0 {
		t.Errorf("count: got %v, want 5.0", result["count"])
	}
	if result["empty"] != "" {
		t.Errorf("empty: got %v, want empty string", result["empty"])
	}
}

func TestParseOptionsBoolFlag(t *testing.T) {
	options := []TaskOption{
		{Name: "verbose", Type: "bool"},
		{Name: "debug", Type: "bool"},
	}
	j := &jobqueue.Job{Arguments: []string{"--verbose"}}
	result := ParseOptions(j, options)

	if result["verbose"] != true {
		t.Errorf("verbose: got %v, want true", result["verbose"])
	}
	if result["debug"] != false {
		t.Errorf("debug: got %v, want false", result["debug"])
	}
}

func TestParseOptionsBoolEquals(t *testing.T) {
	options := []TaskOption{
		{Name: "verbose", Type: "bool"},
		{Name: "debug", Type: "bool"},
	}
	j := &jobqueue.Job{Arguments: []string{"--verbose=true", "--debug=false"}}
	result := ParseOptions(j, options)

	if result["verbose"] != true {
		t.Errorf("verbose: got %v, want true", result["verbose"])
	}
	if result["debug"] != false {
		t.Errorf("debug: got %v, want false", result["debug"])
	}
}

func TestParseOptionsKeyValue(t *testing.T) {
	options := []TaskOption{
		{Name: "target", Type: "string"},
		{Name: "model", Type: "string", Default: "default-model"},
	}
	j := &jobqueue.Job{Arguments: []string{"--target", "/tmp/output", "--model", "gpt-4"}}
	result := ParseOptions(j, options)

	if result["target"] != "/tmp/output" {
		t.Errorf("target: got %v, want /tmp/output", result["target"])
	}
	if result["model"] != "gpt-4" {
		t.Errorf("model: got %v, want gpt-4", result["model"])
	}
}

func TestParseOptionsEqualsString(t *testing.T) {
	options := []TaskOption{
		{Name: "format", Type: "enum", Choices: []string{"mp4", "webm"}},
	}
	j := &jobqueue.Job{Arguments: []string{"--format=webm"}}
	result := ParseOptions(j, options)

	if result["format"] != "webm" {
		t.Errorf("format: got %v, want webm", result["format"])
	}
}

func TestParseOptionsNumber(t *testing.T) {
	options := []TaskOption{
		{Name: "width", Type: "number", Default: 1280.0},
	}

	// With key-value pair
	j := &jobqueue.Job{Arguments: []string{"--width", "640"}}
	result := ParseOptions(j, options)
	if result["width"] != 640.0 {
		t.Errorf("width: got %v, want 640.0", result["width"])
	}

	// With equals
	j2 := &jobqueue.Job{Arguments: []string{"--width=800"}}
	result2 := ParseOptions(j2, options)
	if result2["width"] != 800.0 {
		t.Errorf("width (equals): got %v, want 800.0", result2["width"])
	}
}

func TestParseOptionsUnknownFlagsIgnored(t *testing.T) {
	options := []TaskOption{
		{Name: "known", Type: "string"},
	}
	j := &jobqueue.Job{Arguments: []string{"--unknown", "val", "--known", "yes"}}
	result := ParseOptions(j, options)

	if result["known"] != "yes" {
		t.Errorf("known: got %v, want yes", result["known"])
	}
	if _, exists := result["unknown"]; exists {
		t.Error("unknown flag should not be in result")
	}
}

func TestParseOptionsEmptyArgs(t *testing.T) {
	options := []TaskOption{
		{Name: "mode", Type: "enum", Choices: []string{"a", "b"}, Default: "a"},
		{Name: "flag", Type: "bool"},
		{Name: "num", Type: "number"},
	}
	j := &jobqueue.Job{Arguments: nil}
	result := ParseOptions(j, options)

	if result["mode"] != "a" {
		t.Errorf("mode: got %v, want a", result["mode"])
	}
	if result["flag"] != false {
		t.Errorf("flag: got %v, want false", result["flag"])
	}
	if result["num"] != 0.0 {
		t.Errorf("num: got %v, want 0.0", result["num"])
	}
}

func TestParseOptionsMixed(t *testing.T) {
	options := []TaskOption{
		{Name: "target", Type: "string", Required: true},
		{Name: "recursive", Type: "bool"},
		{Name: "width", Type: "number", Default: 100.0},
		{Name: "format", Type: "enum", Choices: []string{"jpg", "png"}, Default: "jpg"},
	}
	j := &jobqueue.Job{Arguments: []string{
		"--recursive", "--target", "/out", "--width=200", "--format", "png",
	}}
	result := ParseOptions(j, options)

	if result["recursive"] != true {
		t.Errorf("recursive: got %v, want true", result["recursive"])
	}
	if result["target"] != "/out" {
		t.Errorf("target: got %v, want /out", result["target"])
	}
	if result["width"] != 200.0 {
		t.Errorf("width: got %v, want 200.0", result["width"])
	}
	if result["format"] != "png" {
		t.Errorf("format: got %v, want png", result["format"])
	}
}
