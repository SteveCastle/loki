package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func newTestApp() (*App, *bytes.Buffer, *bytes.Buffer) {
	out := &bytes.Buffer{}
	errOut := &bytes.Buffer{}
	return &App{Out: out, ErrOut: errOut}, out, errOut
}

func TestPrintJSON(t *testing.T) {
	a, out, _ := newTestApp()
	if code := a.PrintJSON(map[string]any{"id": "x"}); code != 0 {
		t.Fatalf("exit = %d", code)
	}
	if got := out.String(); got != "{\n  \"id\": \"x\"\n}\n" {
		t.Errorf("output = %q", got)
	}
}

func TestFailWithAPIError401(t *testing.T) {
	a, out, errOut := newTestApp()
	err := &APIError{Status: 401, Body: `{"error":"nope"}`, Hint: "run: lokictl login --password <password>"}
	if code := a.Fail(err); code != 1 {
		t.Fatalf("exit = %d", code)
	}
	if out.Len() != 0 {
		t.Errorf("stdout should be empty, got %q", out.String())
	}
	var parsed struct {
		Error  string `json:"error"`
		Status int    `json:"status"`
		Hint   string `json:"hint"`
		Detail any    `json:"detail"`
	}
	if err := json.Unmarshal(errOut.Bytes(), &parsed); err != nil {
		t.Fatalf("stderr not JSON: %v; got %q", err, errOut.String())
	}
	if parsed.Status != 401 || !strings.Contains(parsed.Hint, "lokictl login") || parsed.Detail == nil {
		t.Errorf("parsed = %+v", parsed)
	}
}

func TestPrintTable(t *testing.T) {
	a, out, _ := newTestApp()
	a.Table = true
	rows := []map[string]any{
		{"id": "1", "state": "completed"},
		{"id": "2", "state": "pending", "extra": true},
	}
	if code := a.PrintJSON(rows); code != 0 {
		t.Fatalf("exit = %d", code)
	}
	lines := strings.Split(strings.TrimSpace(out.String()), "\n")
	if len(lines) != 3 {
		t.Fatalf("lines = %v", lines)
	}
	if !strings.Contains(lines[0], "id") || !strings.Contains(lines[0], "extra") {
		t.Errorf("header = %q", lines[0])
	}
	// Non-list values fall back to JSON even in table mode.
	out.Reset()
	if code := a.PrintJSON(map[string]any{"k": 1}); code != 0 {
		t.Fatalf("exit = %d", code)
	}
	if !strings.HasPrefix(out.String(), "{") {
		t.Errorf("fallback output = %q", out.String())
	}
}

func TestFirstRowKeyOrder(t *testing.T) {
	raw := []byte(`[{"zeta":1,"alpha":{"nested":true},"mid":[1,2]}]`)
	got := firstRowKeyOrder(raw)
	want := []string{"zeta", "alpha", "mid"}
	if len(got) != len(want) {
		t.Fatalf("got %v", got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("got %v, want %v", got, want)
			break
		}
	}
}
