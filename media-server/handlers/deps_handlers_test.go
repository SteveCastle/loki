package handlers

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestGetStatus_ReturnsJSONArray(t *testing.T) {
	rr := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/api/deps/status", nil)
	HandleDepsStatus(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rr.Code, rr.Body)
	}
	var arr []map[string]any
	if err := json.Unmarshal(rr.Body.Bytes(), &arr); err != nil {
		t.Fatalf("body not JSON array: %v", err)
	}
}

func TestPostDownload_UnknownIDReturns404(t *testing.T) {
	rr := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/deps/models/no-such-model/download", nil)
	req.SetPathValue("id", "no-such-model")
	HandleModelDownload(rr, req)
	if rr.Code != http.StatusNotFound {
		t.Errorf("status=%d want 404", rr.Code)
	}
	if !strings.Contains(rr.Body.String(), "unknown") {
		t.Errorf("body=%q want unknown", rr.Body)
	}
}
