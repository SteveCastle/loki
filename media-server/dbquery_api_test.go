package main

import (
	"database/sql"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func newDBQueryTestDeps(t *testing.T) *Dependencies {
	t.Helper()
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatalf("failed to open in-memory db: %v", err)
	}
	t.Cleanup(func() { db.Close() })
	if _, err := db.Exec(`CREATE TABLE t(a INTEGER, b TEXT, c BLOB)`); err != nil {
		t.Fatalf("create table: %v", err)
	}
	if _, err := db.Exec(`INSERT INTO t VALUES (1, 'one', NULL), (2, 'two', x'DEADBEEF')`); err != nil {
		t.Fatalf("insert: %v", err)
	}
	return &Dependencies{DB: db}
}

func postDBQuery(t *testing.T, deps *Dependencies, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/api/db/query", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	dbQueryHandler(deps).ServeHTTP(rr, req)
	return rr
}

func decodeDBQueryResponse(t *testing.T, rr *httptest.ResponseRecorder) dbQueryResponse {
	t.Helper()
	var resp dbQueryResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("response not JSON: %v; body = %s", err, rr.Body.String())
	}
	return resp
}

func TestDBQuery_Select(t *testing.T) {
	deps := newDBQueryTestDeps(t)
	rr := postDBQuery(t, deps, `{"sql":"SELECT a, b FROM t ORDER BY a"}`)
	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body = %s", rr.Code, rr.Body.String())
	}
	resp := decodeDBQueryResponse(t, rr)
	if len(resp.Columns) != 2 || resp.Columns[0] != "a" || resp.Columns[1] != "b" {
		t.Errorf("columns = %v", resp.Columns)
	}
	if resp.RowCount != 2 || resp.Truncated {
		t.Errorf("row_count = %d truncated = %v", resp.RowCount, resp.Truncated)
	}
	if got := resp.Rows[0][1]; got != "one" {
		t.Errorf("rows[0][1] = %v (%T), want \"one\"", got, got)
	}
}

func TestDBQuery_PositionalArgs(t *testing.T) {
	deps := newDBQueryTestDeps(t)
	rr := postDBQuery(t, deps, `{"sql":"SELECT b FROM t WHERE a = ?","args":[1]}`)
	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d; body = %s", rr.Code, rr.Body.String())
	}
	resp := decodeDBQueryResponse(t, rr)
	if resp.RowCount != 1 || resp.Rows[0][0] != "one" {
		t.Errorf("unexpected result: %+v", resp)
	}
}

func TestDBQuery_LimitTruncates(t *testing.T) {
	deps := newDBQueryTestDeps(t)
	rr := postDBQuery(t, deps, `{"sql":"SELECT a FROM t","limit":1}`)
	resp := decodeDBQueryResponse(t, rr)
	if resp.RowCount != 1 || !resp.Truncated {
		t.Errorf("row_count = %d truncated = %v, want 1/true", resp.RowCount, resp.Truncated)
	}
}

func TestDBQuery_BlobPlaceholder(t *testing.T) {
	deps := newDBQueryTestDeps(t)
	rr := postDBQuery(t, deps, `{"sql":"SELECT c FROM t WHERE a = 2"}`)
	resp := decodeDBQueryResponse(t, rr)
	cell, ok := resp.Rows[0][0].(map[string]any)
	if !ok {
		t.Fatalf("blob cell = %v (%T), want object", resp.Rows[0][0], resp.Rows[0][0])
	}
	if n, _ := cell["blob_bytes"].(float64); n != 4 {
		t.Errorf("blob_bytes = %v, want 4", cell["blob_bytes"])
	}
}

func TestDBQuery_RejectsWrites(t *testing.T) {
	deps := newDBQueryTestDeps(t)
	for _, sqlText := range []string{
		`INSERT INTO t VALUES (9, 'x', NULL)`,
		`UPDATE t SET b = 'x'`,
		`DELETE FROM t`,
		`DROP TABLE t`,
		`PRAGMA journal_mode=DELETE`,
		`SELECT 1; SELECT 2`,
		`SELECT 1; DROP TABLE t`,
	} {
		rr := postDBQuery(t, deps, `{"sql":`+mustJSON(sqlText)+`}`)
		if rr.Code != http.StatusBadRequest {
			t.Errorf("%q: status = %d, want 400", sqlText, rr.Code)
		}
	}
	// Table must be intact afterwards.
	var n int
	if err := deps.DB.QueryRow(`SELECT COUNT(*) FROM t`).Scan(&n); err != nil || n != 2 {
		t.Errorf("table damaged: count = %d err = %v", n, err)
	}
}

func TestDBQuery_AllowsCommentsAndWith(t *testing.T) {
	deps := newDBQueryTestDeps(t)
	for _, sqlText := range []string{
		"-- a comment\nSELECT 1",
		"/* block */ SELECT 1",
		"WITH x AS (SELECT 1 AS v) SELECT v FROM x",
		"SELECT 1;",
	} {
		rr := postDBQuery(t, deps, `{"sql":`+mustJSON(sqlText)+`}`)
		if rr.Code != http.StatusOK {
			t.Errorf("%q: status = %d, want 200; body = %s", sqlText, rr.Code, rr.Body.String())
		}
	}
}

func TestDBQuery_MethodAndBodyValidation(t *testing.T) {
	deps := newDBQueryTestDeps(t)
	req := httptest.NewRequest(http.MethodGet, "/api/db/query", nil)
	rr := httptest.NewRecorder()
	dbQueryHandler(deps).ServeHTTP(rr, req)
	if rr.Code != http.StatusMethodNotAllowed {
		t.Errorf("GET status = %d, want 405", rr.Code)
	}
	rr = postDBQuery(t, deps, `{not json`)
	if rr.Code != http.StatusBadRequest {
		t.Errorf("bad json status = %d, want 400", rr.Code)
	}
}

func TestIsReadOnlySQL(t *testing.T) {
	cases := []struct {
		sql  string
		want bool
	}{
		{"SELECT 1", true},
		{"  select a from t  ", true},
		{"WITH x AS (SELECT 1) SELECT * FROM x", true},
		{"-- c\nSELECT 1", true},
		{"/* c */ WITH x AS (SELECT 1) SELECT 1", true},
		{"SELECT 1;", true},
		{"SELECT 1; SELECT 2", false},
		{"INSERT INTO t VALUES (1)", false},
		{"PRAGMA foo", false},
		{"-- only a comment", false},
		{"/* unterminated", false},
		{"", false},
	}
	for _, c := range cases {
		if got := isReadOnlySQL(c.sql); got != c.want {
			t.Errorf("isReadOnlySQL(%q) = %v, want %v", c.sql, got, c.want)
		}
	}
}

func mustJSON(s string) string {
	b, _ := json.Marshal(s)
	return string(b)
}
