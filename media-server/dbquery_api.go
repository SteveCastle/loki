package main

// POST /api/db/query — read-only SQL over the active database, built for the
// lokictl CLI (and any trusted API client). Untagged so every platform main
// registers it. Two layers of protection: a SELECT/WITH prefix check, and the
// query_only pragma on a dedicated pooled connection, which makes SQLite
// itself refuse any write no matter how it is phrased.

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"time"
)

type dbQueryRequest struct {
	SQL       string `json:"sql"`
	Args      []any  `json:"args"`
	Limit     int    `json:"limit"`
	TimeoutMS int    `json:"timeout_ms"`
}

type dbQueryResponse struct {
	Columns   []string `json:"columns"`
	Rows      [][]any  `json:"rows"`
	RowCount  int      `json:"row_count"`
	Truncated bool     `json:"truncated"`
	ElapsedMS int64    `json:"elapsed_ms"`
}

// isReadOnlySQL accepts a single SELECT/WITH statement. Leading SQL comments
// are allowed; one trailing semicolon is allowed. A ';' anywhere else rejects
// the query (this also rejects ';' inside string literals — callers should use
// bind args for data, which sidesteps the issue).
func isReadOnlySQL(sqlText string) bool {
	s := strings.TrimSpace(sqlText)
	for {
		if strings.HasPrefix(s, "--") {
			i := strings.IndexByte(s, '\n')
			if i < 0 {
				return false
			}
			s = strings.TrimSpace(s[i+1:])
			continue
		}
		if strings.HasPrefix(s, "/*") {
			i := strings.Index(s, "*/")
			if i < 0 {
				return false
			}
			s = strings.TrimSpace(s[i+2:])
			continue
		}
		break
	}
	up := strings.ToUpper(s)
	if !strings.HasPrefix(up, "SELECT") && !strings.HasPrefix(up, "WITH") {
		return false
	}
	if i := strings.IndexByte(s, ';'); i >= 0 && strings.TrimSpace(s[i+1:]) != "" {
		return false
	}
	return true
}

func writeDBQueryError(w http.ResponseWriter, status int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]string{"error": msg})
}

func dbQueryHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeDBQueryError(w, http.StatusMethodNotAllowed, "Use POST")
			return
		}
		var req dbQueryRequest
		if err := readJSONBody(r, &req); err != nil {
			writeDBQueryError(w, http.StatusBadRequest, "bad json")
			return
		}
		if !isReadOnlySQL(req.SQL) {
			writeDBQueryError(w, http.StatusBadRequest, "only a single SELECT/WITH statement is allowed")
			return
		}
		limit := req.Limit
		if limit <= 0 {
			limit = 1000
		} else if limit > 10000 {
			limit = 10000
		}
		toMS := req.TimeoutMS
		if toMS <= 0 {
			toMS = 5000
		} else if toMS > 30000 {
			toMS = 30000
		}
		ctx, cancel := context.WithTimeout(r.Context(), time.Duration(toMS)*time.Millisecond)
		defer cancel()

		conn, err := deps.DB.Conn(ctx)
		if err != nil {
			writeDBQueryError(w, http.StatusInternalServerError, "db unavailable")
			return
		}
		defer conn.Close()
		if _, err := conn.ExecContext(ctx, "PRAGMA query_only=ON"); err != nil {
			writeDBQueryError(w, http.StatusInternalServerError, "failed to set read-only mode")
			return
		}
		// Reset before the conn returns to the pool, even on error/timeout —
		// a pooled connection stuck in query_only would break writers.
		defer func() { _, _ = conn.ExecContext(context.Background(), "PRAGMA query_only=OFF") }()

		start := time.Now()
		rows, err := conn.QueryContext(ctx, req.SQL, req.Args...)
		if err != nil {
			writeDBQueryError(w, http.StatusBadRequest, err.Error())
			return
		}
		defer rows.Close()

		cols, err := rows.Columns()
		if err != nil {
			writeDBQueryError(w, http.StatusInternalServerError, "failed to read columns")
			return
		}
		resp := dbQueryResponse{Columns: cols, Rows: [][]any{}}
		for rows.Next() {
			if resp.RowCount >= limit {
				resp.Truncated = true
				break
			}
			raw := make([]any, len(cols))
			ptrs := make([]any, len(cols))
			for i := range raw {
				ptrs[i] = &raw[i]
			}
			if err := rows.Scan(ptrs...); err != nil {
				writeDBQueryError(w, http.StatusInternalServerError, "scan failed")
				return
			}
			cells := make([]any, len(cols))
			for i, v := range raw {
				if b, ok := v.([]byte); ok {
					cells[i] = map[string]int{"blob_bytes": len(b)}
				} else {
					cells[i] = v
				}
			}
			resp.Rows = append(resp.Rows, cells)
			resp.RowCount++
		}
		if err := rows.Err(); err != nil {
			writeDBQueryError(w, http.StatusBadRequest, err.Error())
			return
		}
		resp.ElapsedMS = time.Since(start).Milliseconds()
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}
}
