package main

import (
	"database/sql"
	"math"
	"net/http"
	"time"
)

// -----------------------------------------------------------------------------
// Battle API (shared across all platform mains).
//
//   POST /api/media/battle — record a battle-mode vote
//
// The client sends only which item won; the Elo math runs here inside a
// transaction so concurrent voters can't clobber each other with stale
// ratings, and every vote lands in the append-only battle log. The elo
// column on media is a derived cache of that log.
// -----------------------------------------------------------------------------

const defaultElo = 1500

// battleKFactor returns the Elo K-factor for an item that has fought n
// battles: provisional ratings move fast, established ones stabilise.
// Mirrored in the Electron main process (src/main/media.ts recordBattle) —
// keep the schedules identical or shared databases drift.
func battleKFactor(n int64) float64 {
	switch {
	case n < 10:
		return 48
	case n < 30:
		return 24
	default:
		return 12
	}
}

func mediaBattleHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			httpError(w, "use POST", http.StatusMethodNotAllowed)
			return
		}
		var req struct {
			WinnerPath string   `json:"winnerPath"`
			LoserPath  string   `json:"loserPath"`
			Outcome    *float64 `json:"outcome"` // winner's score: 1 (default) or 0.5 for a draw
		}
		if err := readJSON(r, &req); err != nil || req.WinnerPath == "" || req.LoserPath == "" {
			httpError(w, "bad request: winnerPath and loserPath required", http.StatusBadRequest)
			return
		}
		if req.WinnerPath == req.LoserPath {
			httpError(w, "bad request: an item cannot battle itself", http.StatusBadRequest)
			return
		}
		outcome := 1.0
		if req.Outcome != nil {
			outcome = *req.Outcome
		}
		if outcome != 1 && outcome != 0.5 {
			httpError(w, "bad request: outcome must be 1 or 0.5", http.StatusBadRequest)
			return
		}

		tx, err := deps.DB.Begin()
		if err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		defer tx.Rollback()

		readSide := func(path string) (elo float64, matches int64, err error) {
			elo = defaultElo
			var stored sql.NullFloat64
			err = tx.QueryRow(`SELECT elo FROM media WHERE path = ?`, path).Scan(&stored)
			if err == sql.ErrNoRows {
				err = nil // unseen path: rated on first battle, row created below
			} else if err != nil {
				return
			}
			if stored.Valid {
				elo = stored.Float64
			}
			err = tx.QueryRow(
				`SELECT COUNT(*) FROM battle WHERE winner_path = ? OR loser_path = ?`,
				path, path).Scan(&matches)
			return
		}
		winnerElo, winnerMatches, err := readSide(req.WinnerPath)
		if err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		loserElo, loserMatches, err := readSide(req.LoserPath)
		if err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}

		expectedWinner := 1 / (1 + math.Pow(10, (loserElo-winnerElo)/400))
		newWinnerElo := winnerElo + battleKFactor(winnerMatches)*(outcome-expectedWinner)
		newLoserElo := loserElo + battleKFactor(loserMatches)*((1-outcome)-(1-expectedWinner))

		// wins/losses/battles are convenience counters for display and
		// pairing; draws bump only battles. The battle log is the
		// authoritative record, so battles is (re)set from the fresh log
		// count — self-healing for rows written before the log existed.
		winInc, lossInc := int64(0), int64(0)
		if outcome == 1 {
			winInc, lossInc = 1, 1
		}
		const upsert = `INSERT INTO media (path, elo, wins, losses, battles) VALUES (?, ?, ?, ?, ?)
			ON CONFLICT(path) DO UPDATE SET
				elo = excluded.elo,
				wins = COALESCE(wins, 0) + ?,
				losses = COALESCE(losses, 0) + ?,
				battles = excluded.battles`
		if _, err := tx.Exec(upsert, req.WinnerPath, newWinnerElo, winInc, 0, winnerMatches+1, winInc, 0); err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		if _, err := tx.Exec(upsert, req.LoserPath, newLoserElo, 0, lossInc, loserMatches+1, 0, lossInc); err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		if _, err := tx.Exec(`INSERT INTO battle
			(winner_path, loser_path, outcome,
			 winner_elo_before, loser_elo_before, winner_elo_after, loser_elo_after, created_at)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
			req.WinnerPath, req.LoserPath, outcome,
			winnerElo, loserElo, newWinnerElo, newLoserElo, time.Now().UnixMilli()); err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		if err := tx.Commit(); err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}

		writeJSON(w, map[string]any{
			"winnerPath":    req.WinnerPath,
			"winnerElo":     newWinnerElo,
			"winnerMatches": winnerMatches + 1,
			"loserPath":     req.LoserPath,
			"loserElo":      newLoserElo,
			"loserMatches":  loserMatches + 1,
		})
	}
}
