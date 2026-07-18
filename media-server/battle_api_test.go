package main

import (
	"encoding/json"
	"math"
	"net/http"
	"testing"
)

type battleResp struct {
	WinnerPath    string  `json:"winnerPath"`
	WinnerElo     float64 `json:"winnerElo"`
	WinnerMatches int64   `json:"winnerMatches"`
	LoserPath     string  `json:"loserPath"`
	LoserElo      float64 `json:"loserElo"`
	LoserMatches  int64   `json:"loserMatches"`
}

func postBattle(t *testing.T, deps *Dependencies, body string) (int, battleResp) {
	t.Helper()
	rr := postLibraryJSON(t, mediaBattleHandler(deps), "/api/media/battle", body)
	var resp battleResp
	if rr.Code == http.StatusOK {
		if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
			t.Fatalf("bad JSON: %v; body = %s", err, rr.Body.String())
		}
	}
	return rr.Code, resp
}

func TestBattleFirstMatchFromDefaults(t *testing.T) {
	deps := newLibraryTestDeps(t)

	code, resp := postBattle(t, deps, `{"winnerPath":"a.jpg","loserPath":"b.jpg"}`)
	if code != http.StatusOK {
		t.Fatalf("status = %d", code)
	}
	// Both unrated (1500 vs 1500, expected 0.5), zero prior matches → K=48:
	// winner 1500 + 48*0.5 = 1524, loser 1476.
	if math.Abs(resp.WinnerElo-1524) > 1e-9 || math.Abs(resp.LoserElo-1476) > 1e-9 {
		t.Errorf("elos = %v / %v, want 1524 / 1476", resp.WinnerElo, resp.LoserElo)
	}
	if resp.WinnerMatches != 1 || resp.LoserMatches != 1 {
		t.Errorf("matches = %d / %d, want 1 / 1", resp.WinnerMatches, resp.LoserMatches)
	}

	// Ratings, counters, and the log row all persisted.
	var elo float64
	var wins int64
	deps.DB.QueryRow(`SELECT elo, COALESCE(wins,0) FROM media WHERE path='a.jpg'`).Scan(&elo, &wins)
	if math.Abs(elo-1524) > 1e-9 || wins != 1 {
		t.Errorf("winner row: elo = %v, wins = %d", elo, wins)
	}
	var losses int64
	deps.DB.QueryRow(`SELECT elo, COALESCE(losses,0) FROM media WHERE path='b.jpg'`).Scan(&elo, &losses)
	if math.Abs(elo-1476) > 1e-9 || losses != 1 {
		t.Errorf("loser row: elo = %v, losses = %d", elo, losses)
	}
	var n int64
	var before, after float64
	deps.DB.QueryRow(`SELECT COUNT(*), winner_elo_before, winner_elo_after FROM battle`).Scan(&n, &before, &after)
	if n != 1 || before != 1500 || math.Abs(after-1524) > 1e-9 {
		t.Errorf("battle log: n = %d, before = %v, after = %v", n, before, after)
	}
}

func TestBattleUnseenPathsGetRows(t *testing.T) {
	deps := newLibraryTestDeps(t)

	code, _ := postBattle(t, deps, `{"winnerPath":"new1.jpg","loserPath":"new2.jpg"}`)
	if code != http.StatusOK {
		t.Fatalf("status = %d", code)
	}
	var n int
	deps.DB.QueryRow(`SELECT COUNT(*) FROM media WHERE path IN ('new1.jpg','new2.jpg')`).Scan(&n)
	if n != 2 {
		t.Errorf("media rows created = %d, want 2", n)
	}
}

func TestBattleKDecay(t *testing.T) {
	deps := newLibraryTestDeps(t)
	// a.jpg has fought 10 battles already → K drops to 24; b.jpg is fresh → K=48.
	for i := 0; i < 10; i++ {
		deps.DB.Exec(`INSERT INTO battle (winner_path, loser_path, outcome) VALUES ('a.jpg', 'x.jpg', 1)`)
	}
	deps.DB.Exec(`UPDATE media SET elo = 1500 WHERE path IN ('a.jpg','b.jpg')`)

	code, resp := postBattle(t, deps, `{"winnerPath":"a.jpg","loserPath":"b.jpg"}`)
	if code != http.StatusOK {
		t.Fatalf("status = %d", code)
	}
	// Equal ratings: winner moves +K/2, loser -K/2 with their own K each.
	if math.Abs(resp.WinnerElo-1512) > 1e-9 {
		t.Errorf("veteran winner elo = %v, want 1512 (K=24)", resp.WinnerElo)
	}
	if math.Abs(resp.LoserElo-1476) > 1e-9 {
		t.Errorf("fresh loser elo = %v, want 1476 (K=48)", resp.LoserElo)
	}
	if resp.WinnerMatches != 11 || resp.LoserMatches != 1 {
		t.Errorf("matches = %d / %d, want 11 / 1", resp.WinnerMatches, resp.LoserMatches)
	}
}

func TestBattleDrawOutcome(t *testing.T) {
	deps := newLibraryTestDeps(t)
	deps.DB.Exec(`UPDATE media SET elo = 1600 WHERE path = 'a.jpg'`)
	deps.DB.Exec(`UPDATE media SET elo = 1400 WHERE path = 'b.jpg'`)

	code, resp := postBattle(t, deps, `{"winnerPath":"a.jpg","loserPath":"b.jpg","outcome":0.5}`)
	if code != http.StatusOK {
		t.Fatalf("status = %d", code)
	}
	// A draw pulls the higher rating down and the lower up.
	if resp.WinnerElo >= 1600 || resp.LoserElo <= 1400 {
		t.Errorf("draw elos = %v / %v, want <1600 / >1400", resp.WinnerElo, resp.LoserElo)
	}
	// Draws don't touch the wins/losses counters.
	var wins, losses int64
	deps.DB.QueryRow(`SELECT COALESCE(wins,0), COALESCE(losses,0) FROM media WHERE path='a.jpg'`).Scan(&wins, &losses)
	if wins != 0 || losses != 0 {
		t.Errorf("draw counters: wins = %d, losses = %d, want 0/0", wins, losses)
	}
}

func TestBattleValidation(t *testing.T) {
	deps := newLibraryTestDeps(t)
	if code, _ := postBattle(t, deps, `{"winnerPath":"a.jpg"}`); code != http.StatusBadRequest {
		t.Errorf("missing loser: status = %d, want 400", code)
	}
	if code, _ := postBattle(t, deps, `{"winnerPath":"a.jpg","loserPath":"a.jpg"}`); code != http.StatusBadRequest {
		t.Errorf("self battle: status = %d, want 400", code)
	}
	if code, _ := postBattle(t, deps, `{"winnerPath":"a.jpg","loserPath":"b.jpg","outcome":0.3}`); code != http.StatusBadRequest {
		t.Errorf("bad outcome: status = %d, want 400", code)
	}
	var n int
	deps.DB.QueryRow(`SELECT COUNT(*) FROM battle`).Scan(&n)
	if n != 0 {
		t.Errorf("rejected requests must not log battles, got %d rows", n)
	}
}
