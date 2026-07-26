package main

import (
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"html/template"
	"net/http"
	"net/url"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/stevecastle/shrike/auth"
)

// Browser-based CLI login (RFC 8252 loopback flow with PKCE).
//
// The CLI listens on an ephemeral 127.0.0.1 port and opens the browser at
// GET /auth/cli/authorize?port&state&code_challenge&name. A logged-in
// browser session sees an approve page; approving POSTs back here, which
// mints a one-time authorization code bound to the user and the PKCE
// challenge and redirects to the loopback callback. The CLI then exchanges
// code + verifier at POST /auth/cli/token for a freshly minted lk_ API key
// — the same credential kind header auth already accepts everywhere.
//
// Hardening: codes are single-use and expire after 2 minutes; the redirect
// target is always 127.0.0.1:<port> built server-side (never a caller
// -supplied URL); minting requires a POST from a same-origin page with a
// valid session cookie (GET never mints); the PKCE S256 check stops anything
// that saw the redirect from redeeming the code without the verifier.

const cliAuthCodeTTL = 2 * time.Minute

var (
	cliStateRE    = regexp.MustCompile(`^[A-Za-z0-9_-]{8,128}$`)
	cliChalRE     = regexp.MustCompile(`^[A-Za-z0-9_-]{43}$`)       // base64url SHA-256, no padding
	cliVerifierRE = regexp.MustCompile(`^[A-Za-z0-9._~-]{43,128}$`) // RFC 7636 unreserved
)

type cliAuthCode struct {
	username  string
	challenge string
	keyName   string
	expires   time.Time
}

var cliAuthCodes = struct {
	sync.Mutex
	m map[string]*cliAuthCode
}{m: map[string]*cliAuthCode{}}

func mintCLIAuthCode(username, challenge, keyName string) (string, error) {
	raw := make([]byte, 32)
	if _, err := rand.Read(raw); err != nil {
		return "", err
	}
	code := hex.EncodeToString(raw)
	now := time.Now()
	cliAuthCodes.Lock()
	defer cliAuthCodes.Unlock()
	for c, v := range cliAuthCodes.m {
		if now.After(v.expires) {
			delete(cliAuthCodes.m, c)
		}
	}
	cliAuthCodes.m[code] = &cliAuthCode{username: username, challenge: challenge, keyName: keyName, expires: now.Add(cliAuthCodeTTL)}
	return code, nil
}

// redeemCLIAuthCode removes and returns the code's grant. Single use: the
// code is consumed whether or not the caller's verifier turns out to match.
func redeemCLIAuthCode(code string) *cliAuthCode {
	cliAuthCodes.Lock()
	defer cliAuthCodes.Unlock()
	v := cliAuthCodes.m[code]
	delete(cliAuthCodes.m, code)
	if v == nil || time.Now().After(v.expires) {
		return nil
	}
	return v
}

// cliAuthParams are the validated query/form parameters of the authorize
// endpoint.
type cliAuthParams struct {
	port      int
	state     string
	challenge string
	keyName   string
}

func parseCLIAuthParams(get func(string) string) (cliAuthParams, error) {
	var p cliAuthParams
	port, err := strconv.Atoi(get("port"))
	if err != nil || port < 1 || port > 65535 {
		return p, fmt.Errorf("invalid port")
	}
	p.port = port
	p.state = get("state")
	if !cliStateRE.MatchString(p.state) {
		return p, fmt.Errorf("invalid state")
	}
	p.challenge = get("code_challenge")
	if !cliChalRE.MatchString(p.challenge) {
		return p, fmt.Errorf("invalid code_challenge")
	}
	p.keyName = strings.Map(func(r rune) rune {
		if r < 32 || r == 127 {
			return -1
		}
		return r
	}, strings.TrimSpace(get("name")))
	if p.keyName == "" {
		p.keyName = "lokictl"
	}
	if len(p.keyName) > 64 {
		p.keyName = p.keyName[:64]
	}
	return p, nil
}

// cliSessionUsername resolves the browser session (cookie only — header
// credentials are for API clients, not the approve page).
func cliSessionUsername(deps *Dependencies, r *http.Request) string {
	cookie, err := r.Cookie("auth_token")
	if err != nil {
		return ""
	}
	claims, err := deps.Auth.VerifyToken(cookie.Value)
	if err != nil {
		return ""
	}
	return claims.Username
}

// sameOriginPost rejects cross-site form posts: when the browser sends an
// Origin header it must match the host serving this page.
func sameOriginPost(r *http.Request) bool {
	origin := r.Header.Get("Origin")
	if origin == "" {
		return true // non-browser or same-origin GET-initiated form on old browsers
	}
	u, err := url.Parse(origin)
	if err != nil {
		return false
	}
	return u.Host == r.Host
}

var cliApprovePage = template.Must(template.New("cliapprove").Parse(`<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Authorize CLI – Lowkey Media Server</title>
<style>
  body { margin:0; min-height:100vh; display:flex; align-items:center; justify-content:center;
         background:#101216; color:#d7dae0; font:14px/1.5 "Segoe UI", system-ui, sans-serif; }
  .card { width:min(420px, calc(100vw - 32px)); background:#1a1d23; border:1px solid #30343e;
          border-radius:12px; padding:28px; }
  h1 { font-size:17px; margin:0 0 12px; }
  p { color:#9aa0ab; margin:0 0 10px; }
  code { color:#d7dae0; background:#262a33; border-radius:4px; padding:1px 6px; }
  .actions { display:flex; gap:10px; margin-top:20px; }
  button { flex:1; padding:10px; border-radius:8px; border:1px solid #30343e; cursor:pointer;
           font-size:13px; font-weight:600; background:#262a33; color:#d7dae0; }
  button.approve { background:#00d4aa; border-color:#00d4aa; color:#00281f; }
</style>
</head>
<body>
  <div class="card">
    <h1>Authorize <code>{{.KeyName}}</code>?</h1>
    <p>A command-line tool on this computer is asking for access to your
       library as <strong>{{.Username}}</strong>.</p>
    <p>Approving creates an API key and hands it to the tool listening on
       <code>127.0.0.1:{{.Port}}</code>. Only approve if you just ran
       <code>lokictl login</code> yourself.</p>
    <form method="POST">
      <input type="hidden" name="port" value="{{.Port}}">
      <input type="hidden" name="state" value="{{.State}}">
      <input type="hidden" name="code_challenge" value="{{.Challenge}}">
      <input type="hidden" name="name" value="{{.KeyName}}">
      <div class="actions">
        <button type="submit" name="action" value="deny">Deny</button>
        <button type="submit" name="action" value="approve" class="approve" autofocus>Approve</button>
      </div>
    </form>
  </div>
</body>
</html>`))

// cliAuthAuthorizeHandler implements /auth/cli/authorize: GET renders the
// approve page (bouncing to /login first when there is no session), POST
// mints the one-time code and redirects to the CLI's loopback callback.
func cliAuthAuthorizeHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			p, err := parseCLIAuthParams(r.URL.Query().Get)
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			username := cliSessionUsername(deps, r)
			if username == "" {
				http.Redirect(w, r, "/login?redirect="+url.QueryEscape(r.URL.String()), http.StatusFound)
				return
			}
			if username == auth.DefaultAdminUsername {
				http.Error(w, "Finish first-run setup (create a real account) before authorizing the CLI", http.StatusForbidden)
				return
			}
			w.Header().Set("Content-Type", "text/html; charset=utf-8")
			w.Header().Set("Cache-Control", "no-store")
			_ = cliApprovePage.Execute(w, map[string]any{
				"Username": username, "Port": p.port, "State": p.state,
				"Challenge": p.challenge, "KeyName": p.keyName,
			})

		case http.MethodPost:
			if !sameOriginPost(r) {
				http.Error(w, "Cross-origin request rejected", http.StatusForbidden)
				return
			}
			if err := r.ParseForm(); err != nil {
				http.Error(w, "Invalid form", http.StatusBadRequest)
				return
			}
			p, err := parseCLIAuthParams(r.PostForm.Get)
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			callback := fmt.Sprintf("http://127.0.0.1:%d/callback", p.port)
			if r.PostForm.Get("action") != "approve" {
				http.Redirect(w, r, callback+"?error=access_denied&state="+url.QueryEscape(p.state), http.StatusSeeOther)
				return
			}
			username := cliSessionUsername(deps, r)
			if username == "" {
				http.Error(w, "Not logged in", http.StatusUnauthorized)
				return
			}
			if username == auth.DefaultAdminUsername {
				http.Error(w, "Finish first-run setup before authorizing the CLI", http.StatusForbidden)
				return
			}
			code, err := mintCLIAuthCode(username, p.challenge, p.keyName)
			if err != nil {
				http.Error(w, "Could not create authorization code", http.StatusInternalServerError)
				return
			}
			http.Redirect(w, r, callback+"?code="+url.QueryEscape(code)+"&state="+url.QueryEscape(p.state), http.StatusSeeOther)

		default:
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		}
	}
}

// cliAuthTokenHandler implements /auth/cli/token: the CLI exchanges its
// one-time code plus the PKCE verifier for a newly minted API key.
func cliAuthTokenHandler(deps *Dependencies) http.HandlerFunc {
	fail := func(w http.ResponseWriter, status int, msg string) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_ = json.NewEncoder(w).Encode(map[string]string{"error": msg})
	}
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			fail(w, http.StatusMethodNotAllowed, "Use POST")
			return
		}
		var req struct {
			Code         string `json:"code"`
			CodeVerifier string `json:"code_verifier"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			fail(w, http.StatusBadRequest, "Invalid JSON")
			return
		}
		if !cliVerifierRE.MatchString(req.CodeVerifier) {
			fail(w, http.StatusBadRequest, "Invalid code_verifier")
			return
		}
		grant := redeemCLIAuthCode(req.Code)
		if grant == nil {
			fail(w, http.StatusBadRequest, "Invalid or expired code")
			return
		}
		sum := sha256.Sum256([]byte(req.CodeVerifier))
		want := base64.RawURLEncoding.EncodeToString(sum[:])
		if subtle.ConstantTimeCompare([]byte(want), []byte(grant.challenge)) != 1 {
			fail(w, http.StatusBadRequest, "code_verifier does not match")
			return
		}
		key, meta, err := deps.Auth.CreateAPIKey(grant.username, grant.keyName)
		if err != nil {
			fail(w, http.StatusInternalServerError, err.Error())
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"status":   "ok",
			"key":      key,
			"username": meta.Username,
			"name":     meta.Name,
		})
	}
}
