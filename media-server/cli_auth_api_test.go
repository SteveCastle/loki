package main

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
)

const testVerifier = "verifier-verifier-verifier-verifier-verifier-1"

func testChallenge() string {
	sum := sha256.Sum256([]byte(testVerifier))
	return base64.RawURLEncoding.EncodeToString(sum[:])
}

func authorizeQuery() string {
	return "port=43210&state=test-state-1234&code_challenge=" + url.QueryEscape(testChallenge()) + "&name=lokictl%40box"
}

func doAuthorize(t *testing.T, deps *Dependencies, method, target, body, cookie string, hdr map[string]string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(method, target, strings.NewReader(body))
	if body != "" {
		req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	}
	if cookie != "" {
		req.AddCookie(&http.Cookie{Name: "auth_token", Value: cookie})
	}
	for k, v := range hdr {
		req.Header.Set(k, v)
	}
	rr := httptest.NewRecorder()
	cliAuthAuthorizeHandler(deps).ServeHTTP(rr, req)
	return rr
}

func exchangeToken(t *testing.T, deps *Dependencies, code, verifier string) (*httptest.ResponseRecorder, map[string]string) {
	t.Helper()
	body, _ := json.Marshal(map[string]string{"code": code, "code_verifier": verifier})
	req := httptest.NewRequest(http.MethodPost, "/auth/cli/token", strings.NewReader(string(body)))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	cliAuthTokenHandler(deps).ServeHTTP(rr, req)
	var out map[string]string
	_ = json.Unmarshal(rr.Body.Bytes(), &out)
	return rr, out
}

// approveAndGetCode runs the approve POST and pulls the code out of the
// loopback redirect.
func approveAndGetCode(t *testing.T, deps *Dependencies, cookie string) string {
	t.Helper()
	form := authorizeQuery() + "&action=approve"
	rr := doAuthorize(t, deps, http.MethodPost, "/auth/cli/authorize", form, cookie, nil)
	if rr.Code != http.StatusSeeOther {
		t.Fatalf("approve: got %d, want 303 (body: %s)", rr.Code, rr.Body.String())
	}
	loc, err := url.Parse(rr.Header().Get("Location"))
	if err != nil {
		t.Fatalf("parse redirect: %v", err)
	}
	if loc.Scheme != "http" || loc.Host != "127.0.0.1:43210" || loc.Path != "/callback" {
		t.Fatalf("redirect target = %s, want the loopback callback", loc.String())
	}
	if got := loc.Query().Get("state"); got != "test-state-1234" {
		t.Fatalf("redirect state = %q", got)
	}
	code := loc.Query().Get("code")
	if code == "" {
		t.Fatal("redirect has no code")
	}
	return code
}

func TestCLIAuthAuthorize_RequiresSession(t *testing.T) {
	deps := newAPIKeyTestDeps(t)
	rr := doAuthorize(t, deps, http.MethodGet, "/auth/cli/authorize?"+authorizeQuery(), "", "", nil)
	if rr.Code != http.StatusFound {
		t.Fatalf("got %d, want 302 to /login", rr.Code)
	}
	loc := rr.Header().Get("Location")
	if !strings.HasPrefix(loc, "/login?redirect=") {
		t.Fatalf("redirect = %q, want /login?redirect=...", loc)
	}
	// The round-trip target must survive the login redirect.
	if !strings.Contains(loc, url.QueryEscape("/auth/cli/authorize")) {
		t.Fatalf("redirect %q does not return to the authorize page", loc)
	}
}

func TestCLIAuthAuthorize_ShowsApprovePage(t *testing.T) {
	deps := newAPIKeyTestDeps(t)
	tok := loginToken(t, deps)
	rr := doAuthorize(t, deps, http.MethodGet, "/auth/cli/authorize?"+authorizeQuery(), "", tok, nil)
	if rr.Code != http.StatusOK {
		t.Fatalf("got %d, want 200 (body: %s)", rr.Code, rr.Body.String())
	}
	page := rr.Body.String()
	for _, want := range []string{"lokictl@box", "127.0.0.1:43210", "steve", `name="action" value="approve"`} {
		if !strings.Contains(page, want) {
			t.Errorf("approve page missing %q", want)
		}
	}
}

func TestCLIAuthAuthorize_RejectsBadParams(t *testing.T) {
	deps := newAPIKeyTestDeps(t)
	tok := loginToken(t, deps)
	for name, q := range map[string]string{
		"bad port":      "port=999999&state=test-state-1234&code_challenge=" + testChallenge(),
		"missing state": "port=43210&code_challenge=" + testChallenge(),
		"short state":   "port=43210&state=ab&code_challenge=" + testChallenge(),
		"bad challenge": "port=43210&state=test-state-1234&code_challenge=nope",
	} {
		rr := doAuthorize(t, deps, http.MethodGet, "/auth/cli/authorize?"+q, "", tok, nil)
		if rr.Code != http.StatusBadRequest {
			t.Errorf("%s: got %d, want 400", name, rr.Code)
		}
	}
}

func TestCLIAuthAuthorize_RejectsCrossOriginPost(t *testing.T) {
	deps := newAPIKeyTestDeps(t)
	tok := loginToken(t, deps)
	rr := doAuthorize(t, deps, http.MethodPost, "/auth/cli/authorize",
		authorizeQuery()+"&action=approve", tok, map[string]string{"Origin": "https://evil.example"})
	if rr.Code != http.StatusForbidden {
		t.Fatalf("cross-origin approve: got %d, want 403", rr.Code)
	}
}

func TestCLIAuthAuthorize_DenyRedirectsWithError(t *testing.T) {
	deps := newAPIKeyTestDeps(t)
	tok := loginToken(t, deps)
	rr := doAuthorize(t, deps, http.MethodPost, "/auth/cli/authorize",
		authorizeQuery()+"&action=deny", tok, nil)
	if rr.Code != http.StatusSeeOther {
		t.Fatalf("deny: got %d, want 303", rr.Code)
	}
	loc, _ := url.Parse(rr.Header().Get("Location"))
	if loc.Query().Get("error") != "access_denied" || loc.Query().Get("code") != "" {
		t.Fatalf("deny redirect = %s, want error=access_denied and no code", loc)
	}
}

func TestCLIAuthFlow_ExchangeMintsWorkingKey(t *testing.T) {
	deps := newAPIKeyTestDeps(t)
	tok := loginToken(t, deps)
	code := approveAndGetCode(t, deps, tok)

	rr, out := exchangeToken(t, deps, code, testVerifier)
	if rr.Code != http.StatusOK {
		t.Fatalf("exchange: got %d (body: %s)", rr.Code, rr.Body.String())
	}
	if !strings.HasPrefix(out["key"], "lk_") {
		t.Fatalf("key = %q, want lk_ prefix", out["key"])
	}
	if out["username"] != "steve" || out["name"] != "lokictl@box" {
		t.Fatalf("key metadata = %v", out)
	}
	claims, err := deps.Auth.VerifyAPIKey(out["key"])
	if err != nil || claims.Username != "steve" {
		t.Fatalf("minted key does not verify: claims=%v err=%v", claims, err)
	}

	// Codes are single use.
	rr, _ = exchangeToken(t, deps, code, testVerifier)
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("code replay: got %d, want 400", rr.Code)
	}
}

func TestCLIAuthFlow_WrongVerifierRejected(t *testing.T) {
	deps := newAPIKeyTestDeps(t)
	tok := loginToken(t, deps)
	code := approveAndGetCode(t, deps, tok)

	rr, _ := exchangeToken(t, deps, code, "wrong-verifier-wrong-verifier-wrong-verifier-x")
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("wrong verifier: got %d, want 400", rr.Code)
	}
	// The failed attempt must also burn the code.
	rr, _ = exchangeToken(t, deps, code, testVerifier)
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("code after failed exchange: got %d, want 400", rr.Code)
	}
}

func TestCLIAuthToken_RejectsGarbage(t *testing.T) {
	deps := newAPIKeyTestDeps(t)
	for name, pair := range map[string][2]string{
		"unknown code": {"deadbeef", testVerifier},
		"bad verifier": {"deadbeef", "short"},
		"empty":        {"", ""},
	} {
		rr, _ := exchangeToken(t, deps, pair[0], pair[1])
		if rr.Code != http.StatusBadRequest {
			t.Errorf("%s: got %d, want 400", name, rr.Code)
		}
	}
}

func TestCLIAuthCodeExpiry(t *testing.T) {
	code, err := mintCLIAuthCode("steve", testChallenge(), "lokictl")
	if err != nil {
		t.Fatal(err)
	}
	cliAuthCodes.Lock()
	cliAuthCodes.m[code].expires = cliAuthCodes.m[code].expires.Add(-3 * cliAuthCodeTTL)
	cliAuthCodes.Unlock()
	if grant := redeemCLIAuthCode(code); grant != nil {
		t.Fatal("expired code was redeemed")
	}
}

func TestParseCLIAuthParams_NameDefaultsAndSanitizes(t *testing.T) {
	get := func(m map[string]string) func(string) string {
		return func(k string) string { return m[k] }
	}
	base := map[string]string{"port": "1234", "state": "test-state-1234", "code_challenge": testChallenge()}

	p, err := parseCLIAuthParams(get(base))
	if err != nil || p.keyName != "lokictl" {
		t.Fatalf("default name: %q err=%v", p.keyName, err)
	}

	withName := map[string]string{}
	for k, v := range base {
		withName[k] = v
	}
	withName["name"] = "  evil\x00name\x1b " + strings.Repeat("x", 100)
	p, err = parseCLIAuthParams(get(withName))
	if err != nil {
		t.Fatal(err)
	}
	if strings.ContainsAny(p.keyName, "\x00\x1b") || len(p.keyName) > 64 {
		t.Fatalf("name not sanitized: %q", p.keyName)
	}
	_ = fmt.Sprint(p)
}
