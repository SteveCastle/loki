package main

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"errors"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"runtime"
	"time"
)

func init() {
	register(command{
		group: "login", args: "[--password P [--username U]] [--no-browser]",
		summary: "Log in — opens the browser to authorize (default); --password logs in directly (headless)",
		run:     cmdLogin,
	})
}

func cmdLogin(a *App, args []string) int {
	fs := flag.NewFlagSet("login", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	user := fs.String("username", "admin", "username (with --password)")
	pass := fs.String("password", "", "password — skips the browser flow")
	noBrowser := fs.Bool("no-browser", false, "print the authorization URL instead of opening a browser")
	if err := fs.Parse(args); err != nil {
		return a.Usage(fs, err.Error())
	}
	if *pass != "" {
		return passwordLogin(a, *user, *pass)
	}
	return browserLogin(a, *noBrowser)
}

/* ---- password login (headless / scripted) --------------------------- */

func passwordLogin(a *App, user, pass string) int {
	var resp struct {
		Status        string `json:"status"`
		Token         string `json:"token"`
		SetupRequired bool   `json:"setup_required"`
	}
	err := a.Client.DoJSON("POST", "/auth/login", map[string]string{
		"username": user,
		"password": pass,
	}, &resp)
	if err != nil {
		return a.Fail(err)
	}
	if resp.Token == "" {
		return a.Fail(fmt.Errorf("server did not return a token"))
	}
	path, err := saveLoginToken(a, resp.Token)
	if err != nil {
		return a.Fail(err)
	}
	if resp.SetupRequired {
		fmt.Fprintln(a.ErrOut, `warning: the default admin account is still active — most endpoints return 403 until a real user is created (visit /login?setup=true in a browser)`)
	}
	return a.PrintJSON(map[string]any{
		"status":         "ok",
		"config":         path,
		"setup_required": resp.SetupRequired,
	})
}

/* ---- browser login (loopback + PKCE) -------------------------------- */

// browserLogin implements the RFC 8252 native-app flow: listen on an
// ephemeral 127.0.0.1 port, send the browser to the server's authorize page,
// receive a one-time code on the loopback callback, and exchange it (with
// the PKCE verifier) for an API key.
func browserLogin(a *App, noBrowser bool) int {
	state, err := randToken(16)
	if err != nil {
		return a.Fail(err)
	}
	verifier, err := randToken(32)
	if err != nil {
		return a.Fail(err)
	}
	sum := sha256.Sum256([]byte(verifier))
	challenge := base64.RawURLEncoding.EncodeToString(sum[:])

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return a.Fail(fmt.Errorf("cannot open loopback listener: %w", err))
	}
	defer ln.Close()
	port := ln.Addr().(*net.TCPAddr).Port

	keyName := "lokictl"
	if host, err := os.Hostname(); err == nil && host != "" {
		keyName = "lokictl@" + host
	}
	authURL := fmt.Sprintf("%s/auth/cli/authorize?port=%d&state=%s&code_challenge=%s&name=%s",
		a.Client.Base, port, url.QueryEscape(state), url.QueryEscape(challenge), url.QueryEscape(keyName))

	if noBrowser {
		fmt.Fprintf(a.ErrOut, "Open this URL in your browser to authorize lokictl:\n\n  %s\n\n", authURL)
	} else {
		fmt.Fprintf(a.ErrOut, "Opening your browser to authorize lokictl…\nIf it does not open, visit:\n\n  %s\n\n", authURL)
		if err := openBrowser(authURL); err != nil {
			fmt.Fprintf(a.ErrOut, "(could not launch a browser: %v)\n", err)
		}
	}
	fmt.Fprintln(a.ErrOut, "Waiting for authorization…")

	code, err := waitForCLICallback(ln, state, 3*time.Minute)
	if err != nil {
		return a.Fail(err)
	}

	var resp struct {
		Status   string `json:"status"`
		Key      string `json:"key"`
		Username string `json:"username"`
		Name     string `json:"name"`
	}
	if err := a.Client.DoJSON("POST", "/auth/cli/token", map[string]string{
		"code":          code,
		"code_verifier": verifier,
	}, &resp); err != nil {
		return a.Fail(err)
	}
	if resp.Key == "" {
		return a.Fail(fmt.Errorf("server did not return an API key"))
	}
	path, err := saveLoginToken(a, resp.Key)
	if err != nil {
		return a.Fail(err)
	}
	return a.PrintJSON(map[string]any{
		"status":   "ok",
		"username": resp.Username,
		"key_name": resp.Name,
		"config":   path,
	})
}

func saveLoginToken(a *App, token string) (string, error) {
	cfg := loadCLIConfig()
	cfg.Server = a.Client.Base
	cfg.Token = token
	path, err := saveCLIConfig(cfg)
	if err != nil {
		return "", fmt.Errorf("login succeeded but saving config failed: %w", err)
	}
	a.Client.Token = token
	return path, nil
}

// randToken returns n random bytes as unpadded base64url (safe for both the
// state parameter and the PKCE verifier).
func randToken(n int) (string, error) {
	b := make([]byte, n)
	if _, err := rand.Read(b); err != nil {
		return "", err
	}
	return base64.RawURLEncoding.EncodeToString(b), nil
}

const cliCallbackPage = `<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>lokictl</title>
<style>body{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;
background:#101216;color:#d7dae0;font:15px/1.5 "Segoe UI",system-ui,sans-serif}
div{text-align:center}p{color:#9aa0ab}</style></head>
<body><div><h1>%s</h1><p>%s</p></div></body></html>`

// waitForCLICallback serves the loopback listener until the browser delivers
// the authorization code (or an error / the timeout).
func waitForCLICallback(ln net.Listener, state string, timeout time.Duration) (string, error) {
	type result struct {
		code string
		err  error
	}
	ch := make(chan result, 1)
	deliver := func(r result) {
		select {
		case ch <- r:
		default: // a second hit after the first result is decided — ignore
		}
	}

	srv := &http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/callback" {
			http.NotFound(w, r)
			return
		}
		q := r.URL.Query()
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		switch {
		case q.Get("state") != state:
			fmt.Fprintf(w, cliCallbackPage, "Login failed", "State mismatch — go back to the terminal and try again.")
			deliver(result{err: errors.New("state mismatch in browser callback")})
		case q.Get("error") != "":
			fmt.Fprintf(w, cliCallbackPage, "Login canceled", "You can close this tab.")
			deliver(result{err: fmt.Errorf("authorization was denied (%s)", q.Get("error"))})
		case q.Get("code") == "":
			fmt.Fprintf(w, cliCallbackPage, "Login failed", "No authorization code was returned.")
			deliver(result{err: errors.New("browser callback had no code")})
		default:
			fmt.Fprintf(w, cliCallbackPage, "✓ lokictl is logged in", "You can close this tab and return to the terminal.")
			deliver(result{code: q.Get("code")})
		}
	})}
	go func() { _ = srv.Serve(ln) }()
	defer func() {
		// Graceful shutdown so the success page finishes writing.
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		_ = srv.Shutdown(ctx)
	}()

	select {
	case r := <-ch:
		return r.code, r.err
	case <-time.After(timeout):
		return "", fmt.Errorf("timed out after %s waiting for the browser — for headless machines use: lokictl login --password <password>", timeout)
	}
}

// openBrowser launches the platform's default browser at url.
func openBrowser(u string) error {
	switch runtime.GOOS {
	case "windows":
		return exec.Command("rundll32", "url.dll,FileProtocolHandler", u).Start()
	case "darwin":
		return exec.Command("open", u).Start()
	default:
		return exec.Command("xdg-open", u).Start()
	}
}
