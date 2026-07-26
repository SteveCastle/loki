package main

import (
	"fmt"
	"net"
	"net/http"
	"strings"
	"testing"
	"time"
)

func callbackListener(t *testing.T) (net.Listener, string) {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	return ln, fmt.Sprintf("http://%s/callback", ln.Addr())
}

func TestWaitForCLICallback_DeliversCode(t *testing.T) {
	ln, cb := callbackListener(t)
	type res struct {
		code string
		err  error
	}
	done := make(chan res, 1)
	go func() {
		code, err := waitForCLICallback(ln, "state-abc", 5*time.Second)
		done <- res{code, err}
	}()

	resp, err := http.Get(cb + "?state=state-abc&code=code-123")
	if err != nil {
		t.Fatal(err)
	}
	body := make([]byte, 4096)
	n, _ := resp.Body.Read(body)
	resp.Body.Close()
	if !strings.Contains(string(body[:n]), "logged in") {
		t.Errorf("success page missing confirmation text: %q", string(body[:n]))
	}

	r := <-done
	if r.err != nil || r.code != "code-123" {
		t.Fatalf("got code=%q err=%v", r.code, r.err)
	}
}

func TestWaitForCLICallback_RejectsStateMismatch(t *testing.T) {
	ln, cb := callbackListener(t)
	done := make(chan error, 1)
	go func() {
		_, err := waitForCLICallback(ln, "state-abc", 5*time.Second)
		done <- err
	}()

	resp, err := http.Get(cb + "?state=forged&code=code-123")
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()

	if err := <-done; err == nil || !strings.Contains(err.Error(), "state mismatch") {
		t.Fatalf("err = %v, want state mismatch", err)
	}
}

func TestWaitForCLICallback_DeniedByUser(t *testing.T) {
	ln, cb := callbackListener(t)
	done := make(chan error, 1)
	go func() {
		_, err := waitForCLICallback(ln, "state-abc", 5*time.Second)
		done <- err
	}()

	resp, err := http.Get(cb + "?state=state-abc&error=access_denied")
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()

	if err := <-done; err == nil || !strings.Contains(err.Error(), "denied") {
		t.Fatalf("err = %v, want denial", err)
	}
}
