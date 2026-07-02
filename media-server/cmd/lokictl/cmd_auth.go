package main

import (
	"flag"
	"fmt"
	"io"
)

func init() {
	register(command{
		group: "login", args: "--password P [--username U]",
		summary: "Authenticate (POST /auth/login) and store the token in the CLI config",
		run:     cmdLogin,
	})
}

func cmdLogin(a *App, args []string) int {
	fs := flag.NewFlagSet("login", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	user := fs.String("username", "admin", "username")
	pass := fs.String("password", "", "password (required)")
	if err := fs.Parse(args); err != nil {
		return a.Usage(fs, err.Error())
	}
	if *pass == "" {
		return a.Usage(fs, "--password is required")
	}

	var resp struct {
		Status        string `json:"status"`
		Token         string `json:"token"`
		SetupRequired bool   `json:"setup_required"`
	}
	err := a.Client.DoJSON("POST", "/auth/login", map[string]string{
		"username": *user,
		"password": *pass,
	}, &resp)
	if err != nil {
		return a.Fail(err)
	}
	if resp.Token == "" {
		return a.Fail(fmt.Errorf("server did not return a token"))
	}

	cfg := loadCLIConfig()
	cfg.Server = a.Client.Base
	cfg.Token = resp.Token
	path, err := saveCLIConfig(cfg)
	if err != nil {
		return a.Fail(fmt.Errorf("login succeeded but saving config failed: %w", err))
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
