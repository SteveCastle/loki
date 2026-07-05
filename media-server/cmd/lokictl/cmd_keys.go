package main

import (
	"flag"
	"fmt"
	"io"
	"strconv"
)

func init() {
	register(command{
		group: "key", name: "create", args: "--name N [--username U] [--save]",
		summary: "Create an API key (POST /auth/keys); --save stores it as this CLI's token",
		run:     cmdKeyCreate,
	})
	register(command{
		group: "key", name: "list",
		summary: "List API keys (GET /auth/keys)",
		run:     cmdKeyList,
	})
	register(command{
		group: "key", name: "revoke", args: "--id N",
		summary: "Revoke an API key (DELETE /auth/keys?id=N)",
		run:     cmdKeyRevoke,
	})
}

func cmdKeyCreate(a *App, args []string) int {
	fs := flag.NewFlagSet("key create", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	name := fs.String("name", "", "key name, e.g. \"lokictl\" (required)")
	user := fs.String("username", "", "key owner (default: the authenticated user)")
	save := fs.Bool("save", false, "store the new key as this CLI's token in the config file")
	if err := fs.Parse(args); err != nil {
		return a.Usage(fs, err.Error())
	}
	if *name == "" {
		return a.Usage(fs, "--name is required")
	}

	var resp struct {
		Status   string `json:"status"`
		Key      string `json:"key"`
		ID       int64  `json:"id"`
		Name     string `json:"name"`
		Username string `json:"username"`
		Prefix   string `json:"prefix"`
	}
	err := a.Client.DoJSON("POST", "/auth/keys", map[string]string{
		"name":     *name,
		"username": *user,
	}, &resp)
	if err != nil {
		return a.Fail(err)
	}
	if resp.Key == "" {
		return a.Fail(fmt.Errorf("server did not return a key"))
	}

	out := map[string]any{
		"status":   "ok",
		"key":      resp.Key, // shown once; the server stores only a hash
		"id":       resp.ID,
		"name":     resp.Name,
		"username": resp.Username,
		"prefix":   resp.Prefix,
	}
	if *save {
		cfg := loadCLIConfig()
		cfg.Server = a.Client.Base
		cfg.Token = resp.Key
		path, err := saveCLIConfig(cfg)
		if err != nil {
			return a.Fail(fmt.Errorf("key created but saving config failed (copy the key from this error's detail): %w", err))
		}
		out["config"] = path
	}
	return a.PrintJSON(out)
}

func cmdKeyList(a *App, args []string) int {
	if len(args) > 0 {
		return a.Usage(nil, "key list takes no arguments")
	}
	var resp struct {
		Keys []map[string]any `json:"keys"`
	}
	if err := a.Client.DoJSON("GET", "/auth/keys", nil, &resp); err != nil {
		return a.Fail(err)
	}
	if resp.Keys == nil {
		resp.Keys = []map[string]any{}
	}
	return a.PrintJSON(resp.Keys)
}

func cmdKeyRevoke(a *App, args []string) int {
	fs := flag.NewFlagSet("key revoke", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	id := fs.Int64("id", 0, "key id from \"lokictl key list\" (required)")
	if err := fs.Parse(args); err != nil {
		return a.Usage(fs, err.Error())
	}
	if *id == 0 {
		return a.Usage(fs, "--id is required")
	}

	var resp struct {
		Status string `json:"status"`
	}
	if err := a.Client.DoJSON("DELETE", "/auth/keys?id="+strconv.FormatInt(*id, 10), nil, &resp); err != nil {
		return a.Fail(err)
	}
	return a.PrintJSON(map[string]any{"status": resp.Status, "id": *id})
}
