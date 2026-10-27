package main

import (
	"flag"
	"fmt"
	"io"
	"net/url"
	"strings"
)

// Tag groups are a visual level between a category and its tags (e.g. a
// "TV Shows" category with a "Dirty Pair" group holding per-season tags).
// They never change what a tag query matches.

func init() {
	register(command{group: "group", name: "list", args: "--category C",
		summary: "A category's tag groups in order (GET /api/taxonomy/groups)", run: cmdGroupList})
	register(command{group: "group", name: "create", args: "<label> --category C",
		summary: "Create a tag group at the end of the category's list (POST /api/tag-groups)", run: cmdGroupCreate})
	register(command{group: "group", name: "rename", args: "<old> <new> --category C",
		summary: "Rename a tag group, carrying its tags (POST /api/tag-groups/rename)", run: cmdGroupRename})
	register(command{group: "group", name: "delete", args: "<label> --category C --yes",
		summary: "Delete a tag group; its tags stay, ungrouped (DELETE /api/tag-groups)", run: cmdGroupDelete})
	register(command{group: "group", name: "weight", args: "<label> --category C --weight W",
		summary: "Set a tag group's sort weight (POST /api/tag-groups/weight)", run: cmdGroupWeight})
	register(command{group: "group", name: "hide", args: "<label> --category C",
		summary: "Hide a tag group from the taxonomy panel (PUT /api/tag-groups/hidden)", run: cmdGroupHide})
	register(command{group: "group", name: "show", args: "<label> --category C",
		summary: "Unhide a tag group (PUT /api/tag-groups/hidden)", run: cmdGroupShow})
	register(command{group: "group", name: "view", args: "<label> card|list|auto --category C",
		summary: "Set a tag group's display; auto follows the category (PUT /api/tag-groups/view-mode)", run: cmdGroupView})
	register(command{group: "group", name: "add", args: "<group> --category C [tags... | --stdin]",
		summary: "Put tags into a group, creating it if needed (POST /api/tag-groups + PUT /api/tags/group)", run: cmdGroupAdd})
	register(command{group: "group", name: "remove", args: "[tags... | --stdin]",
		summary: "Ungroup tags (PUT /api/tags/group with an empty groupLabel)", run: cmdGroupRemove})

	register(command{group: "category", name: "grouping", args: "<label> on|off",
		summary: "Show or hide a category's tag groups (POST /api/categories/tag-grouping)", run: cmdCategoryGrouping})
}

// groupBody mirrors the server's tagGroupRequest.
type groupBody struct {
	CategoryLabel string  `json:"categoryLabel"`
	Label         string  `json:"label"`
	NewLabel      string  `json:"newLabel,omitempty"`
	Weight        float64 `json:"weight,omitempty"`
	Hidden        bool    `json:"hidden,omitempty"`
	ViewMode      string  `json:"viewMode,omitempty"`
}

// parseGroupArgs splits nPositional leading labels from the flags; every
// group command needs --category.
func parseGroupArgs(a *App, name, usage string, nPositional int, args []string, setup func(fs *flag.FlagSet)) (pos []string, category string, fs *flag.FlagSet, code int) {
	fs = flag.NewFlagSet(name, flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	cat := fs.String("category", "", "category label (required)")
	fs.Bool("yes", false, "confirm destructive action")
	if setup != nil {
		setup(fs)
	}
	if len(args) < nPositional {
		return nil, "", fs, a.Usage(fs, "usage: lokictl "+name+" "+usage)
	}
	for _, p := range args[:nPositional] {
		if strings.HasPrefix(p, "-") {
			return nil, "", fs, a.Usage(fs, "usage: lokictl "+name+" "+usage)
		}
	}
	if err := fs.Parse(args[nPositional:]); err != nil {
		return nil, "", fs, a.Usage(fs, err.Error())
	}
	if *cat == "" {
		return nil, "", fs, a.Usage(fs, "--category is required")
	}
	return args[:nPositional], *cat, fs, -1
}

func cmdGroupList(a *App, args []string) int {
	_, cat, _, code := parseGroupArgs(a, "group list", "--category C", 0, args, nil)
	if code >= 0 {
		return code
	}
	var out any
	if err := a.Client.DoJSON("GET", "/api/taxonomy/groups?category="+url.QueryEscape(cat), nil, &out); err != nil {
		return a.Fail(err)
	}
	return a.PrintJSON(out)
}

func cmdGroupCreate(a *App, args []string) int {
	pos, cat, _, code := parseGroupArgs(a, "group create", "<label> --category C", 1, args, nil)
	if code >= 0 {
		return code
	}
	if err := a.Client.DoJSON("POST", "/api/tag-groups", groupBody{CategoryLabel: cat, Label: pos[0]}, nil); err != nil {
		return a.Fail(err)
	}
	return a.PrintJSON(map[string]string{"status": "created", "label": pos[0], "category": cat})
}

func cmdGroupRename(a *App, args []string) int {
	pos, cat, _, code := parseGroupArgs(a, "group rename", "<old> <new> --category C", 2, args, nil)
	if code >= 0 {
		return code
	}
	body := groupBody{CategoryLabel: cat, Label: pos[0], NewLabel: pos[1]}
	if err := a.Client.DoJSON("POST", "/api/tag-groups/rename", body, nil); err != nil {
		return a.Fail(err)
	}
	return a.PrintJSON(map[string]string{"status": "renamed", "label": pos[0], "newLabel": pos[1], "category": cat})
}

func cmdGroupDelete(a *App, args []string) int {
	pos, cat, fs, code := parseGroupArgs(a, "group delete", "<label> --category C --yes", 1, args, nil)
	if code >= 0 {
		return code
	}
	if !hasYesFlag(args) {
		return a.Usage(fs, fmt.Sprintf("this deletes group %q (its tags stay, ungrouped) — re-run with --yes to confirm", pos[0]))
	}
	if err := a.Client.DoJSON("DELETE", "/api/tag-groups", groupBody{CategoryLabel: cat, Label: pos[0]}, nil); err != nil {
		return a.Fail(err)
	}
	return a.PrintJSON(map[string]string{"status": "deleted", "label": pos[0], "category": cat})
}

func cmdGroupWeight(a *App, args []string) int {
	var weight *float64
	pos, cat, fs, code := parseGroupArgs(a, "group weight", "<label> --category C --weight W", 1, args, func(fs *flag.FlagSet) {
		weight = fs.Float64("weight", 0, "the weight to set (required)")
	})
	if code >= 0 {
		return code
	}
	set := false
	fs.Visit(func(f *flag.Flag) {
		if f.Name == "weight" {
			set = true
		}
	})
	if !set {
		return a.Usage(fs, "--weight is required")
	}
	body := groupBody{CategoryLabel: cat, Label: pos[0], Weight: *weight}
	if err := a.Client.DoJSON("POST", "/api/tag-groups/weight", body, nil); err != nil {
		return a.Fail(err)
	}
	return a.PrintJSON(map[string]any{"status": "ok", "label": pos[0], "category": cat, "weight": *weight})
}

func cmdGroupHide(a *App, args []string) int { return setGroupHidden(a, "group hide", args, true) }
func cmdGroupShow(a *App, args []string) int { return setGroupHidden(a, "group show", args, false) }

func setGroupHidden(a *App, name string, args []string, hidden bool) int {
	pos, cat, _, code := parseGroupArgs(a, name, "<label> --category C", 1, args, nil)
	if code >= 0 {
		return code
	}
	body := groupBody{CategoryLabel: cat, Label: pos[0], Hidden: hidden}
	if err := a.Client.DoJSON("PUT", "/api/tag-groups/hidden", body, nil); err != nil {
		return a.Fail(err)
	}
	return a.PrintJSON(map[string]any{"status": "ok", "label": pos[0], "category": cat, "hidden": hidden})
}

func cmdGroupView(a *App, args []string) int {
	pos, cat, fs, code := parseGroupArgs(a, "group view", "<label> card|list|auto --category C", 2, args, nil)
	if code >= 0 {
		return code
	}
	mode := pos[1]
	if mode != "card" && mode != "list" && mode != "auto" {
		return a.Usage(fs, "view mode must be card, list or auto")
	}
	body := groupBody{CategoryLabel: cat, Label: pos[0]}
	if mode != "auto" {
		body.ViewMode = mode
	}
	if err := a.Client.DoJSON("PUT", "/api/tag-groups/view-mode", body, nil); err != nil {
		return a.Fail(err)
	}
	return a.PrintJSON(map[string]string{"status": "ok", "label": pos[0], "category": cat, "viewMode": mode})
}

func cmdGroupAdd(a *App, args []string) int {
	var stdin *bool
	pos, cat, fs, code := parseGroupArgs(a, "group add", "<group> --category C [tags... | --stdin]", 1, args, func(fs *flag.FlagSet) {
		stdin = fs.Bool("stdin", false, "read newline-separated tag labels from stdin")
	})
	if code >= 0 {
		return code
	}
	tags, err := bulkPaths(fs.Args(), *stdin)
	if err != nil {
		return a.Fail(err)
	}
	if len(tags) == 0 {
		return a.Usage(fs, "no tags — pass them as arguments or via --stdin")
	}
	// Create explicitly so the group lands in --category even on servers
	// whose PUT /api/tags/group doesn't create missing groups.
	if err := a.Client.DoJSON("POST", "/api/tag-groups", groupBody{CategoryLabel: cat, Label: pos[0]}, nil); err != nil {
		return a.Fail(err)
	}
	for _, tag := range tags {
		if err := a.Client.DoJSON("PUT", "/api/tags/group", map[string]string{"label": tag, "groupLabel": pos[0]}, nil); err != nil {
			return a.Fail(fmt.Errorf("grouping tag %q: %w", tag, err))
		}
	}
	return a.PrintJSON(map[string]any{"status": "grouped", "group": pos[0], "category": cat, "tags": len(tags)})
}

func cmdGroupRemove(a *App, args []string) int {
	fs := flag.NewFlagSet("group remove", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	stdin := fs.Bool("stdin", false, "read newline-separated tag labels from stdin")
	if err := fs.Parse(args); err != nil {
		return a.Usage(fs, err.Error())
	}
	tags, err := bulkPaths(fs.Args(), *stdin)
	if err != nil {
		return a.Fail(err)
	}
	if len(tags) == 0 {
		return a.Usage(fs, "usage: lokictl group remove [tags... | --stdin]")
	}
	for _, tag := range tags {
		if err := a.Client.DoJSON("PUT", "/api/tags/group", map[string]string{"label": tag, "groupLabel": ""}, nil); err != nil {
			return a.Fail(fmt.Errorf("ungrouping tag %q: %w", tag, err))
		}
	}
	return a.PrintJSON(map[string]any{"status": "ungrouped", "tags": len(tags)})
}

func cmdCategoryGrouping(a *App, args []string) int {
	if len(args) != 2 || (args[1] != "on" && args[1] != "off") {
		return a.Usage(nil, "usage: lokictl category grouping <label> on|off")
	}
	body := map[string]string{"label": args[0], "mode": args[1]}
	if err := a.Client.DoJSON("POST", "/api/categories/tag-grouping", body, nil); err != nil {
		return a.Fail(err)
	}
	return a.PrintJSON(map[string]string{"status": "ok", "label": args[0], "tagGrouping": args[1]})
}
