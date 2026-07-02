package main

import (
	"flag"
	"fmt"
	"io"
	"net/url"
	"strings"
)

func init() {
	register(command{group: "taxonomy", args: "[--category C] [categories]",
		summary: "Full taxonomy (GET /api/taxonomy), one category's tags, or the category list",
		run:     cmdTaxonomy})

	register(command{group: "tag", name: "create", args: "<label> --category C [--weight W]",
		summary: "Create a tag (POST /api/tags)", run: cmdTagCreate})
	register(command{group: "tag", name: "delete", args: "<label> --category C --yes",
		summary: "Delete a tag everywhere (DELETE /api/tags)", run: cmdTagDelete})
	register(command{group: "tag", name: "rename", args: "<old> <new>",
		summary: "Rename a tag (POST /api/tags/rename)", run: cmdTagRename})
	register(command{group: "tag", name: "move", args: "<label> --category C",
		summary: "Move a tag to another category (POST /api/tags/move)", run: cmdTagMove})
	register(command{group: "tag", name: "assign", args: "<media-path> <label> --category C [--timestamp S]",
		summary: "Tag a media item (POST /api/assignments)", run: cmdTagAssign})
	register(command{group: "tag", name: "unassign", args: "<media-path> <label>",
		summary: "Remove a tag from a media item (DELETE /api/assignments)", run: cmdTagUnassign})

	register(command{group: "category", name: "create", args: "<label>",
		summary: "Create a category (POST /api/categories)", run: categoryLabelCommand("POST", "/api/categories", false)})
	register(command{group: "category", name: "delete", args: "<label> --yes",
		summary: "Delete a category (DELETE /api/categories)", run: categoryLabelCommand("DELETE", "/api/categories", true)})
	register(command{group: "category", name: "rename", args: "<old> <new>",
		summary: "Rename a category (POST /api/categories/rename)", run: cmdCategoryRename})
}

func cmdTaxonomy(a *App, args []string) int {
	fs := flag.NewFlagSet("taxonomy", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	category := fs.String("category", "", "list only this category's tags")
	if len(args) > 0 && args[0] == "categories" {
		var out any
		if err := a.Client.DoJSON("GET", "/api/taxonomy/categories", nil, &out); err != nil {
			return a.Fail(err)
		}
		return a.PrintJSON(out)
	}
	if err := fs.Parse(args); err != nil {
		return a.Usage(fs, err.Error())
	}
	path := "/api/taxonomy"
	if *category != "" {
		path = "/api/taxonomy/tags?category=" + url.QueryEscape(*category)
	}
	var out any
	if err := a.Client.DoJSON("GET", path, nil, &out); err != nil {
		return a.Fail(err)
	}
	return a.PrintJSON(out)
}

// tagBody mirrors the server's tagRequest.
type tagBody struct {
	Label         string  `json:"label"`
	CategoryLabel string  `json:"categoryLabel"`
	Weight        float64 `json:"weight,omitempty"`
}

func parseLabelCategory(a *App, name string, args []string, needYes bool) (label, category string, weight float64, ok int) {
	fs := flag.NewFlagSet(name, flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	cat := fs.String("category", "", "category label (required)")
	w := fs.Float64("weight", 0, "tag weight")
	fs.Bool("yes", false, "confirm destructive action")
	if len(args) == 0 || strings.HasPrefix(args[0], "-") {
		return "", "", 0, a.Usage(fs, fmt.Sprintf("usage: lokictl %s <label> --category C", name))
	}
	label = args[0]
	if err := fs.Parse(args[1:]); err != nil {
		return "", "", 0, a.Usage(fs, err.Error())
	}
	if *cat == "" {
		return "", "", 0, a.Usage(fs, "--category is required")
	}
	if needYes && !hasYesFlag(args) {
		return "", "", 0, a.Usage(fs, fmt.Sprintf("this deletes tag %q from every media item — re-run with --yes to confirm", label))
	}
	return label, *cat, *w, -1
}

func cmdTagCreate(a *App, args []string) int {
	label, cat, weight, code := parseLabelCategory(a, "tag create", args, false)
	if code >= 0 {
		return code
	}
	if err := a.Client.DoJSON("POST", "/api/tags", tagBody{Label: label, CategoryLabel: cat, Weight: weight}, nil); err != nil {
		return a.Fail(err)
	}
	return a.PrintJSON(map[string]string{"status": "created", "label": label, "category": cat})
}

func cmdTagDelete(a *App, args []string) int {
	label, cat, _, code := parseLabelCategory(a, "tag delete", args, true)
	if code >= 0 {
		return code
	}
	if err := a.Client.DoJSON("DELETE", "/api/tags", tagBody{Label: label, CategoryLabel: cat}, nil); err != nil {
		return a.Fail(err)
	}
	return a.PrintJSON(map[string]string{"status": "deleted", "label": label, "category": cat})
}

func cmdTagRename(a *App, args []string) int {
	if len(args) != 2 {
		return a.Usage(nil, "usage: lokictl tag rename <old> <new>")
	}
	body := map[string]string{"label": args[0], "newLabel": args[1]}
	if err := a.Client.DoJSON("POST", "/api/tags/rename", body, nil); err != nil {
		return a.Fail(err)
	}
	return a.PrintJSON(map[string]string{"status": "renamed", "label": args[0], "newLabel": args[1]})
}

func cmdTagMove(a *App, args []string) int {
	label, cat, _, code := parseLabelCategory(a, "tag move", args, false)
	if code >= 0 {
		return code
	}
	body := map[string]string{"label": label, "categoryLabel": cat}
	if err := a.Client.DoJSON("POST", "/api/tags/move", body, nil); err != nil {
		return a.Fail(err)
	}
	return a.PrintJSON(map[string]string{"status": "moved", "label": label, "category": cat})
}

func cmdTagAssign(a *App, args []string) int {
	fs := flag.NewFlagSet("tag assign", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	cat := fs.String("category", "", "category label (required)")
	ts := fs.Float64("timestamp", 0, "timestamp in seconds (video tags)")
	if len(args) < 2 || strings.HasPrefix(args[0], "-") || strings.HasPrefix(args[1], "-") {
		return a.Usage(fs, "usage: lokictl tag assign <media-path> <label> --category C [--timestamp S]")
	}
	mediaPath, label := args[0], args[1]
	if err := fs.Parse(args[2:]); err != nil {
		return a.Usage(fs, err.Error())
	}
	if *cat == "" {
		return a.Usage(fs, "--category is required")
	}
	body := map[string]any{
		"mediaPath":     mediaPath,
		"tagLabel":      label,
		"categoryLabel": *cat,
	}
	if *ts != 0 {
		body["timeStamp"] = *ts
	}
	if err := a.Client.DoJSON("POST", "/api/assignments", body, nil); err != nil {
		return a.Fail(err)
	}
	return a.PrintJSON(map[string]string{"status": "assigned", "path": mediaPath, "label": label})
}

func cmdTagUnassign(a *App, args []string) int {
	if len(args) != 2 {
		return a.Usage(nil, "usage: lokictl tag unassign <media-path> <label>")
	}
	body := map[string]any{
		"mediaPath": args[0],
		"tag":       map[string]any{"tag_label": args[1]},
	}
	if err := a.Client.DoJSON("DELETE", "/api/assignments", body, nil); err != nil {
		return a.Fail(err)
	}
	return a.PrintJSON(map[string]string{"status": "unassigned", "path": args[0], "label": args[1]})
}

func categoryLabelCommand(method, path string, needYes bool) func(a *App, args []string) int {
	return func(a *App, args []string) int {
		var label string
		for _, arg := range args {
			if arg != "--yes" && arg != "-y" {
				label = arg
			}
		}
		if label == "" {
			return a.Usage(nil, "usage: lokictl category <create|delete> <label>")
		}
		if needYes && !hasYesFlag(args) {
			return a.Usage(nil, fmt.Sprintf("this deletes category %q and its tag links — re-run with --yes to confirm", label))
		}
		if err := a.Client.DoJSON(method, path, map[string]string{"label": label}, nil); err != nil {
			return a.Fail(err)
		}
		return a.PrintJSON(map[string]string{"status": "ok", "label": label})
	}
}

func cmdCategoryRename(a *App, args []string) int {
	if len(args) != 2 {
		return a.Usage(nil, "usage: lokictl category rename <old> <new>")
	}
	body := map[string]string{"label": args[0], "newLabel": args[1]}
	if err := a.Client.DoJSON("POST", "/api/categories/rename", body, nil); err != nil {
		return a.Fail(err)
	}
	return a.PrintJSON(map[string]string{"status": "renamed", "label": args[0], "newLabel": args[1]})
}
