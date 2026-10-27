package main

import (
	"flag"
	"fmt"
	"io"
	"net/url"
	"os"
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
	register(command{group: "tag", name: "assign-bulk", args: "<label> --category C [--timestamp S] [paths... | --stdin]",
		summary: "Tag many media items at once (POST /api/assignments)", run: cmdTagAssignBulk})
	register(command{group: "tag", name: "unassign-bulk", args: "<label> [paths... | --stdin] --yes",
		summary: "Remove a tag from many media items (DELETE /api/assignments)", run: cmdTagUnassignBulk})
	register(command{group: "tag", name: "list", args: "[--category C]",
		summary: "All tags with usage counts (GET /api/tags/list)", run: cmdTagList})
	register(command{group: "tag", name: "count", args: "<label>",
		summary: "Distinct media count for one tag (POST /api/tags/count)", run: cmdTagCount})
	register(command{group: "tag", name: "weight", args: "<label> --weight W",
		summary: "Set a tag's sort weight (POST /api/tags/weight)", run: cmdTagWeight})
	register(command{group: "tag", name: "has", args: "<media-path> <label> --category C",
		summary: "Check whether a media item has a tag (GET /media/has-tag)", run: cmdTagHas})
	register(command{group: "tag", name: "timestamp", args: "<media-path> <label> (--from S --to S | --remove --at S)",
		summary: "Edit or remove a video tag's timestamp (PUT/DELETE /api/tags/timestamp)", run: cmdTagTimestamp})
	register(command{group: "tag", name: "assignment-weight", args: "<media-path> <label> --weight W [--timestamp S]",
		summary: "Set the weight of one media↔tag assignment (POST /api/assignments/weight)", run: cmdTagAssignmentWeight})

	register(command{group: "category", name: "create", args: "<label>",
		summary: "Create a category (POST /api/categories)", run: categoryLabelCommand("POST", "/api/categories", false)})
	register(command{group: "category", name: "delete", args: "<label> [--dry-run] --yes [--allow-nonempty]",
		summary: "Delete a category, its tags, groups and their assignments (GET /api/categories/impact, then DELETE /api/categories); refuses to remove assignments without --allow-nonempty", run: cmdCategoryDelete})
	register(command{group: "category", name: "rename", args: "<old> <new>",
		summary: "Rename a category (POST /api/categories/rename)", run: cmdCategoryRename})
	register(command{group: "category", name: "count", args: "<label>",
		summary: "Distinct media count for one category (GET /api/taxonomy/category-count)", run: cmdCategoryCount})
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
	var removed struct {
		Assignments *int64 `json:"assignments"`
	}
	if err := a.Client.DoJSON("DELETE", "/api/tags", tagBody{Label: label, CategoryLabel: cat}, &removed); err != nil {
		return a.Fail(err)
	}
	return a.PrintJSON(map[string]any{"status": "deleted", "label": label, "category": cat, "assignmentsRemoved": removed.Assignments})
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

// bulkPaths resolves the media paths for a bulk command: positional args, or
// newline-separated stdin when --stdin is passed. A leading UTF-8 BOM is
// dropped: Windows PowerShell prepends one when piping to a native command,
// which would otherwise become part of the first path.
func bulkPaths(positional []string, useStdin bool) ([]string, error) {
	if useStdin {
		b, err := io.ReadAll(os.Stdin)
		if err != nil {
			return nil, fmt.Errorf("reading paths from stdin: %w", err)
		}
		var paths []string
		for _, line := range strings.Split(strings.TrimPrefix(string(b), string(rune(0xFEFF))), "\n") {
			if line = strings.TrimRight(strings.TrimSpace(line), "\r"); line != "" {
				paths = append(paths, line)
			}
		}
		return paths, nil
	}
	return positional, nil
}

func cmdTagAssignBulk(a *App, args []string) int {
	fs := flag.NewFlagSet("tag assign-bulk", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	cat := fs.String("category", "", "category label (required)")
	ts := fs.Float64("timestamp", 0, "timestamp in seconds (video tags)")
	stdin := fs.Bool("stdin", false, "read newline-separated paths from stdin")
	if len(args) == 0 || strings.HasPrefix(args[0], "-") {
		return a.Usage(fs, "usage: lokictl tag assign-bulk <label> --category C [--timestamp S] [paths... | --stdin]")
	}
	label := args[0]
	if err := fs.Parse(args[1:]); err != nil {
		return a.Usage(fs, err.Error())
	}
	if *cat == "" {
		return a.Usage(fs, "--category is required")
	}
	paths, err := bulkPaths(fs.Args(), *stdin)
	if err != nil {
		return a.Fail(err)
	}
	if len(paths) == 0 {
		return a.Usage(fs, "no paths — pass them as arguments or via --stdin")
	}
	body := map[string]any{
		"mediaPaths":    paths,
		"tagLabel":      label,
		"categoryLabel": *cat,
	}
	if *ts != 0 {
		body["timeStamp"] = *ts
	}
	if err := a.Client.DoJSON("POST", "/api/assignments", body, nil); err != nil {
		return a.Fail(err)
	}
	return a.PrintJSON(map[string]any{"status": "assigned", "label": label, "paths": len(paths)})
}

func cmdTagUnassignBulk(a *App, args []string) int {
	fs := flag.NewFlagSet("tag unassign-bulk", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	stdin := fs.Bool("stdin", false, "read newline-separated paths from stdin")
	fs.Bool("yes", false, "confirm destructive action")
	if len(args) == 0 || strings.HasPrefix(args[0], "-") {
		return a.Usage(fs, "usage: lokictl tag unassign-bulk <label> [paths... | --stdin] --yes")
	}
	label := args[0]
	if err := fs.Parse(args[1:]); err != nil {
		return a.Usage(fs, err.Error())
	}
	paths, err := bulkPaths(fs.Args(), *stdin)
	if err != nil {
		return a.Fail(err)
	}
	if len(paths) == 0 {
		return a.Usage(fs, "no paths — pass them as arguments or via --stdin")
	}
	if !hasYesFlag(args) {
		return a.Usage(fs, fmt.Sprintf("this removes tag %q from %d media items — re-run with --yes to confirm", label, len(paths)))
	}
	body := map[string]any{
		"mediaPaths": paths,
		"tag":        map[string]any{"tag_label": label},
	}
	if err := a.Client.DoJSON("DELETE", "/api/assignments", body, nil); err != nil {
		return a.Fail(err)
	}
	return a.PrintJSON(map[string]any{"status": "unassigned", "label": label, "paths": len(paths)})
}

func cmdTagList(a *App, args []string) int {
	fs := flag.NewFlagSet("tag list", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	category := fs.String("category", "", "only this category's tags")
	if err := fs.Parse(args); err != nil {
		return a.Usage(fs, err.Error())
	}
	path := "/api/tags/list"
	if *category != "" {
		path += "?category=" + url.QueryEscape(*category)
	}
	var resp struct {
		Tags []map[string]any `json:"tags"`
	}
	if err := a.Client.DoJSON("GET", path, nil, &resp); err != nil {
		return a.Fail(err)
	}
	return a.PrintJSON(resp.Tags)
}

func cmdTagCount(a *App, args []string) int {
	if len(args) != 1 {
		return a.Usage(nil, "usage: lokictl tag count <label>")
	}
	var out any
	if err := a.Client.DoJSON("POST", "/api/tags/count", map[string]string{"label": args[0]}, &out); err != nil {
		return a.Fail(err)
	}
	return a.PrintJSON(out)
}

func cmdTagWeight(a *App, args []string) int {
	fs := flag.NewFlagSet("tag weight", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	weight := fs.Float64("weight", 0, "the weight to set (required)")
	if len(args) == 0 || strings.HasPrefix(args[0], "-") {
		return a.Usage(fs, "usage: lokictl tag weight <label> --weight W")
	}
	label := args[0]
	if err := fs.Parse(args[1:]); err != nil {
		return a.Usage(fs, err.Error())
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
	body := map[string]any{"label": label, "weight": *weight}
	if err := a.Client.DoJSON("POST", "/api/tags/weight", body, nil); err != nil {
		return a.Fail(err)
	}
	return a.PrintJSON(map[string]any{"status": "ok", "label": label, "weight": *weight})
}

func cmdTagHas(a *App, args []string) int {
	fs := flag.NewFlagSet("tag has", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	cat := fs.String("category", "", "category label (required)")
	if len(args) < 2 || strings.HasPrefix(args[0], "-") || strings.HasPrefix(args[1], "-") {
		return a.Usage(fs, "usage: lokictl tag has <media-path> <label> --category C")
	}
	mediaPath, label := args[0], args[1]
	if err := fs.Parse(args[2:]); err != nil {
		return a.Usage(fs, err.Error())
	}
	if *cat == "" {
		return a.Usage(fs, "--category is required")
	}
	q := url.Values{
		"media_path":     {mediaPath},
		"tag_label":      {label},
		"category_label": {*cat},
	}
	var out any
	if err := a.Client.DoJSON("GET", "/media/has-tag?"+q.Encode(), nil, &out); err != nil {
		return a.Fail(err)
	}
	return a.PrintJSON(out)
}

func cmdTagTimestamp(a *App, args []string) int {
	fs := flag.NewFlagSet("tag timestamp", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	from := fs.Float64("from", 0, "the existing timestamp (seconds)")
	to := fs.Float64("to", 0, "the new timestamp (seconds)")
	remove := fs.Bool("remove", false, "remove the timestamp instead of moving it")
	at := fs.Float64("at", 0, "the timestamp to remove (seconds, with --remove)")
	if len(args) < 2 || strings.HasPrefix(args[0], "-") || strings.HasPrefix(args[1], "-") {
		return a.Usage(fs, "usage: lokictl tag timestamp <media-path> <label> (--from S --to S | --remove --at S)")
	}
	mediaPath, label := args[0], args[1]
	if err := fs.Parse(args[2:]); err != nil {
		return a.Usage(fs, err.Error())
	}

	if *remove {
		body := map[string]any{"mediaPath": mediaPath, "tagLabel": label, "timestamp": *at}
		if err := a.Client.DoJSON("DELETE", "/api/tags/timestamp", body, nil); err != nil {
			return a.Fail(err)
		}
		return a.PrintJSON(map[string]any{"status": "removed", "path": mediaPath, "label": label, "timestamp": *at})
	}
	toSet := false
	fs.Visit(func(f *flag.Flag) {
		if f.Name == "to" {
			toSet = true
		}
	})
	if !toSet {
		return a.Usage(fs, "pass --from S --to S to move a timestamp, or --remove --at S to remove one")
	}
	body := map[string]any{"mediaPath": mediaPath, "tagLabel": label, "oldTimestamp": *from, "newTimestamp": *to}
	if err := a.Client.DoJSON("PUT", "/api/tags/timestamp", body, nil); err != nil {
		return a.Fail(err)
	}
	return a.PrintJSON(map[string]any{"status": "updated", "path": mediaPath, "label": label, "timestamp": *to})
}

func cmdTagAssignmentWeight(a *App, args []string) int {
	fs := flag.NewFlagSet("tag assignment-weight", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	weight := fs.Float64("weight", 0, "the weight to set (required)")
	ts := fs.Float64("timestamp", 0, "only the assignment at this timestamp")
	if len(args) < 2 || strings.HasPrefix(args[0], "-") || strings.HasPrefix(args[1], "-") {
		return a.Usage(fs, "usage: lokictl tag assignment-weight <media-path> <label> --weight W [--timestamp S]")
	}
	mediaPath, label := args[0], args[1]
	if err := fs.Parse(args[2:]); err != nil {
		return a.Usage(fs, err.Error())
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
	body := map[string]any{
		"mediaPath":      mediaPath,
		"tagLabel":       label,
		"weight":         *weight,
		"mediaTimeStamp": *ts,
	}
	if err := a.Client.DoJSON("POST", "/api/assignments/weight", body, nil); err != nil {
		return a.Fail(err)
	}
	return a.PrintJSON(map[string]any{"status": "ok", "path": mediaPath, "label": label, "weight": *weight})
}

func cmdCategoryCount(a *App, args []string) int {
	if len(args) != 1 {
		return a.Usage(nil, "usage: lokictl category count <label>")
	}
	var out any
	if err := a.Client.DoJSON("GET", "/api/taxonomy/category-count?category="+url.QueryEscape(args[0]), nil, &out); err != nil {
		return a.Fail(err)
	}
	return a.PrintJSON(map[string]any{"category": args[0], "count": out})
}

// categoryImpact mirrors the server's GET /api/categories/impact response.
type categoryImpact struct {
	Label       string `json:"label"`
	Exists      bool   `json:"exists"`
	Tags        int64  `json:"tags"`
	Groups      int64  `json:"groups"`
	Assignments int64  `json:"assignments"`
	Media       int64  `json:"media"`
}

// cmdCategoryDelete always previews first. A category that looks empty can
// still own assignments, so removing any needs --allow-nonempty on top of
// --yes; a server that can't preview is refused rather than trusted.
func cmdCategoryDelete(a *App, args []string) int {
	fs := flag.NewFlagSet("category delete", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	dryRun := fs.Bool("dry-run", false, "only report what would be removed")
	allowNonEmpty := fs.Bool("allow-nonempty", false, "also remove the category's tag assignments")
	fs.Bool("yes", false, "confirm destructive action")
	if len(args) == 0 || strings.HasPrefix(args[0], "-") {
		return a.Usage(fs, "usage: lokictl category delete <label> [--dry-run] --yes [--allow-nonempty]")
	}
	label := args[0]
	if err := fs.Parse(args[1:]); err != nil {
		return a.Usage(fs, err.Error())
	}
	var im categoryImpact
	if err := a.Client.DoJSON("GET", "/api/categories/impact?label="+url.QueryEscape(label), nil, &im); err != nil {
		return a.Fail(fmt.Errorf("previewing the delete failed, so nothing was deleted (servers without GET /api/categories/impact can't be previewed — restart on a current build): %w", err))
	}
	if !im.Exists {
		return a.Fail(fmt.Errorf("category %q does not exist", label))
	}
	if *dryRun {
		return a.PrintJSON(map[string]any{"status": "dry-run", "wouldRemove": im})
	}
	summary := fmt.Sprintf("category %q: %d tags, %d groups, %d assignments on %d media", label, im.Tags, im.Groups, im.Assignments, im.Media)
	if !hasYesFlag(args) {
		return a.Usage(fs, "this deletes "+summary+" — re-run with --yes to confirm")
	}
	if im.Assignments > 0 && !*allowNonEmpty {
		return a.Usage(fs, "refusing: deleting "+summary+" would untag media — pass --allow-nonempty as well if that's intended, or move its tags out first")
	}
	var removed categoryImpact
	if err := a.Client.DoJSON("DELETE", "/api/categories", map[string]string{"label": label}, &removed); err != nil {
		return a.Fail(err)
	}
	return a.PrintJSON(map[string]any{"status": "deleted", "removed": removed})
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
