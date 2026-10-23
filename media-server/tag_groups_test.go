package main

import (
	"encoding/json"
	"net/http"
	"testing"

	"github.com/stevecastle/shrike/media"
)

// tagGroupDeps starts from the PRE-groups schema (setupTestDB has no
// tag_group table, tag.group_label or category.tag_grouping) and migrates it,
// the way an existing library is upgraded on server start.
func tagGroupDeps(t *testing.T) *Dependencies {
	t.Helper()
	deps := testDeps(t)
	if err := media.InitializeSchema(deps.DB); err != nil {
		t.Fatalf("migrate: %v", err)
	}
	return deps
}

type groupedTag struct {
	Label    string  `json:"label"`
	Category string  `json:"category"`
	Group    *string `json:"group"`
}

func categoryTags(t *testing.T, deps *Dependencies, category string) map[string]groupedTag {
	t.Helper()
	rr := getJSON(lokiTaxonomyTagsHandler(deps), "/api/taxonomy/tags?category="+category)
	if rr.Code != http.StatusOK {
		t.Fatalf("category tags: %d %s", rr.Code, rr.Body.String())
	}
	var tags []groupedTag
	if err := json.Unmarshal(rr.Body.Bytes(), &tags); err != nil {
		t.Fatalf("decode: %v", err)
	}
	byLabel := map[string]groupedTag{}
	for _, tg := range tags {
		byLabel[tg.Label] = tg
	}
	return byLabel
}

func groupLabels(t *testing.T, deps *Dependencies, category string) []string {
	t.Helper()
	rr := getJSON(lokiTagGroupsHandler(deps), "/api/taxonomy/groups?category="+category)
	if rr.Code != http.StatusOK {
		t.Fatalf("groups: %d %s", rr.Code, rr.Body.String())
	}
	var groups []struct {
		Label string `json:"label"`
	}
	if err := json.Unmarshal(rr.Body.Bytes(), &groups); err != nil {
		t.Fatalf("decode: %v", err)
	}
	labels := []string{}
	for _, g := range groups {
		labels = append(labels, g.Label)
	}
	return labels
}

func groupOf(tg groupedTag) string {
	if tg.Group == nil {
		return ""
	}
	return *tg.Group
}

func mustOK(t *testing.T, what string, code int, body string) {
	t.Helper()
	if code != http.StatusOK {
		t.Fatalf("%s: expected 200, got %d: %s", what, code, body)
	}
}

// A library with no groups: every tag is ungrouped, grouping reads "on", and
// nothing else about the payloads changes.
func TestTagGroups_UpgradedLibraryHasNoGroups(t *testing.T) {
	deps := tagGroupDeps(t)

	rr := getJSON(lokiCategoriesHandler(deps), "/api/taxonomy/categories")
	mustOK(t, "categories", rr.Code, rr.Body.String())
	var cats []struct {
		Label       string `json:"label"`
		TagGrouping string `json:"tagGrouping"`
	}
	json.Unmarshal(rr.Body.Bytes(), &cats)
	// InitializeSchema may seed system categories; every one reads "on".
	if len(cats) < 2 {
		t.Fatalf("expected the 2 seeded categories, got %d", len(cats))
	}
	for _, c := range cats {
		if c.TagGrouping != "on" {
			t.Errorf("%s: tagGrouping = %q, want on", c.Label, c.TagGrouping)
		}
	}

	tags := categoryTags(t, deps, "Subject")
	if len(tags) != 2 {
		t.Fatalf("expected 2 Subject tags, got %d", len(tags))
	}
	for _, tg := range tags {
		if tg.Group != nil {
			t.Errorf("%s: expected no group, got %q", tg.Label, *tg.Group)
		}
	}
	if got := groupLabels(t, deps, "Subject"); len(got) != 0 {
		t.Errorf("expected no groups, got %v", got)
	}
}

func TestTagGroups_Lifecycle(t *testing.T) {
	deps := tagGroupDeps(t)

	for _, g := range []string{"Places", "People"} {
		rr := postJSON(lokiCreateTagGroupHandler(deps), tagGroupRequest{CategoryLabel: "Subject", Label: g})
		mustOK(t, "create "+g, rr.Code, rr.Body.String())
	}
	// Creation order is list order; a duplicate is a no-op.
	postJSON(lokiCreateTagGroupHandler(deps), tagGroupRequest{CategoryLabel: "Subject", Label: "Places"})
	if got := groupLabels(t, deps, "Subject"); len(got) != 2 || got[0] != "Places" || got[1] != "People" {
		t.Fatalf("groups = %v, want [Places People]", got)
	}

	// Reorder: People ahead of Places.
	rr := postJSON(lokiUpdateTagGroupWeightHandler(deps), tagGroupRequest{CategoryLabel: "Subject", Label: "People", Weight: 0.5})
	mustOK(t, "weight", rr.Code, rr.Body.String())
	if got := groupLabels(t, deps, "Subject"); got[0] != "People" {
		t.Fatalf("groups after reorder = %v, want People first", got)
	}

	rr = postJSON(lokiSetTagGroupHandler(deps), map[string]string{"label": "sunset", "groupLabel": "Places"})
	mustOK(t, "set group", rr.Code, rr.Body.String())
	if g := groupOf(categoryTags(t, deps, "Subject")["sunset"]); g != "Places" {
		t.Fatalf("sunset group = %q, want Places", g)
	}

	// Grouping is visual only: the tag query is unchanged.
	mr := postJSON(lokiMediaHandler(deps), mediaRequest{Tags: []string{"sunset"}, Mode: "EXCLUSIVE"})
	var items []map[string]any
	json.Unmarshal(mr.Body.Bytes(), &items)
	if len(items) != 1 || items[0]["path"] != "/photos/a.jpg" {
		t.Fatalf("tag query changed by grouping: %v", items)
	}

	// Renaming a group carries its tags.
	rr = postJSON(lokiRenameTagGroupHandler(deps), tagGroupRequest{CategoryLabel: "Subject", Label: "Places", NewLabel: "Scenes"})
	mustOK(t, "rename group", rr.Code, rr.Body.String())
	if g := groupOf(categoryTags(t, deps, "Subject")["sunset"]); g != "Scenes" {
		t.Fatalf("sunset group after rename = %q, want Scenes", g)
	}

	// Toggle grouping off and back on.
	rr = postJSON(lokiUpdateCategoryTagGroupingHandler(deps), map[string]string{"label": "Subject", "mode": "off"})
	mustOK(t, "grouping off", rr.Code, rr.Body.String())
	var mode string
	deps.DB.QueryRow(`SELECT tag_grouping FROM category WHERE label = 'Subject'`).Scan(&mode)
	if mode != "off" {
		t.Fatalf("tag_grouping = %q, want off", mode)
	}

	// Renaming the category keeps its groups.
	rr = postJSON(lokiRenameCategoryHandler(deps), renameRequest{Label: "Subject", NewLabel: "Topic"})
	mustOK(t, "rename category", rr.Code, rr.Body.String())
	if got := groupLabels(t, deps, "Topic"); len(got) != 2 {
		t.Fatalf("groups after category rename = %v", got)
	}
	if g := groupOf(categoryTags(t, deps, "Topic")["sunset"]); g != "Scenes" {
		t.Fatalf("sunset group after category rename = %q", g)
	}

	// Deleting a group ungroups its tags but keeps them.
	postJSON(lokiSetTagGroupHandler(deps), map[string]string{"label": "portrait", "groupLabel": "People"})
	rr = postJSON(lokiDeleteTagGroupHandler(deps), tagGroupRequest{CategoryLabel: "Topic", Label: "People"})
	mustOK(t, "delete group", rr.Code, rr.Body.String())
	tags := categoryTags(t, deps, "Topic")
	if _, ok := tags["portrait"]; !ok {
		t.Fatal("deleting a group deleted its tag")
	}
	if g := groupOf(tags["portrait"]); g != "" {
		t.Fatalf("portrait still grouped: %q", g)
	}

	// An empty groupLabel ungroups.
	postJSON(lokiSetTagGroupHandler(deps), map[string]string{"label": "portrait", "groupLabel": "Scenes"})
	postJSON(lokiSetTagGroupHandler(deps), map[string]string{"label": "portrait", "groupLabel": ""})
	if g := groupOf(categoryTags(t, deps, "Topic")["portrait"]); g != "" {
		t.Fatalf("portrait not ungrouped: %q", g)
	}

	// Moving a tag to another category drops its group.
	postJSON(lokiMoveTagHandler(deps), moveTagRequest{Label: "sunset", CategoryLabel: "Style"})
	if g := groupOf(categoryTags(t, deps, "Style")["sunset"]); g != "" {
		t.Fatalf("moved tag kept group %q", g)
	}

	// Deleting the category removes its groups.
	rr = postJSON(lokiDeleteCategoryHandler(deps), labelRequest{Label: "Topic"})
	mustOK(t, "delete category", rr.Code, rr.Body.String())
	if got := groupLabels(t, deps, "Topic"); len(got) != 0 {
		t.Fatalf("groups survived category delete: %v", got)
	}
}
