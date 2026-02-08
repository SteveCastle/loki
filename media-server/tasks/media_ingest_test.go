package tasks

import (
	"database/sql"
	"testing"

	_ "modernc.org/sqlite"
)

func TestParseIngestOptions_Tags(t *testing.T) {
	args := []string{"--recursive", "--tag=sunset:subject", "--tag=portrait", "--transcript", "somepath"}
	opts, remaining := parseIngestOptions(args)

	if !opts.Recursive {
		t.Error("expected Recursive to be true")
	}
	if !opts.Transcript {
		t.Error("expected Transcript to be true")
	}
	if len(remaining) != 1 || remaining[0] != "somepath" {
		t.Errorf("unexpected remaining args: %v", remaining)
	}
	if len(opts.Tags) != 2 {
		t.Fatalf("expected 2 tags, got %d", len(opts.Tags))
	}
	if opts.Tags[0].Label != "sunset" || opts.Tags[0].Category != "subject" {
		t.Errorf("tag[0] = %+v; want {sunset subject}", opts.Tags[0])
	}
	if opts.Tags[1].Label != "portrait" || opts.Tags[1].Category != "" {
		t.Errorf("tag[1] = %+v; want {portrait \"\"}", opts.Tags[1])
	}
}

func TestParseTagArg(t *testing.T) {
	tests := []struct {
		input    string
		wantL    string
		wantC    string
	}{
		{"sunset:subject", "sunset", "subject"},
		{"portrait", "portrait", ""},
		{"hello%20world:my%20cat", "hello world", "my cat"},
		{":empty", "", "empty"},
		{"label:", "label", ""},
		{"a:b:c", "a", "b:c"}, // split on first colon only
	}
	for _, tc := range tests {
		l, c := parseTagArg(tc.input)
		if l != tc.wantL || c != tc.wantC {
			t.Errorf("parseTagArg(%q) = (%q, %q); want (%q, %q)", tc.input, l, c, tc.wantL, tc.wantC)
		}
	}
}

func TestResolveTagCategories(t *testing.T) {
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	// Create tag table matching the real schema
	_, err = db.Exec(`CREATE TABLE tag (
		label TEXT PRIMARY KEY,
		category_label TEXT,
		weight REAL
	)`)
	if err != nil {
		t.Fatal(err)
	}

	// Insert a known tag
	_, err = db.Exec(`INSERT INTO tag (label, category_label) VALUES ('sunset', 'Subject')`)
	if err != nil {
		t.Fatal(err)
	}

	tags := []TagInfo{
		{Label: "sunset", Category: ""},        // should resolve to "Subject" from DB
		{Label: "portrait", Category: ""},       // unknown, should default to "General"
		{Label: "beach", Category: "Location"},  // explicit category, should be preserved
	}

	resolved := resolveTagCategories(db, tags)

	if len(resolved) != 3 {
		t.Fatalf("expected 3 resolved tags, got %d", len(resolved))
	}
	if resolved[0].Category != "Subject" {
		t.Errorf("resolved[0].Category = %q; want %q", resolved[0].Category, "Subject")
	}
	if resolved[1].Category != "General" {
		t.Errorf("resolved[1].Category = %q; want %q", resolved[1].Category, "General")
	}
	if resolved[2].Category != "Location" {
		t.Errorf("resolved[2].Category = %q; want %q", resolved[2].Category, "Location")
	}
}
