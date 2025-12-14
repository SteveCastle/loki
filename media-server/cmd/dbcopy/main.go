// copy_tag_rows.go
package main

import (
	"database/sql"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	_ "modernc.org/sqlite"
)

func main() {
	var (
		srcPath    string
		dstPath    string
		tag        string
		onConflict string
		dryRun     bool
		verbose    bool
	)

	flag.StringVar(&srcPath, "source", "", "Path to source SQLite DB")
	flag.StringVar(&dstPath, "dest", "", "Path to destination SQLite DB")
	flag.StringVar(&tag, "tag", "", "Tag value to match in tag_label")
	flag.StringVar(&onConflict, "on-conflict", "ignore", "Conflict behavior: ignore | abort | replace | rollback | fail")
	flag.BoolVar(&dryRun, "dry-run", false, "Show what would happen without writing")
	flag.BoolVar(&verbose, "v", false, "Verbose logging")
	flag.Parse()

	if srcPath == "" || dstPath == "" || tag == "" {
		fmt.Fprintf(os.Stderr, "Usage: %s -source <src.db> -dest <dest.db> -tag <tag> [-on-conflict ignore|abort|replace|rollback|fail] [-dry-run] [-v]\n", os.Args[0])
		os.Exit(2)
	}

	confVerb := strings.ToUpper(onConflict)
	validConf := map[string]bool{
		"IGNORE": true, "ABORT": true, "REPLACE": true, "ROLLBACK": true, "FAIL": true,
	}
	if !validConf[confVerb] {
		log.Fatalf("invalid -on-conflict value %q; use ignore|abort|replace|rollback|fail", onConflict)
	}

	// Open the source DB.
	// DSN notes: _pragma=busy_timeout=5000 helps with locked DBs.
	dsn := fmt.Sprintf("file:%s?_pragma=busy_timeout=5000&_pragma=foreign_keys=ON", srcPath)
	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		log.Fatalf("open source: %v", err)
	}
	defer db.Close()

	// Basic ping
	if err := db.Ping(); err != nil {
		log.Fatalf("ping source: %v", err)
	}

	// Attach the destination DB as schema "dest".
	if _, err := db.Exec(`ATTACH DATABASE ? AS dest`, dstPath); err != nil {
		log.Fatalf("attach dest: %v", err)
	}

	// Check the table exists on both sides.
	requireTable := func(schema string) {
		var cnt int
		row := db.QueryRow(`SELECT count(*) FROM ` + schema + `.` +
			`sqlite_master WHERE type='table' AND name='media_tag_by_category'`)
		if err := row.Scan(&cnt); err != nil {
			log.Fatalf("check table %s.media_tag_by_category: %v", schema, err)
		}
		if cnt == 0 {
			log.Fatalf("table %s.media_tag_by_category not found", schema)
		}
	}
	requireTable("main")
	requireTable("dest")

	// Count rows to copy.
	var toCopy int
	if err := db.QueryRow(
		`SELECT COUNT(*) FROM main.media_tag_by_category WHERE tag_label = ?`, tag,
	).Scan(&toCopy); err != nil {
		log.Fatalf("count rows: %v", err)
	}

	if verbose || dryRun {
		log.Printf("Rows matching tag %q in source: %d", tag, toCopy)
	}

	if dryRun {
		log.Printf("Dry run: no changes written.")
		return
	}

	tx, err := db.Begin()
	if err != nil {
		log.Fatalf("begin tx: %v", err)
	}
	defer func() {
		// In case of panic, rollback; otherwise commit/rollback handled below.
		if p := recover(); p != nil {
			_ = tx.Rollback()
			panic(p)
		}
	}()

	// Perform the copy with desired conflict behavior.
	// We use SELECT * with ATTACH so column orders match exactly.
	// Note: this assumes the destination table has a compatible schema.
	insertSQL := fmt.Sprintf(`
		INSERT OR %s INTO dest.media_tag_by_category
		SELECT * FROM main.media_tag_by_category
		WHERE tag_label = ?
	`, confVerb)

	res, err := tx.Exec(insertSQL, tag)
	if err != nil {
		_ = tx.Rollback()
		log.Fatalf("insert: %v", err)
	}

	affected, _ := res.RowsAffected() // If unsupported, returns 0, nil.
	if err := tx.Commit(); err != nil {
		log.Fatalf("commit: %v", err)
	}

	// changes() only counts changes in the current connection since last reset.
	// We rely on RowsAffected where available.
	if verbose {
		log.Printf("Inserted %d row(s) into dest.media_tag_by_category (conflict=%s).", affected, confVerb)
	} else {
		fmt.Printf("Done. Inserted %d row(s).\n", affected)
	}
}
