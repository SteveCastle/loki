package main

import "fmt"

// sqliteDSN builds the connection string used everywhere the server opens
// the application database. busy_timeout=5000 makes writers wait up to 5s
// for a competing write to finish instead of failing immediately with
// SQLITE_BUSY ("database is locked"), which is the practical fix for write
// contention between the ingest path, job queue, and ad-hoc updates.
func sqliteDSN(dbPath string) string {
	return fmt.Sprintf("file:%s?_pragma=busy_timeout=5000&_pragma=foreign_keys=ON", dbPath)
}
