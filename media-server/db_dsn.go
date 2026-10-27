package main

import "fmt"

// sqliteDSN builds the connection string used everywhere the server opens
// the application database. busy_timeout=5000 makes writers wait up to 5s
// for a competing write to finish instead of failing immediately with
// SQLITE_BUSY ("database is locked"), which is the practical fix for write
// contention between the ingest path, job queue, and ad-hoc updates.
//
// journal_mode=WAL lets readers proceed while a writer commits; in the
// default rollback journal any write (tagging, ingest, a job's progress)
// blocks every query until it finishes. The viewer already switches the
// shared database to WAL, but a database the server creates or opens alone
// stayed in rollback mode. synchronous=NORMAL is the documented safe
// pairing for WAL (durable across application crashes, fsync only at
// checkpoints) and makes the server's many small writes far cheaper.
func sqliteDSN(dbPath string) string {
	return fmt.Sprintf("file:%s?_pragma=busy_timeout=5000&_pragma=foreign_keys=ON&_pragma=journal_mode=WAL&_pragma=synchronous=NORMAL", dbPath)
}
