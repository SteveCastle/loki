// Regression tests for the transaction-leak wedge: a failure between
// BEGIN and COMMIT (e.g. SQLITE_BUSY from the Go media-server writing to the
// shared dream.sqlite) used to leave the connection inside an open
// transaction, after which EVERY write failed with "cannot start a
// transaction within a transaction" until app restart — tag application
// silently dead. withTransaction must roll back, rethrow, and leave the
// connection usable; it must also self-heal an already-wedged connection.
import { Database } from '../main/database';
import { retryAsync, isDatabaseLockedError } from '../main/db-retry';

// Each test gets a fresh in-memory DB with one table.
async function makeDb(): Promise<Database> {
  const db = new Database(':memory:');
  await db.ready;
  await db.run('CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)');
  return db;
}

describe('Database.withTransaction', () => {
  it('commits on success', async () => {
    const db = await makeDb();
    await db.withTransaction(async () => {
      await db.run('INSERT INTO t (val) VALUES (?)', ['a']);
    });
    const row = await db.get('SELECT COUNT(*) AS n FROM t');
    expect(row.n).toBe(1);
    await db.close();
  });

  it('rolls back on error, rethrows, and leaves the connection usable', async () => {
    const db = await makeDb();
    await expect(
      db.withTransaction(async () => {
        await db.run('INSERT INTO t (val) VALUES (?)', ['doomed']);
        throw new Error('SQLITE_BUSY: database is locked');
      })
    ).rejects.toThrow('SQLITE_BUSY');

    // Rolled back: the doomed row must not exist.
    const row = await db.get('SELECT COUNT(*) AS n FROM t');
    expect(row.n).toBe(0);

    // THE regression: the next transaction must not fail with
    // "cannot start a transaction within a transaction".
    await db.withTransaction(async () => {
      await db.run('INSERT INTO t (val) VALUES (?)', ['ok']);
    });
    const after = await db.get('SELECT COUNT(*) AS n FROM t');
    expect(after.n).toBe(1);
    await db.close();
  });

  it('recovers through a transient lock when composed with retryAsync', async () => {
    // The create-assignment write path: a SQLITE_BUSY inside the transaction
    // (Go media-server holding the shared DB) must roll back cleanly and
    // succeed on the retry — tagging has to work without user intervention.
    const db = await makeDb();
    let attempts = 0;
    await retryAsync(
      () =>
        db.withTransaction(async () => {
          attempts += 1;
          if (attempts === 1) {
            throw Object.assign(new Error('SQLITE_BUSY: database is locked'), {
              code: 'SQLITE_BUSY',
            });
          }
          await db.run('INSERT INTO t (val) VALUES (?)', ['retried']);
        }),
      {
        retries: 3,
        isRetryable: isDatabaseLockedError,
        sleep: async () => {},
      }
    );
    expect(attempts).toBe(2);
    const row = await db.get('SELECT COUNT(*) AS n FROM t');
    expect(row.n).toBe(1);
    await db.close();
  });

  it('self-heals a connection already wedged inside an open transaction', async () => {
    const db = await makeDb();
    // Simulate the pre-fix leak: some earlier code path opened a transaction
    // and never closed it.
    await db.run('BEGIN TRANSACTION');

    await db.withTransaction(async () => {
      await db.run('INSERT INTO t (val) VALUES (?)', ['healed']);
    });
    const row = await db.get('SELECT COUNT(*) AS n FROM t');
    expect(row.n).toBe(1);
    await db.close();
  });
});

describe('PreparedStatement', () => {
  it('run awaits completion and resolves changes', async () => {
    const db = await makeDb();
    const stmt = await db.prepare('INSERT INTO t (val) VALUES (?)');
    const result = await stmt.run('x');
    expect(result.changes).toBe(1);
    await stmt.finalize();
    const row = await db.get('SELECT COUNT(*) AS n FROM t');
    expect(row.n).toBe(1);
    await db.close();
  });

  it('run rejects on SQL errors instead of losing them', async () => {
    const db = await makeDb();
    const stmt = await db.prepare('INSERT INTO t (id, val) VALUES (?, ?)');
    await stmt.run(1, 'first');
    // Same PK → constraint violation must reject (pre-fix this surfaced as an
    // uncaughtException and the caller kept going as if it had worked).
    await expect(stmt.run(1, 'dupe')).rejects.toThrow(/SQLITE_CONSTRAINT/);
    await stmt.finalize();
    await db.close();
  });
});
