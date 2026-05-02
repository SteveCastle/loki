import * as sqlite3 from 'sqlite3';
import { logQuery } from './queryLog';

// High-resolution timer that works in Node and Electron without depending on
// the DOM Performance global being typed.
const now = (): number => {
  // Use Number conversion so we always return a plain number in ms.
  const hr = process.hrtime();
  return hr[0] * 1000 + hr[1] / 1e6;
};

export class Database {
  private db: sqlite3.Database;

  constructor(dbPath: string) {
    this.db = new sqlite3.Database(dbPath, (err) => {
      if (err) {
        console.error('Error connecting to the database:', err);
      } else {
        console.log('Connected to the database');
        this.db.run('PRAGMA journal_mode = WAL');
      }
    });
  }

  run(
    query: string,
    params: any[] = [],
    name?: string
  ): Promise<{ id: number; changes: number }> {
    const start = now();
    return new Promise((resolve, reject) => {
      this.db.run(query, params, function (err: Error | null) {
        const duration_ms = now() - start;
        if (err) {
          logQuery({
            name,
            sql: query,
            params,
            duration_ms,
            rows: null,
            error: err.message,
          });
          console.error('Error executing query:', err);
          reject(err);
        } else {
          logQuery({
            name,
            sql: query,
            params,
            duration_ms,
            rows: this.changes ?? null,
            error: null,
          });
          resolve({ id: this.lastID, changes: this.changes });
        }
      });
    });
  }

  get(query: string, params: any[] = [], name?: string): Promise<any> {
    const start = now();
    return new Promise((resolve, reject) => {
      this.db.get(query, params, (err: Error | null, row: any) => {
        const duration_ms = now() - start;
        if (err) {
          logQuery({
            name,
            sql: query,
            params,
            duration_ms,
            rows: null,
            error: err.message,
          });
          console.error('Error executing query:', err);
          reject(err);
        } else {
          logQuery({
            name,
            sql: query,
            params,
            duration_ms,
            rows: row ? 1 : 0,
            error: null,
          });
          resolve(row);
        }
      });
    });
  }

  all(query: string, params: any[] = [], name?: string): Promise<any[]> {
    const start = now();
    return new Promise((resolve, reject) => {
      this.db.all(query, params, (err: Error | null, rows: any[]) => {
        const duration_ms = now() - start;
        if (err) {
          logQuery({
            name,
            sql: query,
            params,
            duration_ms,
            rows: null,
            error: err.message,
          });
          console.error('Error executing query:', err);
          reject(err);
        } else {
          logQuery({
            name,
            sql: query,
            params,
            duration_ms,
            rows: rows ? rows.length : 0,
            error: null,
          });
          resolve(rows);
        }
      });
    });
  }
  prepare(query: string): Promise<sqlite3.Statement> {
    return new Promise((resolve, reject) => {
      const statement = this.db.prepare(query, (err: Error | null) => {
        if (err) {
          console.error('Error preparing query:', err);
          reject(err);
        } else {
          resolve(statement);
        }
      });
    });
  }

  close(): Promise<void> {
    return new Promise((resolve, reject) => {
      this.db.close((err: Error | null) => {
        if (err) {
          console.error('Error closing the database:', err);
          reject(err);
        } else {
          console.log('Database connection closed');
          resolve();
        }
      });
    });
  }
}

export async function initDB(db: Database) {
  if (!db) return;

  // Create category table first (referenced by other tables)
  await db.run(`CREATE TABLE IF NOT EXISTS category (
  label TEXT PRIMARY KEY,
  weight REAL
)`);

  // Create tag table (references category)
  await db.run(`CREATE TABLE IF NOT EXISTS tag (
  label TEXT PRIMARY KEY,
  category_label TEXT,
  weight REAL,
  preview BLOB,
  thumbnail_path_600 TEXT,
  FOREIGN KEY (category_label) REFERENCES category (label)
)`);

  // Create media table
  await db.run(`CREATE TABLE IF NOT EXISTS media (
  path TEXT PRIMARY KEY,
  description TEXT,
  transcript TEXT,
  preview BLOB,
  thumbnail_path_600 TEXT,
  thumbnail_path_1200 TEXT,
  elo REAL,
  views INTEGER,
  wins INTEGER,
  losses INTEGER,
  size INTEGER,
  hash TEXT,
  width INTEGER,
  height INTEGER
)`);

  // Create media_tag_by_category table
  await db.run(`CREATE TABLE IF NOT EXISTS media_tag_by_category (
  media_path TEXT,
  tag_label TEXT,
  category_label TEXT,
  weight REAL,
  time_stamp REAL,
  created_at INTEGER,
  PRIMARY KEY (media_path, tag_label, category_label, time_stamp)
)`);

  // Create cache table
  await db.run(`CREATE TABLE IF NOT EXISTS cache (
  "key" TEXT,
  files TEXT
)`);

  // Migrate existing media table if needed
  const mediaTable = await db.get(
    `SELECT name FROM sqlite_master WHERE type='table' AND name='media'`
  );
  if (mediaTable) {
    const tableInfo = await db.all(`PRAGMA table_info(media)`);
    const columnsToMigrate = [
      { name: 'elo', type: 'REAL' },
      { name: 'views', type: 'INTEGER' },
      { name: 'wins', type: 'INTEGER' },
      { name: 'losses', type: 'INTEGER' },
      { name: 'size', type: 'INTEGER' },
      { name: 'hash', type: 'TEXT' },
      { name: 'width', type: 'INTEGER' },
      { name: 'height', type: 'INTEGER' },
      { name: 'preview', type: 'BLOB' },
      { name: 'thumbnail_path_600', type: 'TEXT' },
      { name: 'thumbnail_path_1200', type: 'TEXT' },
      { name: 'transcript', type: 'TEXT' },
    ];
    for (const column of columnsToMigrate) {
      const columnExists = tableInfo.some(
        (tableColumn: any) => tableColumn.name === column.name
      );
      if (!columnExists) {
        await db.run(
          `ALTER TABLE media ADD COLUMN ${column.name} ${column.type}`
        );
      }
    }
  }

  // Migrate existing tag table if needed
  const tagTable = await db.get(
    `SELECT name FROM sqlite_master WHERE type='table' AND name='tag'`
  );
  if (tagTable) {
    const tableInfo = await db.all(`PRAGMA table_info(tag)`);
    const columnsToMigrate = [
      { name: 'preview', type: 'BLOB' },
      { name: 'thumbnail_path_600', type: 'INTEGER' },
      { name: 'description', type: 'TEXT' },
    ];
    for (const column of columnsToMigrate) {
      const columnExists = tableInfo.some(
        (tableColumn: any) => tableColumn.name === column.name
      );
      if (!columnExists) {
        await db.run(
          `ALTER TABLE tag ADD COLUMN ${column.name} ${column.type}`
        );
      }
    }
  }

  // Migrate existing category table if needed
  const categoryTable = await db.get(
    `SELECT name FROM sqlite_master WHERE type='table' AND name='category'`
  );
  if (categoryTable) {
    const tableInfo = await db.all(`PRAGMA table_info(category)`);
    const columnsToMigrate = [
      { name: 'description', type: 'TEXT' },
      { name: 'tag_view_mode', type: 'TEXT' },
    ];
    for (const column of columnsToMigrate) {
      const columnExists = tableInfo.some(
        (tableColumn: any) => tableColumn.name === column.name
      );
      if (!columnExists) {
        await db.run(
          `ALTER TABLE category ADD COLUMN ${column.name} ${column.type}`
        );
      }
    }
  }

  // Migrate existing media_tag_by_category table if needed
  const mediaTagTable = await db.get(
    `SELECT name FROM sqlite_master WHERE type='table' AND name='media_tag_by_category'`
  );
  if (mediaTagTable) {
    const tableInfo = await db.all(`PRAGMA table_info(media_tag_by_category)`);
    const columnsToMigrate = [
      { name: 'created_at', type: 'INTEGER' },
      { name: 'time_stamp', type: 'REAL' },
    ];
    for (const column of columnsToMigrate) {
      const columnExists = tableInfo.some(
        (tableColumn: any) => tableColumn.name === column.name
      );
      if (!columnExists) {
        await db.run(
          `ALTER TABLE media_tag_by_category ADD COLUMN ${column.name} ${column.type}`
        );
      }
    }
  }

  // Index on tag_label for the typed-query hot paths. Without this, every
  // tag-based filter does a full table scan of media_tag_by_category. The
  // composite PK (media_path, tag_label, ...) only helps queries that lead
  // with media_path, so we still need a standalone index for tag_label.
  //
  // We don't add a separate index on media_path because the PK's leading
  // column already covers `WHERE media_path = ?` lookups; adding one was
  // redundant and contended with the Go media-server when both processes
  // hold the DB open, producing SQLITE_BUSY on every Electron start.
  //
  // Index creation is wrapped in try/catch so a transient lock contention
  // (or any other failure) cannot kill app startup. The app still works
  // without the index — queries are just slower — and the next start will
  // try again.
  try {
    await db.run(
      `CREATE INDEX IF NOT EXISTS idx_mtc_tag_label ON media_tag_by_category(tag_label)`
    );
  } catch (e) {
    console.warn(
      '[initDB] failed to create idx_mtc_tag_label (will retry next start):',
      e
    );
  }
}
