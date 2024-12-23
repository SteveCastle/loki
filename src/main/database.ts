import * as sqlite3 from 'sqlite3';

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
    params: any[] = []
  ): Promise<{ id: number; changes: number }> {
    return new Promise((resolve, reject) => {
      this.db.run(query, params, function (err: Error | null) {
        if (err) {
          console.error('Error executing query:', err);
          reject(err);
        } else {
          resolve({ id: this.lastID, changes: this.changes });
        }
      });
    });
  }

  get(query: string, params: any[] = []): Promise<any> {
    return new Promise((resolve, reject) => {
      this.db.get(query, params, (err: Error | null, row: any) => {
        if (err) {
          console.error('Error executing query:', err);
          reject(err);
        } else {
          resolve(row);
        }
      });
    });
  }

  all(query: string, params: any[] = []): Promise<any[]> {
    return new Promise((resolve, reject) => {
      this.db.all(query, params, (err: Error | null, rows: any[]) => {
        if (err) {
          console.error('Error executing query:', err);
          reject(err);
        } else {
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
  await db.run(`CREATE TABLE IF NOT EXISTS media (
  path TEXT PRIMARY KEY,
  description TEXT,
  transcript TEXT,
  views INTEGER,
  wins INTEGER,
  losses INTEGER,
  size INTEGER,
  hash TEXT,
  elo REAL
)`);

  // If media table exists but does not have elo column create it
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
  await db.run(`CREATE TABLE IF NOT EXISTS category (
  label TEXT PRIMARY KEY,
  weight REAL
)`);

  await db.run(`CREATE TABLE IF NOT EXISTS tag (
  label TEXT PRIMARY KEY,
  category_label TEXT,
  weight REAL,
  thumbnail_path_100 TEXT,
  thumbnail_path_300 TEXT,
  thumbnail_path_600 TEXT,
  thumbnail_path_1200 TEXT,
  FOREIGN KEY (category_label) REFERENCES category (label)
)`);

  await db.run(`CREATE TABLE IF NOT EXISTS media_tag_by_category (
  media_path TEXT,
  tag_label TEXT,
  category_label TEXT,
  weight REAL,
  job_id INTEGER,
  created_at INTEGER,
  time_stamp REAL,
  PRIMARY KEY (media_path, tag_label, category_label, time_stamp),
  FOREIGN KEY (media_path) REFERENCES media (path),
  FOREIGN KEY (tag_label) REFERENCES tag (label),
  FOREIGN KEY (category_label) REFERENCES category (label)
)`);

  const tagTable = await db.get(
    `SELECT name FROM sqlite_master WHERE type='table' AND name='media_tag_by_category'`
  );
  if (tagTable) {
    const tableInfo = await db.all(`PRAGMA table_info(media_tag_by_category)`);
    const columnsToMigrate = [{ name: 'created_at', type: 'INTEGER' }];
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
}
