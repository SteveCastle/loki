// The context palette's synchronous "Merge" action: consolidate a discrete
// multi-selection into its FIRST item. Tags/embeddings are copied additively
// (the target's own rows always win), an empty transcript is filled and the
// .vtt sidecar file moves next to the target, and then the source files are
// DELETED from disk with every database reference erased. A source whose file
// can't be deleted keeps its rows.
jest.mock('electron', () => ({
  shell: { trashItem: jest.fn() },
}));

import { shell } from 'electron';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { Database } from '../main/database';
import { mergeItemMetadata } from '../main/media';

const trashItem = shell.trashItem as jest.Mock;

async function makeDb(): Promise<Database> {
  const db = new Database(':memory:');
  await db.ready;
  await db.run(
    'CREATE TABLE media (path TEXT PRIMARY KEY, description TEXT, transcript TEXT)'
  );
  await db.run(
    `CREATE TABLE media_tag_by_category (
       media_path TEXT, tag_label TEXT, category_label TEXT, weight REAL,
       time_stamp REAL, created_at INTEGER,
       PRIMARY KEY (media_path, tag_label, category_label, time_stamp)
     )`
  );
  await db.run(
    `CREATE TABLE media_embedding (
       media_path TEXT NOT NULL, model TEXT NOT NULL, dim INTEGER,
       vector BLOB, created_at INTEGER,
       PRIMARY KEY (media_path, model)
     )`
  );
  await db.run(
    `CREATE TABLE battle (
       id INTEGER PRIMARY KEY AUTOINCREMENT,
       winner_path TEXT NOT NULL, loser_path TEXT NOT NULL, outcome REAL
     )`
  );
  return db;
}

const tagRow = (db: Database, p: string, tag: string, ts: number | null = 0) =>
  db.run(
    `INSERT INTO media_tag_by_category
       (media_path, tag_label, category_label, weight, time_stamp)
     VALUES (?, ?, 'Scene', 1, ?)`,
    [p, tag, ts]
  );

const embRow = (db: Database, p: string, model: string, byte: number) =>
  db.run(
    `INSERT INTO media_embedding (media_path, model, dim, vector) VALUES (?, ?, 1, ?)`,
    [p, model, Buffer.from([byte])]
  );

const call = (db: Database, paths: string[]) =>
  mergeItemMetadata(db)({} as never, [paths]);

const count = async (db: Database, sql: string, params: unknown[] = []) =>
  (await db.get(`SELECT COUNT(*) AS n FROM ${sql}`, params as never[])).n;

describe('mergeItemMetadata', () => {
  let dir: string;
  let keep: string;
  let dup: string;

  beforeEach(() => {
    // Trash is unavailable in tests: reject so the unlink fallback deletes
    // the REAL temp files, proving the disk side of the merge.
    trashItem.mockReset().mockRejectedValue(new Error('no trash in tests'));
    dir = fs.mkdtempSync(path.join(os.tmpdir(), 'merge-test-'));
    keep = path.join(dir, 'keep.mp4');
    dup = path.join(dir, 'dup.mp4');
    fs.writeFileSync(keep, 'keep-bytes');
    fs.writeFileSync(dup, 'dup-bytes');
  });

  afterEach(() => {
    fs.rmSync(dir, { recursive: true, force: true });
  });

  it('merges metadata onto the first item, moves the sidecar, and deletes sources', async () => {
    fs.writeFileSync(path.join(dir, 'dup.vtt'), 'WEBVTT dup');
    const db = await makeDb();
    await db.run(`INSERT INTO media (path) VALUES (?)`, [keep]);
    await db.run(
      `INSERT INTO media (path, transcript) VALUES (?, 'from dup')`,
      [dup]
    );
    await tagRow(db, keep, 'sunset');
    await tagRow(db, dup, 'sunset'); // overlap — must not duplicate
    await tagRow(db, dup, 'beach');
    await embRow(db, keep, 'm1', 0xaa); // target's own vector must win
    await embRow(db, dup, 'm1', 0xbb);
    await embRow(db, dup, 'm2', 0xcc); // model the target lacks — copied
    await db.run(
      `INSERT INTO battle (winner_path, loser_path, outcome) VALUES (?, ?, 1)`,
      [dup, keep]
    );

    const result = await call(db, [keep, dup]);

    expect(result).toMatchObject({
      target: keep,
      tags: 1,
      embeddings: 1,
      transcript: true,
      deleted: [dup],
      failed: [],
    });

    // Disk: source and its sidecar are gone; the sidecar moved to the target.
    expect(fs.existsSync(dup)).toBe(false);
    expect(fs.existsSync(path.join(dir, 'dup.vtt'))).toBe(false);
    expect(fs.readFileSync(path.join(dir, 'keep.vtt'), 'utf8')).toBe(
      'WEBVTT dup'
    );
    expect(result.transcriptFile).toBe(path.join(dir, 'keep.vtt'));
    expect(fs.existsSync(keep)).toBe(true);

    // Target: union of tags, own m1 vector kept, m2 copied, transcript filled.
    expect(
      await count(db, 'media_tag_by_category WHERE media_path = ?', [keep])
    ).toBe(2);
    const m1 = await db.get(
      `SELECT vector FROM media_embedding WHERE media_path = ? AND model = 'm1'`,
      [keep]
    );
    expect(Buffer.from(m1.vector)[0]).toBe(0xaa);
    expect(
      await count(db, 'media_embedding WHERE media_path = ?', [keep])
    ).toBe(2);
    const t = await db.get(`SELECT transcript FROM media WHERE path = ?`, [
      keep,
    ]);
    expect(t.transcript).toBe('from dup');

    // Source: every database reference erased.
    expect(await count(db, 'media WHERE path = ?', [dup])).toBe(0);
    expect(
      await count(db, 'media_tag_by_category WHERE media_path = ?', [dup])
    ).toBe(0);
    expect(await count(db, 'media_embedding WHERE media_path = ?', [dup])).toBe(
      0
    );
    expect(
      await count(db, 'battle WHERE winner_path = ? OR loser_path = ?', [
        dup,
        dup,
      ])
    ).toBe(0);
    await db.close();
  });

  it('never overwrites the target transcript or its sidecar', async () => {
    fs.writeFileSync(path.join(dir, 'keep.vtt'), 'WEBVTT mine');
    fs.writeFileSync(path.join(dir, 'dup.vtt'), 'WEBVTT theirs');
    const db = await makeDb();
    await db.run(`INSERT INTO media (path, transcript) VALUES (?, 'mine')`, [
      keep,
    ]);
    await db.run(`INSERT INTO media (path, transcript) VALUES (?, 'theirs')`, [
      dup,
    ]);

    const result = await call(db, [keep, dup]);

    expect(result.transcript).toBe(false);
    const t = await db.get(`SELECT transcript FROM media WHERE path = ?`, [
      keep,
    ]);
    expect(t.transcript).toBe('mine');
    expect(fs.readFileSync(path.join(dir, 'keep.vtt'), 'utf8')).toBe(
      'WEBVTT mine'
    );
    // The source's now-orphaned sidecar goes with the deleted source file.
    expect(fs.existsSync(path.join(dir, 'dup.vtt'))).toBe(false);
    await db.close();
  });

  it('keeps rows for a source whose file cannot be deleted', async () => {
    const db = await makeDb();
    const ghost = path.join(dir, 'ghost.mp4'); // never written to disk
    await db.run(`INSERT INTO media (path) VALUES (?)`, [keep]);
    await db.run(`INSERT INTO media (path) VALUES (?)`, [ghost]);
    await tagRow(db, ghost, 'beach');

    const result = await call(db, [keep, ghost]);

    // Metadata still merged, but the source stays recoverable.
    expect(result.tags).toBe(1);
    expect(result.deleted).toEqual([]);
    expect(result.failed).toEqual([ghost]);
    expect(await count(db, 'media WHERE path = ?', [ghost])).toBe(1);
    expect(
      await count(db, 'media_tag_by_category WHERE media_path = ?', [ghost])
    ).toBe(1);
    await db.close();
  });

  it('is a no-op without at least one source', async () => {
    const db = await makeDb();
    await db.run(`INSERT INTO media (path) VALUES (?)`, [keep]);
    const result = await call(db, [keep]);
    expect(result).toMatchObject({
      tags: 0,
      embeddings: 0,
      deleted: [],
      failed: [],
    });
    expect(fs.existsSync(keep)).toBe(true);
    await db.close();
  });
});
