// "Find Media" bookkeeping: after a file moves on disk, every database
// reference has to follow it. This mirrors the media-server's media.MovePath
// (POST /api/media/move) — the viewer runs against its own SQLite connection,
// so the two implementations must agree, and these tests encode the contract
// both sides are held to.
jest.mock('electron', () => ({
  shell: { trashItem: jest.fn() },
}));

import { Database } from '../main/database';
import { moveMedia, MoveConflictError } from '../main/media';

const call = (
  db: Database,
  from: string,
  to: string,
  prefix?: boolean,
  dryRun?: boolean
) => moveMedia(db)({} as never, [from, to, prefix, dryRun]);

// The viewer's own tables (src/main/database.ts) plus the media-server's.
async function makeDb(): Promise<Database> {
  const db = new Database(':memory:');
  await db.ready;
  await db.run('CREATE TABLE media (path TEXT PRIMARY KEY, description TEXT)');
  await db.run(
    `CREATE TABLE media_tag_by_category (
       media_path TEXT, tag_label TEXT, category_label TEXT, weight REAL, time_stamp REAL
     )`
  );
  await db.run(
    `CREATE TABLE battle (
       id INTEGER PRIMARY KEY AUTOINCREMENT,
       winner_path TEXT NOT NULL, loser_path TEXT NOT NULL, outcome REAL
     )`
  );
  await db.run(
    `CREATE TABLE media_embedding (media_path TEXT, model TEXT, dim INTEGER, vector BLOB)`
  );
  await db.run(
    `CREATE TABLE face (
       id INTEGER PRIMARY KEY AUTOINCREMENT, media_path TEXT, model TEXT, person_id INTEGER
     )`
  );
  await db.run(`CREATE TABLE face_scan (media_path TEXT, model TEXT)`);
  return db;
}

// Give a path a row in every table a move has to carry.
async function seed(db: Database, path: string): Promise<void> {
  await db.run('INSERT INTO media (path) VALUES (?)', [path]);
  await db.run(
    `INSERT INTO media_tag_by_category (media_path, tag_label, category_label, weight, time_stamp)
     VALUES (?, 'sunset', 'Scene', 1, 0)`,
    [path]
  );
  await db.run(
    `INSERT INTO media_embedding (media_path, model, dim, vector)
     VALUES (?, 'siglip2', 2, x'0000')`,
    [path]
  );
  await db.run(`INSERT INTO face (media_path, model) VALUES (?, 'sface')`, [path]);
  await db.run(`INSERT INTO face_scan (media_path, model) VALUES (?, 'sface')`, [path]);
  await db.run(
    `INSERT INTO battle (winner_path, loser_path, outcome) VALUES (?, '/other.jpg', 1)`,
    [path]
  );
}

const countAt = async (db: Database, path: string) => {
  const rows = await Promise.all(
    [
      ['media', 'SELECT COUNT(*) AS n FROM media WHERE path = ?'],
      ['tags', 'SELECT COUNT(*) AS n FROM media_tag_by_category WHERE media_path = ?'],
      ['embeddings', 'SELECT COUNT(*) AS n FROM media_embedding WHERE media_path = ?'],
      ['faces', 'SELECT COUNT(*) AS n FROM face WHERE media_path = ?'],
      ['scans', 'SELECT COUNT(*) AS n FROM face_scan WHERE media_path = ?'],
      ['battles', 'SELECT COUNT(*) AS n FROM battle WHERE winner_path = ? OR loser_path = ?'],
    ].map(async ([key, sql]) => {
      const params = key === 'battles' ? [path, path] : [path];
      const row = await db.get(sql, params);
      return [key, row.n as number] as const;
    })
  );
  return Object.fromEntries(rows) as Record<string, number>;
};

describe('moveMedia', () => {
  it('carries every reference to the new path', async () => {
    const db = await makeDb();
    await seed(db, '/photos/old.jpg');

    const result = await call(db, '/photos/old.jpg', '/archive/new.jpg');
    expect(result.items).toBe(1);
    expect(result.rows).toMatchObject({
      'media.path': 1,
      'media_tag_by_category.media_path': 1,
      'media_embedding.media_path': 1,
      'face.media_path': 1,
      'face_scan.media_path': 1,
      'battle.winner_path': 1,
    });

    expect(await countAt(db, '/photos/old.jpg')).toEqual({
      media: 0, tags: 0, embeddings: 0, faces: 0, scans: 0, battles: 0,
    });
    expect(await countAt(db, '/archive/new.jpg')).toEqual({
      media: 1, tags: 1, embeddings: 1, faces: 1, scans: 1, battles: 1,
    });
    await db.close();
  });

  it('moves a whole folder in prefix mode, segment-aligned', async () => {
    const db = await makeDb();
    await seed(db, '/photos/2023/a.jpg');
    await seed(db, '/photos/2023/sub/b.jpg');
    await seed(db, '/photos/2023extra/c.jpg'); // must NOT be dragged along
    await seed(db, '/photos/2024/d.jpg');

    const result = await call(db, '/photos/2023', '/archive/2023', true);
    expect(result.items).toBe(2);
    expect((await countAt(db, '/archive/2023/a.jpg')).media).toBe(1);
    expect((await countAt(db, '/archive/2023/sub/b.jpg')).faces).toBe(1);
    expect((await countAt(db, '/photos/2023extra/c.jpg')).media).toBe(1);
    expect((await countAt(db, '/photos/2024/d.jpg')).media).toBe(1);
    await db.close();
  });

  it('preserves Windows sub-paths and ignores trailing separators', async () => {
    const db = await makeDb();
    await seed(db, 'C:\\media\\shoot\\a.jpg');
    await seed(db, 'C:\\media\\shoot-b\\keep.jpg');

    const result = await call(db, 'C:\\media\\shoot\\', 'D:\\archive\\shoot', true);
    expect(result.items).toBe(1);
    expect((await countAt(db, 'D:\\archive\\shoot\\a.jpg')).media).toBe(1);
    expect((await countAt(db, 'C:\\media\\shoot-b\\keep.jpg')).media).toBe(1);
    await db.close();
  });

  it('reports real counts for a dry run without writing', async () => {
    const db = await makeDb();
    await seed(db, '/photos/old.jpg');

    const result = await call(db, '/photos/old.jpg', '/archive/new.jpg', false, true);
    expect(result.dryRun).toBe(true);
    expect(result.items).toBe(1);
    expect(result.total).toBe(6);
    expect((await countAt(db, '/photos/old.jpg')).media).toBe(1);
    expect((await countAt(db, '/archive/new.jpg')).media).toBe(0);
    await db.close();
  });

  it('refuses an occupied destination and rolls nothing forward', async () => {
    const db = await makeDb();
    await seed(db, '/photos/a.jpg');
    await seed(db, '/photos/b.jpg');

    await expect(call(db, '/photos/a.jpg', '/photos/b.jpg')).rejects.toBeInstanceOf(
      MoveConflictError
    );
    // The source is intact — a refused move must not half-apply.
    expect(await countAt(db, '/photos/a.jpg')).toEqual({
      media: 1, tags: 1, embeddings: 1, faces: 1, scans: 1, battles: 1,
    });
    await db.close();
  });

  it('rejects nonsense arguments', async () => {
    const db = await makeDb();
    await expect(call(db, '', '/b.jpg')).rejects.toThrow();
    await expect(call(db, '/a.jpg', '   ')).rejects.toThrow();
    await expect(call(db, '/a.jpg', '/a.jpg')).rejects.toThrow();
    await expect(call(db, '/a/', '/a')).rejects.toThrow(); // same after trimming
    await expect(call(db, '/a', '/a/b', true)).rejects.toThrow(); // dest inside source
    await db.close();
  });

  it('is a no-op for a path the library does not have', async () => {
    const db = await makeDb();
    await seed(db, '/photos/a.jpg');

    const result = await call(db, '/photos/ghost.jpg', '/archive/ghost.jpg');
    expect(result.items).toBe(0);
    expect(result.total).toBe(0);
    expect((await countAt(db, '/photos/a.jpg')).media).toBe(1);
    await db.close();
  });

  it('skips server-only tables a viewer library does not have', async () => {
    const db = new Database(':memory:');
    await db.ready;
    await db.run('CREATE TABLE media (path TEXT PRIMARY KEY)');
    await db.run(
      `CREATE TABLE media_tag_by_category (media_path TEXT, tag_label TEXT, category_label TEXT, weight REAL, time_stamp REAL)`
    );
    await db.run('INSERT INTO media (path) VALUES (?)', ['/photos/a.jpg']);

    const result = await call(db, '/photos/a.jpg', '/archive/a.jpg');
    expect(result.rows['media.path']).toBe(1);
    // Absent tables are reported as absent, not as "0 rows moved".
    expect(result.rows).not.toHaveProperty('face.media_path');
    const row = await db.get('SELECT COUNT(*) AS n FROM media WHERE path = ?', [
      '/archive/a.jpg',
    ]);
    expect(row.n).toBe(1);
    await db.close();
  });
});
