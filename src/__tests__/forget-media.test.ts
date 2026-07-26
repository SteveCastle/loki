// "Remove from Library" on the media-unavailable panel: erase every database
// reference to a path WITHOUT touching the file. deleteMedia can't do this job
// — it trashes/unlinks first, and a missing file makes that throw before any
// row is cleaned up, which is exactly the case this runs in.
//
// The face and embedding tables belong to the Go media-server's schema, so a
// viewer-only library may not have them; the handler must clean what exists
// and skip the rest instead of aborting.
jest.mock('electron', () => ({
  shell: { trashItem: jest.fn() },
}));

import { Database } from '../main/database';
import { forgetMedia } from '../main/media';

const GONE = 'C:\\media\\gone.jpg';
const KEPT = 'C:\\media\\kept.jpg';

// The viewer's own initDB tables (see src/main/database.ts).
async function makeViewerDb(): Promise<Database> {
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
  for (const p of [GONE, KEPT]) {
    await db.run('INSERT INTO media (path) VALUES (?)', [p]);
    await db.run(
      `INSERT INTO media_tag_by_category
         (media_path, tag_label, category_label, weight, time_stamp)
       VALUES (?, 'sunset', 'Scene', 1, 0)`,
      [p]
    );
  }
  await db.run(
    `INSERT INTO battle (winner_path, loser_path, outcome) VALUES (?, ?, 1)`,
    [GONE, KEPT]
  );
  await db.run(
    `INSERT INTO battle (winner_path, loser_path, outcome) VALUES (?, ?, 1)`,
    [KEPT, GONE]
  );
  return db;
}

// The extra tables the Go media-server adds to the same file.
async function addServerTables(db: Database): Promise<void> {
  await db.run(
    `CREATE TABLE media_embedding (media_path TEXT, model TEXT, dim INTEGER, vector BLOB)`
  );
  await db.run(
    `CREATE TABLE face (
       id INTEGER PRIMARY KEY AUTOINCREMENT, media_path TEXT, model TEXT,
       person_id INTEGER, assigned_by TEXT
     )`
  );
  await db.run(`CREATE TABLE face_scan (media_path TEXT, model TEXT)`);
  await db.run(
    `CREATE TABLE person (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, cover_face_id INTEGER)`
  );
  await db.run(`CREATE TABLE face_veto (face_id INTEGER, person_id INTEGER)`);
  await db.run(`CREATE TABLE face_cannot_link (face_a INTEGER, face_b INTEGER)`);
  await db.run(
    `CREATE TABLE face_group_ban_member (ban_id INTEGER, face_id INTEGER)`
  );

  for (const p of [GONE, KEPT]) {
    await db.run(
      `INSERT INTO media_embedding (media_path, model, dim, vector)
       VALUES (?, 'siglip2', 2, x'0000')`,
      [p]
    );
    await db.run(`INSERT INTO face_scan (media_path, model) VALUES (?, 'sface')`, [p]);
  }
  await db.run(
    `INSERT INTO face (id, media_path, model, person_id, assigned_by)
     VALUES (1, ?, 'sface', 7, 'auto')`,
    [GONE]
  );
  await db.run(
    `INSERT INTO face (id, media_path, model, person_id, assigned_by)
     VALUES (2, ?, 'sface', 7, 'user')`,
    [KEPT]
  );
  // Person 7's cover is the doomed face, and the curation assertions are keyed
  // by that face id — all of it has to go with the face row.
  await db.run(`INSERT INTO person (id, name, cover_face_id) VALUES (7, 'Alice', 1)`);
  await db.run(`INSERT INTO face_veto (face_id, person_id) VALUES (1, 9)`);
  await db.run(`INSERT INTO face_cannot_link (face_a, face_b) VALUES (1, 2)`);
  await db.run(`INSERT INTO face_group_ban_member (ban_id, face_id) VALUES (3, 1)`);
}

const call = (db: Database, path: string) =>
  forgetMedia(db)({} as never, [path]);

const count = async (db: Database, sql: string, params: unknown[] = []) =>
  (await db.get(`SELECT COUNT(*) AS n FROM ${sql}`, params as never[])).n;

describe('forgetMedia', () => {
  it('erases the media row, tags, and battle history for the path only', async () => {
    const db = await makeViewerDb();
    const result = await call(db, GONE);

    expect(result.media).toBe(1);
    expect(result.tags).toBe(1);
    expect(result.battles).toBe(2); // one win, one loss
    expect(await count(db, 'media WHERE path = ?', [GONE])).toBe(0);
    expect(await count(db, 'media_tag_by_category WHERE media_path = ?', [GONE])).toBe(0);
    expect(
      await count(db, 'battle WHERE winner_path = ? OR loser_path = ?', [GONE, GONE])
    ).toBe(0);

    // The other item is untouched.
    expect(await count(db, 'media WHERE path = ?', [KEPT])).toBe(1);
    expect(await count(db, 'media_tag_by_category WHERE media_path = ?', [KEPT])).toBe(1);
    await db.close();
  });

  it('erases embeddings, faces, scan markers, and every face-keyed assertion', async () => {
    const db = await makeViewerDb();
    await addServerTables(db);
    const result = await call(db, GONE);

    expect(result.embeddings).toBe(1);
    expect(result.faces).toBe(1);
    expect(await count(db, 'media_embedding WHERE media_path = ?', [GONE])).toBe(0);
    expect(await count(db, 'face WHERE media_path = ?', [GONE])).toBe(0);
    expect(await count(db, 'face_scan WHERE media_path = ?', [GONE])).toBe(0);
    // Assertions keyed by the deleted face id would otherwise be permanent
    // orphans — nothing can ever resolve face 1 again.
    expect(await count(db, 'face_veto WHERE face_id = 1')).toBe(0);
    expect(await count(db, 'face_cannot_link WHERE face_a = 1 OR face_b = 1')).toBe(0);
    expect(await count(db, 'face_group_ban_member WHERE face_id = 1')).toBe(0);
    // The person survives; only the dangling cover pointer is cleared.
    const person = await db.get('SELECT cover_face_id FROM person WHERE id = 7');
    expect(person.cover_face_id).toBeNull();

    // The other item's face data is untouched.
    expect(await count(db, 'face WHERE media_path = ?', [KEPT])).toBe(1);
    expect(await count(db, 'media_embedding WHERE media_path = ?', [KEPT])).toBe(1);
    await db.close();
  });

  it('skips server-only tables a viewer library does not have', async () => {
    const db = await makeViewerDb();
    const result = await call(db, GONE);
    expect(result.embeddings).toBe(0);
    expect(result.faces).toBe(0);
    expect(result.media).toBe(1);
    await db.close();
  });

  it('is a no-op for a path with no rows', async () => {
    const db = await makeViewerDb();
    const result = await call(db, 'C:\\media\\never-imported.jpg');
    expect(result).toMatchObject({ media: 0, tags: 0, battles: 0 });
    // Nothing else was collateral damage.
    expect(await count(db, 'media')).toBe(2);
    await db.close();
  });
});
