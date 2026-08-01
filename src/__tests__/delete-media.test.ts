// The Delete action must erase EVERY database reference to the path, not just
// the media row and tags — leaving embedding, face, scan-marker, or battle
// rows behind kept a deleted file "covered" in every index and leaked face
// assertions forever. deleteMedia now shares eraseMediaReferences with
// forgetMedia, so both clean the same table list.
//
// The DB cleanup must only run when the file is actually gone: if both trash
// and unlink fail, the rows stay (the media-unavailable panel + forgetMedia is
// the recovery path for that case).
jest.mock('electron', () => ({
  shell: { trashItem: jest.fn() },
}));

import { shell } from 'electron';
import fs from 'fs';
import { Database } from '../main/database';
import { deleteMedia } from '../main/media';

const trashItem = shell.trashItem as jest.Mock;

const DOOMED = 'C:\\media\\doomed.jpg';
const KEPT = 'C:\\media\\kept.jpg';

// The viewer's own initDB tables plus the Go media-server's additions — same
// shape as forget-media.test.ts, since the two actions share one cleanup.
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

  for (const p of [DOOMED, KEPT]) {
    await db.run('INSERT INTO media (path) VALUES (?)', [p]);
    await db.run(
      `INSERT INTO media_tag_by_category
         (media_path, tag_label, category_label, weight, time_stamp)
       VALUES (?, 'sunset', 'Scene', 1, 0)`,
      [p]
    );
    await db.run(
      `INSERT INTO media_embedding (media_path, model, dim, vector)
       VALUES (?, 'siglip2', 2, x'0000')`,
      [p]
    );
    await db.run(
      `INSERT INTO face_scan (media_path, model) VALUES (?, 'sface')`,
      [p]
    );
  }
  await db.run(
    `INSERT INTO battle (winner_path, loser_path, outcome) VALUES (?, ?, 1)`,
    [DOOMED, KEPT]
  );
  await db.run(
    `INSERT INTO face (id, media_path, model, person_id, assigned_by)
     VALUES (1, ?, 'sface', 7, 'auto')`,
    [DOOMED]
  );
  await db.run(
    `INSERT INTO person (id, name, cover_face_id) VALUES (7, 'Alice', 1)`
  );
  await db.run(`INSERT INTO face_veto (face_id, person_id) VALUES (1, 9)`);
  return db;
}

const call = (db: Database, path: string) =>
  deleteMedia(db)({} as never, [path]);

const count = async (db: Database, sql: string, params: unknown[] = []) =>
  (await db.get(`SELECT COUNT(*) AS n FROM ${sql}`, params as never[])).n;

const countAllReferences = async (db: Database, p: string) =>
  (await count(db, 'media WHERE path = ?', [p])) +
  (await count(db, 'media_tag_by_category WHERE media_path = ?', [p])) +
  (await count(db, 'media_embedding WHERE media_path = ?', [p])) +
  (await count(db, 'face WHERE media_path = ?', [p])) +
  (await count(db, 'face_scan WHERE media_path = ?', [p])) +
  (await count(db, 'battle WHERE winner_path = ? OR loser_path = ?', [p, p]));

describe('deleteMedia', () => {
  beforeEach(() => {
    trashItem.mockReset().mockResolvedValue(undefined);
  });

  it('trashes the file and erases every index reference to the path', async () => {
    const db = await makeDb();
    const result = await call(db, DOOMED);

    expect(trashItem).toHaveBeenCalledWith(DOOMED);
    expect(result).toMatchObject({
      media: 1,
      tags: 1,
      embeddings: 1,
      faces: 1,
      battles: 1,
    });
    expect(await countAllReferences(db, DOOMED)).toBe(0);
    // Face-id-keyed assertions and the cover pointer go with the face row.
    expect(await count(db, 'face_veto WHERE face_id = 1')).toBe(0);
    const person = await db.get('SELECT cover_face_id FROM person WHERE id = 7');
    expect(person.cover_face_id).toBeNull();

    // The other item keeps all of its rows (battle row named both paths, so
    // it is legitimately gone for KEPT too).
    expect(await count(db, 'media WHERE path = ?', [KEPT])).toBe(1);
    expect(await count(db, 'media_embedding WHERE media_path = ?', [KEPT])).toBe(1);
    expect(await count(db, 'face_scan WHERE media_path = ?', [KEPT])).toBe(1);
    await db.close();
  });

  it('falls back to unlink when trash fails, then still cleans the database', async () => {
    const db = await makeDb();
    trashItem.mockRejectedValue(new Error('no trash on this volume'));
    const unlink = jest
      .spyOn(fs.promises, 'unlink')
      .mockResolvedValue(undefined);

    await call(db, DOOMED);

    expect(unlink).toHaveBeenCalledWith(DOOMED);
    expect(await countAllReferences(db, DOOMED)).toBe(0);
    unlink.mockRestore();
    await db.close();
  });

  it('leaves all rows intact when the file cannot be deleted', async () => {
    const db = await makeDb();
    trashItem.mockRejectedValue(new Error('no trash'));
    const unlink = jest
      .spyOn(fs.promises, 'unlink')
      .mockRejectedValue(new Error('EBUSY'));

    await expect(call(db, DOOMED)).rejects.toThrow('EBUSY');
    // File still exists → the item must stay recoverable, nothing erased.
    expect(await countAllReferences(db, DOOMED)).toBeGreaterThan(0);
    expect(await count(db, 'media WHERE path = ?', [DOOMED])).toBe(1);
    unlink.mockRestore();
    await db.close();
  });
});
