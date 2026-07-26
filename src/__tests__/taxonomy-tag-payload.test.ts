// The tag list that backs the fuzzy-search index is the whole `tag` table —
// ~190K rows on a large library. It crosses the IPC boundary and is then
// structured-cloned again into the tag-search worker, and BOTH hops serialize
// every property name once per row. Carrying `description` (empty on all but a
// handful of rows) and `thumbnail_path_600` (null on all but a few hundred)
// therefore cost ~29 bytes of key text per row per hop for data no consumer of
// this list reads.
//
// These tests pin the split: loadAllTags stays thin, and getTag is the way to
// reach the fields it dropped.
import { Database } from '../main/database';
import { loadAllTags, getTag } from '../main/taxonomy';

async function makeDb(): Promise<Database> {
  const db = new Database(':memory:');
  await db.ready;
  await db.run(
    `CREATE TABLE tag (
       label TEXT PRIMARY KEY,
       category_label TEXT,
       weight REAL,
       description TEXT,
       thumbnail_path_600 TEXT
     )`
  );
  await db.run(
    `INSERT INTO tag (label, category_label, weight, description, thumbnail_path_600)
     VALUES ('moody', 'Style', 0, 'Low-key, heavy shadows', '/thumbs/moody.jpg')`
  );
  await db.run(
    `INSERT INTO tag (label, category_label, weight) VALUES ('sunset', 'Subject', 1)`
  );
  return db;
}

describe('loadAllTags payload', () => {
  it('returns only label, category and weight', async () => {
    const db = await makeDb();
    const rows = (await loadAllTags(db)()) as Record<string, unknown>[];

    expect(rows).toHaveLength(2);
    for (const row of rows) {
      expect(Object.keys(row).sort()).toEqual(['category', 'label', 'weight']);
    }
    // Ordered by category then weight, as the search index expects.
    expect(rows.map((r) => r.label)).toEqual(['moody', 'sunset']);
    await db.close();
  });
});

describe('getTag', () => {
  it('returns the full detail for a single tag', async () => {
    const db = await makeDb();
    const tag = (await getTag(db)({} as any, ['moody'])) as any;

    expect(tag).toEqual({
      label: 'moody',
      category: 'Style',
      weight: 0,
      description: 'Low-key, heavy shadows',
      thumbnail_path_600: '/thumbs/moody.jpg',
    });
    await db.close();
  });

  it('normalizes a missing description to an empty string', async () => {
    const db = await makeDb();
    const tag = (await getTag(db)({} as any, ['sunset'])) as any;

    expect(tag.description).toBe('');
    expect(tag.thumbnail_path_600).toBeNull();
    await db.close();
  });

  it('returns null for an unknown tag and for an empty label', async () => {
    const db = await makeDb();
    expect(await getTag(db)({} as any, ['nope'])).toBeNull();
    expect(await getTag(db)({} as any, [''])).toBeNull();
    await db.close();
  });
});
