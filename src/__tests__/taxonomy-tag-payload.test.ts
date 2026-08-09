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
  // The autotagger's catch-all bucket. On a real library this is ~97% of the
  // table (183,548 of 189,122), which is why the command palette excludes it.
  await db.run(
    `INSERT INTO tag (label, category_label, weight) VALUES ('1girl', 'Suggested', 0)`
  );
  await db.run(
    `INSERT INTO tag (label, category_label, weight) VALUES ('outdoors', 'Suggested', 0)`
  );
  // A tag with no category at all — must survive an exclusion filter, since
  // NULL NOT IN (...) is NULL in SQL, not true.
  await db.run(`INSERT INTO tag (label, weight) VALUES ('orphan', 2)`);
  return db;
}

const labelsOf = (rows: Record<string, unknown>[]) =>
  rows.map((r) => r.label).sort();

describe('loadAllTags payload', () => {
  it('returns only label, category and weight', async () => {
    const db = await makeDb();
    const rows = (await loadAllTags(db)({} as any)) as Record<
      string,
      unknown
    >[];

    expect(rows).toHaveLength(5);
    for (const row of rows) {
      expect(Object.keys(row).sort()).toEqual(['category', 'label', 'weight']);
    }
    await db.close();
  });

  it('excludes the requested categories, keeping uncategorised tags', async () => {
    const db = await makeDb();
    const rows = (await loadAllTags(db)({} as any, [
      ['Suggested'],
    ])) as Record<string, unknown>[];

    // 'orphan' has a NULL category: `NULL NOT IN ('Suggested')` evaluates to
    // NULL (not true), so without an explicit IS NULL branch it would vanish.
    expect(labelsOf(rows)).toEqual(['moody', 'orphan', 'sunset']);
    await db.close();
  });

  it('excludes several categories at once', async () => {
    const db = await makeDb();
    const rows = (await loadAllTags(db)({} as any, [
      ['Suggested', 'Subject'],
    ])) as Record<string, unknown>[];

    expect(labelsOf(rows)).toEqual(['moody', 'orphan']);
    await db.close();
  });

  it('ignores an empty or malformed exclusion list', async () => {
    const db = await makeDb();
    expect(
      labelsOf((await loadAllTags(db)({} as any, [[]])) as any[])
    ).toHaveLength(5);
    expect(
      labelsOf((await loadAllTags(db)({} as any, [['', null as any]])) as any[])
    ).toHaveLength(5);
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
