// Tag groups: a visual level between a category and its tags. These pin the
// two promises the feature makes — a library without groups behaves exactly as
// before, and grouping never touches anything but layout — plus the cascades
// that keep groups attached to their category through renames and moves.
import { Database, initDB } from '../main/database';
import {
  loadCategories,
  loadCategoryTags,
  loadTagGroups,
  createTagGroup,
  renameTagGroup,
  deleteTagGroup,
  updateTagGroupWeight,
  updateTagGroupHidden,
  updateTagGroupViewMode,
  setTagGroup,
  updateCategoryTagGrouping,
  renameCategory,
  deleteCategory,
  moveTag,
} from '../main/taxonomy';
import {
  groupTags,
  flattenGroupedRows,
  hiddenGroupLabels,
  weightBeside,
  loadCollapsedGroups,
  saveCollapsedGroups,
  renameCollapsedGroup,
  UNGROUPED_KEY,
} from '../renderer/components/taxonomy/tag-groups';

const ev = {} as any;

const tag = (label: string, weight: number, group?: string | null) => ({
  label,
  weight,
  category: 'Subject',
  group,
});

describe('groupTags', () => {
  it('returns null without groups, so the flat layout is kept', () => {
    expect(groupTags([tag('a', 1)], [])).toBeNull();
    expect(groupTags([tag('a', 1)], undefined)).toBeNull();
  });

  it('orders sections by group weight and keeps tag order inside each', () => {
    const sections = groupTags(
      [tag('a', 1, 'B'), tag('b', 2), tag('c', 3, 'A'), tag('d', 4, 'B')],
      [
        { label: 'B', weight: 2 },
        { label: 'A', weight: 1 },
        { label: 'Empty', weight: 3 },
      ]
    )!;
    expect(sections.map((s) => s.group)).toEqual(['A', 'B', 'Empty', null]);
    expect(sections[1].tags.map((t) => t.label)).toEqual(['a', 'd']);
    // Empty groups stay: they are drop targets.
    expect(sections[2].tags).toEqual([]);
    expect(sections[3].key).toBe(UNGROUPED_KEY);
    expect(sections[3].tags.map((t) => t.label)).toEqual(['b']);
  });

  it('puts tags whose group no longer exists in Ungrouped', () => {
    const sections = groupTags(
      [tag('a', 1, 'Gone')],
      [{ label: 'Here', weight: 1 }]
    )!;
    expect(sections[1].tags.map((t) => t.label)).toEqual(['a']);
  });
});

describe('flattenGroupedRows', () => {
  const sections = groupTags(
    [tag('a', 1, 'G'), tag('b', 2, 'G'), tag('c', 3, 'G'), tag('d', 4)],
    [{ label: 'G', weight: 1 }]
  )!;

  it('chunks each section by the column count under its header', () => {
    const rows = flattenGroupedRows(sections, new Set(), 2);
    expect(rows.map((r) => r.type)).toEqual([
      'header',
      'tags',
      'tags',
      'header',
      'tags',
    ]);
    expect(rows[2].type === 'tags' && rows[2].tags.map((t) => t.label)).toEqual(
      ['c']
    );
  });

  it('collapsed sections contribute only their header', () => {
    const rows = flattenGroupedRows(sections, new Set(['G']), 2);
    expect(rows.map((r) => r.type)).toEqual(['header', 'header', 'tags']);
    expect(rows[0].type === 'header' && rows[0].collapsed).toBe(true);
  });
});

describe('group display settings', () => {
  const tags = [tag('a', 1, 'Secret'), tag('b', 2, 'Shown'), tag('c', 3)];
  const groups = [
    { label: 'Secret', weight: 1, hidden: true },
    { label: 'Shown', weight: 2, viewMode: 'list' as const },
  ];

  it('drops hidden groups and their tags unless asked to show them', () => {
    const hidden = groupTags(tags, groups, false)!;
    expect(hidden.map((s) => s.group)).toEqual(['Shown', null]);
    // A hidden group's tags do not leak into Ungrouped.
    expect(hidden[1].tags.map((t) => t.label)).toEqual(['c']);

    const shown = groupTags(tags, groups, true)!;
    expect(shown.map((s) => s.group)).toEqual(['Secret', 'Shown', null]);
    expect(shown[0].hidden).toBe(true);
    expect([...hiddenGroupLabels(groups)]).toEqual(['Secret']);
  });

  it('lays a list-mode group out one tag per row inside a card category', () => {
    const sections = groupTags(
      [tag('a', 1, 'L'), tag('b', 2, 'L'), tag('c', 3), tag('d', 4)],
      [{ label: 'L', weight: 1, viewMode: 'list' }]
    )!;
    const rows = flattenGroupedRows(sections, new Set(), 4, 'card');
    const tagRows = rows.filter((r) => r.type === 'tags') as Extract<
      typeof rows[number],
      { type: 'tags' }
    >[];
    expect(tagRows.map((r) => [r.mode, r.tags.length])).toEqual([
      ['list', 1],
      ['list', 1],
      ['card', 2],
    ]);
    // A list category's override-free sections follow it.
    const listRows = flattenGroupedRows(sections, new Set(), 4, 'list');
    expect(listRows.filter((r) => r.type === 'tags')).toHaveLength(4);
  });
});

describe('weightBeside', () => {
  const groups = [
    { label: 'A', weight: 1 },
    { label: 'B', weight: 2 },
  ];
  it('takes the midpoint with the neighbour, or steps past the ends', () => {
    expect(weightBeside(groups, 'B', true)).toBe(1.5);
    expect(weightBeside(groups, 'A', true)).toBe(0);
    expect(weightBeside(groups, 'B', false)).toBe(3);
  });
});

describe('collapsed group persistence', () => {
  beforeEach(() => window.localStorage.clear());

  it('round-trips per category and follows a rename', () => {
    saveCollapsedGroups('Subject', new Set(['A', UNGROUPED_KEY]));
    expect([...loadCollapsedGroups('Subject')].sort()).toEqual(
      ['A', UNGROUPED_KEY].sort()
    );
    expect(loadCollapsedGroups('Style').size).toBe(0);
    renameCollapsedGroup('Subject', 'A', 'Z');
    expect(loadCollapsedGroups('Subject').has('Z')).toBe(true);
  });

  it('survives garbage in storage', () => {
    window.localStorage.setItem('taxonomy.collapsedTagGroups', '{not json');
    expect(loadCollapsedGroups('Subject').size).toBe(0);
  });
});

// A library created before tag groups existed: no tag_group table, no
// tag.group_label, no category.tag_grouping. initDB migrates it in place.
async function legacyDb(): Promise<Database> {
  const db = new Database(':memory:');
  await db.ready;
  await db.run(`CREATE TABLE category (label TEXT PRIMARY KEY, weight REAL)`);
  await db.run(
    `CREATE TABLE tag (label TEXT PRIMARY KEY, category_label TEXT, weight REAL)`
  );
  await db.run(`INSERT INTO category (label, weight) VALUES ('Subject', 0)`);
  await db.run(`INSERT INTO category (label, weight) VALUES ('Style', 1)`);
  await db.run(
    `INSERT INTO tag (label, category_label, weight) VALUES ('sunset', 'Subject', 1)`
  );
  await db.run(
    `INSERT INTO tag (label, category_label, weight) VALUES ('portrait', 'Subject', 2)`
  );
  await initDB(db);
  return db;
}

const tagsOf = async (db: Database, category: string) =>
  (await loadCategoryTags(db)(ev, [category])) as {
    label: string;
    group: string | null;
  }[];
const groupOf = async (db: Database, category: string, label: string) =>
  (await tagsOf(db, category)).find((t) => t.label === label)?.group;
const groupLabels = async (db: Database, category: string) =>
  ((await loadTagGroups(db)(ev, [category])) as { label: string }[]).map(
    (g) => g.label
  );

describe('tag group IPC handlers', () => {
  it('an upgraded library has no groups and grouping reads on', async () => {
    const db = await legacyDb();
    const cats = (await loadCategories(db)()) as { tagGrouping: string }[];
    expect(cats.every((c) => c.tagGrouping === 'on')).toBe(true);
    const tags = await tagsOf(db, 'Subject');
    expect(tags.map((t) => t.label)).toEqual(['sunset', 'portrait']);
    expect(tags.every((t) => t.group === null)).toBe(true);
    expect(await groupLabels(db, 'Subject')).toEqual([]);
  });

  it('creates, orders, assigns, renames and deletes groups', async () => {
    const db = await legacyDb();
    await createTagGroup(db)(ev, ['Subject', 'Places']);
    await createTagGroup(db)(ev, ['Subject', 'People']);
    await createTagGroup(db)(ev, ['Subject', 'Places']); // duplicate: no-op
    expect(await groupLabels(db, 'Subject')).toEqual(['Places', 'People']);

    await updateTagGroupWeight(db)(ev, ['Subject', 'People', 0.5]);
    expect(await groupLabels(db, 'Subject')).toEqual(['People', 'Places']);

    await setTagGroup(db)(ev, ['sunset', 'Places']);
    expect(await groupOf(db, 'Subject', 'sunset')).toBe('Places');

    await renameTagGroup(db)(ev, ['Subject', 'Places', 'Scenes']);
    expect(await groupOf(db, 'Subject', 'sunset')).toBe('Scenes');

    await setTagGroup(db)(ev, ['portrait', 'People']);
    await deleteTagGroup(db)(ev, ['Subject', 'People']);
    expect(await groupOf(db, 'Subject', 'portrait')).toBeNull(); // kept, ungrouped

    await setTagGroup(db)(ev, ['sunset', null]);
    expect(await groupOf(db, 'Subject', 'sunset')).toBeNull();
  });

  it('stores per-group hidden and view mode, migrating older tables', async () => {
    const db = await legacyDb();
    await createTagGroup(db)(ev, ['Subject', 'Places']);
    const read = async () =>
      ((await loadTagGroups(db)(ev, ['Subject'])) as any[])[0];
    expect(await read()).toMatchObject({ hidden: false, viewMode: null });

    await updateTagGroupHidden(db)(ev, ['Subject', 'Places', true]);
    await updateTagGroupViewMode(db)(ev, ['Subject', 'Places', 'list']);
    await renameTagGroup(db)(ev, ['Subject', 'Places', 'Scenes']);
    expect(await read()).toMatchObject({
      label: 'Scenes',
      hidden: true,
      viewMode: 'list',
    });

    await updateTagGroupViewMode(db)(ev, ['Subject', 'Scenes', null]);
    await updateTagGroupHidden(db)(ev, ['Subject', 'Scenes', false]);
    expect(await read()).toMatchObject({ hidden: false, viewMode: null });
  });

  it('stores the grouping toggle', async () => {
    const db = await legacyDb();
    await updateCategoryTagGrouping(db)(ev, ['Subject', 'off']);
    const cats = (await loadCategories(db)()) as {
      label: string;
      tagGrouping: string;
    }[];
    expect(cats.find((c) => c.label === 'Subject')?.tagGrouping).toBe('off');
    expect(cats.find((c) => c.label === 'Style')?.tagGrouping).toBe('on');
  });

  it('keeps groups with their category through rename, move and delete', async () => {
    const db = await legacyDb();
    await createTagGroup(db)(ev, ['Subject', 'Places']);
    await setTagGroup(db)(ev, ['sunset', 'Places']);

    await renameCategory(db)(ev, ['Subject', 'Topic']);
    expect(await groupLabels(db, 'Topic')).toEqual(['Places']);
    expect(await groupOf(db, 'Topic', 'sunset')).toBe('Places');

    // Groups belong to a category: a moved tag leaves its group behind.
    await moveTag(db)(ev, ['sunset', 'Style']);
    expect(await groupOf(db, 'Style', 'sunset')).toBeNull();

    await deleteCategory(db)(ev, ['Topic']);
    expect(await groupLabels(db, 'Topic')).toEqual([]);
  });
});
