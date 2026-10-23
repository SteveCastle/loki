// Tag groups: a purely visual level between a category and its tags. A tag
// names its group in `group` (tag.group_label); the category's groups come from
// `load-tag-groups`. Nothing here touches queries — grouping only decides how
// the taxonomy panel lays out one category's tags.
//
// Backwards compatible by construction: a category with no groups (or with
// grouping switched off) never reaches this layout — groupTags returns null and
// the panel renders the flat, weight-sorted list it always has.

export type TagGroup = { label: string; weight: number };

// Per-category toggle stored on the category row (category.tag_grouping).
// 'off' lists the tags flat again, ignoring their groups.
export type TagGrouping = 'on' | 'off';

export type GroupableTag = {
  label: string;
  weight: number;
  group?: string | null;
};

// One group's slice of the category. `group` is null for the trailing
// "Ungrouped" section.
export type GroupSection<T extends GroupableTag> = {
  key: string;
  group: string | null;
  tags: T[];
};

// Collapse-state key of the Ungrouped section. A NUL can't come from a typed
// group name, so it never collides with a real group.
export const UNGROUPED_KEY = '\u0000ungrouped';

export function sortGroups(groups: TagGroup[]): TagGroup[] {
  return [...groups].sort(
    (a, b) =>
      (a.weight ?? 0) - (b.weight ?? 0) ||
      a.label.localeCompare(b.label, undefined, { sensitivity: 'base' })
  );
}

// Splits weight-sorted tags into their groups (in group order), then one
// Ungrouped section. Empty groups are kept — they are drop targets. A tag
// naming a group that no longer exists (e.g. renamed by an older app version)
// falls back to Ungrouped rather than disappearing. Returns null when the
// category has no groups: the caller keeps its flat layout.
export function groupTags<T extends GroupableTag>(
  tags: T[],
  groups: TagGroup[] | undefined | null
): GroupSection<T>[] | null {
  if (!groups || groups.length === 0) return null;
  const sections = sortGroups(groups).map(
    (g): GroupSection<T> => ({ key: g.label, group: g.label, tags: [] })
  );
  const byLabel = new Map(sections.map((s) => [s.group as string, s]));
  const ungrouped: GroupSection<T> = {
    key: UNGROUPED_KEY,
    group: null,
    tags: [],
  };
  for (const tag of tags) {
    const section = (tag.group && byLabel.get(tag.group)) || ungrouped;
    section.tags.push(tag);
  }
  return [...sections, ungrouped];
}

export type GroupedRow<T extends GroupableTag> =
  | {
      type: 'header';
      key: string;
      section: GroupSection<T>;
      collapsed: boolean;
    }
  | { type: 'tags'; key: string; section: GroupSection<T>; tags: T[] };

// Flattens sections into virtualizer rows: a header per section, then its
// tags chunked `perRow` at a time (the card grid's column count, or 1 for the
// list style). Collapsed sections contribute only their header.
export function flattenGroupedRows<T extends GroupableTag>(
  sections: GroupSection<T>[],
  collapsed: ReadonlySet<string>,
  perRow: number
): GroupedRow<T>[] {
  const step = Math.max(1, Math.floor(perRow));
  const rows: GroupedRow<T>[] = [];
  for (const section of sections) {
    const isCollapsed = collapsed.has(section.key);
    rows.push({
      type: 'header',
      key: `h:${section.key}`,
      section,
      collapsed: isCollapsed,
    });
    if (isCollapsed) continue;
    for (let i = 0; i < section.tags.length; i += step) {
      rows.push({
        type: 'tags',
        key: `t:${section.key}:${i}`,
        section,
        tags: section.tags.slice(i, i + step),
      });
    }
  }
  return rows;
}

// Weight that drops `label` just before/after `target` in the ordered list
// (midpoint with the neighbour, or ±1 at either end). Shared by group
// reordering; tags keep their own inline version in tag.tsx.
export function weightBeside(
  ordered: TagGroup[],
  target: string,
  before: boolean
): number {
  const index = ordered.findIndex((g) => g.label === target);
  if (index === -1) return 0;
  const targetWeight = ordered[index].weight ?? 0;
  const neighbour = ordered[before ? index - 1 : index + 1];
  if (!neighbour) return before ? targetWeight - 1 : targetWeight + 1;
  return (targetWeight + (neighbour.weight ?? 0)) / 2;
}

// ---- Collapse state ----
// Per viewer, not per library: collapsing is a way of looking, so it lives in
// localStorage (works for read-only public visitors too) rather than the DB.

const COLLAPSED_STORAGE_KEY = 'taxonomy.collapsedTagGroups';

type CollapsedMap = Record<string, string[]>;

function readCollapsedMap(): CollapsedMap {
  try {
    const raw = window.localStorage.getItem(COLLAPSED_STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : null;
    return parsed && typeof parsed === 'object' ? parsed : {};
  } catch {
    return {};
  }
}

function writeCollapsedMap(map: CollapsedMap) {
  try {
    window.localStorage.setItem(COLLAPSED_STORAGE_KEY, JSON.stringify(map));
  } catch {
    // Storage unavailable (private window, quota): collapse just won't persist.
  }
}

export function loadCollapsedGroups(category: string): Set<string> {
  const list = readCollapsedMap()[category];
  return new Set(Array.isArray(list) ? list : []);
}

export function saveCollapsedGroups(category: string, keys: Set<string>) {
  const map = readCollapsedMap();
  if (keys.size === 0) delete map[category];
  else map[category] = [...keys];
  writeCollapsedMap(map);
}

// Keeps a renamed group collapsed under its new name.
export function renameCollapsedGroup(
  category: string,
  from: string,
  to: string
) {
  const keys = loadCollapsedGroups(category);
  if (!keys.delete(from)) return;
  keys.add(to);
  saveCollapsedGroups(category, keys);
}
