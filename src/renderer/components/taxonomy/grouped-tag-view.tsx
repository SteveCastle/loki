import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import useComponentSize from '@rehooks/component-size';
import { useVirtualizer } from '@tanstack/react-virtual';
import Tag from './tag';
import TagListRow from './tag-list-row';
import TagGroupHeader from './tag-group-header';
import {
  TagGroup,
  groupTags,
  flattenGroupedRows,
  sortGroups,
  loadCollapsedGroups,
  saveCollapsedGroups,
} from './tag-groups';

type Concept = {
  label: string;
  weight: number;
  category: string;
  group?: string | null;
  thumbnail_path_600?: string | null;
};

type Props = {
  category: string;
  tags: Concept[];
  groups: TagGroup[];
  viewMode: 'card' | 'list';
  selectedTags: string[];
  isDisabled: boolean;
  handleEditAction: (tag: string) => void;
  onAddTagToGroup: (group: string) => void;
  onRenameGroup: (group: string) => void;
};

// Same metrics as VirtualizedTagGrid / TagListView so a grouped category
// looks like the flat one with headers slotted in.
const MIN_COLUMN_WIDTH = 100;
const CARD_HEIGHT = 60;
const CARD_GAP = 8;
const LIST_ROW_HEIGHT = 30;
const HEADER_HEIGHT = 30;

// One category's tags under collapsible group headers. Always virtualized —
// header rows and tag rows share one virtualizer, so a 10k-tag category
// renders the same handful of rows whether it is grouped or not.
export default function GroupedTagView({
  category,
  tags,
  groups,
  viewMode,
  selectedTags,
  isDisabled,
  handleEditAction,
  onAddTagToGroup,
  onRenameGroup,
}: Props) {
  const parentRef = useRef<HTMLDivElement>(null);
  const { width } = useComponentSize(parentRef);
  const columns =
    viewMode === 'list'
      ? 1
      : Math.max(
          1,
          Math.floor((width || MIN_COLUMN_WIDTH) / MIN_COLUMN_WIDTH)
        );

  const [collapsed, setCollapsed] = useState<Set<string>>(() =>
    loadCollapsedGroups(category)
  );
  useEffect(() => {
    setCollapsed(loadCollapsedGroups(category));
  }, [category]);
  const toggleCollapsed = useCallback(
    (key: string) => {
      setCollapsed((prev) => {
        const next = new Set(prev);
        if (!next.delete(key)) next.add(key);
        saveCollapsedGroups(category, next);
        return next;
      });
    },
    [category]
  );

  const orderedGroups = useMemo(() => sortGroups(groups), [groups]);
  const sections = useMemo(
    () => groupTags(tags, orderedGroups) ?? [],
    [tags, orderedGroups]
  );
  const rows = useMemo(
    () => flattenGroupedRows(sections, collapsed, columns),
    [sections, collapsed, columns]
  );

  const rowHeight = viewMode === 'list' ? LIST_ROW_HEIGHT : CARD_HEIGHT + CARD_GAP;
  // Both callbacks must be stable between renders: the virtualizer recomputes
  // its measurements whenever either identity changes, and an inline
  // getItemKey re-renders forever. They change exactly when the rows do (a
  // group collapses, the column count changes), which is when the header
  // positions move.
  const getItemKey = useCallback(
    (index: number) => rows[index]?.key ?? index,
    [rows]
  );
  const estimateSize = useCallback(
    (index: number) =>
      rows[index]?.type === 'header' ? HEADER_HEIGHT : rowHeight,
    [rows, rowHeight]
  );
  const rowVirtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => parentRef.current,
    getItemKey,
    estimateSize,
    overscan: 6,
  });

  return (
    <div
      className={`${
        viewMode === 'list' ? 'tag-list-view' : 'tags virtualized'
      } tag-groups`}
      ref={parentRef}
    >
      <div
        style={{
          height: `${rowVirtualizer.getTotalSize()}px`,
          width: '100%',
          position: 'relative',
        }}
      >
        {rowVirtualizer.getVirtualItems().map((virtualRow) => {
          const row = rows[virtualRow.index];
          if (!row) return null;
          const style = {
            position: 'absolute' as const,
            top: 0,
            left: 0,
            width: '100%',
            transform: `translateY(${virtualRow.start}px)`,
          };
          if (row.type === 'header') {
            return (
              <div
                key={virtualRow.key}
                style={{ ...style, height: `${HEADER_HEIGHT}px` }}
              >
                <TagGroupHeader
                  category={category}
                  group={row.section.group}
                  groups={orderedGroups}
                  count={row.section.tags.length}
                  collapsed={row.collapsed}
                  onToggleCollapsed={() => toggleCollapsed(row.section.key)}
                  onAddTag={onAddTagToGroup}
                  onRename={onRenameGroup}
                  isDisabled={isDisabled}
                />
              </div>
            );
          }
          if (viewMode === 'list') {
            const tag = row.tags[0];
            return (
              <div
                key={virtualRow.key}
                style={{ ...style, height: `${LIST_ROW_HEIGHT}px` }}
              >
                <TagListRow
                  tag={tag}
                  tags={row.section.tags}
                  group={row.section.group}
                  active={selectedTags.includes(tag.label)}
                  isDisabled={isDisabled}
                  handleEditAction={handleEditAction}
                />
              </div>
            );
          }
          return (
            <div
              key={virtualRow.key}
              className="virtualized-row"
              style={{
                ...style,
                height: `${CARD_HEIGHT}px`,
                display: 'grid',
                gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))`,
                gap: `${CARD_GAP}px`,
              }}
            >
              {row.tags.map((tag) => (
                <Tag
                  key={tag.label}
                  tag={tag}
                  tags={row.section.tags}
                  group={row.section.group}
                  active={selectedTags.includes(tag.label)}
                  isDisabled={isDisabled}
                  handleEditAction={handleEditAction}
                />
              ))}
            </div>
          );
        })}
      </div>
    </div>
  );
}
