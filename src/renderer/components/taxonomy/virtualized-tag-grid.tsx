import { useMemo, useRef } from 'react';
import useComponentSize from '@rehooks/component-size';
import { useVirtualizer } from '@tanstack/react-virtual';
import Tag from './tag';
import { ThumbnailShape, tagCardHeight } from './tag-groups';

type Concept = {
  label: string;
  weight: number;
  category: string;
};

type Props = {
  tags: Concept[];
  selectedTags: string[];
  isDisabled: boolean;
  handleEditAction: (tag: string) => void;
  disableReorder?: boolean;
  thumbnailShape?: ThumbnailShape;
};

const MIN_COLUMN_WIDTH = 100;
const ROW_GAP = 8;

export default function VirtualizedTagGrid({
  tags,
  selectedTags,
  isDisabled,
  handleEditAction,
  disableReorder = false,
  thumbnailShape = 'landscape',
}: Props) {
  const parentRef = useRef<HTMLDivElement>(null);
  const { width } = useComponentSize(parentRef);

  const columns = Math.max(
    1,
    Math.floor((width || MIN_COLUMN_WIDTH) / MIN_COLUMN_WIDTH)
  );
  const rowHeight = tagCardHeight(thumbnailShape, width, columns, ROW_GAP);
  const rowCount = useMemo(
    () => Math.ceil(tags.length / columns),
    [tags.length, columns]
  );

  const rowVirtualizer = useVirtualizer({
    count: rowCount,
    getScrollElement: () => parentRef.current,
    estimateSize: () => rowHeight + ROW_GAP,
    overscan: 4,
  });

  return (
    <div className="tags virtualized" ref={parentRef}>
      <div
        style={{
          height: `${rowVirtualizer.getTotalSize()}px`,
          width: '100%',
          position: 'relative',
        }}
      >
        {rowVirtualizer.getVirtualItems().map((virtualRow) => (
          <div
            key={virtualRow.key}
            className="virtualized-row"
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: `${rowHeight}px`,
              transform: `translateY(${virtualRow.start}px)`,
              display: 'grid',
              gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))`,
              gap: `${ROW_GAP}px`,
            }}
          >
            {Array.from({ length: columns }).map((_, colIdx) => {
              const tagIdx = virtualRow.index * columns + colIdx;
              const tag = tags[tagIdx];
              if (!tag) return <div key={`empty-${colIdx}`} />;
              return (
                <Tag
                  key={tag.label}
                  tag={tag}
                  tags={tags}
                  active={selectedTags.includes(tag.label)}
                  isDisabled={isDisabled}
                  handleEditAction={handleEditAction}
                  disableReorder={disableReorder}
                />
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}
