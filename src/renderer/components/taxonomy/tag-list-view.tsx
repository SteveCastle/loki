import { useRef } from 'react';
import { useVirtualizer } from '@tanstack/react-virtual';
import TagListRow from './tag-list-row';

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
};

const ROW_HEIGHT = 30;

export default function TagListView({
  tags,
  selectedTags,
  isDisabled,
  handleEditAction,
  disableReorder = false,
}: Props) {
  const parentRef = useRef<HTMLDivElement>(null);

  const rowVirtualizer = useVirtualizer({
    count: tags.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => ROW_HEIGHT,
    overscan: 8,
  });

  return (
    <div className="tag-list-view" ref={parentRef}>
      <div
        style={{
          height: `${rowVirtualizer.getTotalSize()}px`,
          position: 'relative',
          width: '100%',
        }}
      >
        {rowVirtualizer.getVirtualItems().map((virtualRow) => {
          const tag = tags[virtualRow.index];
          if (!tag) return null;
          return (
            <div
              key={tag.label}
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: `${virtualRow.size}px`,
                transform: `translateY(${virtualRow.start}px)`,
              }}
            >
              <TagListRow
                tag={tag}
                tags={tags}
                active={selectedTags.includes(tag.label)}
                isDisabled={isDisabled}
                handleEditAction={handleEditAction}
                disableReorder={disableReorder}
              />
            </div>
          );
        })}
      </div>
    </div>
  );
}
