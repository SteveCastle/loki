// Virtualized person-card grid. The People category can hold thousands of
// clusters, and rendering them all at once (each a react-dnd drag+drop target
// with a face-crop fetch) made the panel crawl. Same row-chunking approach as
// virtualized-tag-grid.tsx, with two extra wrinkles: section headings are
// virtual rows too (the named/unnamed split lives INSIDE one scroll list),
// and row height tracks the panel width because cards are square-ish
// (face image ≈ column width, plus the name bar).
import { ReactNode, RefObject, useEffect, useMemo } from 'react';
import useComponentSize from '@rehooks/component-size';
import { useVirtualizer } from '@tanstack/react-virtual';
import type { Person } from './people-grid';

const MIN_COLUMN_WIDTH = 96; // matches the old auto-fill minmax(96px, 1fr)
const GAP = 8;
const CARD_INFO_HEIGHT = 25; // name/count bar under the face image
const HEADING_HEIGHT = 26;

type Row =
  | { kind: 'heading'; label: string }
  | { kind: 'cards'; people: Person[] };

export default function VirtualizedPeopleGrid({
  named,
  unknown,
  renderCard,
  scrollRef,
  after,
}: {
  named: Person[];
  unknown: Person[];
  renderCard: (person: Person) => ReactNode;
  // Owned by the caller so drag auto-scroll can target the same element.
  scrollRef: RefObject<HTMLDivElement>;
  // Non-virtualized trailing content (the "Not grouped yet" section). Sits
  // AFTER the virtual body in normal flow, so it needs no scrollMargin math.
  after?: ReactNode;
}) {
  const { width } = useComponentSize(scrollRef);

  // Same column math as CSS auto-fill: how many (col + gap) units fit.
  const columns = Math.max(
    1,
    Math.floor(((width || MIN_COLUMN_WIDTH) + GAP) / (MIN_COLUMN_WIDTH + GAP))
  );
  const colWidth = ((width || MIN_COLUMN_WIDTH) - GAP * (columns - 1)) / columns;
  // Cards are forced to exactly this height by CSS (the face image flexes),
  // so the estimate IS the layout — no measurement pass needed.
  const rowHeight = Math.round(colWidth) + CARD_INFO_HEIGHT;

  const rows = useMemo<Row[]>(() => {
    const out: Row[] = [];
    const chunk = (people: Person[]) => {
      for (let i = 0; i < people.length; i += columns) {
        out.push({ kind: 'cards', people: people.slice(i, i + columns) });
      }
    };
    chunk(named);
    if (unknown.length > 0) {
      out.push({ kind: 'heading', label: 'Unnamed clusters' });
      chunk(unknown);
    }
    return out;
  }, [named, unknown, columns]);

  const virtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: (i) =>
      rows[i].kind === 'heading' ? HEADING_HEIGHT + GAP : rowHeight + GAP,
    overscan: 4,
  });

  // estimateSize is a closure over rows/rowHeight — the virtualizer caches
  // its results, so force a re-measure when the inputs change (panel resize,
  // people added/removed).
  useEffect(() => {
    virtualizer.measure();
  }, [rowHeight, rows, virtualizer]);

  return (
    <div className="people-virtual-scroll" ref={scrollRef}>
      <div
        className="people-virtual-body"
        style={{ height: `${virtualizer.getTotalSize()}px` }}
      >
        {virtualizer.getVirtualItems().map((virtualRow) => {
          const row = rows[virtualRow.index];
          if (row.kind === 'heading') {
            return (
              <div
                key={virtualRow.key}
                className="people-virtual-row people-virtual-heading"
                style={{
                  height: `${HEADING_HEIGHT}px`,
                  transform: `translateY(${virtualRow.start}px)`,
                }}
              >
                <div className="people-grid-heading">{row.label}</div>
              </div>
            );
          }
          return (
            <div
              key={virtualRow.key}
              className="people-virtual-row"
              style={{
                height: `${rowHeight}px`,
                transform: `translateY(${virtualRow.start}px)`,
                display: 'grid',
                gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))`,
                gap: `${GAP}px`,
              }}
            >
              {row.people.map(renderCard)}
            </div>
          );
        })}
      </div>
      {after}
    </div>
  );
}
