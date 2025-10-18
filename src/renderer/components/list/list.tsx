import {
  useContext,
  useEffect,
  useRef,
  useState,
  useMemo,
  useCallback,
} from 'react';
import { useSelector } from '@xstate/react';
import { GlobalStateContext } from '../../state';
import { useVirtualizer } from '@tanstack/react-virtual';
import { useDragLayer } from 'react-dnd';
import filter from '../../filter';
import { ListItem } from './list-item';
import { PERFORMANCE_CONSTANTS } from '../../constants/performance';
import './list-item.css';
import './list.css';

export function List() {
  const [initialLoad, setInitialLoad] = useState(true);
  const { libraryService } = useContext(GlobalStateContext);

  const base = useSelector(
    libraryService,
    (state) => {
      return {
        library: state.context.library,
        libraryLoadId: state.context.libraryLoadId,
        textFilter: state.context.textFilter,
        filters: state.context.settings.filters,
        sortBy: state.context.settings.sortBy,
        gridSize: state.context.settings.gridSize as [number, number],
      };
    },
    (a, b) =>
      a.library === b.library &&
      a.libraryLoadId === b.libraryLoadId &&
      a.textFilter === b.textFilter &&
      a.filters === b.filters &&
      a.sortBy === b.sortBy &&
      a.gridSize[0] === b.gridSize[0] &&
      a.gridSize[1] === b.gridSize[1]
  );

  const items = useMemo(() => {
    return filter(
      base.libraryLoadId,
      base.textFilter,
      base.library,
      base.filters,
      base.sortBy
    );
  }, [
    base.libraryLoadId,
    base.textFilter,
    base.library,
    base.filters,
    base.sortBy,
  ]);

  console.log('rendering list');
  // Read initial scroll position once
  const initialScrollPositionRef = useRef(0);
  useEffect(() => {
    initialScrollPositionRef.current =
      libraryService.getSnapshot().context.scrollPosition;
  }, [libraryService]);
  const cursor = useSelector(
    libraryService,
    (state) => state.context.cursor,
    (a, b) => a === b
  );

  const [columns, rows] = useMemo(() => {
    const columns = Math.max(1, base.gridSize[0] || 0);
    let rows = Math.max(1, base.gridSize[1] || 0);
    const totalNumberOfRows = Math.ceil(items.length / columns);
    if (totalNumberOfRows < rows) {
      rows = Math.max(1, totalNumberOfRows);
    }
    return [columns, rows];
  }, [base.gridSize, items.length]);

  const [height, setHeight] = useState(window.innerHeight / rows);

  useEffect(() => {
    setHeight(window.innerHeight / rows);
    const handleResize = () => {
      setHeight(window.innerHeight / rows);
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [rows]);

  const debounceIdRef = useRef<number | null>(null);
  const pendingScrollTopRef = useRef<number>(0);
  const lastSentScrollTopRef = useRef<number>(-1);
  const handleScroll = useCallback(
    (e: React.UIEvent<HTMLDivElement>) => {
      const target = e.target as HTMLDivElement;
      pendingScrollTopRef.current = target.scrollTop;
      if (debounceIdRef.current != null) {
        clearTimeout(debounceIdRef.current);
      }
      debounceIdRef.current = window.setTimeout(() => {
        const next = pendingScrollTopRef.current;
        if (next !== lastSentScrollTopRef.current) {
          libraryService.send('SET_SCROLL_POSITION', { position: next });
          lastSentScrollTopRef.current = next;
        }
        debounceIdRef.current = null;
      }, PERFORMANCE_CONSTANTS.SCROLL_DEBOUNCE);
    },
    [libraryService]
  );

  useEffect(() => {
    return () => {
      if (debounceIdRef.current != null) {
        clearTimeout(debounceIdRef.current);
        debounceIdRef.current = null;
      }
    };
  }, []);

  return (
    <VirtualGrid
      items={items}
      columns={columns}
      height={height}
      overscan={PERFORMANCE_CONSTANTS.LIST_OVERSCAN}
      initialScrollOffset={initialScrollPositionRef.current}
      cursor={cursor}
      onScroll={handleScroll}
      onDidInitialScroll={() => setInitialLoad(false)}
      shouldDoInitialScroll={initialLoad}
    />
  );
}

type VirtualGridProps = {
  items: ReturnType<typeof filter>;
  columns: number;
  height: number;
  overscan: number;
  initialScrollOffset: number;
  cursor: number | null;
  onScroll: (e: React.UIEvent<HTMLDivElement>) => void;
  onDidInitialScroll: () => void;
  shouldDoInitialScroll: boolean;
};

function VirtualGrid({
  items,
  columns,
  height,
  overscan,
  initialScrollOffset,
  cursor,
  onScroll,
  onDidInitialScroll,
  shouldDoInitialScroll,
}: VirtualGridProps) {
  const parentRef = useRef<HTMLDivElement>(null);
  const listLength = useMemo(
    () => Math.ceil(items.length / columns),
    [items.length, columns]
  );
  const rowVirtualizer = useVirtualizer({
    count: listLength,
    getScrollElement: () => parentRef.current,
    estimateSize: () => height,
    overscan,
  });

  // Recalculate measurements when row height changes
  useEffect(() => {
    rowVirtualizer.measure();
  }, [height, rowVirtualizer]);

  // Handle initial restore of scroll position and scrolling to cursor updates
  const didInitialScrollRef = useRef(false);
  useEffect(() => {
    if (!parentRef.current) return;

    if (!didInitialScrollRef.current && shouldDoInitialScroll) {
      rowVirtualizer.scrollToOffset(initialScrollOffset);
      didInitialScrollRef.current = true;
      onDidInitialScroll();
      return;
    }

    if (cursor != null) {
      const scrollTarget = Math.floor(cursor / columns) || 0;
      if (
        rowVirtualizer.getVirtualItems()[0] &&
        scrollTarget !== rowVirtualizer.getVirtualItems()[0].index
      ) {
        rowVirtualizer.scrollToIndex(scrollTarget, {
          align: 'auto',
        });
      }
    }
  }, [
    cursor,
    columns,
    initialScrollOffset,
    onDidInitialScroll,
    shouldDoInitialScroll,
    rowVirtualizer,
  ]);

  // Auto-scroll while dragging near edges
  const { isDragging, offset, type } = useDragLayer((monitor) => ({
    isDragging: monitor.isDragging(),
    offset: monitor.getClientOffset(),
    type: monitor.getItemType(),
  }));

  const mapRange = useCallback(
    (
      value: number,
      low1: number,
      high1: number,
      low2: number,
      high2: number
    ) => {
      return low2 + ((high2 - low2) * (value - low1)) / (high1 - low1);
    },
    []
  );

  const getScrollSpeed = useCallback(
    (yOffset: number, containerHeight: number) => {
      let scrollSpeed = 0;
      const threshold = PERFORMANCE_CONSTANTS.SCROLL_SPEED_THRESHOLD;
      const minSpeed = PERFORMANCE_CONSTANTS.SCROLL_SPEED_RANGE.MIN;
      const maxSpeed = PERFORMANCE_CONSTANTS.SCROLL_SPEED_RANGE.MAX;

      if (yOffset < threshold) {
        scrollSpeed = mapRange(yOffset, 0, threshold, minSpeed, 0);
      } else if (
        yOffset > containerHeight - threshold &&
        yOffset < containerHeight
      ) {
        scrollSpeed = mapRange(
          yOffset,
          containerHeight - threshold,
          containerHeight,
          0,
          maxSpeed
        );
      }

      return scrollSpeed;
    },
    [mapRange]
  );

  useEffect(() => {
    let animationFrameId: number;
    const containerHeight = parentRef.current?.clientHeight;

    const scroll = () => {
      if (
        isDragging &&
        type === 'MEDIA' &&
        offset &&
        parentRef.current &&
        containerHeight
      ) {
        const mousePosition = offset.y;
        const scrollSpeed = getScrollSpeed(mousePosition, containerHeight);
        parentRef.current.scrollBy(0, scrollSpeed);
        animationFrameId = requestAnimationFrame(scroll);
      } else {
        cancelAnimationFrame(animationFrameId);
      }
    };

    animationFrameId = requestAnimationFrame(scroll);

    return () => cancelAnimationFrame(animationFrameId);
  }, [isDragging, offset, type, getScrollSpeed]);

  return (
    <div className="List" ref={parentRef} onScroll={onScroll}>
      <div
        className="ListContainer"
        style={{
          height: `${rowVirtualizer.getTotalSize()}px`,
          position: 'relative',
          width: '100%',
        }}
      >
        {rowVirtualizer.getVirtualItems().map((virtualItem) => (
          <div
            className="ListRow"
            key={virtualItem.key}
            style={{
              height: `${virtualItem.size}px`,
              transform: `translateY(${virtualItem.start}px)`,
              gridAutoFlow: 'column',
            }}
          >
            {new Array(columns).fill(columns).map((_, i) => {
              const item = items[columns * virtualItem.index + i];
              return items[columns * virtualItem.index + i] ? (
                <ListItem
                  scaleMode={'cover'}
                  height={height}
                  key={
                    item?.path +
                    (item?.timeStamp != null ? String(item?.timeStamp) : 'null')
                  }
                  item={item}
                  idx={columns * virtualItem.index + i}
                />
              ) : null;
            })}
          </div>
        ))}
      </div>
    </div>
  );
}
