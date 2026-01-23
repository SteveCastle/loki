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
        layoutMode: state.context.settings.layoutMode || 'grid',
      };
    },
    (a, b) =>
      a.library === b.library &&
      a.libraryLoadId === b.libraryLoadId &&
      a.textFilter === b.textFilter &&
      a.filters === b.filters &&
      a.sortBy === b.sortBy &&
      a.gridSize[0] === b.gridSize[0] &&
      a.gridSize[1] === b.gridSize[1] &&
      a.layoutMode === b.layoutMode
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
  // Read initial scroll position synchronously on mount (not in useEffect which runs after render)
  const initialScrollPositionRef = useRef(
    libraryService.getSnapshot().context.scrollPosition
  );
  const cursor = useSelector(
    libraryService,
    (state) => state.context.cursor,
    (a, b) => a === b
  );

  const scrollToCursorEventId = useSelector(
    libraryService,
    (state) => state.context.scrollToCursorEventId,
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

  if (base.layoutMode === 'masonry') {
    return (
      <MasonryGrid
        items={items}
        columns={columns}
        initialScrollOffset={initialScrollPositionRef.current}
        cursor={cursor}
        scrollToCursorEventId={scrollToCursorEventId}
        libraryService={libraryService}
        onScroll={handleScroll}
        onDidInitialScroll={() => setInitialLoad(false)}
        shouldDoInitialScroll={initialLoad}
      />
    );
  }

  return (
    <VirtualGrid
      items={items}
      columns={columns}
      height={height}
      overscan={PERFORMANCE_CONSTANTS.LIST_OVERSCAN}
      initialScrollOffset={initialScrollPositionRef.current}
      cursor={cursor}
      scrollToCursorEventId={scrollToCursorEventId}
      libraryService={libraryService}
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
  scrollToCursorEventId: string | null;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  libraryService: any;
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
  scrollToCursorEventId,
  libraryService,
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

  // Handle initial restore of scroll position
  const didInitialScrollRef = useRef(false);
  useEffect(() => {
    if (!parentRef.current) return;

    if (!didInitialScrollRef.current && shouldDoInitialScroll) {
      rowVirtualizer.scrollToOffset(initialScrollOffset);
      didInitialScrollRef.current = true;
      onDidInitialScroll();
    }
  }, [
    initialScrollOffset,
    onDidInitialScroll,
    shouldDoInitialScroll,
    rowVirtualizer,
  ]);

  // Scroll to cursor when explicitly requested via scrollToCursorEventId
  // Use refs to avoid effect re-running on every render
  const cursorRef = useRef(cursor);
  const columnsRef = useRef(columns);
  const rowVirtualizerRef = useRef(rowVirtualizer);
  cursorRef.current = cursor;
  columnsRef.current = columns;
  rowVirtualizerRef.current = rowVirtualizer;

  useEffect(() => {
    if (!scrollToCursorEventId) return;

    const currentCursor = cursorRef.current;
    if (currentCursor != null) {
      const scrollTarget = Math.floor(currentCursor / columnsRef.current) || 0;
      rowVirtualizerRef.current.scrollToIndex(scrollTarget, {
        align: 'auto',
      });
    }

    // Clear the event after processing to prevent duplicate scroll on remount
    libraryService.send('CLEAR_SCROLL_TO_CURSOR');
  }, [scrollToCursorEventId, libraryService]);

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

type MasonryGridProps = {
  items: ReturnType<typeof filter>;
  columns: number;
  initialScrollOffset: number;
  cursor: number | null;
  scrollToCursorEventId: string | null;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  libraryService: any;
  onScroll: (e: React.UIEvent<HTMLDivElement>) => void;
  onDidInitialScroll: () => void;
  shouldDoInitialScroll: boolean;
};

// Generate a unique key for an item (path + optional timestamp for video clips)
function getItemKey(item: { path: string; timeStamp?: number }): string {
  return item.timeStamp != null ? `${item.path}::${item.timeStamp}` : item.path;
}

function MasonryGrid({
  items,
  columns,
  initialScrollOffset,
  cursor,
  scrollToCursorEventId,
  libraryService,
  onScroll,
  onDidInitialScroll,
  shouldDoInitialScroll,
}: MasonryGridProps) {
  const parentRef = useRef<HTMLDivElement>(null);
  const [containerWidth, setContainerWidth] = useState(0);
  const [scrollTop, setScrollTop] = useState(initialScrollOffset);

  // Local state for fast synchronous updates during current session
  const [localDimensions, setLocalDimensions] = useState<
    Map<string, { width: number; height: number }>
  >(() => new Map());

  // On mount, load any persisted dimensions from state machine (for view switch stability)
  const hasLoadedPersistedRef = useRef(false);
  useEffect(() => {
    if (hasLoadedPersistedRef.current) return;
    hasLoadedPersistedRef.current = true;

    const persistedCache = libraryService.getSnapshot().context.masonryDimensionsCache;
    if (persistedCache && Object.keys(persistedCache).length > 0) {
      setLocalDimensions(new Map(Object.entries(persistedCache)));
    }
  }, [libraryService]);

  // Callback for ListItem to report dimensions when media loads
  const handleDimensionsLoaded = useCallback(
    (itemKey: string, width: number, height: number) => {
      // Update local state immediately for fast re-render
      setLocalDimensions((prev) => {
        if (prev.has(itemKey)) return prev;
        const next = new Map(prev);
        next.set(itemKey, { width, height });
        return next;
      });
      // Also persist to state machine for view switch stability
      libraryService.send('CACHE_MASONRY_DIMENSIONS', { itemKey, width, height });
    },
    [libraryService]
  );

  // Resize observer to get container width
  useEffect(() => {
    if (!parentRef.current) return;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setContainerWidth(entry.contentRect.width);
      }
    });
    observer.observe(parentRef.current);
    return () => observer.disconnect();
  }, []);

  const layout = useMemo(() => {
    if (containerWidth === 0) return { height: 0, items: [] };
    const columnWidth = containerWidth / columns;
    const colHeights = new Array(columns).fill(0);
    const itemPositions = items.map((item, index) => {
      const minColIndex = colHeights.indexOf(Math.min(...colHeights));
      const x = minColIndex * columnWidth;
      const y = colHeights[minColIndex];

      let itemHeight = 200; // Default fallback

      // First try to use pre-calculated dimensions from item
      if (item.width && item.height) {
        itemHeight = columnWidth / (item.width / item.height);
      } else {
        // Fall back to local dimensions cache (fast updates + initialized from persisted cache)
        const itemKey = getItemKey(item);
        const cached = localDimensions.get(itemKey);
        if (cached && cached.width && cached.height) {
          itemHeight = columnWidth / (cached.width / cached.height);
        }
      }

      colHeights[minColIndex] += itemHeight;

      return { index, x, y, width: columnWidth, height: itemHeight, item };
    });
    return { height: Math.max(...colHeights), items: itemPositions };
  }, [items, columns, containerWidth, localDimensions]);

  const visibleItems = useMemo(() => {
    // Determine viewport height
    const viewportHeight =
      parentRef.current?.clientHeight || window.innerHeight;
    const overscan = 500;
    const start = Math.max(0, scrollTop - overscan);
    const end = scrollTop + viewportHeight + overscan;

    return layout.items.filter((p) => p.y + p.height > start && p.y < end);
  }, [layout.items, scrollTop]);

  const handleScrollLocal = useCallback(
    (e: React.UIEvent<HTMLDivElement>) => {
      setScrollTop(e.currentTarget.scrollTop);
      onScroll(e);
    },
    [onScroll]
  );

  // Initial scroll: restore the previous scroll position for stable view switching
  const didInitialScrollRef = useRef(false);
  useEffect(() => {
    if (!parentRef.current) return;
    if (containerWidth === 0) return;

    if (!didInitialScrollRef.current && shouldDoInitialScroll) {
      // Restore scroll position - clamp to valid range
      const maxScroll = Math.max(
        0,
        layout.height - parentRef.current.clientHeight
      );
      const clampedOffset = Math.max(
        0,
        Math.min(initialScrollOffset, maxScroll)
      );
      parentRef.current.scrollTop = clampedOffset;
      setScrollTop(clampedOffset);
      didInitialScrollRef.current = true;
      onDidInitialScroll();
    }
  }, [
    initialScrollOffset,
    onDidInitialScroll,
    shouldDoInitialScroll,
    containerWidth,
    layout.height,
  ]);

  // Scroll to cursor when explicitly requested via scrollToCursorEventId
  useEffect(() => {
    if (containerWidth === 0) return;
    if (layout.items.length === 0) return;
    if (!scrollToCursorEventId) return;

    // Scroll to cursor item position
    if (cursor != null && cursor >= 0 && cursor < layout.items.length) {
      const itemPos = layout.items[cursor];
      if (itemPos && parentRef.current) {
        const targetScrollTop = Math.max(0, itemPos.y);
        parentRef.current.scrollTop = targetScrollTop;
        setScrollTop(targetScrollTop);
      }
    }

    libraryService.send('CLEAR_SCROLL_TO_CURSOR');
  }, [
    scrollToCursorEventId,
    libraryService,
    layout.items,
    containerWidth,
    cursor,
  ]);

  // Auto-scroll logic (copied from VirtualGrid)
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
        setScrollTop(parentRef.current.scrollTop); // Update state to trigger render
        animationFrameId = requestAnimationFrame(scroll);
      } else {
        cancelAnimationFrame(animationFrameId);
      }
    };

    animationFrameId = requestAnimationFrame(scroll);

    return () => cancelAnimationFrame(animationFrameId);
  }, [isDragging, offset, type, getScrollSpeed]);

  return (
    <div className="List" ref={parentRef} onScroll={handleScrollLocal}>
      <div
        className="ListContainer"
        style={{
          height: `${layout.height}px`,
          position: 'relative',
          width: '100%',
        }}
      >
        {visibleItems.map((p) => {
          // Determine if this item needs dimension loading
          const itemKey = getItemKey(p.item);
          const needsDimensionLoad =
            !p.item.width && !p.item.height && !localDimensions.has(itemKey);

          return (
            <div
              className="ListRow"
              key={
                p.item?.path +
                (p.item?.timeStamp != null ? String(p.item?.timeStamp) : 'null')
              }
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: `${p.width}px`,
                height: `${p.height}px`,
                transform: `translate3d(${p.x}px, ${p.y}px, 0)`,
              }}
            >
              <ListItem
                scaleMode={'cover'}
                height={p.height}
                item={p.item}
                idx={p.index}
                onDimensionsLoaded={
                  needsDimensionLoad ? handleDimensionsLoaded : undefined
                }
              />
            </div>
          );
        })}
      </div>
    </div>
  );
}
