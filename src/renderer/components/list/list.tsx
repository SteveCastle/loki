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
  // Read initial scroll position once to avoid re-renders on scroll
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
    const columns = base.gridSize[0];
    let rows = base.gridSize[1];
    const totalNumberOfRows = Math.ceil(items.length / columns);
    if (totalNumberOfRows < rows) {
      rows = totalNumberOfRows;
    }
    return [columns, rows];
  }, [base.gridSize, items.length]);

  const [height, setHeight] = useState(window.innerHeight / rows);

  const parentRef = useRef<HTMLDivElement>(null);
  const listLength = useMemo(
    () => Math.ceil(items.length / columns),
    [items.length, columns]
  );
  const rowVirtualizer = useVirtualizer({
    count: listLength,
    getScrollElement: () => parentRef.current,
    estimateSize: () => height,
    overscan: PERFORMANCE_CONSTANTS.LIST_OVERSCAN,
  });

  useEffect(() => {
    setHeight(window.innerHeight / rows);
    const handleResize = () => {
      setHeight(window.innerHeight / rows);
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [rows]);

  // if height changes run   rowVirtualizer.measure();
  useEffect(() => {
    rowVirtualizer.measure();
  }, [height]);

  useEffect(() => {
    if (initialLoad && parentRef.current && cursor) {
      rowVirtualizer.scrollToOffset(initialScrollPositionRef.current);
      setInitialLoad(false);
    } else if (parentRef.current && cursor) {
      rowVirtualizer.scrollToIndex(Math.floor(cursor / columns), {
        align: 'auto',
      });
    }
  }, [rowVirtualizer, cursor]);

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
    (offset: number, height: number) => {
      let scrollSpeed = 0;
      const threshold = PERFORMANCE_CONSTANTS.SCROLL_SPEED_THRESHOLD;
      const minSpeed = PERFORMANCE_CONSTANTS.SCROLL_SPEED_RANGE.MIN;
      const maxSpeed = PERFORMANCE_CONSTANTS.SCROLL_SPEED_RANGE.MAX;

      if (offset < threshold) {
        // The smaller the offset the faster it scrolls.
        scrollSpeed = mapRange(offset, 0, threshold, minSpeed, 0);
      } else if (offset > height - threshold && offset < height) {
        // The closer to the bottom the faster it scrolls.
        scrollSpeed = mapRange(offset, height - threshold, height, 0, maxSpeed);
      }

      return scrollSpeed;
    },
    [mapRange]
  );

  useEffect(() => {
    let animationFrameId: number;
    const height = parentRef.current?.clientHeight;

    const scroll = () => {
      if (
        isDragging &&
        type === 'MEDIA' &&
        offset &&
        parentRef.current &&
        height
      ) {
        const mousePosition = offset.y;
        const scrollSpeed = getScrollSpeed(mousePosition, height);
        parentRef.current.scrollBy(0, scrollSpeed);
        // Call requestAnimationFrame again to keep the loop going
        animationFrameId = requestAnimationFrame(scroll);
      } else {
        // If dragging has stopped, cancel the animation frame
        cancelAnimationFrame(animationFrameId);
      }
    };

    // Call scroll for the first time
    animationFrameId = requestAnimationFrame(scroll);

    return () => cancelAnimationFrame(animationFrameId);
  }, [isDragging, offset, getScrollSpeed]); // Dependency array means this effect runs whenever isDragging or offset changes

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
    <div className="List" ref={parentRef} onScroll={handleScroll}>
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
                  visibleLibrary={items}
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
