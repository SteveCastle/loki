import { useContext, useEffect, useRef, useState } from 'react';
import { useSelector } from '@xstate/react';
import { GlobalStateContext } from '../../state';
import { useVirtualizer } from '@tanstack/react-virtual';
import { useDragLayer } from 'react-dnd';
import filter from '../../filter';
import { ListItem } from './list-item';
import './list-item.css';
import './list.css';

const OVERSCAN = 2;

export function List() {
  const [initialLoad, setInitialLoad] = useState(true);
  const { libraryService } = useContext(GlobalStateContext);

  const { library } = useSelector(
    libraryService,
    (state) => {
      return {
        filters: state.context.settings.filters,
        sortBy: state.context.settings.sortBy,
        library: filter(
          state.context.libraryLoadId,
          state.context.textFilter,
          state.context.library,
          state.context.settings.filters,
          state.context.settings.sortBy
        ),
        libraryLoadId: state.context.libraryLoadId,
      };
    },
    (a, b) =>
      a.libraryLoadId === b.libraryLoadId &&
      a.filters === b.filters &&
      a.sortBy === b.sortBy
  );
  const scrollPosition = useSelector(
    libraryService,
    (state) => state.context.scrollPosition
  );
  const cursor = useSelector(
    libraryService,
    (state) => state.context.cursor,
    (a, b) => a === b
  );
  const items = library;

  const [columns, rows] = useSelector(libraryService, (state) => {
    const columns = state.context.settings.gridSize[0];
    let rows = state.context.settings.gridSize[1];
    const totalNumberOfRows = Math.ceil(items.length / columns);
    if (totalNumberOfRows < rows) {
      rows = totalNumberOfRows;
    }
    return [columns, rows];
  });

  const [height, setHeight] = useState(window.innerHeight / rows);

  const parentRef = useRef<HTMLDivElement>(null);
  const listLength = Math.ceil(items.length / columns);
  // The virtualizer
  const rowVirtualizer = useVirtualizer({
    count: listLength,
    getScrollElement: () => parentRef.current,
    estimateSize: () => height,
    overscan: OVERSCAN,
  });

  useEffect(() => {
    setHeight(window.innerHeight / rows);
    rowVirtualizer.measure();
    const handleResize = () => {
      setHeight(window.innerHeight / rows);
      rowVirtualizer.measure();
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [columns, rows, library, rowVirtualizer]);

  useEffect(() => {
    if (initialLoad && parentRef.current && cursor) {
      rowVirtualizer.scrollToOffset(scrollPosition);
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

  function mapRange(
    value: number,
    low1: number,
    high1: number,
    low2: number,
    high2: number
  ) {
    return low2 + ((high2 - low2) * (value - low1)) / (high1 - low1);
  }

  function getScrollSpeed(offset: number, height: number) {
    let scrollSpeed = 0;
    if (offset < 200) {
      // The smaller the offset the faster it scrolls.
      scrollSpeed = mapRange(offset, 0, 200, -200, 0);
    } else if (offset > height - 200 && offset < height) {
      // The closer to the bottom the faster it scrolls.
      scrollSpeed = mapRange(offset, height - 200, height, 0, 200);
    }

    return scrollSpeed;
  }

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
  }, [isDragging, offset]); // Dependency array means this effect runs whenever isDragging or offset changes
  return (
    <div
      className="List"
      ref={parentRef}
      onScroll={(e) => {
        const target = e.target as HTMLDivElement;
        libraryService.send('SET_SCROLL_POSITION', {
          position: target.scrollTop,
        });
      }}
    >
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
                    (item?.timeStamp
                      ? item?.timeStamp.toString()
                      : isNaN(item?.timeStamp)
                      ? 'null'
                      : item?.timeStamp)
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
