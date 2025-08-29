import { useContext, useMemo, useRef } from 'react';
import { GlobalStateContext, Item } from '../state';
import filter from '../filter';
import { useQueryClient } from '@tanstack/react-query';
import { useSelector } from '@xstate/react';

import { DropTargetMonitor, useDrop } from 'react-dnd';

function getIsLeft(
  monitor: DropTargetMonitor,
  containerRef: React.RefObject<HTMLDivElement>
): boolean {
  const hoverBoundingRect = containerRef.current?.getBoundingClientRect();
  const hoverMiddleX =
    (hoverBoundingRect?.left || 0) + (hoverBoundingRect?.width || 0) / 2;
  const mousePosition = monitor.getClientOffset()?.x;
  const isLeft = (mousePosition || 0) < hoverMiddleX;
  return isLeft;
}

export default function useTagDrop(
  item: any,
  location: 'DETAIL' | 'LIST',
  visibleLibrary?: Item[]
) {
  const { libraryService } = useContext(GlobalStateContext);
  const queryClient = useQueryClient();
  const { applyTagPreview, applyTagToAll, sortBy } = useSelector(
    libraryService,
    (state) => {
      return state.context.settings;
    }
  );
  // Active library: prefer provided list from parent to avoid recomputation per item
  const library = useSelector(
    libraryService,
    (state) => {
      return {
        library: state.context.library,
        libraryLoadId: state.context.libraryLoadId,
        textFilter: state.context.textFilter,
        filters: state.context.settings.filters,
        sortBy: state.context.settings.sortBy,
      };
    },
    (a, b) =>
      a.library === b.library &&
      a.libraryLoadId === b.libraryLoadId &&
      a.textFilter === b.textFilter &&
      a.filters === b.filters &&
      a.sortBy === b.sortBy
  );

  const activeLibrary: Item[] = useMemo(() => {
    if (visibleLibrary) return visibleLibrary;
    return filter(
      library.libraryLoadId,
      library.textFilter,
      library.library,
      library.filters,
      library.sortBy
    );
  }, [
    visibleLibrary,
    library.libraryLoadId,
    library.textFilter,
    library.library,
    library.filters,
    library.sortBy,
  ]);

  const actualVideoTime = useSelector(libraryService, (state) => {
    return state.context.videoPlayer.actualVideoTime;
  });

  const activeTag = useSelector(libraryService, (state) => {
    return state.context.dbQuery.tags[0];
  });

  const libraryPaths = useMemo(
    () => activeLibrary.map((i: Item) => i.path),
    [activeLibrary]
  );

  const containerRef = useRef<HTMLDivElement>(null);

  type DropProps = {
    isOver: boolean;
    isLeft: boolean;
    isSelf: boolean;
    itemType: string | symbol | null | undefined;
  };

  const [collectedProps, drop] = useDrop<Item, unknown, DropProps>(
    () => ({
      accept: ['TAG', 'MEDIA'],
      collect: (monitor) => ({
        isOver: monitor.isOver(),
        isLeft: getIsLeft(monitor, containerRef),
        isSelf:
          monitor.getItem()?.path === item?.path &&
          monitor.getItem()?.timeStamp === item?.timeStamp,
        itemType: monitor.getItemType(),
      }),
      drop: (droppedItem: any, monitor) => {
        // Check if dropped item is type Tag
        async function createAssignment() {
          console.log('INVOKING ASSIGNMENT WITH', droppedItem);
          await window.electron.ipcRenderer.invoke('create-assignment', [
            applyTagToAll ? libraryPaths : [item.path],
            droppedItem.label,
            droppedItem.category,
            location === 'DETAIL' ? actualVideoTime : null,
            applyTagPreview,
          ]);
          libraryService.send('SET_MOST_RECENT_TAG', {
            tag: droppedItem.label,
            category: droppedItem.category,
          });
          queryClient.invalidateQueries({ queryKey: ['metadata'] });
          queryClient.invalidateQueries({
            queryKey: ['taxonomy', 'tag', droppedItem.label],
          });
          queryClient.invalidateQueries({
            queryKey: ['tags-by-path', item.path],
          });
        }
        if (droppedItem.label && item.path) {
          createAssignment();
        }

        async function updateAssignmentWeight() {
          console.log('DROPPED MEDIA ON MEDIA');

          const index = activeLibrary.findIndex(
            (i: Item) => i.path === item.path
          );
          const targetWeight = activeLibrary[index]?.weight || 0;
          const previousItemWeight = activeLibrary[index - 1]?.weight || 0;
          const nextItemWeight =
            activeLibrary[index + 1]?.weight || activeLibrary.length + 1;
          const isLeft = getIsLeft(monitor, containerRef);
          const newWeight = isLeft
            ? (previousItemWeight + targetWeight) / 2
            : (nextItemWeight + targetWeight) / 2;
          console.log(
            'calling update-assignment-weight with:',
            droppedItem.path,
            activeTag,
            newWeight,
            droppedItem.timeStamp
          );
          await window.electron.ipcRenderer.invoke('update-assignment-weight', [
            droppedItem.path,
            activeTag,
            newWeight,
            droppedItem.timeStamp,
          ]);
          libraryService.send({ type: 'SORTED_WEIGHTS' });
        }
        if (droppedItem.path && item.path) {
          updateAssignmentWeight();
        }
      },
    }),
    [
      item,
      applyTagPreview,
      activeLibrary,
      applyTagToAll,
      activeTag,
      actualVideoTime,
    ]
  );

  return { drop, collectedProps, containerRef };
}
