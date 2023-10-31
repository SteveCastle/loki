import { useContext, useMemo, useRef } from 'react';
import { GlobalStateContext, Item } from '../state';
import filter from '../filter';
import { useQueryClient, useMutation } from '@tanstack/react-query';
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

export default function useTagDrop(item: any, location: 'DETAIL' | 'LIST') {
  const { libraryService } = useContext(GlobalStateContext);
  const queryClient = useQueryClient();
  const { applyTagPreview, applyTagToAll, sortBy } = useSelector(
    libraryService,
    (state) => {
      return state.context.settings;
    }
  );
  // Get Entire active library
  const library = useSelector(
    libraryService,
    (state) => {
      return filter(
        state.context.libraryLoadId,
        state.context.textFilter,
        state.context.library,
        state.context.settings.filters,
        state.context.settings.sortBy
      );
    },
    (a, b) =>
      a.libraryLoadId === b.libraryLoadId &&
      a.filters === b.filters &&
      a.sortBy === b.sortBy
  );

  const actualVideoTime = useSelector(libraryService, (state) => {
    return state.context.videoPlayer.actualVideoTime;
  });

  const activeTag = useSelector(libraryService, (state) => {
    return state.context.dbQuery.tags[0];
  });

  const libraryPaths = useMemo(
    () => library.map((item: Item) => item.path),
    [library]
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
          queryClient.invalidateQueries({ queryKey: ['metadata'] });
          queryClient.invalidateQueries({
            queryKey: ['taxonomy', 'tag', droppedItem.label],
          });
          console.log('invalidated tag', droppedItem.label);
        }
        if (droppedItem.label && item.path) {
          createAssignment();
        }

        async function updateAssignmentWeight() {
          console.log('DROPPED MEDIA ON MEDIA');

          const index = library.findIndex((i: Item) => i.path === item.path);
          const targetWeight = library[index]?.weight || 0;
          const previousItemWeight = library[index - 1]?.weight || 0;
          const nextItemWeight =
            library[index + 1]?.weight || library.length + 1;
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
    [item, applyTagPreview, library, applyTagToAll, activeTag, actualVideoTime]
  );

  return { drop, collectedProps, containerRef };
}
