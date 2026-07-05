import { useContext, useRef, useState } from 'react';
import { GlobalStateContext, Item } from '../state';
import { invoke, mediaServerBase } from '../platform';
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

export default function useTagDrop(item: Item, location: 'DETAIL' | 'LIST') {
  const { libraryService } = useContext(GlobalStateContext);
  const queryClient = useQueryClient();
  const [isLeft, setIsLeft] = useState<boolean>(false);
  // Lightweight selectors for hover behavior, others are read from snapshot on drop
  const actualVideoTime = useSelector(libraryService, (state) => {
    return state.context.videoPlayer.actualVideoTime;
  });

  const activeTag = useSelector(libraryService, (state) => {
    return state.context.dbQuery.tags[0];
  });

  const containerRef = useRef<HTMLDivElement>(null);

  type DropProps = {
    isOver: boolean;
    isSelf: boolean;
    itemType: string | symbol | null | undefined;
  };

  type DroppedTag = { label: string; category: string };
  type DroppedMedia = { path: string; timeStamp?: number };
  // A person card dragged from the taxonomy People grid.
  type DroppedPerson = { id: number; name: string };
  const isDroppedTag = (v: unknown): v is DroppedTag =>
    typeof v === 'object' && v != null && 'label' in v && 'category' in v;
  const isDroppedMedia = (v: unknown): v is DroppedMedia =>
    typeof v === 'object' && v != null && 'path' in v;

  const [collectedProps, drop] = useDrop<
    DroppedTag | DroppedMedia | DroppedPerson,
    unknown,
    DropProps
  >(
    () => ({
      accept: ['TAG', 'MEDIA', 'PERSON'],
      collect: (monitor) => ({
        isOver: monitor.isOver({ shallow: true }),
        isSelf: (() => {
          const dragged = monitor.getItem();
          if (isDroppedMedia(dragged)) {
            return (
              dragged.path === item?.path &&
              dragged.timeStamp === item?.timeStamp
            );
          }
          return false;
        })(),
        itemType: monitor.getItemType(),
      }),
      hover: (_droppedItem, monitor) => {
        const nextIsLeft = getIsLeft(monitor, containerRef);
        setIsLeft((prev) => (prev !== nextIsLeft ? nextIsLeft : prev));
      },
      drop: (droppedItem, monitor) => {
        // Get latest snapshot to compute library only when needed
        const snapshot = libraryService.getSnapshot();
        const ctx = snapshot.context;
        const { applyTagPreview, applyTagToAll } = ctx.settings;

        async function createAssignment(tag: DroppedTag) {
          let targetPaths: string[] = [item.path];
          if (applyTagToAll) {
            const activeLibrary: Item[] = filter(
              ctx.libraryLoadId,
              ctx.textFilter,
              ctx.library,
              ctx.settings.filters,
              ctx.settings.sortBy
            );
            targetPaths = activeLibrary.map((i: Item) => i.path);
          }
          if (applyTagToAll && targetPaths.length > 1) {
            libraryService.send({
              type: 'ADD_TOAST',
              data: {
                type: 'info',
                title: `Applying "${tag.label}" to ${targetPaths.length} items`,
                message: 'Bulk tagging in progress…',
                durationMs: 3000,
              },
            });
          }
          await invoke('create-assignment', [
            targetPaths,
            tag.label,
            tag.category,
            location === 'DETAIL' ? actualVideoTime : null,
            applyTagPreview,
          ]);
          libraryService.send('SET_MOST_RECENT_TAG', {
            tag: tag.label,
            category: tag.category,
          });
          queryClient.invalidateQueries({ queryKey: ['metadata'] });
          queryClient.invalidateQueries({
            queryKey: ['taxonomy', 'tag', tag.label],
          });
          queryClient.invalidateQueries({
            queryKey: ['tags-by-path'],
          });
        }
        if (isDroppedTag(droppedItem) && item.path) {
          // Surface failures — a rejected create-assignment (e.g. the shared
          // DB briefly locked by the Go media-server) used to vanish as an
          // unhandled rejection, making the drop look like it did nothing.
          createAssignment(droppedItem).catch((err) => {
            libraryService.send({
              type: 'ADD_TOAST',
              data: {
                type: 'error',
                title: `Could not apply "${droppedItem.label}"`,
                message:
                  err instanceof Error ? err.message : 'Tagging failed — try again.',
              },
            });
          });
        }

        // A person card dropped on media: assign this item's face to that
        // person (the server scans the item on the fly if it has no face
        // vectors yet). Shift at drop time additionally makes the face the
        // person's preview crop — analogous to the tag-preview behavior.
        async function assignPersonToMedia(person: DroppedPerson) {
          const setCover = !!(window as any).__shiftHeld;
          const headers: HeadersInit = { 'Content-Type': 'application/json' };
          if (ctx.authToken) {
            headers['Authorization'] = `Bearer ${ctx.authToken}`;
          }
          const res = await fetch(`${mediaServerBase}/api/media/assign-person`, {
            method: 'POST',
            headers,
            credentials: 'include',
            body: JSON.stringify({
              path: item.path,
              personId: person.id,
              setCover,
            }),
          });
          if (!res.ok) {
            let msg = `HTTP ${res.status}`;
            try {
              const body = await res.json();
              if (body?.error) msg = body.error;
            } catch {
              /* keep status message */
            }
            throw new Error(msg);
          }
          libraryService.send({
            type: 'ADD_TOAST',
            data: {
              type: 'success',
              title: setCover
                ? `Assigned to ${person.name} + preview updated`
                : `Assigned to ${person.name}`,
              message: item.path.split(/[/\\]/).pop() || item.path,
            },
          });
          queryClient.invalidateQueries({ queryKey: ['taxonomy'] });
          queryClient.invalidateQueries({ queryKey: ['metadata'] });
          queryClient.invalidateQueries({ queryKey: ['tags-by-path'] });
        }
        if (monitor.getItemType() === 'PERSON' && item.path) {
          const person = droppedItem as DroppedPerson;
          // Scanning on the fly (video frame + ONNX) can take a few seconds —
          // acknowledge the drop immediately so it doesn't feel dead.
          libraryService.send({
            type: 'ADD_TOAST',
            data: {
              type: 'info',
              title: `Matching face for ${person.name}…`,
              durationMs: 2500,
            },
          });
          assignPersonToMedia(person).catch((err) => {
            libraryService.send({
              type: 'ADD_TOAST',
              data: {
                type: 'error',
                title: `Could not assign ${person.name}`,
                message:
                  err instanceof Error ? err.message : 'Assignment failed — try again.',
              },
            });
          });
          return;
        }

        async function updateAssignmentWeight(media: DroppedMedia) {
          const activeLibrary: Item[] = filter(
            ctx.libraryLoadId,
            ctx.textFilter,
            ctx.library,
            ctx.settings.filters,
            ctx.settings.sortBy
          );

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
          await invoke('update-assignment-weight', [
            media.path,
            activeTag,
            newWeight,
            media.timeStamp,
          ]);
          libraryService.send({ type: 'SORTED_WEIGHTS' });
        }
        if (isDroppedMedia(droppedItem) && item.path) {
          updateAssignmentWeight(droppedItem);
        }
      },
    }),
    [item, libraryService, activeTag, actualVideoTime, location, queryClient]
  );

  return { drop, collectedProps, containerRef, isLeft };
}
