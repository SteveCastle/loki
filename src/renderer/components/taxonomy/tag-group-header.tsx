import React, { useContext, useState } from 'react';
import { useDrag, useDrop, DropTargetMonitor } from 'react-dnd';
import { useQueryClient } from '@tanstack/react-query';
import { useSelector } from '@xstate/react';
import editPencil from '../../../../assets/edit-pencil.svg';
import deleteIcon from '../../../../assets/delete.svg';
import { GlobalStateContext } from '../../state';
import { invoke } from '../../platform';
import { TagGroup, weightBeside } from './tag-groups';

// Drag type for reordering groups by their header. Separate from 'TAG' so a
// group can never be dropped onto a category, a tag, or media.
export const TAG_GROUP_DND_TYPE = 'TAG_GROUP';

type DraggedTag = { label: string; category: string; group?: string | null };
type DraggedGroup = { category: string; label: string };

type Props = {
  category: string;
  // null = the Ungrouped section: not renamable, deletable or draggable, but
  // still a drop target (dropping a tag there ungroups it).
  group: string | null;
  // The category's groups in display order, for reorder midpoints.
  groups: TagGroup[];
  count: number;
  collapsed: boolean;
  onToggleCollapsed: () => void;
  onAddTag: (group: string) => void;
  onRename: (group: string) => void;
  isDisabled: boolean;
};

function TagGroupHeader({
  category,
  group,
  groups,
  count,
  collapsed,
  onToggleCollapsed,
  onAddTag,
  onRename,
  isDisabled,
}: Props) {
  const { libraryService } = useContext(GlobalStateContext);
  const canWrite = useSelector(
    libraryService,
    (state) => state.context.canWrite
  );
  const queryClient = useQueryClient();
  const ref = React.useRef<HTMLDivElement>(null);
  const [confirmingDelete, setConfirmingDelete] = useState(false);
  const [isAbove, setIsAbove] = useState(false);

  function getIsAbove(monitor: DropTargetMonitor): boolean {
    const rect = ref.current?.getBoundingClientRect();
    const middleY = (rect?.top || 0) + (rect?.height || 0) / 2;
    return (monitor.getClientOffset()?.y || 0) < middleY;
  }

  function reportError(title: string, err: unknown) {
    libraryService.send({
      type: 'ADD_TOAST',
      data: { type: 'error', title, message: String(err) },
    });
  }

  const [{ isDragging }, drag] = useDrag(
    () => ({
      type: TAG_GROUP_DND_TYPE,
      item: { category, label: group } as DraggedGroup,
      canDrag: () => canWrite && group !== null,
      collect: (monitor) => ({ isDragging: monitor.isDragging() }),
    }),
    [category, group, canWrite]
  );

  const [{ isOver, itemType }, drop] = useDrop(
    () => ({
      accept: ['TAG', TAG_GROUP_DND_TYPE],
      canDrop: (item: DraggedTag | DraggedGroup, monitor) => {
        if (!canWrite) return false;
        if (monitor.getItemType() === TAG_GROUP_DND_TYPE) {
          const dragged = item as DraggedGroup;
          return (
            group !== null &&
            dragged.category === category &&
            dragged.label !== group
          );
        }
        return true;
      },
      collect: (monitor) => ({
        isOver: monitor.isOver() && monitor.canDrop(),
        itemType: monitor.getItemType(),
      }),
      hover: (_item, monitor) => {
        if (monitor.getItemType() !== TAG_GROUP_DND_TYPE) return;
        const next = getIsAbove(monitor);
        setIsAbove((prev) => (prev === next ? prev : next));
      },
      drop: (item: DraggedTag | DraggedGroup, monitor) => {
        async function apply() {
          if (monitor.getItemType() === TAG_GROUP_DND_TYPE) {
            // Reorder: land just above or below this header.
            const dragged = item as DraggedGroup;
            const others = groups.filter((g) => g.label !== dragged.label);
            const weight = weightBeside(
              others,
              group as string,
              getIsAbove(monitor)
            );
            await invoke('update-tag-group-weight', [
              category,
              dragged.label,
              weight,
            ]);
          } else {
            // A tag dropped on a header joins that group (or, on Ungrouped,
            // leaves its group). From another category it moves here first.
            const tag = item as DraggedTag;
            if (tag.category !== category) {
              await invoke('move-tag', [tag.label, category]);
            }
            await invoke('set-tag-group', [tag.label, group]);
          }
          queryClient.invalidateQueries({ queryKey: ['taxonomy'] });
        }
        apply().catch((err) => reportError('Failed to update group', err));
      },
    }),
    [category, group, groups, canWrite]
  );

  drag(drop(ref));

  async function handleDelete() {
    if (group === null) return;
    setConfirmingDelete(false);
    try {
      await invoke('delete-tag-group', [category, group]);
      queryClient.invalidateQueries({ queryKey: ['taxonomy'] });
    } catch (err) {
      reportError('Failed to delete group', err);
    }
  }

  const reordering = isOver && itemType === TAG_GROUP_DND_TYPE;
  return (
    <div
      ref={ref}
      className={[
        'tag-group-header',
        group === null ? 'ungrouped' : '',
        collapsed ? 'collapsed' : '',
        isDragging ? 'dragging' : '',
        isOver && !reordering ? 'hovered' : '',
        reordering && isAbove ? 'above' : '',
        reordering && !isAbove ? 'below' : '',
      ]
        .filter(Boolean)
        .join(' ')}
      onClick={onToggleCollapsed}
      title={collapsed ? 'Expand group' : 'Collapse group'}
    >
      <span className="tag-group-chevron" aria-hidden="true">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
          <path d="M6 9l6 6 6-6" />
        </svg>
      </span>
      <span className="tag-group-label">{group ?? 'Ungrouped'}</span>
      <span className="tag-group-count">{count}</span>
      {confirmingDelete ? (
        <div className="tag-group-confirm" onClick={(e) => e.stopPropagation()}>
          <span>Delete group? Its tags are kept.</span>
          <button className="confirm" onClick={handleDelete}>
            Delete
          </button>
          <button className="cancel" onClick={() => setConfirmingDelete(false)}>
            Cancel
          </button>
        </div>
      ) : (
        canWrite &&
        group !== null && (
          <div className="actions">
            <button
              disabled={isDisabled}
              title="New tag in this group"
              onClick={(e) => {
                e.stopPropagation();
                if (!isDisabled) onAddTag(group);
              }}
            >
              +
            </button>
            <button
              disabled={isDisabled}
              title="Rename group"
              onClick={(e) => {
                e.stopPropagation();
                if (!isDisabled) onRename(group);
              }}
            >
              <img src={editPencil} />
            </button>
            <button
              disabled={isDisabled}
              title="Delete group (keeps its tags)"
              onClick={(e) => {
                e.stopPropagation();
                if (!isDisabled) setConfirmingDelete(true);
              }}
            >
              <img src={deleteIcon} />
            </button>
          </div>
        )
      )}
    </div>
  );
}

export default React.memo(TagGroupHeader);
