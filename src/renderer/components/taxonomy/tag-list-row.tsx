import React, { useContext } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { useDrag, useDrop, DropTargetMonitor } from 'react-dnd';
import { useSelector } from '@xstate/react';
import deleteIcon from '../../../../assets/delete.svg';
import editPencil from '../../../../assets/edit-pencil.svg';
import { GlobalStateContext } from '../../state';
import ConfirmDeleteTag from './confirm-delete-tag';
import TagCount from './tag-count';
import { invoke } from '../../platform';

type Concept = {
  label: string;
  weight: number;
  category: string;
};

type Props = {
  tag: Concept;
  active: boolean;
  isDisabled: boolean;
  tags: Concept[];
  handleEditAction: (tag: string) => void;
  disableReorder?: boolean;
};

function TagListRow({
  tag,
  tags,
  active,
  handleEditAction,
  isDisabled,
  disableReorder = false,
}: Props) {
  const { libraryService } = useContext(GlobalStateContext);
  const showTagCount = useSelector(
    libraryService,
    (state) => state.context.settings.showTagCount
  );
  const queryClient = useQueryClient();
  const ref = React.useRef<HTMLDivElement>(null);
  const [showDeleteModal, setShowDeleteModal] = React.useState(false);

  function getIsAbove(
    monitor: DropTargetMonitor,
    containerRef: React.RefObject<HTMLDivElement>
  ): boolean {
    const rect = containerRef.current?.getBoundingClientRect();
    const middleY = (rect?.top || 0) + (rect?.height || 0) / 2;
    const mouseY = monitor.getClientOffset()?.y;
    return (mouseY || 0) < middleY;
  }

  const [, drag] = useDrag(
    () => ({
      collect: (monitor) => ({ isDragging: monitor.isDragging() }),
      type: 'TAG',
      item: tag,
    }),
    [tag]
  );

  type DropProps = {
    isOver: boolean;
    isAbove: boolean;
    isSelf: boolean;
  };
  const [isAbove, setIsAbove] = React.useState(false);
  const [collectedProps, drop] = useDrop<Concept, unknown, DropProps>(
    () => ({
      accept: ['TAG'],
      canDrop: () => !disableReorder,
      collect: (monitor) => ({
        isOver: monitor.isOver(),
        isAbove: getIsAbove(monitor, ref),
        isSelf: monitor.getItem()?.label === tag.label,
      }),
      hover: (_item: Concept, monitor) => {
        setIsAbove(getIsAbove(monitor, ref));
      },
      drop: (droppedTag: Concept, monitor) => {
        async function updateWeight() {
          const above = getIsAbove(monitor, ref);
          const index = tags.findIndex((i) => i.label === tag.label);
          const targetWeight = tag.weight || 0;
          const previousItemWeight = tags[index - 1]?.weight || 0;
          const nextItemWeight = tags[index + 1]?.weight || targetWeight + 10;
          const newWeight = above
            ? (previousItemWeight + targetWeight) / 2
            : (nextItemWeight + targetWeight) / 2;
          await invoke('update-tag-weight', [droppedTag.label, newWeight]);
          queryClient.invalidateQueries({ queryKey: ['taxonomy'] });
        }
        updateWeight();
      },
    }),
    [tag, tags, disableReorder]
  );
  drag(drop(ref));

  return (
    <div
      ref={ref}
      className={[
        'tag-list-row',
        active ? 'active' : '',
        collectedProps.isOver && !collectedProps.isSelf ? 'hovered' : '',
        collectedProps.isOver && isAbove ? 'above' : '',
        collectedProps.isOver && !isAbove ? 'below' : '',
        isDisabled ? 'disabled' : '',
      ]
        .filter(Boolean)
        .join(' ')}
      onClick={() => {
        if (!isDisabled) {
          libraryService.send({
            type: 'SET_QUERY_TAG',
            data: { tag: tag.label },
          });
        }
      }}
      onContextMenu={(e) => {
        if (e.shiftKey) {
          e.preventDefault();
          e.stopPropagation();
          libraryService.send('SHOW_CONTEXT_PALETTE', {
            position: { x: e.clientX, y: e.clientY },
            target: { type: 'tag', tag: tag.label },
          });
        }
      }}
    >
      <div className="label" title={tag.label}>
        {tag.label}
      </div>
      <div className="actions">
        {showTagCount ? <TagCount tag={tag} /> : null}
        <button
          disabled={isDisabled}
          className={isDisabled ? 'disabled' : ''}
          onClick={(e) => {
            e.stopPropagation();
            if (!isDisabled) handleEditAction(tag.label);
          }}
        >
          <img src={editPencil} />
        </button>
        <button
          disabled={isDisabled}
          className={isDisabled ? 'disabled' : ''}
          onClick={(e) => {
            e.stopPropagation();
            if (!isDisabled) setShowDeleteModal(true);
          }}
        >
          <img src={deleteIcon} />
        </button>
      </div>
      {showDeleteModal ? (
        <ConfirmDeleteTag
          handleClose={() => setShowDeleteModal(false)}
          currentValue={tag.label}
        />
      ) : null}
    </div>
  );
}

export default React.memo(TagListRow, (prev, next) => {
  return (
    prev.tag.label === next.tag.label &&
    prev.tag.weight === next.tag.weight &&
    prev.tag.category === next.tag.category &&
    prev.active === next.active &&
    prev.isDisabled === next.isDisabled &&
    prev.tags === next.tags &&
    prev.handleEditAction === next.handleEditAction
  );
});
