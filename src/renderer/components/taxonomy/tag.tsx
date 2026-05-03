import React, { useContext } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { NativeTypes } from 'react-dnd-html5-backend';
import deleteIcon from '../../../../assets/delete.svg';
import editPencil from '../../../../assets/edit-pencil.svg';
import checkCircle from '../../../../assets/check-circle.svg';
import { GlobalStateContext } from '../../state';
import { useDrag, useDrop, DropTargetMonitor } from 'react-dnd';
import ConfirmDeleteTag from './confirm-delete-tag';
import './taxonomy.css';
import { getFileType } from 'file-types';
import { useSelector } from '@xstate/react';
import TagCount from './tag-count';
import { invoke, mediaUrl, fetchTagPreview } from '../../platform';
import { isClickWithinThreshold, Point } from './click-vs-drag';

type Concept = {
  label: string;
  weight: number;
  category: string;
  // Bundled in the taxonomy load so we don't have to issue a separate
  // fetchTagPreview IPC per tag at startup.
  thumbnail_path_600?: string | null;
};

type Props = {
  tag: Concept;
  active: boolean;
  isDisabled: boolean;
  tags: Concept[];
  handleEditAction: (tag: string) => void;
  disableReorder?: boolean;
};

const fetchTagPreviewFn = (tag: string) => async (): Promise<string> => {
  const path = await fetchTagPreview(tag);
  return path ?? '';
};

export default function Tag({
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
  // Tracks the mousedown coordinate so we can fire selection only on real
  // clicks. Same DOM node is the drag source, and HTML5 dnd suppresses
  // synthetic `click` events once any cursor movement starts a drag, so
  // relying on `onClick` here would drop intentional taps with mild jitter.
  const mouseDownPosRef = React.useRef<Point | null>(null);
  const [showDeleteModal, setShowDeleteModal] = React.useState(false);
  // Prefer the thumbnail bundled in the taxonomy payload. Fall back to the
  // dedicated fetchTagPreview IPC only when the bundled value is missing
  // (e.g. legacy callers that haven't loaded taxonomy through the updated
  // query). When the bundled value is present, `enabled: false` skips the
  // network roundtrip entirely.
  const bundledPreview = tag.thumbnail_path_600 || '';
  const { data: fetchedPreview } = useQuery<string, Error>(
    ['taxonomy', 'tag', tag.label],
    fetchTagPreviewFn(tag.label),
    { enabled: !bundledPreview }
  );
  const previewImage = bundledPreview || fetchedPreview;

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

  const [{ isDragging, offset }, drag, dragPreview] = useDrag(
    () => ({
      collect: (monitor) => ({
        isDragging: monitor.isDragging(),
        offset: monitor.getClientOffset(),
      }),
      type: 'TAG',
      item: tag,
    }),
    [tag]
  );

  type DropProps = {
    isOver: boolean;
    isLeft: boolean;
    isSelf: boolean;
    itemType: string | symbol | null | undefined;
  };
  const [isLeft, setIsLeft] = React.useState(false);
  const [collectedProps, drop] = useDrop<Concept, unknown, DropProps>(
    () => ({
      accept: ['TAG', NativeTypes.FILE],
      canDrop: () => !disableReorder,
      collect: (monitor) => ({
        isOver: monitor.isOver(),
        isLeft: getIsLeft(monitor, ref),
        isSelf: monitor.getItem()?.label === tag.label,
        itemType: monitor.getItemType(),
      }),
      hover: (item: Concept, monitor) => {
        // Calculate isLeft value.
        const isLeft = getIsLeft(monitor, ref);
        setIsLeft(isLeft);
      },
      drop: (droppedTag: Concept, monitor) => {
        async function updateWeight() {
          const isLeft = getIsLeft(monitor, ref);
          const index = tags.findIndex((i) => i.label === tag.label);
          const targetWeight = tag.weight || 0;
          const previousItemWeight = tags[index - 1]?.weight || 0;
          const nextItemWeight = tags[index + 1]?.weight || targetWeight + 10;
          const newWeight = isLeft
            ? (previousItemWeight + targetWeight) / 2
            : (nextItemWeight + targetWeight) / 2;
          await invoke('update-tag-weight', [
            droppedTag.label,
            newWeight,
          ]);
          queryClient.invalidateQueries({ queryKey: ['taxonomy'] });
        }
        updateWeight();
      },
    }),
    [tag, disableReorder]
  );
  drag(drop(ref));
  return (
    <div
      key={tag.label}
      className={[
        'tag',
        active ? 'active' : '',
        collectedProps.isOver && !collectedProps.isSelf ? 'hovered' : '',
        collectedProps.isOver && isLeft ? 'left' : '',
        collectedProps.isOver && !isLeft ? 'right' : '',
        isDisabled ? 'disabled' : '',
      ].join(' ')}
      ref={ref}
      onMouseDown={(e) => {
        if (e.button !== 0) return;
        mouseDownPosRef.current = { x: e.clientX, y: e.clientY };
      }}
      onMouseUp={(e) => {
        if (e.button !== 0) return;
        const start = mouseDownPosRef.current;
        mouseDownPosRef.current = null;
        if (isDisabled) return;
        if (!isClickWithinThreshold(start, { x: e.clientX, y: e.clientY })) {
          return;
        }
        libraryService.send({
          type: 'SET_QUERY_TAG',
          data: { tag: tag.label },
        });
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
      {previewImage ? (
        getFileType(previewImage) !== 'video' ? (
          <img
            src={mediaUrl(previewImage)}
          />
        ) : (
          <video
            src={mediaUrl(previewImage)}
            controls={false}
            autoPlay
            loop
          />
        )
      ) : null}
      <div className="label">{tag.label}</div>
      {active && <img className="check" src={checkCircle} />}
      <div className="actions">
        {showTagCount ? <TagCount tag={tag} /> : null}
        <button
          disabled={isDisabled}
          className={isDisabled ? 'disabled' : ''}
          onMouseDown={(e) => e.stopPropagation()}
          onMouseUp={(e) => e.stopPropagation()}
          onClick={(e) => {
            e.stopPropagation();
            if (!isDisabled) {
              handleEditAction(tag.label);
            }
          }}
        >
          <img src={editPencil} />
        </button>
        <button
          disabled={isDisabled}
          className={isDisabled ? 'disabled' : ''}
          onMouseDown={(e) => e.stopPropagation()}
          onMouseUp={(e) => e.stopPropagation()}
          onClick={(e) => {
            e.stopPropagation();
            if (!isDisabled) {
              setShowDeleteModal(true);
            }
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
