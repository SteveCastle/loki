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

type Concept = {
  label: string;
  weight: number;
  category: string;
};

type Props = {
  tag: Concept;
  active: boolean;
  tags: Concept[];
  handleEditAction: (tag: string) => void;
};

const fetchTagPreview = (tag: string) => async (): Promise<string> => {
  const path = await window.electron.fetchTagPreview(tag);
  return path;
};

export default function Tag({ tag, tags, active, handleEditAction }: Props) {
  const { libraryService } = useContext(GlobalStateContext);
  const queryClient = useQueryClient();
  const ref = React.useRef<HTMLDivElement>(null);
  const [showDeleteModal, setShowDeleteModal] = React.useState(false);
  const { data: previewImage } = useQuery<string, Error>(
    ['taxonomy', 'tag', tag.label],
    fetchTagPreview(tag.label)
  );

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

  const [{ isDragging, offset }, drag, dragPreview] = useDrag(() => ({
    collect: (monitor) => ({
      isDragging: monitor.isDragging(),
      offset: monitor.getClientOffset(),
    }),
    type: 'TAG',
    item: tag,
  }));

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
          await window.electron.ipcRenderer.invoke('update-tag-weight', [
            droppedTag.label,
            newWeight,
          ]);
          queryClient.invalidateQueries({ queryKey: ['taxonomy'] });
        }
        updateWeight();
      },
    }),
    [tag]
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
      ].join(' ')}
      ref={ref}
      onClick={() => {
        libraryService.send({
          type: 'SET_QUERY_TAG',
          data: { tag: tag.label },
        });
        // Set the sortBy setting to 'weight' when a tag is clicked
        libraryService.send({
          type: 'CHANGE_SETTING',
          data: { sortBy: 'weight' },
        });
      }}
    >
      {previewImage && getFileType(previewImage) !== 'video' ? (
        <img
          src={window.electron.url.format({
            protocol: 'gsm',
            pathname: previewImage,
          })}
        />
      ) : (
        <video
          src={window.electron.url.format({
            protocol: 'gsm',
            pathname: previewImage,
          })}
          controls={false}
          autoPlay
          loop
        />
      )}
      <div className="label">{tag.label}</div>
      {active && <img className="check" src={checkCircle} />}
      <div className="actions">
        <button
          onClick={(e) => {
            e.stopPropagation();
            handleEditAction(tag.label);
          }}
        >
          <img src={editPencil} />
        </button>
        <button
          onClick={(e) => {
            e.stopPropagation();
            setShowDeleteModal(true);
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
