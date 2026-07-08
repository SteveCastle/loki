import React, { useContext } from 'react';
import { useDragLayer } from 'react-dnd';
import { useSelector } from '@xstate/react';
import { getFileType } from 'file-types';
import { GlobalStateContext } from '../../state';
import { mediaUrl } from '../../platform';
import { displayTagLabel } from '../../tag-display';
import { FaceCrop } from '../taxonomy/people-grid';
import './drag-layer.css';

// Custom drag preview for TAG and PERSON drags. The drag sources suppress
// the browser's native drag ghost (a translucent snapshot of the whole
// source element — a full tag card or list row) via
// useHideNativeDragPreview, and this layer renders a compact chip that
// follows the cursor instead. MEDIA and native-file drags keep their
// default previews, so this renders nothing for them.

type TagItem = {
  label: string;
  category: string;
  thumbnail_path_600?: string | null;
};

type PersonItem = {
  id: number;
  name: string;
  coverFaceId?: number;
};

function TagChip({ tag }: { tag: TagItem }) {
  const preview = tag.thumbnail_path_600 || '';
  return (
    <div className="drag-chip">
      {preview ? (
        getFileType(preview) !== 'video' ? (
          <img className="drag-chip-thumb" src={mediaUrl(preview)} alt="" />
        ) : (
          <video
            className="drag-chip-thumb"
            src={mediaUrl(preview)}
            muted
            autoPlay
            loop
          />
        )
      ) : (
        <span className="drag-chip-glyph" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M20.6 13.4 11 3.8A2 2 0 0 0 9.6 3.2H5a2 2 0 0 0-2 2v4.6c0 .5.2 1 .6 1.4l9.6 9.6a2 2 0 0 0 2.8 0l4.6-4.6a2 2 0 0 0 0-2.8Z" />
            <circle cx="7.5" cy="7.5" r="0.5" fill="currentColor" />
          </svg>
        </span>
      )}
      <span className="drag-chip-label">{displayTagLabel(tag.label)}</span>
    </div>
  );
}

function PersonChip({
  person,
  authToken,
}: {
  person: PersonItem;
  authToken: string | null;
}) {
  return (
    <div className="drag-chip drag-chip-person">
      <span className="drag-chip-thumb drag-chip-face">
        <FaceCrop faceId={person.coverFaceId} authToken={authToken} size={64} />
      </span>
      <span className="drag-chip-label">{displayTagLabel(person.name)}</span>
    </div>
  );
}

export default function DragChipLayer() {
  const { libraryService } = useContext(GlobalStateContext);
  const authToken = useSelector(libraryService, (s) => s.context.authToken);
  const { itemType, item, offset, isDragging } = useDragLayer((monitor) => ({
    item: monitor.getItem(),
    itemType: monitor.getItemType(),
    offset: monitor.getClientOffset(),
    isDragging: monitor.isDragging(),
  }));

  if (
    !isDragging ||
    !offset ||
    (itemType !== 'TAG' && itemType !== 'PERSON')
  ) {
    return null;
  }

  return (
    <div className="drag-chip-layer" aria-hidden="true">
      <div
        className="drag-chip-anchor"
        style={{ transform: `translate(${offset.x}px, ${offset.y}px)` }}
      >
        {itemType === 'TAG' ? (
          <TagChip tag={item as TagItem} />
        ) : (
          <PersonChip person={item as PersonItem} authToken={authToken} />
        )}
      </div>
    </div>
  );
}
