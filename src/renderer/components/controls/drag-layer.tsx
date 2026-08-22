import React, { useContext, useLayoutEffect, useRef } from 'react';
import { useDragDropManager, useDragLayer } from 'react-dnd';
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

// Follows the cursor without going through React. Offset updates arrive on
// every dragover event; collecting them via useDragLayer would re-render the
// layer (and reconcile the chip subtree) per mousemove, which reads as drag
// jank. Instead the chip renders once per drag and a monitor subscription
// writes the transform straight to the anchor element, coalesced to one
// write per animation frame. The anchor starts visibility:hidden (see CSS)
// so it never flashes at 0,0 before the first position write.
function ChipFollower({ children }: { children: React.ReactNode }) {
  const manager = useDragDropManager();
  const anchorRef = useRef<HTMLDivElement>(null);

  useLayoutEffect(() => {
    const monitor = manager.getMonitor();
    let frameId = 0;
    const position = () => {
      frameId = 0;
      const el = anchorRef.current;
      const offset = monitor.getClientOffset();
      if (!el || !offset) return;
      el.style.transform = `translate3d(${offset.x}px, ${offset.y}px, 0)`;
      el.style.visibility = 'visible';
    };
    position();
    const unsubscribe = monitor.subscribeToOffsetChange(() => {
      if (!frameId) frameId = requestAnimationFrame(position);
    });
    return () => {
      unsubscribe();
      if (frameId) cancelAnimationFrame(frameId);
    };
  }, [manager]);

  return (
    <div className="drag-chip-layer" aria-hidden="true">
      <div className="drag-chip-anchor" ref={anchorRef}>
        {children}
      </div>
    </div>
  );
}

export default function DragChipLayer() {
  const { libraryService } = useContext(GlobalStateContext);
  const authToken = useSelector(libraryService, (s) => s.context.authToken);
  // Deliberately does NOT collect the cursor offset — everything collected
  // here is stable for the duration of a drag, so this component only
  // renders at drag start/end. ChipFollower tracks the cursor imperatively.
  const { itemType, item, isDragging } = useDragLayer((monitor) => ({
    item: monitor.getItem(),
    itemType: monitor.getItemType(),
    isDragging: monitor.isDragging(),
  }));

  if (!isDragging || (itemType !== 'TAG' && itemType !== 'PERSON')) {
    return null;
  }

  return (
    <ChipFollower>
      {itemType === 'TAG' ? (
        <TagChip tag={item as TagItem} />
      ) : (
        <PersonChip person={item as PersonItem} authToken={authToken} />
      )}
    </ChipFollower>
  );
}
