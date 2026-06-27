import React, { useContext } from 'react';
import { GlobalStateContext } from '../../state';
import Layout from './layout';
import useFileDrop from '../../hooks/useFileDrop';
import { useWarmTagSearch } from '../../hooks/useWarmTagSearch';

export function Panels() {
  const { libraryService } = useContext(GlobalStateContext);
  const { dropRef, isOver, canDrop } = useFileDrop();
  // Build the shared tag-search index at startup so the first type-ahead from
  // the taxonomy sidebar or command palette is instant.
  useWarmTagSearch();

  return (
    <>
      <div className="drag-handle" />
      <div
        ref={dropRef}
        className={`Panels${isOver && canDrop ? ' file-drop-active' : ''}`}
        onContextMenu={(e) => {
          e.preventDefault();
          if (e.shiftKey) {
            libraryService.send('SHOW_CONTEXT_PALETTE', {
              position: { x: e.clientX, y: e.clientY },
              target: { type: 'library' },
            });
          } else {
            libraryService.send('SHOW_COMMAND_PALETTE', {
              position: { x: e.clientX, y: e.clientY },
            });
          }
        }}
      >
        <Layout />
      </div>
    </>
  );
}
